"""Integrated cross-platform paper opportunity scanner.

Paper-only: this module never places orders.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.api.kalshi_client import get_market_orderbook, get_series_info, get_market_details
from src.api.polymarket_client import (
    get_active_markets,
    get_market_by_id,
    get_orderbook,
    get_orderbooks,
    parse_token_ids,
)
from src.arbitrage.exact_fees import kalshi_fee, polymarket_taker_fee
from src.arbitrage.execution_utils import (
    choose_maker_price,
    consume_asks,
    floor_contracts,
    kalshi_tick_size,
)
from src.arbitrage.universal_matcher import find_universal_matches
from src.api.market_metadata import get_open_kalshi_events, build_kalshi_metadata_indexes
from src.data.paper_v2_logger import log_opportunities


CACHE_PATH = Path("data/processed/open_markets.parquet")
UNIVERSE_PATH = Path("data/processed/validated_cross_platform.csv")
EQUIVALENCE_AUDIT_PATH = Path("data/processed/equivalence_audit.csv")
EQUIVALENCE_SUMMARY_PATH = Path("data/processed/equivalence_summary.csv")


@dataclass
class EngineConfig:
    bankroll: float = 100.0
    max_bankroll_fraction: float = 0.25
    max_contracts: int = 500
    min_net_per_contract: float = 0.0025
    min_return_on_capital: float = 0.001
    max_quote_skew_seconds: float = 3.0
    improve_maker_by_one_tick: bool = True
    allow_taker_taker: bool = True
    min_poly_liquidity: float = 0.0
    min_poly_volume24hr: float = 0.0
    max_results: int = 30


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series_ticker(ticker):
    return ticker.split("-")[0]


def build_validated_universe(kalshi: pd.DataFrame, polymarket: list[dict]):
    # Metadata-first matching.  If the public event metadata endpoint is
    # temporarily unavailable, the matcher safely falls back to the fields in
    # the cached Kalshi dataframe and the previously validated legacy matcher.
    market_metadata = {}
    try:
        events, milestones = get_open_kalshi_events(with_nested_markets=True, with_milestones=True)
        _, market_metadata = build_kalshi_metadata_indexes(events, milestones)
        print(f"Kalshi metadata: {len(events)} open events | {len(market_metadata)} nested markets")
    except Exception as exc:
        print(f"Kalshi metadata enrichment unavailable; conservative fallback used: {type(exc).__name__}: {exc}")

    matches, audits = find_universal_matches(
        kalshi, polymarket, market_metadata, include_legacy=True,
        kalshi_detail_fetcher=get_market_details,
        polymarket_detail_fetcher=get_market_by_id,
    )
    rows = []
    for match in matches:
        sig = match["signature"]
        pm = match["polymarket_market"]
        rows.append({
            "kalshi_ticker": match["kalshi_ticker"],
            "kalshi_title": match["kalshi_title"],
            "subject": sig.get("subject"),
            "topic": sig.get("topic"),
            "domain": sig.get("domain"),
            "competition": sig.get("competition"),
            "stage": sig.get("stage"),
            "year": sig.get("year"),
            "poly_id": pm.get("id"),
            "poly_question": match["polymarket_question"],
            "match_source": match.get("match_source"),
            "match_tier": str(match.get("match_source") or "").split(":",1)[0],
            "equivalence_score": match.get("equivalence_score"),
            "equivalence_reasons": match.get("equivalence_reasons"),
            "resolution_lane": (match.get("equivalence_certificate") or {}).get("resolution_lane", "STRICT_ARB"),
            "resolution_rule_status": (match.get("equivalence_certificate") or {}).get("resolution_rule_status"),
            "basis_risk_reserve_per_contract": (match.get("equivalence_certificate") or {}).get("basis_risk_reserve_per_contract", 0.0),
            "equivalence_certificate": repr(match.get("equivalence_certificate") or {}),
            "poly_volume24hr": _safe_float(pm.get("volume24hr")),
            "poly_liquidity": _safe_float(pm.get("liquidityNum", pm.get("liquidity"))),
        })
    pd.DataFrame(rows).to_csv(UNIVERSE_PATH, index=False)

    # V8.9: audits is deliberately bounded in memory.  The expanded market
    # universe can generate millions of rejected candidate pairs; materializing
    # all of them into pandas caused V8.8's ~9.7 GB Arrow allocation failure.
    # Save a representative sample for inspection and exact aggregate counters
    # for diagnostics/GitHub figures.
    audit_rows = [{
        "verdict": a.verdict, "score": a.score,
        "kalshi_ticker": a.kalshi_ticker, "kalshi_title": a.kalshi_title,
        "polymarket_question": a.polymarket_question,
        "reasons": "; ".join(a.reasons),
        "kalshi_signature": repr(a.kalshi_signature),
        "polymarket_signature": repr(a.polymarket_signature),
    } for a in audits]
    pd.DataFrame(audit_rows).to_csv(EQUIVALENCE_AUDIT_PATH, index=False)
    counts = getattr(audits, "verdict_counts", {})
    exact = int(counts.get("EXACT", sum(a.verdict == "EXACT" for a in audits)))
    high_conf = int(counts.get("HIGH_CONFIDENCE", sum(a.verdict == "HIGH_CONFIDENCE" for a in audits)))
    review = int(counts.get("REVIEW", sum(a.verdict == "REVIEW" for a in audits)))
    rejected = int(counts.get("REJECT", sum(a.verdict == "REJECT" for a in audits)))
    summary_rows = []
    for verdict_name, count in (("EXACT", exact), ("HIGH_CONFIDENCE", high_conf), ("REVIEW", review), ("REJECT", rejected)):
        summary_rows.append({"section": "verdict", "key": verdict_name, "count": count})
    reason_counts = getattr(audits, "reason_counts", {})
    for reason, count in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]:
        summary_rows.append({"section": "reason", "key": reason, "count": int(count)})
    summary_rows.append({"section": "audit", "key": "candidate_pairs_evaluated", "count": int(getattr(audits, "total_seen", len(audits)))})
    summary_rows.append({"section": "audit", "key": "rows_retained_in_memory", "count": len(audits)})
    stage_counts = getattr(audits, "stage_counts", {}) or {}
    for stage, count in sorted(stage_counts.items(), key=lambda kv: kv[0]):
        summary_rows.append({"section": "funnel", "key": stage, "count": int(count)})
    pd.DataFrame(summary_rows).to_csv(EQUIVALENCE_SUMMARY_PATH, index=False)
    low_basis = sum(
        1 for m in matches
        if str((m.get("equivalence_certificate") or {}).get("resolution_lane") or "").upper() == "LOW_BASIS"
    )
    strict_accepted = len(matches) - low_basis
    print(
        f"V27 MATCHER: EXACT={exact} | HIGH_CONFIDENCE={high_conf} | REVIEW={review} | REJECT={rejected} "
        f"| accepted={len(matches)} | STRICT_ARB={strict_accepted} | LOW_BASIS={low_basis}"
    )
    if stage_counts:
        ordered = ["polymarket_signatures", "kalshi_signatures", "kalshi_with_candidates", "retrieved_topk_pairs", "unique_pairs_verified", "rule_hydration_candidates", "accepted_pairs"]
        print("V27 MATCH FUNNEL | " + " | ".join(f"{k}={int(stage_counts.get(k,0))}" for k in ordered))
        focused = [
            "semantic_relaxed_recovered", "v84_ambiguous_key_skipped",
        ]
        print("V27 RETRIEVAL | " + " | ".join(f"{k}={int(stage_counts.get(k,0))}" for k in focused))
        hydration = {k:v for k,v in stage_counts.items() if str(k).startswith("hydration_")}
        if hydration:
            print("V27 rule hydration | " + " ".join(f"{k}={v}" for k,v in sorted(hydration.items())))
    print(f"Equivalence audit sample: {EQUIVALENCE_AUDIT_PATH} | retained={len(audits)} / evaluated={getattr(audits, 'total_seen', len(audits))}")
    print(f"Equivalence summary: {EQUIVALENCE_SUMMARY_PATH}")
    return matches


def _poly_ask_levels(token_id):
    book = get_orderbook(token_id)
    return [
        {"price": float(level["price"]), "size": float(level["size"])}
        for level in book.get("asks", [])
    ]


def _kalshi_levels(book, side):
    key = f"{side}_dollars"
    return [
        {"price": float(price), "size": float(size)}
        for price, size in book.get(key, [])
    ]


def _best_bid(levels):
    if not levels:
        return None, 0.0
    level = max(levels, key=lambda x: x["price"])
    return level["price"], level["size"]


def _best_ask_from_opposite_bid(opposite_bid):
    return round(1.0 - opposite_bid, 4) if opposite_bid is not None else None


def _activity_score(poly_market):
    volume24 = max(_safe_float(poly_market.get("volume24hr")), 0.0)
    liquidity = max(_safe_float(poly_market.get("liquidityNum", poly_market.get("liquidity"))), 0.0)
    # bounded, deliberately modest: activity helps ranking but never manufactures an edge
    return 1.0 + min(math.log1p(volume24) / 10.0, 0.75) + min(math.log1p(liquidity) / 20.0, 0.50)


def _maker_candidate(match, row, poly_market, kbook, poly_levels, maker_side, config, series_info, quote_skew):
    yes_levels = _kalshi_levels(kbook, "yes")
    no_levels = _kalshi_levels(kbook, "no")
    yes_bid, yes_queue = _best_bid(yes_levels)
    no_bid, no_queue = _best_bid(no_levels)
    yes_ask = _best_ask_from_opposite_bid(no_bid)
    no_ask = _best_ask_from_opposite_bid(yes_bid)

    if maker_side == "yes":
        bid, ask, queue = yes_bid, yes_ask, yes_queue
        poly_token_levels = poly_levels["no"]
        strategy = "K YES maker + P NO taker"
    else:
        bid, ask, queue = no_bid, no_ask, no_queue
        poly_token_levels = poly_levels["yes"]
        strategy = "K NO maker + P YES taker"

    if bid is None or not poly_token_levels:
        return None

    tick = kalshi_tick_size(row, bid)
    maker_price = choose_maker_price(bid, ask, tick, config.improve_maker_by_one_tick)
    if maker_price is None:
        return None

    # If we improve the best bid by a valid tick, there is no displayed
    # quantity ahead of us at the new price at the instant of the snapshot.
    # If we merely join the best bid, the displayed size is a conservative
    # estimate of queue ahead.
    if maker_price > bid + 1e-12:
        queue = 0.0

    max_capital = config.bankroll * config.max_bankroll_fraction
    max_qty = min(config.max_contracts, floor_contracts(max_capital / max(maker_price + poly_token_levels[0]["price"], 1e-9)))
    if max_qty < 1:
        return None

    best = None
    for qty in sorted(set([1, 2, 5, 10, 20, 25, 50, 100, 250, 500, max_qty])):
        if qty < 1 or qty > max_qty:
            continue
        depth = consume_asks(poly_token_levels, qty)
        if not depth.fully_filled:
            continue
        kfee = kalshi_fee(
            price=maker_price,
            contracts=qty,
            fee_type=series_info.get("fee_type"),
            fee_multiplier=series_info.get("fee_multiplier") or 0,
            maker=True,
        )
        if kfee is None:
            continue
        pfee = polymarket_taker_fee(depth.average_price, qty, poly_market)
        gross = qty - qty * maker_price - depth.cost
        net = gross - kfee["cash_fee_upper"] - pfee
        capital = qty * maker_price + depth.cost + kfee["cash_fee_upper"] + pfee
        npc = net / qty
        roc = net / capital if capital > 0 else 0.0
        if npc < config.min_net_per_contract or roc < config.min_return_on_capital:
            continue
        queue_ratio = queue / qty if qty else float("inf")
        queue_score = 1.0 / (1.0 + queue_ratio)
        act = _activity_score(poly_market)
        execution_score = net * queue_score * act
        result = {
            "ticker": match["kalshi_ticker"],
            "subject": match["signature"].get("subject"),
            "topic": match["signature"].get("topic"),
            "strategy": strategy,
            "quantity": qty,
            "capital": round(capital, 6),
            "gross_profit": round(gross, 6),
            "kalshi_fee": round(kfee["cash_fee_upper"], 6),
            "poly_fee": round(pfee, 6),
            "net_profit": round(net, 6),
            "net_per_contract": round(npc, 6),
            "return_on_capital": round(roc, 6),
            "kalshi_price": maker_price,
            "poly_avg_price": round(depth.average_price, 6),
            "poly_worst_price": round(depth.worst_price, 6),
            "queue_ahead": queue,
            "poly_volume24hr": _safe_float(poly_market.get("volume24hr")),
            "poly_liquidity": _safe_float(poly_market.get("liquidityNum", poly_market.get("liquidity"))),
            "activity_score": round(act, 6),
            "execution_score": round(execution_score, 6),
            "quote_skew_seconds": round(quote_skew, 6),
            "kalshi_title": match["kalshi_title"],
            "poly_question": match["polymarket_question"],
            "match_source": match.get("match_source"),
            "match_tier": str(match.get("match_source") or "").split(":",1)[0],
            "equivalence_score": match.get("equivalence_score"),
            "equivalence_reasons": match.get("equivalence_reasons"),
            "equivalence_certificate": match.get("equivalence_certificate") or {},
            "kalshi_signature": match.get("kalshi_signature") or {},
            "polymarket_signature": match.get("polymarket_signature") or {},
        }
        if best is None or result["execution_score"] > best["execution_score"]:
            best = result
    return best


def _taker_candidate(match, poly_market, kbook, poly_levels, kalshi_side, config, series_info, quote_skew):
    yes_levels = _kalshi_levels(kbook, "yes")
    no_levels = _kalshi_levels(kbook, "no")
    yes_bid, yes_size = _best_bid(yes_levels)
    no_bid, no_size = _best_bid(no_levels)

    if kalshi_side == "yes":
        # Kalshi YES ask is generated by NO bids; each NO bid p is a YES ask at 1-p.
        k_ask_levels = sorted(
            ({"price": round(1.0 - x["price"], 4), "size": x["size"]} for x in no_levels),
            key=lambda x: x["price"],
        )
        p_levels = poly_levels["no"]
        strategy = "K YES taker + P NO taker"
    else:
        k_ask_levels = sorted(
            ({"price": round(1.0 - x["price"], 4), "size": x["size"]} for x in yes_levels),
            key=lambda x: x["price"],
        )
        p_levels = poly_levels["yes"]
        strategy = "K NO taker + P YES taker"

    if not k_ask_levels or not p_levels:
        return None

    max_capital = config.bankroll * config.max_bankroll_fraction
    first_cost = k_ask_levels[0]["price"] + p_levels[0]["price"]
    max_qty = min(config.max_contracts, floor_contracts(max_capital / max(first_cost, 1e-9)))
    if max_qty < 1:
        return None

    best = None
    for qty in sorted(set([1, 2, 5, 10, 20, 25, 50, 100, max_qty])):
        if qty < 1 or qty > max_qty:
            continue
        kdepth = consume_asks(k_ask_levels, qty)
        pdepth = consume_asks(p_levels, qty)
        if not kdepth.fully_filled or not pdepth.fully_filled:
            continue
        kfee = kalshi_fee(
            price=kdepth.average_price,
            contracts=qty,
            fee_type=series_info.get("fee_type"),
            fee_multiplier=series_info.get("fee_multiplier") or 0,
            maker=False,
        )
        if kfee is None:
            continue
        pfee = polymarket_taker_fee(pdepth.average_price, qty, poly_market)
        gross = qty - kdepth.cost - pdepth.cost
        net = gross - kfee["cash_fee_upper"] - pfee
        capital = kdepth.cost + pdepth.cost + kfee["cash_fee_upper"] + pfee
        npc = net / qty
        roc = net / capital if capital > 0 else 0.0
        if npc < config.min_net_per_contract or roc < config.min_return_on_capital:
            continue
        act = _activity_score(poly_market)
        result = {
            "ticker": match["kalshi_ticker"],
            "subject": match["signature"].get("subject"),
            "topic": match["signature"].get("topic"),
            "strategy": strategy,
            "quantity": qty,
            "capital": round(capital, 6),
            "gross_profit": round(gross, 6),
            "kalshi_fee": round(kfee["cash_fee_upper"], 6),
            "poly_fee": round(pfee, 6),
            "net_profit": round(net, 6),
            "net_per_contract": round(npc, 6),
            "return_on_capital": round(roc, 6),
            "kalshi_price": round(kdepth.average_price, 6),
            "poly_avg_price": round(pdepth.average_price, 6),
            "poly_worst_price": round(pdepth.worst_price, 6),
            "queue_ahead": 0.0,
            "poly_volume24hr": _safe_float(poly_market.get("volume24hr")),
            "poly_liquidity": _safe_float(poly_market.get("liquidityNum", poly_market.get("liquidity"))),
            "activity_score": round(act, 6),
            "execution_score": round(net * act, 6),
            "quote_skew_seconds": round(quote_skew, 6),
            "kalshi_title": match["kalshi_title"],
            "poly_question": match["polymarket_question"],
            "match_source": match.get("match_source"),
            "match_tier": str(match.get("match_source") or "").split(":",1)[0],
            "equivalence_score": match.get("equivalence_score"),
            "equivalence_reasons": match.get("equivalence_reasons"),
            "equivalence_certificate": match.get("equivalence_certificate") or {},
            "kalshi_signature": match.get("kalshi_signature") or {},
            "polymarket_signature": match.get("polymarket_signature") or {},
        }
        if best is None or result["net_profit"] > best["net_profit"]:
            best = result
    return best


def _poly_levels_from_book(book):
    return [
        {"price": float(level["price"]), "size": float(level["size"])}
        for level in (book or {}).get("asks", [])
    ]


def _snapshot_failure_reason(exc):
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403): return f"auth_http_{status}", False
        if status == 404: return "market_not_found_404", False
        if status in {408, 425, 429, 500, 502, 503, 504}: return f"transient_http_{status}", True
        return f"http_{status or 'unknown'}", False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return type(exc).__name__.lower(), True
    return type(exc).__name__, False


def _fetch_kalshi_books_snapshot(tickers, workers=6, retries=4):
    """Fetch each unique Kalshi book once with bounded transient retries.

    V28 records *why* books are unavailable. Empty/closed markets are not
    confused with auth failures, and transient 429/5xx/timeouts get another
    chance without changing matching or profitability criteria.
    """
    books = {}; timestamps = {}; failures = {}; categories = Counter()

    def fetch(ticker):
        last = None
        for attempt in range(max(0, int(retries)) + 1):
            try:
                book = get_market_orderbook(ticker, depth=100)
                return ticker, book, time.monotonic(), None
            except Exception as exc:
                last = exc
                reason, transient = _snapshot_failure_reason(exc)
                if not transient or attempt >= retries:
                    return ticker, None, None, reason
                retry_after = None
                if isinstance(exc, requests.HTTPError):
                    raw = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
                    try:
                        retry_after = float(raw) if raw else None
                    except (TypeError, ValueError):
                        retry_after = None
                delay = retry_after if retry_after is not None else (0.75 * (2 ** attempt))
                time.sleep(min(20.0, max(0.25, delay)))
        return ticker, None, None, _snapshot_failure_reason(last)[0]

    tickers = list(dict.fromkeys(str(x) for x in tickers if x))
    if not tickers: return books, timestamps, failures
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(tickers)))) as pool:
        futures = {pool.submit(fetch, ticker): ticker for ticker in tickers}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            ticker = futures[future]
            try:
                key, book, ts, reason = future.result()
                if book is not None:
                    books[key] = book; timestamps[key] = ts
                else:
                    failures[key] = reason or "unknown"; categories[reason or "unknown"] += 1
            except Exception as exc:
                reason, _ = _snapshot_failure_reason(exc)
                failures[ticker] = reason; categories[reason] += 1
            if completed % 500 == 0 or completed == len(tickers):
                print(f"Kalshi book snapshot {completed}/{len(tickers)} | success={len(books)} fail={len(failures)}")
    if failures:
        print("V28 Kalshi book failure reasons | " + " | ".join(f"{k}={v}" for k,v in categories.most_common(8)))
    return books, timestamps, failures


def _fetch_poly_books_snapshot(token_ids, chunk_size=100):
    """Fetch unique Polymarket books through the public batch endpoint."""
    ids = list(dict.fromkeys(str(x) for x in token_ids if x is not None))
    books = {}
    timestamps = {}
    failures = 0
    if not ids:
        return books, timestamps, failures
    chunk_size = max(1, int(chunk_size))
    total_chunks = (len(ids) + chunk_size - 1) // chunk_size
    for chunk_idx in range(total_chunks):
        chunk = ids[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        try:
            got = get_orderbooks(chunk, chunk_size=chunk_size)
            ts = time.monotonic()
            for token_id, book in got.items():
                token_id = str(token_id)
                books[token_id] = book
                timestamps[token_id] = ts
        except Exception as exc:
            failures += len(chunk)
            print(f"Polymarket batch-book warning chunk {chunk_idx + 1}/{total_chunks}: {type(exc).__name__}: {exc}")
        if (chunk_idx + 1) % 10 == 0 or chunk_idx + 1 == total_chunks:
            print(
                f"Polymarket book snapshot {chunk_idx + 1}/{total_chunks} chunks | "
                f"books={len(books)} fail_tokens≈{failures}"
            )
    return books, timestamps, failures


def _fetch_series_snapshot(series_tickers, workers=12):
    out = {}
    failures = {}
    series_tickers = list(dict.fromkeys(str(x) for x in series_tickers if x))
    if not series_tickers:
        return out, failures
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(series_tickers)))) as pool:
        futures = {pool.submit(get_series_info, series): series for series in series_tickers}
        for future in as_completed(futures):
            series = futures[future]
            try:
                out[series] = future.result()
            except Exception as exc:
                failures[series] = f"{type(exc).__name__}: {exc}"
    return out, failures


def scan_once(config: EngineConfig):
    kalshi = pd.read_parquet(CACHE_PATH)
    polymarket = get_active_markets(limit=None)
    matches = build_validated_universe(kalshi, polymarket)
    row_by_ticker = {str(row["ticker"]): row for _, row in kalshi.iterrows()}
    results = []

    print(f"Validated structural pairs: {len(matches)}")

    # V25: preserve every structurally accepted pair, but collapse network I/O
    # to unique market/token snapshots. The old implementation re-fetched the
    # same book for every pair, which becomes pathological once recall rises
    # into the thousands. No structural match is discarded here.
    priceable = []
    unique_tickers = []
    unique_tokens = []
    unique_series = []
    for match in matches:
        ticker = str(match["kalshi_ticker"])
        pm = match["polymarket_market"]
        if _safe_float(pm.get("liquidityNum", pm.get("liquidity"))) < config.min_poly_liquidity:
            continue
        if _safe_float(pm.get("volume24hr")) < config.min_poly_volume24hr:
            continue
        tokens = parse_token_ids(pm)
        if len(tokens) != 2:
            continue
        tokens = (str(tokens[0]), str(tokens[1]))
        priceable.append((match, ticker, pm, tokens))
        unique_tickers.append(ticker)
        unique_tokens.extend(tokens)
        unique_series.append(_series_ticker(ticker))

    print(
        f"V25 pricing snapshot | pairs={len(priceable)} | "
        f"unique_kalshi={len(set(unique_tickers))} | "
        f"unique_poly_tokens={len(set(unique_tokens))} | "
        f"unique_series={len(set(unique_series))}"
    )

    # Fetch Polymarket in native 100-book batches and Kalshi concurrently.
    # These are screening snapshots only; v3 immediately refreshes shortlisted
    # taker/taker positives from both venues using the original strict skew rule.
    poly_books, poly_ts, poly_failures = _fetch_poly_books_snapshot(unique_tokens, chunk_size=100)
    kalshi_books, kalshi_ts, kalshi_failures = _fetch_kalshi_books_snapshot(unique_tickers, workers=16)
    series_cache, series_failures = _fetch_series_snapshot(unique_series, workers=12)

    print(
        f"V25 pricing cache ready | kalshi_books={len(kalshi_books)} "
        f"poly_books={len(poly_books)} series={len(series_cache)} | "
        f"kalshi_fail={len(kalshi_failures)} poly_fail_tokens≈{poly_failures} "
        f"series_fail={len(series_failures)}"
    )

    # The snapshot stage is a broad economic screen, not the execution proof.
    # Permit modest snapshot asynchrony so a large universe is not discarded
    # merely because its cached books were fetched seconds apart. The existing
    # v3 confirmation stage still enforces config.max_quote_skew_seconds on a
    # fresh pair-level re-fetch before CONFIRMED status.
    screening_skew_ceiling = max(float(config.max_quote_skew_seconds), 30.0)
    priced = 0
    for match, ticker, pm, tokens in priceable:
        kbook = kalshi_books.get(ticker)
        pyes_book = poly_books.get(tokens[0])
        pno_book = poly_books.get(tokens[1])
        if kbook is None or pyes_book is None or pno_book is None:
            continue
        series = _series_ticker(ticker)
        sinfo = series_cache.get(series)
        row = row_by_ticker.get(ticker)
        if sinfo is None or row is None:
            continue

        pyes_ts = poly_ts.get(tokens[0])
        pno_ts = poly_ts.get(tokens[1])
        kts = kalshi_ts.get(ticker)
        if kts is None or pyes_ts is None or pno_ts is None:
            continue
        quote_skew = max(abs(kts - pyes_ts), abs(kts - pno_ts), abs(pyes_ts - pno_ts))
        if quote_skew > screening_skew_ceiling:
            continue

        plevels = {
            "yes": _poly_levels_from_book(pyes_book),
            "no": _poly_levels_from_book(pno_book),
        }
        for side in ("yes", "no"):
            candidate = _maker_candidate(match, row, pm, kbook, plevels, side, config, sinfo, quote_skew)
            if candidate:
                candidate["pricing_snapshot_mode"] = "V25_BATCH_CACHE"
                results.append(candidate)
            if config.allow_taker_taker:
                candidate = _taker_candidate(match, pm, kbook, plevels, side, config, sinfo, quote_skew)
                if candidate:
                    candidate["pricing_snapshot_mode"] = "V25_BATCH_CACHE"
                    results.append(candidate)
        priced += 1
        if priced % 500 == 0 or priced == len(priceable):
            print(f"Priced locally {priced}/{len(priceable)} | positives={len(results)}")

    results.sort(key=lambda x: x["execution_score"], reverse=True)
    log_opportunities(results)
    return results


def print_results(results, limit=30):
    print("\n" + "=" * 72)
    print("PAPER ENGINE V2 - RANKED OPPORTUNITIES")
    print("=" * 72)
    print(f"Positive strategies: {len(results)}")
    for i, row in enumerate(results[:limit], 1):
        print(
            f"#{i:02d} {row['subject']} | {row['strategy']} | "
            f"qty={row['quantity']} | net=${row['net_profit']:.4f} | "
            f"net/ct={row['net_per_contract']:.4f} | ROC={row['return_on_capital']:.2%} | "
            f"queue={row['queue_ahead']:.2f} | v24={row['poly_volume24hr']:.2f} | "
            f"score={row['execution_score']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--min-edge", type=float, default=0.0025)
    parser.add_argument("--max-skew", type=float, default=3.0)
    parser.add_argument("--no-taker", action="store_true")
    args = parser.parse_args()

    config = EngineConfig(
        bankroll=args.bankroll,
        min_net_per_contract=args.min_edge,
        max_quote_skew_seconds=args.max_skew,
        allow_taker_taker=not args.no_taker,
    )

    for cycle in range(args.cycles):
        print(f"\n=== Cycle {cycle + 1}/{args.cycles} ===")
        results = scan_once(config)
        print_results(results, config.max_results)
        if cycle + 1 < args.cycles:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
