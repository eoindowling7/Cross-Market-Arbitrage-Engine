"""V28 confirmation engine: separates immediate and passive opportunities.

Paper-only. It never places orders. Taker/taker candidates are repeatedly
re-priced from fresh full books before they are labelled CONFIRMED.
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass

from src.api.kalshi_client import get_market_orderbook, get_series_info
from src.api.polymarket_client import get_active_markets, parse_token_ids
from src.arbitrage.exact_fees import kalshi_fee, polymarket_taker_fee
from src.arbitrage.execution_utils import consume_asks
from src.arbitrage.paper_engine_v2 import (
    EngineConfig as V2Config,
    _kalshi_levels,
    _poly_ask_levels,
    scan_once as scan_v2_once,
)


@dataclass
class ConfirmationConfig:
    samples: int = 5
    delay_seconds: float = 0.75
    max_candidates: int = 15
    maker_unknown_fill_penalty: float = 0.05
    # V11: keep a broad confirmation set. One event family cannot consume the
    # entire expensive refresh budget while unrelated short/high-APR pairs are
    # never checked. If no alternatives exist, spare slots are still filled.
    max_per_topic: int = 8




def _safe_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            x = float(value)
            return x if math.isfinite(x) and x > 0 else None
        except Exception:
            return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _preliminary_settlement_days(candidate: dict, now: float | None = None) -> float:
    """Latest known cross-venue horizon used only to prioritize confirmations."""
    now = float(now or time.time())
    timestamps = []
    cert = candidate.get("equivalence_certificate") or {}
    for key in ("latest_cross_settlement_ts", "kalshi_latest_settlement_ts", "polymarket_latest_settlement_ts"):
        ts = _safe_ts(cert.get(key))
        if ts:
            timestamps.append(ts)
    for sig_key in ("kalshi_signature", "polymarket_signature"):
        sig = candidate.get(sig_key) or {}
        for field in ("latest_settlement_ts", "end_ts", "resolution_rule_deadline_ts"):
            ts = _safe_ts(sig.get(field))
            if ts:
                timestamps.append(ts)
    if not timestamps:
        return 365.0
    return max(1.0 / 24.0, (max(timestamps) - now) / 86400.0)


def _confirmation_priority(candidate: dict, now: float | None = None) -> tuple[float, float, float]:
    """Rank scarce confirmation calls by credible capital efficiency.

    Raw dollar profit alone systematically favors deep, long-dated books. V12
    ranks by simple annualized return first, with modest dollar/liquidity terms
    so short, recyclable opportunities get checked before low-velocity locks.
    This changes discovery priority, never the edge calculation itself.
    """
    days = _preliminary_settlement_days(candidate, now=now)
    net = max(0.0, float(candidate.get("net_profit") or 0.0))
    capital = max(0.0, float(candidate.get("capital") or 0.0))
    cert = candidate.get("equivalence_certificate") or {}
    if str(cert.get("resolution_lane") or "").upper() == "LOW_BASIS":
        try:
            reserve_npc = max(0.0, float(cert.get("basis_risk_reserve_per_contract") or 0.0))
            qty = max(0, int(candidate.get("quantity") or 0))
        except (TypeError, ValueError):
            reserve_npc, qty = 0.0, 0
        net = max(0.0, net - reserve_npc * qty)
    roc = net / capital if capital > 0 else max(0.0, float(candidate.get("return_on_capital") or 0.0))
    apr = roc * 365.0 / max(days, 1.0 / 24.0)
    volume = max(0.0, float(candidate.get("poly_volume24hr") or 0.0))
    liquidity = max(0.0, float(candidate.get("poly_liquidity") or 0.0))
    score = (
        4.0 * min(apr, 2.0)
        + 0.75 * min(roc, 1.0)
        + 0.08 * math.log1p(net)
        + 0.015 * math.log1p(volume)
        + 0.010 * math.log1p(liquidity)
        - 0.025 * math.log1p(days / 30.0)
    )
    return score, days, apr


def _select_confirmation_candidates(immediate: list[dict], config: ConfirmationConfig, now: float | None = None) -> list[dict]:
    """Profit-velocity ranking with event-family diversification."""
    ranked = []
    for row in immediate:
        score, days, apr = _confirmation_priority(row, now=now)
        row["confirmation_priority"] = score
        row["preliminary_settlement_days"] = days
        row["preliminary_hold_apr"] = apr
        ranked.append(row)
    ranked.sort(key=lambda x: (x.get("confirmation_priority", -999.0), x.get("net_profit", -999.0)), reverse=True)

    selected = []
    selected_ids = set()
    counts = {}
    cap = max(1, int(config.max_per_topic))
    limit = max(1, int(config.max_candidates))
    for row in ranked:
        topic = str(row.get("topic") or row.get("subject") or row.get("ticker") or "unknown")
        if counts.get(topic, 0) >= cap:
            continue
        selected.append(row); selected_ids.add(id(row))
        counts[topic] = counts.get(topic, 0) + 1
        if len(selected) >= limit:
            return selected

    # If the universe genuinely contains only a few event families, do not
    # waste confirmation capacity: fill the spare slots by global priority.
    for row in ranked:
        if id(row) in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected

def _series_ticker(ticker: str) -> str:
    return ticker.split("-")[0]


def _fixed_taker_price(candidate, poly_market, series_info, max_skew_seconds):
    """Re-price the candidate's exact quantity from fresh depth on both venues."""
    tokens = parse_token_ids(poly_market)
    if len(tokens) != 2:
        return None, "invalid Polymarket token ids"

    qty = int(candidate["quantity"])
    kalshi_side = "yes" if candidate["strategy"].startswith("K YES") else "no"

    # V28: fetch both venues concurrently.  This reduces quote skew without
    # weakening the stale-quote limit.  Polymarket YES/NO are fetched together
    # in the same worker because they are one venue snapshot.
    def _poly_pair():
        started = time.monotonic()
        yes = _poly_ask_levels(tokens[0])
        no = _poly_ask_levels(tokens[1])
        return yes, no, started, time.monotonic()

    with ThreadPoolExecutor(max_workers=2) as pool:
        kf = pool.submit(get_market_orderbook, candidate["ticker"], 100)
        pf = pool.submit(_poly_pair)
        kbook = kf.result()
        kalshi_received = time.monotonic()
        poly_yes, poly_no, _, poly_received = pf.result()
    fetch_skew = abs(poly_received - kalshi_received)

    if fetch_skew > max_skew_seconds:
        return None, f"cross-venue fetch skew {fetch_skew:.3f}s"

    yes_bids = _kalshi_levels(kbook, "yes")
    no_bids = _kalshi_levels(kbook, "no")

    if kalshi_side == "yes":
        kalshi_asks = sorted(
            ({"price": round(1.0 - x["price"], 4), "size": x["size"]} for x in no_bids),
            key=lambda x: x["price"],
        )
        poly_asks = poly_no
    else:
        kalshi_asks = sorted(
            ({"price": round(1.0 - x["price"], 4), "size": x["size"]} for x in yes_bids),
            key=lambda x: x["price"],
        )
        poly_asks = poly_yes

    if not kalshi_asks or not poly_asks:
        return None, "empty executable book"

    kfill = consume_asks(kalshi_asks, qty)
    pfill = consume_asks(poly_asks, qty)
    if not kfill.fully_filled or not pfill.fully_filled:
        return None, "insufficient depth for original quantity"

    kfee = kalshi_fee(
        price=kfill.average_price,
        contracts=qty,
        fee_type=series_info.get("fee_type"),
        fee_multiplier=series_info.get("fee_multiplier") or 0,
        maker=False,
    )
    if kfee is None:
        return None, "unknown Kalshi fee schedule"

    pfee = polymarket_taker_fee(pfill.average_price, qty, poly_market)
    gross = qty - kfill.cost - pfill.cost
    net = gross - kfee["cash_fee_upper"] - pfee
    capital = kfill.cost + pfill.cost + kfee["cash_fee_upper"] + pfee

    return {
        "net_profit": net,
        "net_per_contract": net / qty,
        "return_on_capital": net / capital if capital > 0 else 0.0,
        "kalshi_average_price": kfill.average_price,
        "kalshi_worst_price": kfill.worst_price,
        "poly_average_price": pfill.average_price,
        "poly_worst_price": pfill.worst_price,
        "kalshi_fee": kfee["cash_fee_upper"],
        "poly_fee": pfee,
        "fetch_skew_seconds": fetch_skew,
    }, None


def confirm_candidate(candidate, poly_market, series_info, engine_config, confirmation_config):
    observations = []
    failures = []

    for sample in range(confirmation_config.samples):
        try:
            priced, error = _fixed_taker_price(
                candidate, poly_market, series_info, engine_config.max_quote_skew_seconds
            )
            if error:
                failures.append(f"sample {sample + 1}: {error}")
            else:
                observations.append(priced)
                if priced["net_per_contract"] < engine_config.min_net_per_contract:
                    failures.append(
                        f"sample {sample + 1}: edge fell to {priced['net_per_contract']:.6f}"
                    )
        except Exception as exc:
            failures.append(f"sample {sample + 1}: {type(exc).__name__}: {exc}")

        if sample + 1 < confirmation_config.samples:
            time.sleep(confirmation_config.delay_seconds)

    positive = [
        x for x in observations
        if x["net_per_contract"] >= engine_config.min_net_per_contract
    ]
    confirmed = len(positive) == confirmation_config.samples and not failures

    result = dict(candidate)
    result.update({
        "confirmation_status": "CONFIRMED" if confirmed else "REJECTED",
        "confirmations_positive": len(positive),
        "confirmations_required": confirmation_config.samples,
        "confirmation_reason": "all samples positive" if confirmed else "; ".join(failures),
    })

    if observations:
        result.update({
            "worst_net_per_contract": min(x["net_per_contract"] for x in observations),
            "average_net_per_contract": sum(x["net_per_contract"] for x in observations) / len(observations),
            "worst_net_profit": min(x["net_profit"] for x in observations),
            "max_fetch_skew_seconds": max(x["fetch_skew_seconds"] for x in observations),
        })

    return result


def run_v3(engine_config, confirmation_config):
    print("Running initial full-depth paper scan...")
    results = scan_v2_once(engine_config)

    immediate = [x for x in results if "taker + P" in x["strategy"]]
    passive = [x for x in results if "maker + P" in x["strategy"]]

    # V11 confirmation budget is allocated by annualized capital efficiency
    # and event diversity rather than raw dollars locked.

    # v2's maker score treats an improved quote's zero displayed queue too
    # generously. Until actual Kalshi-side trade flow supports a fill estimate,
    # heavily discount every passive opportunity.
    for row in passive:
        row["raw_execution_score"] = row["execution_score"]
        row["execution_score"] *= confirmation_config.maker_unknown_fill_penalty
        row["fill_confidence"] = "UNKNOWN"
    passive.sort(key=lambda x: x["execution_score"], reverse=True)

    if not immediate:
        return results, [], passive

    print("Refreshing Polymarket universe for confirmation lookup...")
    polymarket = get_active_markets(limit=None)
    by_question = {str(m.get("question")): m for m in polymarket}
    series_cache = {}
    confirmations = []

    selected = _select_confirmation_candidates(immediate, confirmation_config)
    for idx, candidate in enumerate(selected, 1):
        print(f"Confirming {idx}/{len(selected)}: {candidate['subject']} | {candidate['strategy']}")
        pm = by_question.get(candidate["poly_question"])
        if pm is None:
            rejected = dict(candidate)
            rejected.update({
                "confirmation_status": "REJECTED",
                "confirmations_positive": 0,
                "confirmations_required": confirmation_config.samples,
                "confirmation_reason": "Polymarket contract missing on refresh",
            })
            confirmations.append(rejected)
            continue

        series = _series_ticker(candidate["ticker"])
        if series not in series_cache:
            series_cache[series] = get_series_info(series)

        confirmed_row = confirm_candidate(
            candidate, pm, series_cache[series], engine_config, confirmation_config
        )
        # V28.1: preserve the exact Polymarket record used during successful
        # confirmation.  The live watch loop can continue to reprice this
        # contract directly even if a later global Gamma refresh temporarily
        # fails.  The private key is in-memory only and is not required for
        # trading/authentication.
        confirmed_row["_poly_market_snapshot"] = dict(pm)
        # V29 split-watch support: persist the fee/series metadata already
        # fetched during confirmation so watch-only startup does not need to
        # refetch every Kalshi series and trigger avoidable rate limits.
        confirmed_row["_series_info_snapshot"] = dict(series_cache[series]) if isinstance(series_cache[series], dict) else {}
        confirmations.append(confirmed_row)

    return results, confirmations, passive


def print_v3(confirmations, passive, limit=30):
    print("\n" + "=" * 78)
    print("PAPER ENGINE V3 - CONFIRMED IMMEDIATE ARBITRAGE")
    print("=" * 78)
    confirmed = [x for x in confirmations if x["confirmation_status"] == "CONFIRMED"]
    confirmed.sort(key=lambda x: x.get("worst_net_profit", -999.0), reverse=True)
    print(f"Confirmed: {len(confirmed)} / {len(confirmations)} tested")

    if not confirmed:
        print("No immediate opportunity survived every confirmation sample.")
    for idx, row in enumerate(confirmed[:limit], 1):
        print(
            f"#{idx:02d} {row['subject']} | {row['strategy']} | qty={row['quantity']} | "
            f"confirm={row['confirmations_positive']}/{row['confirmations_required']} | "
            f"worst net/ct={row['worst_net_per_contract']:.4f} | "
            f"avg net/ct={row['average_net_per_contract']:.4f} | "
            f"worst net=${row['worst_net_profit']:.4f} | "
            f"max fetch skew={row['max_fetch_skew_seconds']:.3f}s | CONFIRMED"
        )

    rejected = [x for x in confirmations if x["confirmation_status"] != "CONFIRMED"]
    print("\n" + "-" * 78)
    print(f"REJECTED IMMEDIATE CANDIDATES: {len(rejected)}")
    for row in rejected[:15]:
        print(
            f"{row['subject']} | {row['strategy']} | "
            f"{row.get('confirmations_positive', 0)}/{row.get('confirmations_required', 0)} | "
            f"{row.get('confirmation_reason', 'unknown')}"
        )

    print("\n" + "=" * 78)
    print("PASSIVE MAKER WATCHLIST - FILL NOT ASSUMED")
    print("=" * 78)
    print(f"Passive opportunities: {len(passive)}")
    for idx, row in enumerate(passive[:limit], 1):
        print(
            f"#{idx:02d} {row['subject']} | {row['strategy']} | qty={row['quantity']} | "
            f"theoretical net=${row['net_profit']:.4f} | net/ct={row['net_per_contract']:.4f} | "
            f"queue={row['queue_ahead']:.2f} | fill=UNKNOWN | score={row['execution_score']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--min-edge", type=float, default=0.0025)
    parser.add_argument("--max-skew", type=float, default=3.0)
    parser.add_argument("--confirmations", type=int, default=5)
    parser.add_argument("--confirm-delay", type=float, default=0.75)
    parser.add_argument("--confirm-top", type=int, default=15)
    args = parser.parse_args()

    engine_config = V2Config(
        bankroll=args.bankroll,
        min_net_per_contract=args.min_edge,
        max_quote_skew_seconds=args.max_skew,
        allow_taker_taker=True,
    )
    confirmation_config = ConfirmationConfig(
        samples=max(1, args.confirmations),
        delay_seconds=max(0.0, args.confirm_delay),
        max_candidates=max(1, args.confirm_top),
    )

    _, confirmations, passive = run_v3(engine_config, confirmation_config)
    print_v3(confirmations, passive, engine_config.max_results)


if __name__ == "__main__":
    main()
