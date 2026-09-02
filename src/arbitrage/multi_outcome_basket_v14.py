"""Strict Polymarket multi-outcome basket arbitrage scanner (paper only).

For an exhaustive mutually-exclusive event with N outcomes:

* buying one YES token for every outcome pays exactly $1 in total;
* buying one NO token for every outcome pays exactly $(N-1) in total.

The scanner only admits Gamma events that explicitly use negative-risk /
show-all-outcomes metadata and whose complete child-market set is active and
order-book enabled. This is intentionally conservative: ambiguous event groups
are skipped rather than treated as arbitrage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.api.polymarket_client import parse_token_ids, get_orderbooks, get_event_by_id
from src.arbitrage.polymarket_fees import get_polymarket_fee_rate, polymarket_taker_fee
from src.arbitrage.near_miss import push_top


@dataclass
class MultiOutcomeConfig:
    enabled: bool = True
    max_settlement_days: float = 60.0
    max_events_per_scan: int = 250
    max_capital_fraction_per_event: float = 0.10
    min_profit_dollars: float = 0.01
    min_return_on_capital: float = 0.0010
    safety_buffer_per_leg: float = 0.00035
    max_qty: int = 250
    min_outcomes: int = 3


def _event_stub(market: dict) -> dict | None:
    events = market.get("events") or []
    return events[0] if events and isinstance(events[0], dict) else None


def _parse_end_ts(market: dict) -> float | None:
    raw = market.get("endDate") or market.get("endDateIso")
    if not raw:
        return None
    try:
        text = str(raw)
        if len(text) == 10:
            text += "T23:59:59+00:00"
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _asks(book: dict) -> list[tuple[float, float]]:
    rows = []
    for level in book.get("asks", []) or []:
        try:
            p, q = float(level["price"]), float(level["size"])
            if 0 < p < 1 and q > 0:
                rows.append((p, q))
        except Exception:
            pass
    return sorted(rows)


def _cost(levels, qty: int):
    remaining = float(qty)
    total = 0.0
    for price, size in levels:
        take = min(remaining, size)
        total += take * price
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        return None
    return total


def _valid_event_snapshot(event: dict, grouped_markets: list[dict], cfg: MultiOutcomeConfig) -> bool:
    if not isinstance(event, dict):
        return False
    if not bool(event.get("enableNegRisk") or event.get("negRisk")):
        return False
    if event.get("showAllOutcomes") is False:
        return False
    if event.get("closed") is True or event.get("active") is False:
        return False
    children = event.get("markets") or []
    if len(children) < cfg.min_outcomes:
        return False
    child_ids = {str(m.get("id")) for m in children if m.get("id") is not None}
    group_ids = {str(m.get("id")) for m in grouped_markets if m.get("id") is not None}
    if child_ids != group_ids:
        return False
    for m in children:
        if m.get("closed") is True or m.get("active") is False or m.get("enableOrderBook") is False:
            return False
        if len(parse_token_ids(m)) != 2:
            return False
    return True


def _best_qty_for_side(markets, books, token_index, payout_per_set, capital_cap, cfg):
    legs = []
    max_depth = float(cfg.max_qty)
    for m in markets:
        ids = parse_token_ids(m)
        levels = _asks(books.get(str(ids[token_index]), {}))
        if not levels:
            return None
        max_depth = min(max_depth, sum(size for _, size in levels))
        legs.append((m, levels))
    hi = int(max_depth)
    best = None
    for qty in range(1, hi + 1):
        leg_costs = []
        fees = 0.0
        ok = True
        for m, levels in legs:
            c = _cost(levels, qty)
            if c is None:
                ok = False
                break
            avg = c / qty
            rate = get_polymarket_fee_rate(m)
            fees += polymarket_taker_fee(avg, qty, rate)
            leg_costs.append(c)
        if not ok:
            break
        cost = sum(leg_costs) + fees
        if cost > capital_cap + 1e-9:
            break
        payout = payout_per_set * qty
        safety = cfg.safety_buffer_per_leg * len(legs) * qty
        net = payout - cost - safety
        roc = net / max(cost, 1e-9)
        if net >= cfg.min_profit_dollars and roc >= cfg.min_return_on_capital:
            best = {
                "quantity": qty, "capital": cost, "fees": fees, "payout": payout,
                "safety_buffer": safety, "net_profit": net, "return_on_capital": roc,
            }
    return best


def scan_polymarket_multi_outcome(markets: list[dict], available_cash: float, bankroll: float, cfg: MultiOutcomeConfig, *, return_diagnostics: bool = False):
    stats = {
        "markets_input": len(markets), "neg_risk_near_dated_markets": 0,
        "event_groups": 0, "events_fetched": 0, "valid_exhaustive_events": 0,
        "book_complete_events": 0, "raw_positive_sides": 0,
        "positive_after_fees": 0, "qualified": 0, "best_near_miss": None, "near_misses": [],
    }
    if not cfg.enabled or available_cash <= 0:
        return ([], stats) if return_diagnostics else []
    now = time.time()
    grouped = {}
    for m in markets:
        if m.get("closed") is True or m.get("enableOrderBook") is False:
            continue
        stub = _event_stub(m)
        event_id = None if stub is None else stub.get("id")
        if event_id is None:
            continue
        if not bool(m.get("negRisk") or (stub and (stub.get("enableNegRisk") or stub.get("negRisk")))):
            continue
        end_ts = _parse_end_ts(m)
        if end_ts is None or end_ts <= now or (end_ts - now) / 86400.0 > cfg.max_settlement_days:
            continue
        grouped.setdefault(str(event_id), []).append(m)
        stats["neg_risk_near_dated_markets"] += 1

    stats["event_groups"] = len(grouped)
    candidates = sorted(
        grouped.items(),
        key=lambda kv: min(_parse_end_ts(m) or 1e30 for m in kv[1]),
    )[: cfg.max_events_per_scan]
    out = []
    for event_id, group in candidates:
        try:
            event = get_event_by_id(event_id)
            stats["events_fetched"] += 1
        except Exception:
            continue
        if not _valid_event_snapshot(event, group, cfg):
            continue
        stats["valid_exhaustive_events"] += 1
        children = event.get("markets") or []
        token_ids = []
        for m in children:
            token_ids.extend(parse_token_ids(m))
        try:
            books = get_orderbooks(token_ids)
        except Exception:
            continue
        if not all((books.get(str(tid)) or {}).get("asks") for tid in token_ids):
            continue
        stats["book_complete_events"] += 1
        capital_cap = min(available_cash, bankroll * cfg.max_capital_fraction_per_event)
        n = len(children)
        for token_index, strategy, payout in ((0, "buy_all_yes", 1.0), (1, "buy_all_no", float(n - 1))):
            # One-set near-miss diagnostic before minimum-profit thresholds.
            one_cost = 0.0
            one_fees = 0.0
            one_ok = True
            for child in children:
                ids = parse_token_ids(child)
                levels = _asks(books.get(str(ids[token_index]), {}))
                c = _cost(levels, 1)
                if c is None:
                    one_ok = False
                    break
                one_cost += c
                one_fees += polymarket_taker_fee(c, 1, get_polymarket_fee_rate(child))
            if one_ok:
                raw = payout - one_cost
                after_fees = raw - one_fees
                after_buffer = after_fees - cfg.safety_buffer_per_leg * len(children)
                if raw > 0:
                    stats["raw_positive_sides"] += 1
                if after_fees > 0:
                    stats["positive_after_fees"] += 1
                near = {
                    "event_id": event_id, "event_title": event.get("title") or event.get("slug"),
                    "strategy": strategy, "outcomes": n, "raw_per_set": raw,
                    "fees_per_set": one_fees, "net_per_set_after_buffer": after_buffer,
                }
                prev = stats["best_near_miss"]
                if prev is None or near["net_per_set_after_buffer"] > prev["net_per_set_after_buffer"]:
                    stats["best_near_miss"] = near
                push_top(stats["near_misses"], near, key="net_per_set_after_buffer", limit=10)

            best = _best_qty_for_side(children, books, token_index, payout, capital_cap, cfg)
            if best is None:
                continue
            stats["qualified"] += 1
            end_ts = min(_parse_end_ts(m) or 1e30 for m in children)
            days = max((end_ts - now) / 86400.0, 1.0 / 24.0)
            out.append({
                "timestamp": now, "venue": "polymarket", "strategy": strategy,
                "event_id": event_id, "event_title": event.get("title") or event.get("slug"),
                "outcomes": n, "settlement_ts": end_ts, "settlement_days": days,
                "annualized_return": best["return_on_capital"] * 365.0 / days,
                **best,
            })
    out = sorted(out, key=lambda x: (x["annualized_return"], x["net_profit"]), reverse=True)
    return (out, stats) if return_diagnostics else out
