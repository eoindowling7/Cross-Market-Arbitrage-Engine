"""Short-horizon / immediately recyclable Polymarket paper arbitrage.

Paper only. No order placement.

The scanner is deliberately depth-aware and fee-aware. V8.4 also exposes a
near-miss diagnostic funnel so a zero-trade run can distinguish market
efficiency from broken discovery, missing books, insufficient depth, or fees.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from src.api.polymarket_client import parse_token_ids, get_orderbook, get_orderbooks
from src.arbitrage.polymarket_fees import get_polymarket_fee_rate, polymarket_taker_fee


@dataclass
class FastRecycleConfig:
    enabled: bool = True
    scan_seconds: float = 30.0
    max_markets_per_scan: int = 1200
    min_profit_dollars: float = 0.03
    min_profit_per_set: float = 0.0015
    safety_buffer_per_set: float = 0.0010
    max_capital_fraction_per_trade: float = 0.05
    max_depth_sets: int = 250
    cooldown_seconds: float = 60.0


def _asks(book: dict) -> list[tuple[float, float]]:
    out = []
    for x in book.get("asks", []) or []:
        try:
            p, s = float(x["price"]), float(x["size"])
            if 0 < p < 1 and s > 0:
                out.append((p, s))
        except Exception:
            continue
    return sorted(out)


def _cost_for_qty(levels: list[tuple[float, float]], qty: int) -> tuple[float, float] | None:
    remain = float(qty)
    cost = 0.0
    filled = 0.0
    for price, size in levels:
        take = min(remain, size)
        cost += take * price
        filled += take
        remain -= take
        if remain <= 1e-9:
            break
    if remain > 1e-9 or filled <= 0:
        return None
    return cost, cost / filled


def _max_qty(levels_a, levels_b, cap: float, max_depth_sets: int) -> int:
    if not levels_a or not levels_b or cap <= 0:
        return 0
    depth = min(sum(s for _, s in levels_a), sum(s for _, s in levels_b), float(max_depth_sets))
    hi = int(depth)
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ca = _cost_for_qty(levels_a, mid)
        cb = _cost_for_qty(levels_b, mid)
        if ca is None or cb is None or ca[0] + cb[0] > cap:
            hi = mid - 1
        else:
            lo = mid
    return lo


def _near_miss_key(row: dict) -> float:
    return float(row.get("net_per_set_after_buffer", -999.0))


def scan_polymarket_complete_sets(
    markets: Iterable[dict],
    available_cash: float,
    bankroll: float,
    cfg: FastRecycleConfig,
    *,
    return_diagnostics: bool = False,
):
    """Return complete-set opportunities; optionally return diagnostic stats.

    Diagnostics intentionally evaluate one executable set at the top of book as
    well as the bankroll-sized trade. This means a zero result can still report
    the closest live market to profitability instead of silently returning [].
    """
    stats = {
        "markets_input": 0,
        "binary_orderbook_markets": 0,
        "markets_scanned": 0,
        "books_available": 0,
        "two_sided_depth": 0,
        "raw_positive": 0,
        "positive_after_fees": 0,
        "positive_after_buffer": 0,
        "qualified": 0,
        "best_near_miss": None,
    }
    market_list = list(markets)
    stats["markets_input"] = len(market_list)
    if not cfg.enabled or available_cash <= 0:
        return ([], stats) if return_diagnostics else []

    cap = min(available_cash, bankroll * cfg.max_capital_fraction_per_trade)
    eligible = []
    for market in market_list:
        if market.get("closed") is True or market.get("enableOrderBook") is False:
            continue
        ids = parse_token_ids(market)
        if len(ids) != 2:
            continue
        eligible.append(market)
    stats["binary_orderbook_markets"] = len(eligible)
    eligible.sort(
        key=lambda m: float(m.get("volume24hr") or 0)
        + 0.10 * float(m.get("liquidityNum") or m.get("liquidity") or 0),
        reverse=True,
    )
    eligible = eligible[: cfg.max_markets_per_scan]
    stats["markets_scanned"] = len(eligible)

    token_ids = []
    for m in eligible:
        token_ids.extend(parse_token_ids(m))
    try:
        books = get_orderbooks(token_ids)
    except Exception:
        books = {}

    results = []
    best_near = None
    for market in eligible:
        ids = parse_token_ids(market)
        try:
            yes_book = books.get(str(ids[0])) or get_orderbook(ids[0])
            no_book = books.get(str(ids[1])) or get_orderbook(ids[1])
        except Exception:
            continue
        stats["books_available"] += 1
        ya, na = _asks(yes_book), _asks(no_book)
        if not ya or not na:
            continue
        stats["two_sided_depth"] += 1

        # Diagnostic at a single executable set. This is the cleanest measure
        # of whether the market is intrinsically crossed before sizing effects.
        yc1 = _cost_for_qty(ya, 1)
        nc1 = _cost_for_qty(na, 1)
        if yc1 is not None and nc1 is not None:
            fee_rate = get_polymarket_fee_rate(market)
            raw1 = 1.0 - yc1[0] - nc1[0]
            fee1 = polymarket_taker_fee(yc1[1], 1, fee_rate) + polymarket_taker_fee(nc1[1], 1, fee_rate)
            after_fee1 = raw1 - fee1
            after_buf1 = after_fee1 - cfg.safety_buffer_per_set
            if raw1 > 0:
                stats["raw_positive"] += 1
            if after_fee1 > 0:
                stats["positive_after_fees"] += 1
            if after_buf1 > 0:
                stats["positive_after_buffer"] += 1
            near = {
                "question": market.get("question") or market.get("title"),
                "market_id": market.get("id"),
                "yes_ask": yc1[1],
                "no_ask": nc1[1],
                "raw_per_set": raw1,
                "fees_per_set": fee1,
                "net_per_set_after_buffer": after_buf1,
            }
            if best_near is None or _near_miss_key(near) > _near_miss_key(best_near):
                best_near = near

        qty = _max_qty(ya, na, cap, cfg.max_depth_sets)
        if qty < 1:
            continue
        yc = _cost_for_qty(ya, qty)
        nc = _cost_for_qty(na, qty)
        if yc is None or nc is None:
            continue
        yes_cost, yes_avg = yc
        no_cost, no_avg = nc
        fee_rate = get_polymarket_fee_rate(market)
        fees = polymarket_taker_fee(yes_avg, qty, fee_rate) + polymarket_taker_fee(no_avg, qty, fee_rate)
        gross = qty - yes_cost - no_cost
        net = gross - fees - cfg.safety_buffer_per_set * qty
        per_set = net / qty
        capital = yes_cost + no_cost + fees
        if net < cfg.min_profit_dollars or per_set < cfg.min_profit_per_set:
            continue
        stats["qualified"] += 1
        results.append({
            "timestamp": time.time(), "venue": "polymarket", "strategy": "complete_set_merge",
            "market_id": market.get("id"), "question": market.get("question") or market.get("title"),
            "slug": market.get("slug"), "quantity": qty, "yes_avg": yes_avg, "no_avg": no_avg,
            "fee_rate": fee_rate, "fees": fees, "capital": capital, "gross_profit": gross,
            "net_profit": net, "net_per_set": per_set,
            "return_on_capital": net / max(capital, 1e-9), "capital_lock_days": 0.0,
            "recyclable": True,
            "yes_book_hash": yes_book.get("hash"), "no_book_hash": no_book.get("hash"),
        })

    stats["best_near_miss"] = best_near
    results = sorted(results, key=lambda x: (x["net_profit"], x["return_on_capital"]), reverse=True)
    return (results, stats) if return_diagnostics else results
