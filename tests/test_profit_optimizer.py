"""Offline regression tests for V10.0 Final profit/realism upgrades."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from src.arbitrage.paper_engine_v8 import (
    V8Config,
    V8Portfolio,
    _cross_settlement_metrics,
    _select_economic_signal,
)
from src.arbitrage.polymarket_fee_policy import resolve_fee_rate


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def test_later_venue_controls_lock_horizon():
    now = time.time()
    poly_end = now + 30 * 86400
    kalshi_end = now + 90 * 86400
    pm = {"endDate": _iso(poly_end)}
    candidate = {
        "kalshi_signature": {"end_ts": kalshi_end},
        "polymarket_signature": {"end_ts": poly_end},
    }
    cfg = V8Config()
    ts, days, apr = _cross_settlement_metrics(candidate, pm, 100.0, 2.0, cfg, now=now)
    assert abs(ts - kalshi_end) < 2
    assert 89.9 <= days <= 90.1
    assert 0.080 <= apr <= 0.082  # 2% simple return annualized over ~90d


def test_economic_depth_sizing_prefers_surplus_not_raw_size():
    now = time.time()
    end = now + 60 * 86400
    pm = {"endDate": _iso(end)}
    candidate = {
        "kalshi_signature": {"end_ts": end},
        "polymarket_signature": {"end_ts": end},
    }
    cfg = V8Config(base_required_hold_apr=0.03, lock_horizon_hurdle_apr=0.0)
    # The deep size earns more raw dollars but too little return for the hurdle.
    signals = [
        {"quantity": 20, "capital": 20.0, "net_profit": 0.30, "net_per_contract": 0.015},
        {"quantity": 100, "capital": 100.0, "net_profit": 0.35, "net_per_contract": 0.0035},
    ]
    chosen = _select_economic_signal(signals, candidate, pm, cfg)
    assert chosen is not None
    assert chosen["quantity"] == 20
    assert chosen["sizing_economic_surplus"] > 0


def test_maker_reservation_reduces_deployable_cash():
    p = V8Portfolio(1000.0, 0.25, 0.12, 0.25)
    assert abs(p.deployable_cash - 750.0) < 1e-9
    assert p.reserve_maker_probe(100.0)
    assert abs(p.deployable_cash - 650.0) < 1e-9
    p.release_maker_probe(100.0)
    assert abs(p.deployable_cash - 750.0) < 1e-9


def test_fee_policy_known_categories_and_explicit_free():
    assert resolve_fee_rate({"feesEnabled": True, "category": "Politics"}) == 0.04
    assert resolve_fee_rate({"feesEnabled": True, "category": "Crypto"}) == 0.07
    assert resolve_fee_rate({"feesEnabled": True, "category": "Sports"}) == 0.05
    assert resolve_fee_rate({"feesEnabled": False, "category": "Politics"}) == 0.0
    assert resolve_fee_rate({"feesEnabled": True, "feeSchedule": {"rate": 0.0123}}) == 0.0123


def main():
    test_later_venue_controls_lock_horizon()
    test_economic_depth_sizing_prefers_surplus_not_raw_size()
    test_maker_reservation_reduces_deployable_cash()
    test_fee_policy_known_categories_and_explicit_free()
    print("V10.0 Final profit-optimizer tests passed")


if __name__ == "__main__":
    main()
