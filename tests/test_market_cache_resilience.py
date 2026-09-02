import json
import time

import src.api.polymarket_client as pc
import src.arbitrage.paper_engine_v28 as v28


def _market(i):
    return {
        "id": str(i),
        "question": f"market {i}",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "clobTokenIds": [f"y{i}", f"n{i}"],
        "endDate": "2026-12-31T00:00:00Z",
    }


def test_keyset_failure_preserves_stale_expanded_cache(tmp_path, monkeypatch):
    cache = tmp_path / "poly.json"
    stale = [_market(i) for i in range(20)]
    cache.write_text(json.dumps({"created_at": time.time() - 99999, "markets": stale}))
    monkeypatch.setattr(pc, "EXPANDED_CACHE", cache)
    monkeypatch.setattr(pc, "_flat_active_markets", lambda limit=None: [_market(999)])
    monkeypatch.setattr(pc, "get_active_market_stubs_keyset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    result = pc.get_active_markets_expanded(force_refresh=True, cache_ttl_seconds=1)
    assert len(result) == 20
    assert result[0]["question"] == "market 0"


def test_refresh_downgrade_guard_keeps_large_previous_map(monkeypatch):
    previous = {f"q{i}": {"question": f"q{i}"} for i in range(10000)}
    monkeypatch.setattr(v28, "get_active_markets", lambda limit=None: [{"question": f"new{i}"} for i in range(2100)])
    result = v28._refresh_poly_map_v281(previous)
    assert result is previous
    assert len(result) == 10000


def test_confirmation_snapshot_can_backstop_global_lookup():
    snap = _market(7)
    candidate = {"poly_question": snap["question"], "_poly_market_snapshot": snap}
    assert v28._candidate_poly_snapshot(candidate) is snap
