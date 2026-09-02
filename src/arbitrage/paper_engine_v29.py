"""V29 split discovery/watch paper engine.

Paper-only research build. V29 deliberately treats unresolved/missing rule evidence
as non-blocking in the semantic-trust lane used for experimentation. Explicit
structural contradictions, missing executable liquidity, fees, depth, quote skew,
and positive-edge checks remain enforced.

Modes
-----
  discover  Expensive full-universe semantic discovery + confirmation, then save.
  watch     Load the saved confirmed watchlist and monitor it immediately.
  full      Run discover, save, then watch in the same process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Enable the relaxed V29 paper lane before importing V28 implementation.
os.environ.setdefault("V29_RELAX_UNRESOLVED", "1")
os.environ.setdefault("V29_SKIP_LATE_RULE_HYDRATION", "1")
os.environ.setdefault("V29_ALLOW_ONE_SIDED_STRUCTURAL", "1")

from src.arbitrage import paper_engine_v28 as v28
from src.arbitrage.paper_engine_v2 import EngineConfig as V2Config
from src.arbitrage.paper_engine_v3_v28 import ConfirmationConfig, run_v3
from src.arbitrage.paper_engine_v13 import refresh_kalshi_open_market_cache


def _default_watchlist_path() -> Path:
    env = os.getenv("V29_WATCHLIST_PATH")
    if env:
        return Path(env)
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        return drive / "kalshi-market-cache" / "v29_confirmed_watchlist.json"
    return Path("data/processed/v29_confirmed_watchlist.json")


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def save_watchlist(confirmations: list[dict], path: Path, *, limit: int) -> list[dict]:
    confirmed = [dict(x) for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    confirmed.sort(
        key=lambda x: (
            float(x.get("confirmation_priority", -999.0)),
            float(x.get("worst_net_profit", -999.0)),
        ),
        reverse=True,
    )
    confirmed = confirmed[: max(1, int(limit))]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v29-confirmed-watchlist-1",
        "paper_only": True,
        "count": len(confirmed),
        "candidates": _json_safe(confirmed),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return confirmed


def load_watchlist(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Invalid V29 watchlist: {path}")
    return [dict(x) for x in rows if isinstance(x, dict)]


def _engine_config(a) -> V2Config:
    return V2Config(
        bankroll=max(1.0, a.bankroll),
        min_net_per_contract=max(0.0, a.min_edge),
        max_quote_skew_seconds=max(0.01, a.max_skew),
        allow_taker_taker=True,
    )


def _confirmation_config(a) -> ConfirmationConfig:
    return ConfirmationConfig(
        samples=max(1, a.confirmations),
        delay_seconds=max(0.0, a.confirm_delay),
        max_candidates=max(1, a.confirm_top),
    )


def _watch_config(a) -> v28.V8Config:
    # Watch mode intentionally disables full-universe auxiliary scans by default.
    # They were responsible for long pre-cycle delays and are unrelated to the
    # already-confirmed candidate watchlist. Live books for each candidate are
    # still refreshed every cycle and again immediately before simulated fill.
    aux = bool(a.with_auxiliary_lanes)
    return v28.V8Config(
        run_minutes=max(0.0, a.minutes),
        poll_seconds=max(1.0, a.poll),
        max_watch_candidates=max(1, a.confirm_top),
        max_watch_universe=max(1, max(a.confirm_top, 220)),
        latency_mean_seconds=max(0.0, a.latency_mean),
        latency_jitter_seconds=max(0.0, a.latency_jitter),
        max_quote_skew_seconds=max(0.01, a.max_skew),
        minimum_execution_net_per_contract=max(0.0, a.min_edge),
        minimum_signal_net_per_contract=max(0.0, a.min_edge + a.safety_buffer),
        minimum_safety_buffer_per_contract=max(0.0, a.safety_buffer),
        min_trade_net_dollars=max(0.0, a.min_profit),
        max_event_fraction=min(1.0, max(0.01, a.event_cap)),
        max_market_fraction=min(1.0, max(0.01, a.market_cap)),
        reserve_cash_fraction=min(0.95, max(0.0, a.reserve_cash)),
        # Avoid expensive global metadata/discovery refreshes in a short watch run.
        # Candidate-level executable books remain live on every cycle.
        metadata_refresh_minutes=1440.0,
        universe_refresh_minutes=0.0,
        max_signal_age_seconds=max(0.25, a.max_signal_age),
        random_seed=a.seed,
        opportunity_cost_apr=max(0.0, a.opportunity_cost_apr),
        base_required_hold_apr=max(0.0, a.base_hold_apr),
        lock_horizon_hurdle_apr=max(0.0, a.lock_hurdle_apr),
        hard_duration_hurdle=bool(a.hard_duration_hurdle),
        subhurdle_total_fraction=min(0.20, max(0.0, a.subhurdle_cap)),
        early_unwind_check_seconds=max(5.0, a.unwind_check),
        early_unwind_capture_fraction=min(1.0, max(0.0, a.unwind_capture)),
        early_unwind_enabled=not a.no_early_unwind,
        auto_figures=not a.no_figures,
        fast_recycle_enabled=aux,
        multi_outcome_enabled=aux,
        limitless_cross_enabled=aux,
        maker_probe_enabled=False,
        optimized_quantity_sizing=True,
        low_basis_enabled=True,
        low_basis_total_fraction=min(0.10, max(0.0, a.low_basis_cap)),
        low_basis_min_adjusted_apr=max(0.0, a.low_basis_min_apr),
    )


def discover(a, path: Path) -> list[dict]:
    print("V29 DISCOVERY | expensive full-universe stage")
    refresh_kalshi_open_market_cache()
    _, confirmations, _ = run_v3(_engine_config(a), _confirmation_config(a))
    confirmed = save_watchlist(confirmations, path, limit=a.confirm_top)
    print(f"V29 discovery complete | confirmed={len(confirmed)} | saved={path}")
    print("You can now stop the GPU runtime and run V29 watch on CPU.")
    return confirmed


def watch(a, path: Path, preloaded: list[dict] | None = None):
    confirmed = preloaded if preloaded is not None else load_watchlist(path)
    if not confirmed:
        raise RuntimeError(f"V29 watchlist is empty: {path}")

    # Preserve only the exact confirmed candidates. No global Polymarket refresh
    # is required to begin monitoring because V28.1 stored the confirmation-time
    # Polymarket market snapshot on each candidate.
    snapshot_map = {}
    for c in confirmed:
        snap = c.get("_poly_market_snapshot")
        if isinstance(snap, dict) and snap.get("question"):
            snapshot_map[str(snap["question"])] = snap

    missing = sum(1 for c in confirmed if not isinstance(c.get("_poly_market_snapshot"), dict))
    if missing:
        print(f"V29 watch warning | candidates missing confirmation snapshot={missing}")

    series_snapshot_map = {}
    for c in confirmed:
        ticker = str(c.get("ticker") or "")
        if not ticker:
            continue
        series = v28._series_ticker(ticker)
        snap = c.get("_series_info_snapshot")
        if isinstance(snap, dict) and snap:
            series_snapshot_map[series] = snap

    original_run_v3 = v28.run_v3
    original_refresh = v28._refresh_poly_map_v281
    original_get_series = v28.get_series_info
    try:
        v28.run_v3 = lambda engine_config, confirmation_config: ([], confirmed, [])
        v28._refresh_poly_map_v281 = lambda previous=None: dict(snapshot_map if snapshot_map else (previous or {}))
        v28.get_series_info = lambda series: dict(series_snapshot_map.get(series) or original_get_series(series))
        print(
            f"V29 WATCH | loaded={len(confirmed)} | unresolved rule evidence NON-BLOCKING | "
            "explicit known contradictions still rejected"
        )
        portfolio, results, log, rejects = v28.run_v8(
            _engine_config(a), _confirmation_config(a), _watch_config(a)
        )
        v28.print_summary(portfolio, results, log, rejects, a.minutes, _watch_config(a))
        return portfolio, results, log, rejects
    finally:
        v28.run_v3 = original_run_v3
        v28._refresh_poly_map_v281 = original_refresh
        v28.get_series_info = original_get_series


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="V29 split semantic-trust paper engine")
    ap.add_argument("mode", choices=("discover", "watch", "full"), nargs="?", default="watch")
    ap.add_argument("--watchlist", default=str(_default_watchlist_path()))
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--min-edge", type=float, default=0.0015)
    ap.add_argument("--safety-buffer", type=float, default=0.0010)
    ap.add_argument("--min-profit", type=float, default=0.01)
    ap.add_argument("--confirmations", type=int, default=5)
    ap.add_argument("--confirm-delay", type=float, default=0.75)
    ap.add_argument("--confirm-top", type=int, default=220)
    ap.add_argument("--latency-mean", type=float, default=0.25)
    ap.add_argument("--latency-jitter", type=float, default=0.15)
    ap.add_argument("--max-skew", type=float, default=0.50)
    ap.add_argument("--event-cap", type=float, default=0.30)
    ap.add_argument("--market-cap", type=float, default=0.15)
    ap.add_argument("--reserve-cash", type=float, default=0.20)
    ap.add_argument("--max-signal-age", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opportunity-cost-apr", type=float, default=0.060)
    ap.add_argument("--base-hold-apr", type=float, default=0.025)
    ap.add_argument("--lock-hurdle-apr", type=float, default=0.030)
    ap.add_argument("--hard-duration-hurdle", action="store_true")
    ap.add_argument("--subhurdle-cap", type=float, default=0.02)
    ap.add_argument("--unwind-check", type=float, default=60.0)
    ap.add_argument("--unwind-capture", type=float, default=0.75)
    ap.add_argument("--no-early-unwind", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--low-basis-cap", type=float, default=0.03)
    ap.add_argument("--low-basis-min-apr", type=float, default=0.060)
    ap.add_argument("--with-auxiliary-lanes", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    path = Path(a.watchlist)
    if a.mode == "discover":
        discover(a, path)
    elif a.mode == "watch":
        watch(a, path)
    else:
        confirmed = discover(a, path)
        watch(a, path, preloaded=confirmed)


if __name__ == "__main__":
    main()
