"""V6 long-run paper engine: portfolio allocator, reliability and GitHub reporting.

Paper only. There is deliberately no order-placement call in this module.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.api.kalshi_client import get_series_info
from src.api.polymarket_client import get_active_markets
from src.arbitrage.paper_engine_v2 import EngineConfig as V2Config
from src.arbitrage.paper_engine_v3 import ConfirmationConfig, run_v3
from src.arbitrage.paper_engine_v5 import (
    PaperPortfolio, PaperPosition, V5Config, _event_key, _series_ticker,
    _fresh_signal, _book_state, _materially_replenished, _choose_route,
    _sample_latency, simulate_selected_route,
)


@dataclass
class V6Config(V5Config):
    run_minutes: float = 240.0
    allocation_window_seconds: float = 1.0
    metadata_refresh_minutes: float = 30.0
    max_signal_age_seconds: float = 3.0
    min_trade_net_dollars: float = 0.02
    api_retries: int = 2
    api_retry_backoff_seconds: float = 0.35
    max_consecutive_api_failures: int = 8
    # ranking: prioritize return, dollars, robustness, then depth
    roc_weight: float = 1.0
    profit_weight: float = 0.35
    depth_weight: float = 0.10
    confirmation_weight: float = 0.15
    auto_figures: bool = True


class RunLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.root = Path("data/runs") / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "signals": self.root / "signals.csv",
            "trades": self.root / "trades.csv",
            "equity": self.root / "equity.csv",
            "latency": self.root / "latency_stress.csv",
            "summary": self.root / "summary.csv",
        }

    def append(self, kind: str, row: dict):
        path = self.paths[kind]
        row = {"run_id": self.run_id, **row}
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)


def _safe_signal(candidate, pm, series_info, *, max_qty, capital_limit, cfg: V6Config):
    last = None
    for attempt in range(cfg.api_retries + 1):
        try:
            return _fresh_signal(candidate, pm, series_info, max_qty, capital_limit, cfg.max_quote_skew_seconds)
        except Exception as exc:
            last = exc
            if attempt < cfg.api_retries:
                time.sleep(cfg.api_retry_backoff_seconds * (2 ** attempt))
    return None, f"API failure after retries: {type(last).__name__}: {last}"


def _confirmation_ratio(candidate: dict) -> float:
    samples = float(candidate.get("samples") or candidate.get("confirmation_samples") or 0)
    positive = float(candidate.get("positive_samples") or candidate.get("confirmations") or 0)
    if samples > 0 and positive > 0:
        return min(1.0, positive / samples)
    # V3 CONFIRMED means every required sample passed.
    return 1.0 if candidate.get("confirmation_status") == "CONFIRMED" else 0.5


def _allocation_score(candidate: dict, signal: dict, cfg: V6Config) -> float:
    capital = max(float(signal["capital"]), 1e-9)
    roc = float(signal["net_profit"]) / capital
    profit = max(0.0, float(signal["net_profit"]))
    # depth coverage >1 means more displayed depth than requested; cap its influence.
    q = max(float(signal["quantity"]), 1.0)
    coverage = min(5.0, min(signal["kalshi_total_size"], signal["poly_total_size"]) / q)
    depth_term = math.log1p(max(0.0, coverage))
    return (
        cfg.roc_weight * roc
        + cfg.profit_weight * profit
        + cfg.depth_weight * depth_term
        + cfg.confirmation_weight * _confirmation_ratio(candidate)
    )


def _signal_row(candidate, signal, eligible, reason, score=None):
    return {
        "timestamp": time.time(), "ticker": candidate.get("ticker"),
        "subject": candidate.get("subject"), "topic": candidate.get("topic"),
        "strategy": candidate.get("strategy"), "eligible": eligible, "reason": reason,
        "allocation_score": score,
        "quantity": None if signal is None else signal.get("quantity"),
        "capital": None if signal is None else signal.get("capital"),
        "net_profit": None if signal is None else signal.get("net_profit"),
        "net_per_contract": None if signal is None else signal.get("net_per_contract"),
        "quote_skew_seconds": None if signal is None else signal.get("fetch_skew_seconds"),
    }


def _trade_row(trade_id, candidate, signal, rr, portfolio, score):
    return {
        "timestamp": time.time(), "trade_id": trade_id, "ticker": candidate.get("ticker"),
        "subject": candidate.get("subject"), "topic": candidate.get("topic"),
        "strategy": candidate.get("strategy"), "route": rr.route, "status": rr.status,
        "allocation_score": score, "quantity": rr.quantity_requested,
        "hedged_quantity": rr.hedged_quantity, "residual_unhedged": rr.residual_unhedged,
        "signal_net": signal.get("net_profit"), "signal_net_per_contract": signal.get("net_per_contract"),
        "latency_seconds": rr.latency_seconds, "second_leg_price_move": rr.second_leg_price_move,
        "locked_capital": rr.locked_capital, "conservative_pnl": rr.conservative_pnl,
        "portfolio_cash": portfolio.available_cash, "portfolio_locked": portfolio.locked_capital,
        "portfolio_pnl": portfolio.realized_pnl, "portfolio_equity": portfolio.equity,
        "reason": rr.reason,
    }


def _equity_row(portfolio, cycle, elapsed_minutes):
    return {
        "timestamp": time.time(), "cycle": cycle, "elapsed_minutes": elapsed_minutes,
        "available_cash": portfolio.available_cash, "locked_capital": portfolio.locked_capital,
        "realized_pnl": portfolio.realized_pnl, "equity": portfolio.equity,
        "positions": len(portfolio.positions), "residual_contracts": portfolio.residual_contracts,
        "max_drawdown": portfolio.max_drawdown,
        "capital_utilization": portfolio.locked_capital / portfolio.starting_bankroll,
    }


def _refresh_poly_map():
    markets = get_active_markets(limit=None)
    return {str(m.get("question")): m for m in markets}


def run_v6(engine_config: V2Config, confirmation_config: ConfirmationConfig, cfg: V6Config):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLogger(run_id)
    print(f"V6 run id: {run_id}")
    print("Building and confirming immediate paper universe...")
    _, confirmations, _ = run_v3(engine_config, confirmation_config)
    confirmed = [x for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    confirmed.sort(key=lambda x: x.get("worst_net_profit", -999), reverse=True)
    confirmed = confirmed[:cfg.max_watch_candidates]
    print(f"Confirmed watch candidates: {len(confirmed)}")

    portfolio = PaperPortfolio(engine_config.bankroll, cfg.reserve_cash_fraction, cfg.max_market_fraction, cfg.max_event_fraction)
    if not confirmed:
        return portfolio, [], log

    by_question = _refresh_poly_map()
    last_metadata_refresh = time.monotonic()
    series_cache = {}
    watch = []
    for c in confirmed:
        pm = by_question.get(c.get("poly_question"))
        if pm is None:
            continue
        series = _series_ticker(c["ticker"])
        if series not in series_cache:
            series_cache[series] = get_series_info(series)
        watch.append([c, pm, series_cache[series]])

    rng = random.Random(cfg.random_seed)
    previous_executed_state = {}
    results = []
    rejection_counts = Counter()
    api_failure_streak = 0
    trade_id = 0
    started = time.monotonic()
    deadline = started + cfg.run_minutes * 60
    cycle = 0

    print(f"Watching {len(watch)} candidates for {cfg.run_minutes:g} minutes | bankroll=${engine_config.bankroll:.2f}")
    while time.monotonic() < deadline:
        cycle += 1
        cycle_start = time.monotonic()

        # Periodically refresh market metadata so fees/status/token IDs cannot silently go stale.
        if (time.monotonic() - last_metadata_refresh) >= cfg.metadata_refresh_minutes * 60:
            try:
                by_question = _refresh_poly_map()
                for item in watch:
                    refreshed = by_question.get(item[0].get("poly_question"))
                    if refreshed is not None:
                        item[1] = refreshed
                last_metadata_refresh = time.monotonic()
                print("Metadata refreshed.")
            except Exception as exc:
                print(f"Metadata refresh warning: {type(exc).__name__}: {exc}")

        # Phase A: snapshot every candidate before allocating scarce cash.
        proposals = []
        for candidate, pm, series_info in watch:
            ticker = candidate["ticker"]
            event_key = _event_key(candidate)
            cap = portfolio.max_capital_for(ticker, event_key)
            if cap < 0.50:
                reason = "portfolio/event capital limit"
                rejection_counts[reason] += 1; log.append("signals", _signal_row(candidate, None, False, reason)); continue
            signal, reason = _safe_signal(candidate, pm, series_info, max_qty=int(candidate["quantity"]), capital_limit=cap, cfg=cfg)
            if signal is None:
                rejection_counts[reason] += 1; log.append("signals", _signal_row(candidate, None, False, reason));
                if reason.startswith("API failure"): api_failure_streak += 1
                continue
            api_failure_streak = 0
            required = max(cfg.minimum_signal_net_per_contract, cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract)
            if signal["net_per_contract"] < required:
                reason = "edge lacks safety buffer"; rejection_counts[reason] += 1; log.append("signals", _signal_row(candidate, signal, False, reason)); continue
            if signal["net_profit"] < cfg.min_trade_net_dollars:
                reason = "profit dollars below minimum"; rejection_counts[reason] += 1; log.append("signals", _signal_row(candidate, signal, False, reason)); continue
            state = _book_state(signal); key = (ticker, candidate["strategy"])
            if cfg.require_material_book_change and not _materially_replenished(previous_executed_state.get(key), state, int(signal["quantity"]), cfg):
                reason = "book not materially replenished/changed"; rejection_counts[reason] += 1; log.append("signals", _signal_row(candidate, signal, False, reason)); continue
            score = _allocation_score(candidate, signal, cfg)
            proposals.append({"candidate": candidate, "pm": pm, "series": series_info, "signal": signal, "state": state, "score": score, "snapshot_time": time.monotonic()})
            log.append("signals", _signal_row(candidate, signal, True, "proposal", score))

        # Brief batch window reduces discovery-order bias; ranking chooses best use of capital.
        if proposals and cfg.allocation_window_seconds > 0:
            time.sleep(min(cfg.allocation_window_seconds, max(0.0, deadline-time.monotonic())))
        proposals.sort(key=lambda x: x["score"], reverse=True)

        for p in proposals:
            candidate, pm, series_info = p["candidate"], p["pm"], p["series"]
            ticker, event_key = candidate["ticker"], _event_key(candidate)
            age = time.monotonic() - p["snapshot_time"]
            if age > cfg.max_signal_age_seconds:
                reason = "proposal expired before execution"; rejection_counts[reason] += 1; continue

            # Re-price after ranking using the *remaining* portfolio capital. This prevents
            # a stale proposal from over-allocating cash after a higher-ranked trade books.
            cap = portfolio.max_capital_for(ticker, event_key)
            signal, reason = _safe_signal(candidate, pm, series_info, max_qty=int(p["signal"]["quantity"]), capital_limit=cap, cfg=cfg)
            if signal is None:
                rejection_counts["failed pre-execution revalidation"] += 1; continue
            required = max(cfg.minimum_signal_net_per_contract, cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract)
            if signal["net_per_contract"] < required or signal["net_profit"] < cfg.min_trade_net_dollars:
                rejection_counts["edge disappeared on revalidation"] += 1; continue

            route = _choose_route(signal, cfg)
            latency = _sample_latency(cfg, rng)
            expected_second = signal["poly_avg"] if route == "kalshi_first" else signal["kalshi_avg"]
            rr = simulate_selected_route(candidate=candidate, pm=pm, series_info=series_info, route=route,
                qty=int(signal["quantity"]), latency_seconds=latency, expected_second_price=expected_second,
                max_second_leg_move=cfg.max_second_leg_move)

            # Stress ladder is diagnostic and never booked.
            for stress_latency in cfg.stress_latencies:
                sr = simulate_selected_route(candidate=candidate, pm=pm, series_info=series_info, route=route,
                    qty=int(signal["quantity"]), latency_seconds=stress_latency, expected_second_price=expected_second,
                    max_second_leg_move=999.0)
                log.append("latency", {"timestamp": time.time(), "ticker": ticker, "subject": candidate.get("subject"),
                    "topic": candidate.get("topic"), "route": route, "quantity": int(signal["quantity"]),
                    "latency_seconds": stress_latency, "status": sr.status, "net_profit": sr.conservative_pnl,
                    "net_per_contract": sr.net_per_requested_contract, "residual_unhedged": sr.residual_unhedged,
                    "second_leg_price_move": sr.second_leg_price_move})

            trade_id += 1
            if rr.status in ("ERROR", "FIRST_LEG_NO_FILL"):
                rejection_counts[rr.status] += 1
                row = _trade_row(trade_id, candidate, signal, rr, portfolio, p["score"]); results.append(row); log.append("trades", row)
                previous_executed_state[(ticker, candidate["strategy"])] = _book_state(signal)
                continue

            if rr.locked_capital > portfolio.available_cash + 1e-9:
                rr.status = "PORTFOLIO_REJECT_AFTER_MOVE"; rr.reason = "execution capital exceeded remaining cash"
                rejection_counts[rr.status] += 1
            else:
                pos = PaperPosition(trade_id, time.time(), ticker, event_key, str(candidate.get("subject")), str(candidate.get("strategy")), route,
                    int(signal["quantity"]), rr.locked_capital, rr.conservative_pnl, rr.residual_unhedged)
                portfolio.book(pos)
                previous_executed_state[(ticker, candidate["strategy"])] = _book_state(signal)
            row = _trade_row(trade_id, candidate, signal, rr, portfolio, p["score"]); results.append(row); log.append("trades", row)
            print(f"BOOK {candidate['subject']} | score={p['score']:.4f} | {route} | qty={signal['quantity']} | net=${rr.conservative_pnl:.4f} | cash=${portfolio.available_cash:.2f}")

            if portfolio.realized_pnl <= -abs(cfg.max_run_loss) or portfolio.residual_contracts > cfg.max_total_residual_contracts:
                print("Risk circuit breaker reached; ending V6.")
                deadline = time.monotonic(); break

        if api_failure_streak >= cfg.max_consecutive_api_failures:
            print("API-health circuit breaker reached; ending V6."); break

        elapsed = (time.monotonic()-started)/60
        log.append("equity", _equity_row(portfolio, cycle, elapsed))
        print(f"Cycle {cycle} | {elapsed:.1f}m | proposals={len(proposals)} | positions={len(portfolio.positions)} | cash=${portfolio.available_cash:.2f} | locked=${portfolio.locked_capital:.2f} | P&L=${portfolio.realized_pnl:.4f}")
        sleep_for = cfg.poll_seconds - (time.monotonic()-cycle_start)
        if sleep_for > 0 and time.monotonic() < deadline:
            time.sleep(min(sleep_for, deadline-time.monotonic()))

    return portfolio, results, log, rejection_counts


def print_summary(portfolio, results, log, rejection_counts, run_minutes):
    positions = portfolio.positions
    wins = [p for p in positions if p.conservative_pnl > 0]
    losses = [p for p in positions if p.conservative_pnl <= 0]
    hours = max(run_minutes/60, 1e-9)
    print("\n" + "="*92)
    print("PAPER ENGINE V6 - LONG-RUN REALISM SUMMARY")
    print("="*92)
    print(f"Run ID:                   {log.run_id}")
    print(f"Starting bankroll:        ${portfolio.starting_bankroll:.2f}")
    print(f"Available / locked:       ${portfolio.available_cash:.2f} / ${portfolio.locked_capital:.2f}")
    print(f"Conservative paper P&L:   ${portfolio.realized_pnl:.4f}")
    print(f"Paper return:             {portfolio.realized_pnl/portfolio.starting_bankroll:.2%}")
    print(f"Profit/hour (sample):     ${portfolio.realized_pnl/hours:.4f}")
    print(f"Booked positions:         {len(positions)}")
    print(f"Positive / nonpositive:   {len(wins)} / {len(losses)}")
    print(f"Unique markets/events:    {len({p.ticker for p in positions})} / {len({p.event_key for p in positions})}")
    print(f"Residual contracts:       {portfolio.residual_contracts:.4f}")
    print(f"Max drawdown:             ${portfolio.max_drawdown:.4f}")
    print(f"Capital utilization:      {portfolio.locked_capital/portfolio.starting_bankroll:.2%}")
    if rejection_counts:
        print("\nTop rejection reasons:")
        for reason, n in rejection_counts.most_common(8): print(f"  {n:5d}  {reason}")
    summary = {"timestamp": time.time(), "starting_bankroll": portfolio.starting_bankroll,
        "available_cash": portfolio.available_cash, "locked_capital": portfolio.locked_capital,
        "pnl": portfolio.realized_pnl, "return_pct": portfolio.realized_pnl/portfolio.starting_bankroll,
        "positions": len(positions), "wins": len(wins), "losses": len(losses),
        "unique_markets": len({p.ticker for p in positions}), "unique_events": len({p.event_key for p in positions}),
        "residual_contracts": portfolio.residual_contracts, "max_drawdown": portfolio.max_drawdown,
        "capital_utilization": portfolio.locked_capital/portfolio.starting_bankroll, "sample_profit_per_hour": portfolio.realized_pnl/hours}
    log.append("summary", summary)
    print(f"\nRun data: {log.root}")
    print("Paper only: V6 contains no order-placement call.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=100); ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--poll", type=float, default=10); ap.add_argument("--min-edge", type=float, default=.0025)
    ap.add_argument("--safety-buffer", type=float, default=.0025); ap.add_argument("--min-profit", type=float, default=.02)
    ap.add_argument("--confirmations", type=int, default=5); ap.add_argument("--confirm-delay", type=float, default=.75)
    ap.add_argument("--confirm-top", type=int, default=12); ap.add_argument("--latency-mean", type=float, default=.25)
    ap.add_argument("--latency-jitter", type=float, default=.15); ap.add_argument("--max-skew", type=float, default=.50)
    ap.add_argument("--event-cap", type=float, default=.35); ap.add_argument("--market-cap", type=float, default=.25)
    ap.add_argument("--reserve-cash", type=float, default=.10); ap.add_argument("--metadata-refresh", type=float, default=30)
    ap.add_argument("--max-signal-age", type=float, default=3); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    ec = V2Config(bankroll=max(1,a.bankroll), min_net_per_contract=max(0,a.min_edge), max_quote_skew_seconds=max(.01,a.max_skew), allow_taker_taker=True)
    cc = ConfirmationConfig(samples=max(1,a.confirmations), delay_seconds=max(0,a.confirm_delay), max_candidates=max(1,a.confirm_top))
    cfg = V6Config(run_minutes=max(0,a.minutes), poll_seconds=max(1,a.poll), max_watch_candidates=max(1,a.confirm_top),
        latency_mean_seconds=max(0,a.latency_mean), latency_jitter_seconds=max(0,a.latency_jitter), max_quote_skew_seconds=max(.01,a.max_skew),
        minimum_execution_net_per_contract=max(0,a.min_edge), minimum_signal_net_per_contract=max(0,a.min_edge+a.safety_buffer),
        minimum_safety_buffer_per_contract=max(0,a.safety_buffer), min_trade_net_dollars=max(0,a.min_profit),
        max_event_fraction=min(1,max(.01,a.event_cap)), max_market_fraction=min(1,max(.01,a.market_cap)), reserve_cash_fraction=min(.95,max(0,a.reserve_cash)),
        metadata_refresh_minutes=max(1,a.metadata_refresh), max_signal_age_seconds=max(.25,a.max_signal_age), random_seed=a.seed, auto_figures=not a.no_figures)
    out = run_v6(ec, cc, cfg)
    if len(out)==3: portfolio, results, log = out; rejects=Counter()
    else: portfolio, results, log, rejects = out
    print_summary(portfolio, results, log, rejects, cfg.run_minutes)
    if cfg.auto_figures:
        try:
            from src.reporting.paper_v6_figures import generate_figures
            generate_figures(log.root)
        except Exception as exc:
            print(f"Figure generation warning: {type(exc).__name__}: {exc}")

if __name__ == "__main__": main()
