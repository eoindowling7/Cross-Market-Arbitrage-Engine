"""V7 settlement-aware long-run paper engine.

Paper only: this module never places orders.

V7 extends the V6 execution/allocator architecture with the realism issue that
matters most for long-dated prediction markets: capital can remain locked for
months or years.  It therefore separates locked-to-resolution profit from
realized cash P&L, estimates settlement horizons and annualized returns, checks
whether positions can be unwound early at executable bids, and writes richer
research logs for GitHub reporting.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.api.kalshi_client import get_market_orderbook, get_series_info
from src.api.polymarket_client import get_active_markets, get_orderbook, parse_token_ids
from src.arbitrage.exact_fees import kalshi_fee, polymarket_taker_fee
from src.arbitrage.execution_utils import consume_asks
from src.arbitrage.paper_engine_v2 import EngineConfig as V2Config, _kalshi_levels
from src.arbitrage.paper_engine_v3 import ConfirmationConfig, run_v3
from src.arbitrage.paper_engine_v5 import (
    _event_key, _series_ticker, _fresh_signal, _book_state,
    _materially_replenished, _choose_route, _sample_latency,
    simulate_selected_route, _strategy_sides,
)
from src.arbitrage.paper_engine_v6 import V6Config, _safe_signal, _confirmation_ratio


@dataclass
class V7Config(V6Config):
    run_minutes: float = 480.0
    # Settlement / capital-efficiency assumptions.
    opportunity_cost_apr: float = 0.10
    annualized_score_weight: float = 0.30
    minimum_hold_apr: float = 0.0
    unknown_settlement_days: float = 730.0
    # Early unwind: only exit when most of the locked profit is preserved AND
    # the value of releasing the capital compensates for surrendered profit.
    early_unwind_enabled: bool = True
    early_unwind_check_seconds: float = 60.0
    early_unwind_capture_fraction: float = 0.75
    early_unwind_min_profit: float = 0.01
    # Capacity analysis is counterfactual and never affects booked paper P&L.
    capacity_bankrolls: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0)
    auto_figures: bool = True


@dataclass
class V7Position:
    trade_id: int
    entry_time: float
    ticker: str
    event_key: str
    subject: str
    topic: str
    strategy: str
    route: str
    quantity: int
    locked_capital: float
    hold_profit: float
    residual_unhedged: float
    kalshi_side: str
    poly_side: str
    settlement_ts: float
    settlement_days_at_entry: float
    annualized_hold_return: float
    status: str = "OPEN"
    exit_time: float | None = None
    exit_pnl: float | None = None
    exit_proceeds: float | None = None
    exit_reason: str | None = None


@dataclass
class V7Portfolio:
    starting_bankroll: float
    reserve_cash_fraction: float
    max_market_fraction: float
    max_event_fraction: float
    available_cash: float = field(init=False)
    locked_capital: float = 0.0
    locked_profit: float = 0.0
    realized_pnl: float = 0.0
    residual_contracts: float = 0.0
    positions: list[V7Position] = field(default_factory=list)
    closed_positions: list[V7Position] = field(default_factory=list)
    event_capital: dict[str, float] = field(default_factory=dict)
    market_capital: dict[str, float] = field(default_factory=dict)
    peak_equity: float = field(init=False)
    max_drawdown: float = 0.0

    def __post_init__(self):
        self.available_cash = float(self.starting_bankroll)
        self.peak_equity = float(self.starting_bankroll)

    @property
    def reserve_cash(self) -> float:
        return self.starting_bankroll * self.reserve_cash_fraction

    @property
    def deployable_cash(self) -> float:
        return max(0.0, self.available_cash - self.reserve_cash)

    @property
    def equity(self) -> float:
        # Locked profit is included because a fully hedged equivalent-contract
        # position has a fixed $1 maturity payoff in the paper model.  It is
        # reported separately from realized cash P&L to avoid pretending the
        # capital is available for reuse.
        return self.available_cash + self.locked_capital + self.locked_profit

    def max_capital_for(self, ticker: str, event_key: str) -> float:
        market_left = self.starting_bankroll * self.max_market_fraction - self.market_capital.get(ticker, 0.0)
        event_left = self.starting_bankroll * self.max_event_fraction - self.event_capital.get(event_key, 0.0)
        return max(0.0, min(self.deployable_cash, market_left, event_left))

    def book(self, position: V7Position) -> None:
        if position.locked_capital > self.available_cash + 1e-9:
            raise ValueError("attempted to lock more paper cash than available")
        self.available_cash -= position.locked_capital
        self.locked_capital += position.locked_capital
        self.locked_profit += position.hold_profit
        self.residual_contracts += position.residual_unhedged
        self.positions.append(position)
        self.event_capital[position.event_key] = self.event_capital.get(position.event_key, 0.0) + position.locked_capital
        self.market_capital[position.ticker] = self.market_capital.get(position.ticker, 0.0) + position.locked_capital
        self._update_drawdown()

    def close(self, position: V7Position, *, proceeds: float, pnl: float, reason: str, when: float | None = None) -> None:
        if position not in self.positions:
            return
        self.positions.remove(position)
        self.closed_positions.append(position)
        self.locked_capital -= position.locked_capital
        self.locked_profit -= position.hold_profit
        self.available_cash += proceeds
        self.realized_pnl += pnl
        self.residual_contracts = max(0.0, self.residual_contracts - position.residual_unhedged)
        self.event_capital[position.event_key] = max(0.0, self.event_capital.get(position.event_key, 0.0) - position.locked_capital)
        self.market_capital[position.ticker] = max(0.0, self.market_capital.get(position.ticker, 0.0) - position.locked_capital)
        position.status = "CLOSED"
        position.exit_time = when or time.time()
        position.exit_pnl = pnl
        position.exit_proceeds = proceeds
        position.exit_reason = reason
        self._update_drawdown()

    def settle_due(self, now: float) -> list[V7Position]:
        settled = []
        for p in list(self.positions):
            if p.settlement_ts <= now:
                proceeds = p.locked_capital + p.hold_profit
                self.close(p, proceeds=proceeds, pnl=p.hold_profit, reason="paper settlement", when=now)
                settled.append(p)
        return settled

    def _update_drawdown(self):
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)


class RunLoggerV7:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.root = Path("data/runs") / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {k: self.root / v for k, v in {
            "signals": "signals.csv", "trades": "trades.csv", "equity": "equity.csv",
            "latency": "latency_stress.csv", "positions": "positions.csv",
            "unwinds": "early_unwinds.csv", "capacity": "capacity_analysis.csv",
            "summary": "summary.csv",
        }.items()}

    def append(self, kind: str, row: dict):
        path = self.paths[kind]
        row = {"run_id": self.run_id, **row}
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)


def _parse_iso_ts(value, default_days: float) -> float:
    if not value:
        return time.time() + default_days * 86400
    text = str(value).strip()
    try:
        if len(text) == 10:
            dt = datetime.fromisoformat(text + "T23:59:59+00:00")
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time() + default_days * 86400


def _settlement_metrics(pm: dict, capital: float, hold_profit: float, cfg: V7Config, now: float | None = None):
    now = now or time.time()
    settlement_ts = _parse_iso_ts(pm.get("endDate") or pm.get("endDateIso"), cfg.unknown_settlement_days)
    days = max(1.0 / 24.0, (settlement_ts - now) / 86400.0)
    roi = hold_profit / max(capital, 1e-9)
    annualized = roi * 365.0 / days
    return settlement_ts, days, annualized


def _allocation_score_v7(candidate: dict, signal: dict, pm: dict, cfg: V7Config) -> tuple[float, float, float]:
    capital = max(float(signal["capital"]), 1e-9)
    roi = float(signal["net_profit"]) / capital
    profit = max(0.0, float(signal["net_profit"]))
    q = max(float(signal["quantity"]), 1.0)
    coverage = min(5.0, min(signal["kalshi_total_size"], signal["poly_total_size"]) / q)
    depth_term = math.log1p(max(0.0, coverage))
    _, days, annualized = _settlement_metrics(pm, capital, profit, cfg)
    # Bounded annualized component prevents a very short-dated tiny trade from
    # dominating the whole allocator solely because its APR explodes.
    annualized_term = min(1.0, max(0.0, annualized))
    score = (
        cfg.roc_weight * roi
        + cfg.profit_weight * profit
        + cfg.depth_weight * depth_term
        + cfg.confirmation_weight * _confirmation_ratio(candidate)
        + cfg.annualized_score_weight * annualized_term
    )
    return score, days, annualized


def _poly_bids(token_id: str) -> list[dict]:
    book = get_orderbook(token_id)
    return sorted([
        {"price": float(x["price"]), "size": float(x["size"])}
        for x in book.get("bids", []) if float(x.get("size", 0)) > 0
    ], key=lambda x: x["price"], reverse=True)


def _kalshi_bids(ticker: str, side: str) -> list[dict]:
    book = get_market_orderbook(ticker, depth=100)
    return sorted(_kalshi_levels(book, side), key=lambda x: x["price"], reverse=True)


def _consume_bids(levels: list[dict], quantity: float):
    remaining = float(quantity); proceeds = 0.0; filled = 0.0; worst = None
    for level in levels:
        if remaining <= 1e-12:
            break
        take = min(remaining, float(level["size"]))
        proceeds += take * float(level["price"])
        filled += take; remaining -= take; worst = float(level["price"])
    avg = proceeds / filled if filled > 0 else None
    return {"quantity": filled, "proceeds": proceeds, "average_price": avg, "worst_price": worst, "fully_filled": remaining <= 1e-9}


def _early_unwind_quote(position: V7Position, pm: dict, series_info: dict):
    tokens = parse_token_ids(pm)
    if len(tokens) != 2:
        return None, "invalid token ids"
    poly_token = tokens[0] if position.poly_side == "yes" else tokens[1]
    kbids = _kalshi_bids(position.ticker, position.kalshi_side)
    pbids = _poly_bids(poly_token)
    if not kbids or not pbids:
        return None, "empty exit book"
    k = _consume_bids(kbids, position.quantity)
    p = _consume_bids(pbids, position.quantity)
    if not k["fully_filled"] or not p["fully_filled"]:
        return None, "insufficient full exit depth"
    kfee_obj = kalshi_fee(price=k["average_price"], contracts=position.quantity,
        fee_type=series_info.get("fee_type"), fee_multiplier=series_info.get("fee_multiplier") or 0, maker=False)
    if kfee_obj is None:
        return None, "unknown Kalshi exit fee"
    kfee = float(kfee_obj["cash_fee_upper"])
    pfee = float(polymarket_taker_fee(p["average_price"], position.quantity, pm))
    proceeds = k["proceeds"] + p["proceeds"] - kfee - pfee
    pnl = proceeds - position.locked_capital
    return {"proceeds": proceeds, "pnl": pnl, "kalshi_exit_avg": k["average_price"],
        "poly_exit_avg": p["average_price"], "kalshi_exit_fee": kfee, "poly_exit_fee": pfee}, "ok"


def _should_unwind(position: V7Position, quote: dict, cfg: V7Config, now: float):
    exit_pnl = float(quote["pnl"])
    if exit_pnl < cfg.early_unwind_min_profit:
        return False, "exit profit below minimum"
    capture = exit_pnl / max(position.hold_profit, 1e-9)
    if capture < cfg.early_unwind_capture_fraction:
        return False, "exit captures too little locked profit"
    remaining_days = max(0.0, (position.settlement_ts - now) / 86400.0)
    opportunity_value = position.locked_capital * cfg.opportunity_cost_apr * remaining_days / 365.0
    # Release capital when the profit preserved plus conservative opportunity
    # value is at least as attractive as simply waiting to settlement.
    if exit_pnl + opportunity_value + 1e-9 < position.hold_profit:
        return False, "holding dominates after opportunity-cost adjustment"
    return True, "capital-efficient early unwind"


def _signal_row(candidate, signal, eligible, reason, score=None, settlement_days=None, annualized_return=None):
    return {
        "timestamp": time.time(), "ticker": candidate.get("ticker"), "subject": candidate.get("subject"),
        "topic": candidate.get("topic"), "strategy": candidate.get("strategy"), "eligible": eligible,
        "reason": reason, "allocation_score": score,
        "quantity": None if signal is None else signal.get("quantity"),
        "capital": None if signal is None else signal.get("capital"),
        "net_profit": None if signal is None else signal.get("net_profit"),
        "net_per_contract": None if signal is None else signal.get("net_per_contract"),
        "quote_skew_seconds": None if signal is None else signal.get("fetch_skew_seconds"),
        "kalshi_total_size": None if signal is None else signal.get("kalshi_total_size"),
        "poly_total_size": None if signal is None else signal.get("poly_total_size"),
        "settlement_days": settlement_days, "annualized_hold_return": annualized_return,
    }


def _equity_row(portfolio: V7Portfolio, cycle: int, elapsed_minutes: float):
    return {"timestamp": time.time(), "cycle": cycle, "elapsed_minutes": elapsed_minutes,
        "available_cash": portfolio.available_cash, "locked_capital": portfolio.locked_capital,
        "locked_profit": portfolio.locked_profit, "realized_pnl": portfolio.realized_pnl,
        "equity": portfolio.equity, "open_positions": len(portfolio.positions),
        "closed_positions": len(portfolio.closed_positions), "residual_contracts": portfolio.residual_contracts,
        "max_drawdown": portfolio.max_drawdown,
        "capital_utilization": portfolio.locked_capital / portfolio.starting_bankroll}


def _refresh_poly_map():
    markets = get_active_markets(limit=None)
    return {str(m.get("question")): m for m in markets}


def _capacity_analysis(log: RunLoggerV7, cfg: V7Config):
    """Observed-book capacity estimate from best eligible signal per strategy.

    This is deliberately labelled counterfactual: it assumes the displayed
    depth observed in the run were available to a larger paper bankroll and
    does not add any result to trading P&L.
    """
    path = log.paths["signals"]
    if not path.exists():
        return []
    try:
        import pandas as pd
        d = pd.read_csv(path)
    except Exception:
        return []
    d = d[(d["eligible"] == True) & d["capital"].notna() & (d["capital"] > 0) & d["net_profit"].notna()]
    if d.empty:
        return []
    # Keep strongest observed signal for each ticker/strategy, avoiding repeated
    # snapshots from inflating capacity.
    d = d.sort_values("net_profit", ascending=False).drop_duplicates(["ticker", "strategy"])
    rows = []
    for bankroll in cfg.capacity_bankrolls:
        reserve = bankroll * cfg.reserve_cash_fraction
        cash = bankroll - reserve
        event_used = {}; market_used = {}; profit = 0.0; deployed = 0.0; trades = 0
        for _, r in d.sort_values("allocation_score", ascending=False).iterrows():
            ticker = str(r["ticker"]); event = str(r["topic"])
            unit_cost = float(r["capital"]) / max(float(r["quantity"]), 1.0)
            unit_profit = float(r["net_profit"]) / max(float(r["quantity"]), 1.0)
            depth_qty = int(max(0.0, min(float(r.get("kalshi_total_size", 0) or 0), float(r.get("poly_total_size", 0) or 0))))
            if depth_qty < 1 or unit_cost <= 0 or unit_profit <= 0:
                continue
            cap = min(cash, bankroll*cfg.max_market_fraction-market_used.get(ticker,0), bankroll*cfg.max_event_fraction-event_used.get(event,0))
            qty = min(depth_qty, int(max(0, cap // unit_cost)))
            if qty < 1:
                continue
            cost = qty*unit_cost; pnl = qty*unit_profit
            cash -= cost; deployed += cost; profit += pnl; trades += 1
            market_used[ticker] = market_used.get(ticker,0)+cost; event_used[event] = event_used.get(event,0)+cost
        row = {"timestamp": time.time(), "bankroll": bankroll, "estimated_deployed": deployed,
            "estimated_locked_profit": profit, "estimated_return": profit/bankroll if bankroll else 0,
            "estimated_positions": trades, "method": "observed-book counterfactual"}
        log.append("capacity", row); rows.append(row)
    return rows


def run_v7(engine_config: V2Config, confirmation_config: ConfirmationConfig, cfg: V7Config):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLoggerV7(run_id)
    print(f"V7 run id: {run_id}")
    print("Building and confirming immediate paper universe...")
    _, confirmations, _ = run_v3(engine_config, confirmation_config)
    confirmed = [x for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    confirmed.sort(key=lambda x: x.get("worst_net_profit", -999), reverse=True)
    confirmed = confirmed[:cfg.max_watch_candidates]
    print(f"Confirmed watch candidates: {len(confirmed)}")

    portfolio = V7Portfolio(engine_config.bankroll, cfg.reserve_cash_fraction, cfg.max_market_fraction, cfg.max_event_fraction)
    if not confirmed:
        return portfolio, [], log, Counter()

    by_question = _refresh_poly_map(); last_metadata_refresh = time.monotonic(); series_cache = {}; watch = []
    for c in confirmed:
        pm = by_question.get(c.get("poly_question"))
        if pm is None: continue
        series = _series_ticker(c["ticker"])
        if series not in series_cache: series_cache[series] = get_series_info(series)
        watch.append([c, pm, series_cache[series]])

    rng = random.Random(cfg.random_seed); previous_executed_state = {}; results = []; rejection_counts = Counter()
    trade_id = 0; api_failure_streak = 0; started = time.monotonic(); deadline = started + cfg.run_minutes*60; cycle = 0
    last_unwind_check = 0.0
    print(f"Watching {len(watch)} candidates for {cfg.run_minutes:g} minutes | bankroll=${engine_config.bankroll:.2f}")

    while time.monotonic() < deadline:
        cycle += 1; cycle_start = time.monotonic(); now_wall = time.time()
        for p in portfolio.settle_due(now_wall):
            log.append("unwinds", {"timestamp": now_wall, "trade_id": p.trade_id, "ticker": p.ticker, "subject": p.subject,
                "action": "SETTLEMENT", "hold_profit": p.hold_profit, "exit_pnl": p.exit_pnl,
                "capital_released": p.exit_proceeds, "reason": p.exit_reason})

        if (time.monotonic()-last_metadata_refresh) >= cfg.metadata_refresh_minutes*60:
            try:
                by_question = _refresh_poly_map()
                for item in watch:
                    refreshed = by_question.get(item[0].get("poly_question"))
                    if refreshed is not None: item[1] = refreshed
                last_metadata_refresh = time.monotonic(); print("Metadata refreshed.")
            except Exception as exc:
                print(f"Metadata refresh warning: {type(exc).__name__}: {exc}")

        # Early-unwind scan is deliberately infrequent to avoid manufacturing
        # turnover from rapid quote polling.
        if cfg.early_unwind_enabled and (time.monotonic()-last_unwind_check) >= cfg.early_unwind_check_seconds:
            last_unwind_check = time.monotonic()
            pm_by_question = {str(x[1].get("question")): x[1] for x in watch}
            candidate_by_ticker = {x[0]["ticker"]: x for x in watch}
            for pos in list(portfolio.positions):
                item = candidate_by_ticker.get(pos.ticker)
                if item is None: continue
                candidate, pm, series_info = item
                try:
                    quote, reason = _early_unwind_quote(pos, pm, series_info)
                except Exception as exc:
                    quote, reason = None, f"exit API error: {type(exc).__name__}"
                if quote is None:
                    log.append("unwinds", {"timestamp": time.time(), "trade_id": pos.trade_id, "ticker": pos.ticker,
                        "subject": pos.subject, "action": "HOLD", "hold_profit": pos.hold_profit, "exit_pnl": None,
                        "capital_released": None, "reason": reason}); continue
                should, why = _should_unwind(pos, quote, cfg, time.time())
                log.append("unwinds", {"timestamp": time.time(), "trade_id": pos.trade_id, "ticker": pos.ticker,
                    "subject": pos.subject, "action": "EXIT" if should else "HOLD", "hold_profit": pos.hold_profit,
                    "exit_pnl": quote["pnl"], "capital_released": quote["proceeds"], "reason": why})
                if should:
                    portfolio.close(pos, proceeds=quote["proceeds"], pnl=quote["pnl"], reason=why)
                    print(f"UNWIND {pos.subject} | realized=${quote['pnl']:.4f} | released=${quote['proceeds']:.2f}")

        proposals = []
        for candidate, pm, series_info in watch:
            ticker = candidate["ticker"]; event_key = _event_key(candidate); cap = portfolio.max_capital_for(ticker,event_key)
            if cap < 0.50:
                reason="portfolio/event capital limit"; rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,None,False,reason)); continue
            signal, reason = _safe_signal(candidate,pm,series_info,max_qty=int(candidate["quantity"]),capital_limit=cap,cfg=cfg)
            if signal is None:
                rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,None,False,reason));
                if reason.startswith("API failure"): api_failure_streak += 1
                continue
            api_failure_streak=0
            required=max(cfg.minimum_signal_net_per_contract,cfg.minimum_execution_net_per_contract+cfg.minimum_safety_buffer_per_contract)
            if signal["net_per_contract"]<required:
                reason="edge lacks safety buffer"; rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,signal,False,reason)); continue
            if signal["net_profit"]<cfg.min_trade_net_dollars:
                reason="profit dollars below minimum"; rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,signal,False,reason)); continue
            state=_book_state(signal); key=(ticker,candidate["strategy"])
            if cfg.require_material_book_change and not _materially_replenished(previous_executed_state.get(key),state,int(signal["quantity"]),cfg):
                reason="book not materially replenished/changed"; rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,signal,False,reason)); continue
            score, days, annualized = _allocation_score_v7(candidate,signal,pm,cfg)
            if annualized < cfg.minimum_hold_apr:
                reason="annualized hold return below minimum"; rejection_counts[reason]+=1; log.append("signals",_signal_row(candidate,signal,False,reason,score,days,annualized)); continue
            proposals.append({"candidate":candidate,"pm":pm,"series":series_info,"signal":signal,"state":state,"score":score,"snapshot_time":time.monotonic(),"settlement_days":days,"annualized":annualized})
            log.append("signals",_signal_row(candidate,signal,True,"proposal",score,days,annualized))

        if proposals and cfg.allocation_window_seconds>0:
            time.sleep(min(cfg.allocation_window_seconds,max(0.0,deadline-time.monotonic())))
        proposals.sort(key=lambda x:x["score"],reverse=True)

        for p in proposals:
            candidate,pm,series_info=p["candidate"],p["pm"],p["series"]; ticker=candidate["ticker"]; event_key=_event_key(candidate)
            if time.monotonic()-p["snapshot_time"]>cfg.max_signal_age_seconds:
                rejection_counts["proposal expired before execution"]+=1; continue
            cap=portfolio.max_capital_for(ticker,event_key)
            signal,reason=_safe_signal(candidate,pm,series_info,max_qty=int(p["signal"]["quantity"]),capital_limit=cap,cfg=cfg)
            if signal is None:
                rejection_counts["failed pre-execution revalidation"]+=1; continue
            required=max(cfg.minimum_signal_net_per_contract,cfg.minimum_execution_net_per_contract+cfg.minimum_safety_buffer_per_contract)
            if signal["net_per_contract"]<required or signal["net_profit"]<cfg.min_trade_net_dollars:
                rejection_counts["edge disappeared on revalidation"]+=1; continue
            score,days,annualized=_allocation_score_v7(candidate,signal,pm,cfg)
            if annualized<cfg.minimum_hold_apr:
                rejection_counts["annualized hold return below minimum"]+=1; continue
            route=_choose_route(signal,cfg); latency=_sample_latency(cfg,rng)
            expected_second=signal["poly_avg"] if route=="kalshi_first" else signal["kalshi_avg"]
            rr=simulate_selected_route(candidate=candidate,pm=pm,series_info=series_info,route=route,qty=int(signal["quantity"]),latency_seconds=latency,expected_second_price=expected_second,max_second_leg_move=cfg.max_second_leg_move)
            for stress_latency in cfg.stress_latencies:
                sr=simulate_selected_route(candidate=candidate,pm=pm,series_info=series_info,route=route,qty=int(signal["quantity"]),latency_seconds=stress_latency,expected_second_price=expected_second,max_second_leg_move=999.0)
                log.append("latency",{"timestamp":time.time(),"ticker":ticker,"subject":candidate.get("subject"),"topic":candidate.get("topic"),"route":route,"quantity":int(signal["quantity"]),"latency_seconds":stress_latency,"status":sr.status,"net_profit":sr.conservative_pnl,"net_per_contract":sr.net_per_requested_contract,"residual_unhedged":sr.residual_unhedged,"second_leg_price_move":sr.second_leg_price_move})
            trade_id+=1
            if rr.status in ("ERROR","FIRST_LEG_NO_FILL"):
                rejection_counts[rr.status]+=1; continue
            if rr.locked_capital>portfolio.available_cash+1e-9:
                rejection_counts["PORTFOLIO_REJECT_AFTER_MOVE"]+=1; continue
            settlement_ts,days,annualized=_settlement_metrics(pm,rr.locked_capital,rr.conservative_pnl,cfg)
            kside,pside=_strategy_sides(candidate["strategy"])
            pos=V7Position(trade_id,time.time(),ticker,event_key,str(candidate.get("subject")),str(candidate.get("topic")),str(candidate.get("strategy")),route,int(signal["quantity"]),rr.locked_capital,rr.conservative_pnl,rr.residual_unhedged,kside,pside,settlement_ts,days,annualized)
            portfolio.book(pos); previous_executed_state[(ticker,candidate["strategy"])]=_book_state(signal)
            row={"timestamp":time.time(),"trade_id":trade_id,"ticker":ticker,"subject":candidate.get("subject"),"topic":candidate.get("topic"),"strategy":candidate.get("strategy"),"route":route,"status":rr.status,"allocation_score":score,"quantity":int(signal["quantity"]),"latency_seconds":latency,"locked_capital":rr.locked_capital,"hold_profit":rr.conservative_pnl,"net_per_contract":rr.net_per_requested_contract,"settlement_ts":settlement_ts,"settlement_days":days,"annualized_hold_return":annualized,"residual_unhedged":rr.residual_unhedged,"portfolio_cash":portfolio.available_cash,"portfolio_locked":portfolio.locked_capital,"portfolio_locked_profit":portfolio.locked_profit,"portfolio_realized_pnl":portfolio.realized_pnl,"portfolio_equity":portfolio.equity}
            results.append(row); log.append("trades",row); log.append("positions",row)
            print(f"BOOK {candidate['subject']} | score={score:.4f} | qty={signal['quantity']} | locked=${rr.locked_capital:.2f} | hold=${rr.conservative_pnl:.4f} | {days:.0f}d | annualized={annualized:.2%}")
            if portfolio.realized_pnl<=-abs(cfg.max_run_loss) or portfolio.residual_contracts>cfg.max_total_residual_contracts:
                print("Risk circuit breaker reached; ending V7."); deadline=time.monotonic(); break

        if api_failure_streak>=cfg.max_consecutive_api_failures:
            print("API-health circuit breaker reached; ending V7."); break
        elapsed=(time.monotonic()-started)/60; log.append("equity",_equity_row(portfolio,cycle,elapsed))
        print(f"Cycle {cycle} | {elapsed:.1f}m | proposals={len(proposals)} | open={len(portfolio.positions)} | closed={len(portfolio.closed_positions)} | cash=${portfolio.available_cash:.2f} | locked=${portfolio.locked_capital:.2f} | locked profit=${portfolio.locked_profit:.4f} | realized=${portfolio.realized_pnl:.4f}")
        sleep_for=cfg.poll_seconds-(time.monotonic()-cycle_start)
        if sleep_for>0 and time.monotonic()<deadline: time.sleep(min(sleep_for,deadline-time.monotonic()))

    _capacity_analysis(log,cfg)
    return portfolio,results,log,rejection_counts


def print_summary(portfolio: V7Portfolio, results, log: RunLoggerV7, rejection_counts, run_minutes):
    open_pos=portfolio.positions; closed=portfolio.closed_positions; hours=max(run_minutes/60,1e-9)
    weighted_days=sum(p.locked_capital*p.settlement_days_at_entry for p in open_pos)/max(sum(p.locked_capital for p in open_pos),1e-9) if open_pos else 0
    weighted_apr=sum(p.locked_capital*p.annualized_hold_return for p in open_pos)/max(sum(p.locked_capital for p in open_pos),1e-9) if open_pos else 0
    print("\n"+"="*96); print("PAPER ENGINE V7 - SETTLEMENT-AWARE LONG-RUN SUMMARY"); print("="*96)
    print(f"Run ID:                         {log.run_id}")
    print(f"Starting bankroll:              ${portfolio.starting_bankroll:.2f}")
    print(f"Available / locked:             ${portfolio.available_cash:.2f} / ${portfolio.locked_capital:.2f}")
    print(f"Locked-to-resolution profit:    ${portfolio.locked_profit:.4f}")
    print(f"Realized early-exit/settle P&L: ${portfolio.realized_pnl:.4f}")
    print(f"Paper equity:                   ${portfolio.equity:.4f}")
    print(f"Open / closed positions:        {len(open_pos)} / {len(closed)}")
    print(f"Capital utilization:            {portfolio.locked_capital/portfolio.starting_bankroll:.2%}")
    print(f"Capital-weighted lock horizon:  {weighted_days:.1f} days")
    print(f"Capital-weighted hold APR:      {weighted_apr:.2%}")
    print(f"Residual contracts:             {portfolio.residual_contracts:.4f}")
    print(f"Max drawdown:                   ${portfolio.max_drawdown:.4f}")
    print(f"New locked profit/hour sample:  ${sum(float(x.get('hold_profit',0)) for x in results)/hours:.4f} (NOT reusable cash)")
    if rejection_counts:
        print("\nTop rejection reasons:")
        for reason,n in rejection_counts.most_common(10): print(f"  {n:5d}  {reason}")
    summary={"timestamp":time.time(),"starting_bankroll":portfolio.starting_bankroll,"available_cash":portfolio.available_cash,"locked_capital":portfolio.locked_capital,"locked_profit":portfolio.locked_profit,"realized_pnl":portfolio.realized_pnl,"equity":portfolio.equity,"open_positions":len(open_pos),"closed_positions":len(closed),"capital_utilization":portfolio.locked_capital/portfolio.starting_bankroll,"weighted_settlement_days":weighted_days,"weighted_hold_apr":weighted_apr,"residual_contracts":portfolio.residual_contracts,"max_drawdown":portfolio.max_drawdown,"sample_new_locked_profit_per_hour":sum(float(x.get('hold_profit',0)) for x in results)/hours}
    log.append("summary",summary)
    print(f"\nRun data: {log.root}"); print("Paper only: V7 contains no order-placement call.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bankroll",type=float,default=100); ap.add_argument("--minutes",type=float,default=480); ap.add_argument("--poll",type=float,default=10)
    ap.add_argument("--min-edge",type=float,default=.0025); ap.add_argument("--safety-buffer",type=float,default=.0025); ap.add_argument("--min-profit",type=float,default=.02)
    ap.add_argument("--confirmations",type=int,default=5); ap.add_argument("--confirm-delay",type=float,default=.75); ap.add_argument("--confirm-top",type=int,default=12)
    ap.add_argument("--latency-mean",type=float,default=.25); ap.add_argument("--latency-jitter",type=float,default=.15); ap.add_argument("--max-skew",type=float,default=.50)
    ap.add_argument("--event-cap",type=float,default=.35); ap.add_argument("--market-cap",type=float,default=.25); ap.add_argument("--reserve-cash",type=float,default=.10)
    ap.add_argument("--metadata-refresh",type=float,default=30); ap.add_argument("--max-signal-age",type=float,default=3); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--opportunity-cost-apr",type=float,default=.10); ap.add_argument("--min-hold-apr",type=float,default=0.0)
    ap.add_argument("--unwind-check",type=float,default=60); ap.add_argument("--unwind-capture",type=float,default=.75); ap.add_argument("--no-early-unwind",action="store_true"); ap.add_argument("--no-figures",action="store_true")
    a=ap.parse_args()
    ec=V2Config(bankroll=max(1,a.bankroll),min_net_per_contract=max(0,a.min_edge),max_quote_skew_seconds=max(.01,a.max_skew),allow_taker_taker=True)
    cc=ConfirmationConfig(samples=max(1,a.confirmations),delay_seconds=max(0,a.confirm_delay),max_candidates=max(1,a.confirm_top))
    cfg=V7Config(run_minutes=max(0,a.minutes),poll_seconds=max(1,a.poll),max_watch_candidates=max(1,a.confirm_top),latency_mean_seconds=max(0,a.latency_mean),latency_jitter_seconds=max(0,a.latency_jitter),max_quote_skew_seconds=max(.01,a.max_skew),minimum_execution_net_per_contract=max(0,a.min_edge),minimum_signal_net_per_contract=max(0,a.min_edge+a.safety_buffer),minimum_safety_buffer_per_contract=max(0,a.safety_buffer),min_trade_net_dollars=max(0,a.min_profit),max_event_fraction=min(1,max(.01,a.event_cap)),max_market_fraction=min(1,max(.01,a.market_cap)),reserve_cash_fraction=min(.95,max(0,a.reserve_cash)),metadata_refresh_minutes=max(1,a.metadata_refresh),max_signal_age_seconds=max(.25,a.max_signal_age),random_seed=a.seed,opportunity_cost_apr=max(0,a.opportunity_cost_apr),minimum_hold_apr=max(0,a.min_hold_apr),early_unwind_check_seconds=max(5,a.unwind_check),early_unwind_capture_fraction=min(1,max(0,a.unwind_capture)),early_unwind_enabled=not a.no_early_unwind,auto_figures=not a.no_figures)
    portfolio,results,log,rejects=run_v7(ec,cc,cfg); print_summary(portfolio,results,log,rejects,cfg.run_minutes)
    if cfg.auto_figures:
        try:
            from src.reporting.paper_v7_figures import generate_figures
            generate_figures(log.root)
        except Exception as exc:
            print(f"Figure generation warning: {type(exc).__name__}: {exc}")

if __name__=="__main__": main()
