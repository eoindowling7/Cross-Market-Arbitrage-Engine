"""V8 capital-velocity paper engine (paper only).

V13.0 Presentation Final is the payoff-safe, presentation-oriented paper architecture built on V12.0.  It keeps V7's
settlement-aware accounting and adds the main economic correction exposed by
the V7 long run: a dollar of profit is not equally attractive when the capital
is locked for 7 days versus 700 days.

Major additions
---------------
- $1,000 default paper bankroll
- capital-velocity / duration-aware allocator
- dynamic minimum hold-APR hurdle that rises with lock duration
- duration-bucket portfolio caps so long-dated politics cannot consume the
  whole bankroll
- wider confirmed watch universe for a larger bankroll
- periodic full-universe refresh during long runs
- duration-aware early-unwind rule to recycle capital more aggressively when
  a long lock has already captured most of its profit
- post-latency APR recheck before a position is booked
- counterfactual allocator-policy analysis for GitHub/reporting
- counterfactual capacity analysis that respects V8 duration buckets

There is deliberately no order-placement call anywhere in this module.
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

from src.api.kalshi_client import get_series_info, get_market_trades
from src.api.polymarket_client import get_active_markets, parse_token_ids
from src.arbitrage.paper_engine_v2 import EngineConfig as V2Config
from src.arbitrage.paper_engine_v3 import ConfirmationConfig, run_v3
from src.arbitrage.paper_engine_v5 import (
    _event_key,
    _series_ticker,
    _book_state,
    _materially_replenished,
    _choose_route,
    _sample_latency,
    simulate_selected_route,
    _strategy_sides,
    _kalshi_taker_asks,
    _poly_asks,
    _signal_from_levels,
)
from src.arbitrage.paper_engine_v6 import _safe_signal, _confirmation_ratio
from src.arbitrage.exact_fees import kalshi_fee, polymarket_taker_fee
from src.arbitrage.execution_utils import consume_asks
from src.arbitrage.fast_recycle import FastRecycleConfig, scan_polymarket_complete_sets
from src.arbitrage.multi_venue_discovery import audit_extra_venues
from src.arbitrage.multi_outcome_basket import MultiOutcomeConfig, scan_polymarket_multi_outcome
from src.arbitrage.limitless_polymarket import LimitlessPolyConfig, scan_limitless_polymarket
from src.arbitrage.paper_engine_v7 import (
    V7Config,
    V7Position,
    V7Portfolio,
    RunLoggerV7,
    _settlement_metrics,
    _early_unwind_quote,
    _refresh_poly_map,
)


@dataclass
class V8Config(V7Config):
    # Long-run research defaults.
    run_minutes: float = 480.0
    max_watch_candidates: int = 100
    # V12 does not discard a promising proposal merely because earlier
    # candidates took a few seconds to process: every execution path fetches
    # fresh books again immediately before the simulated fill. The short
    # ranking window reduces avoidable ageing; a much larger hard ceiling only
    # prevents recycling genuinely obsolete proposals.
    allocation_window_seconds: float = 0.25
    max_revalidation_age_seconds: float = 45.0

    # Allocation: prioritize capital velocity instead of raw locked dollars.
    capital_velocity_weight: float = 2.50
    velocity_time_exponent: float = 0.65
    max_velocity_term: float = 0.75
    roc_weight: float = 0.70
    profit_weight: float = 0.12
    depth_weight: float = 0.08
    confirmation_weight: float = 0.12
    annualized_score_weight: float = 0.20
    lock_penalty_weight: float = 0.08

    # Dynamic hold hurdle.  The minimum acceptable simple annualized return is
    # base + premium * sqrt(days / 365), capped to avoid pathological values.
    base_required_hold_apr: float = 0.025
    lock_horizon_hurdle_apr: float = 0.030
    maximum_required_hold_apr: float = 0.250
    # The duration hurdle is primarily a ranking target. Hard-rejecting every
    # sub-hurdle trade caused an all-cash portfolio when only long-dated edges
    # were available. Keep a small ultra-long sleeve instead of blocking them all.
    hard_duration_hurdle: bool = False

    # V8.4: sub-hurdle settlement trades are research/demo positions only.
    # They can no longer absorb the 91-365d and >365d sleeves merely because
    # short-term arbitrage is absent. The small sleeve keeps execution data
    # flowing while protecting capital for higher-velocity opportunities.
    subhurdle_total_fraction: float = 0.02
    # V12 strict-yield sleeve: verified STRICT_ARB trades can still be worth
    # taking below the dynamic opportunity hurdle when cash would otherwise sit
    # idle. This sleeve is bounded separately so low-APR locks cannot crowd out
    # future high-velocity opportunities.
    strict_yield_sleeve_fraction: float = 0.30
    strict_yield_min_apr: float = 0.035
    strict_yield_max_days: float = 270.0

    # Duration diversification.  These are caps, not targets.  Shorter capital
    # may use the rest of the portfolio if attractive opportunities exist.
    short_bucket_fraction: float = 0.70       # <= 30 days
    medium_bucket_fraction: float = 0.50      # 31-90 days
    long_bucket_fraction: float = 0.30        # 91-365 days
    ultra_long_bucket_fraction: float = 0.05  # > 365 days; deliberately tiny sleeve

    # A larger bankroll needs concentration limits that still allow useful
    # sizing without letting one election consume the portfolio.
    max_market_fraction: float = 0.12
    max_event_fraction: float = 0.25
    reserve_cash_fraction: float = 0.30

    # Avoid tying up meaningful capital for trivial locked dollars.
    min_trade_net_dollars: float = 0.05

    # Long-run discovery refresh.  A new full scan can add newly emerged
    # opportunities instead of freezing the watch universe at startup.
    universe_refresh_minutes: float = 60.0
    max_watch_universe: int = 100

    # Long locks can be unwound after capturing a smaller fraction of their
    # eventual profit because released capital has greater option value.
    early_unwind_capture_fraction: float = 0.75
    minimum_unwind_capture_fraction: float = 0.55
    long_lock_unwind_discount: float = 0.18
    early_unwind_min_profit: float = 0.02
    opportunity_cost_apr: float = 0.055

    # Counterfactual analysis only.
    capacity_bankrolls: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0)
    auto_figures: bool = True
    # Research target only; not a guarantee or forecast. Used for GitHub benchmarking.
    benchmark_target_apr: float = 0.055

    # V8.8 defensive equivalence controls. Extremely large apparent arbitrage
    # is disproportionately likely to be a semantic mismatch, stale market, or
    # resolution-rule discrepancy. Such trades need a complete strict-match
    # certificate and are absolutely quarantined above the hard edge cap.
    extreme_edge_review_npc: float = 0.15
    extreme_roc_review: float = 0.30
    absolute_edge_quarantine_npc: float = 0.35
    require_structured_certificate_for_extreme: bool = True
    # HIGH_CONFIDENCE pairs can trade normal-sized edges, but spectacular
    # apparent arbitrage must be EXACT. This protects recall without bringing
    # back the V8.6 semantic-leak failure mode.
    high_confidence_max_npc: float = 0.10
    high_confidence_max_roc: float = 0.20

    # Fast-recycle lane: executable complete-set arbitrage on Polymarket can be
    # merged back to collateral instead of waiting for resolution.
    fast_recycle_enabled: bool = True
    fast_recycle_scan_seconds: float = 30.0
    fast_recycle_market_limit: int = 3000
    fast_recycle_max_trade_fraction: float = 0.05

    # Short-duration exhaustive multi-outcome baskets on Polymarket. These are
    # settlement trades (not instant merges), so they use the normal portfolio
    # ledger but are deliberately restricted to near-dated events.
    multi_outcome_enabled: bool = True
    multi_outcome_scan_seconds: float = 60.0
    multi_outcome_max_days: float = 45.0
    multi_outcome_max_events: int = 100
    multi_outcome_event_fraction: float = 0.10
    multi_outcome_min_roc: float = 0.0025

    # Strict Limitless × Polymarket short-horizon lane. Exact threshold,
    # deadline and explicit oracle/source agreement are required. Limitless
    # taker fees use the published maximum buy rate for conservative paper P&L.
    limitless_cross_enabled: bool = True
    limitless_cross_scan_seconds: float = 120.0
    limitless_cross_max_days: float = 14.0
    limitless_cross_trade_fraction: float = 0.08

    # V12.0 Final retains V10 economic depth sizing: quantity selection is not "largest affordable size".
    # Each candidate is re-priced across feasible quantities and the engine
    # chooses the quantity with the largest dollar surplus above the dynamic
    # duration hurdle. This prevents deep, low-edge levels from diluting APR.
    optimized_quantity_sizing: bool = True
    max_sizing_quantity: int = 500
    sizing_profit_tiebreak_weight: float = 0.10

    # V12 graded-resolution basis lane. Strict arbitrage remains the primary
    # ledger. A separately labelled LOW_BASIS pair may trade only when the
    # matcher found no fundamental payoff conflict and attached an explicit
    # per-contract risk reserve. This lane is deliberately tiny and its reserve
    # is subtracted from paper P&L before ranking or booking.
    low_basis_enabled: bool = True
    low_basis_total_fraction: float = 0.04
    low_basis_market_fraction: float = 0.015
    low_basis_event_fraction: float = 0.025
    low_basis_max_reserve_per_contract: float = 0.06
    low_basis_min_adjusted_npc: float = 0.005
    low_basis_min_adjusted_apr: float = 0.055

    # Strict paper maker-taker lane. No maker fill is counted from a static
    # quote: a subsequent public Kalshi trade must consume/print through the
    # hypothetical resting order, after which the Polymarket hedge is repriced.
    maker_probe_enabled: bool = True
    maker_probe_max_active: int = 10
    maker_probe_max_market_fraction: float = 0.06
    maker_probe_total_fraction: float = 0.15
    maker_probe_ttl_seconds: float = 300.0
    maker_probe_min_expected_npc: float = 0.0075
    maker_probe_min_expected_apr: float = 0.055
    maker_probe_max_queue_ratio: float = 2.0
    maker_probe_hedge_reserve_per_contract: float = 0.04
    maker_probe_poll_seconds: float = 10.0



@dataclass
class V8Portfolio(V7Portfolio):
    bucket_capital: dict[str, float] = field(default_factory=dict)
    subhurdle_capital: float = 0.0
    maker_probe_reserved: float = 0.0
    maker_market_reserved: dict[str, float] = field(default_factory=dict)
    maker_event_reserved: dict[str, float] = field(default_factory=dict)
    maker_bucket_reserved: dict[str, float] = field(default_factory=dict)

    @property
    def deployable_cash(self) -> float:
        # Resting maker orders reserve collateral in a real account. Paper
        # probes therefore reduce deployable cash even before they fill.
        return max(0.0, self.available_cash - self.reserve_cash - self.maker_probe_reserved)

    def reserve_maker_probe(self, amount: float, *, ticker: str | None = None, event_key: str | None = None, bucket: str | None = None) -> bool:
        amount = max(0.0, float(amount))
        if amount > self.deployable_cash + 1e-9:
            return False
        self.maker_probe_reserved += amount
        if ticker:
            self.maker_market_reserved[ticker] = self.maker_market_reserved.get(ticker, 0.0) + amount
        if event_key:
            self.maker_event_reserved[event_key] = self.maker_event_reserved.get(event_key, 0.0) + amount
        if bucket:
            self.maker_bucket_reserved[bucket] = self.maker_bucket_reserved.get(bucket, 0.0) + amount
        return True

    def release_maker_probe(self, amount: float, *, ticker: str | None = None, event_key: str | None = None, bucket: str | None = None) -> None:
        amount = max(0.0, float(amount))
        self.maker_probe_reserved = max(0.0, self.maker_probe_reserved - amount)
        if ticker:
            self.maker_market_reserved[ticker] = max(0.0, self.maker_market_reserved.get(ticker, 0.0) - amount)
        if event_key:
            self.maker_event_reserved[event_key] = max(0.0, self.maker_event_reserved.get(event_key, 0.0) - amount)
        if bucket:
            self.maker_bucket_reserved[bucket] = max(0.0, self.maker_bucket_reserved.get(bucket, 0.0) - amount)

    def max_capital_for_v8(self, ticker: str, event_key: str, settlement_days: float, cfg: V8Config, *, subhurdle: bool = False) -> float:
        bucket = _duration_bucket(settlement_days)
        market_left = self.starting_bankroll * self.max_market_fraction - self.market_capital.get(ticker, 0.0) - self.maker_market_reserved.get(ticker, 0.0)
        event_left = self.starting_bankroll * self.max_event_fraction - self.event_capital.get(event_key, 0.0) - self.maker_event_reserved.get(event_key, 0.0)
        cash_left = self.deployable_cash
        base = max(0.0, min(cash_left, market_left, event_left))
        bucket_limit = self.starting_bankroll * _bucket_fraction(bucket, cfg)
        bucket_left = bucket_limit - self.bucket_capital.get(bucket, 0.0) - self.maker_bucket_reserved.get(bucket, 0.0)
        cap = max(0.0, min(base, bucket_left))
        if subhurdle:
            research_left = self.starting_bankroll * cfg.subhurdle_total_fraction - self.subhurdle_capital
            cap = max(0.0, min(cap, research_left))
        return cap

    def book(self, position: V7Position) -> None:
        super().book(position)
        bucket = _duration_bucket(position.settlement_days_at_entry)
        self.bucket_capital[bucket] = self.bucket_capital.get(bucket, 0.0) + position.locked_capital
        if bool(getattr(position, "subhurdle", False)):
            self.subhurdle_capital += position.locked_capital

    def close(self, position: V7Position, *, proceeds: float, pnl: float, reason: str, when: float | None = None) -> None:
        bucket = _duration_bucket(position.settlement_days_at_entry)
        super().close(position, proceeds=proceeds, pnl=pnl, reason=reason, when=when)
        self.bucket_capital[bucket] = max(0.0, self.bucket_capital.get(bucket, 0.0) - position.locked_capital)
        if bool(getattr(position, "subhurdle", False)):
            self.subhurdle_capital = max(0.0, self.subhurdle_capital - position.locked_capital)


class RunLoggerV8(RunLoggerV7):
    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.paths.update({
            "policy": self.root / "allocator_policy_analysis.csv",
            "duration": self.root / "duration_buckets.csv",
            "fast_recycle": self.root / "fast_recycle_trades.csv",
            "venues": self.root / "venue_audit.csv",
            "multi_outcome": self.root / "multi_outcome_baskets.csv",
            "limitless_cross": self.root / "limitless_cross_trades.csv",
            "short_diag": self.root / "short_lane_diagnostics.csv",
            "maker_probes": self.root / "maker_probe_events.csv",
            "maker_trades": self.root / "maker_probe_trades.csv",
        })


def _duration_bucket(days: float) -> str:
    days = float(days)
    if days <= 30:
        return "0-30d"
    if days <= 90:
        return "31-90d"
    if days <= 365:
        return "91-365d"
    return ">365d"


def _bucket_fraction(bucket: str, cfg: V8Config) -> float:
    return {
        "0-30d": cfg.short_bucket_fraction,
        "31-90d": cfg.medium_bucket_fraction,
        "91-365d": cfg.long_bucket_fraction,
        ">365d": cfg.ultra_long_bucket_fraction,
    }[bucket]


def _required_hold_apr(days: float, cfg: V8Config) -> float:
    horizon = max(float(days), 1.0 / 24.0)
    required = cfg.base_required_hold_apr + cfg.lock_horizon_hurdle_apr * math.sqrt(horizon / 365.0)
    return min(cfg.maximum_required_hold_apr, max(cfg.minimum_hold_apr, required))


def _cross_settlement_metrics(candidate: dict, pm: dict, capital: float, hold_profit: float, cfg: V8Config, now: float | None = None):
    """Conservative cross-venue capital-release horizon.

    Principal is treated as unavailable until the latest *known possible*
    settlement clock across both venues. V11 uses rule-parser deadlines and
    Kalshi ``latest_expiration_time``/settlement timers in addition to the
    ordinary expected/end timestamps. This intentionally lowers annualized
    returns when one venue can legally settle much later.
    """
    now = float(now or time.time())
    poly_ts, _, _ = _settlement_metrics(pm, capital, hold_profit, cfg, now=now)
    timestamps = []
    if poly_ts:
        timestamps.append(float(poly_ts))

    for key in ("kalshi_signature", "polymarket_signature"):
        sig = candidate.get(key) or {}
        for field in ("latest_settlement_ts", "end_ts", "resolution_rule_deadline_ts"):
            raw = sig.get(field)
            try:
                ts = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ts) and ts > 0:
                timestamps.append(ts)

    cert = candidate.get("equivalence_certificate") or {}
    for field in (
        "latest_cross_settlement_ts",
        "kalshi_latest_settlement_ts",
        "polymarket_latest_settlement_ts",
    ):
        try:
            ts = float(cert.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(ts) and ts > 0:
            timestamps.append(ts)

    settlement_ts = max(timestamps) if timestamps else now + cfg.unknown_settlement_days * 86400.0
    days = max(1.0 / 24.0, (settlement_ts - now) / 86400.0)
    roi = float(hold_profit) / max(float(capital), 1e-9)
    annualized = roi * 365.0 / days
    return settlement_ts, days, annualized

def _resolution_lane(candidate: dict) -> str:
    cert = candidate.get("equivalence_certificate") or {}
    lane = str(cert.get("resolution_lane") or "STRICT_ARB").upper()
    return "LOW_BASIS" if lane == "LOW_BASIS" else "STRICT_ARB"


def _basis_reserve_per_contract(candidate: dict, cfg: V8Config) -> float:
    if _resolution_lane(candidate) != "LOW_BASIS":
        return 0.0
    cert = candidate.get("equivalence_certificate") or {}
    try:
        reserve = float(cert.get("basis_risk_reserve_per_contract") or 0.0)
    except (TypeError, ValueError):
        reserve = 0.0
    return max(0.0, min(float(cfg.low_basis_max_reserve_per_contract), reserve))


def _basis_cap_limit(portfolio: V8Portfolio, candidate: dict, event_key: str, cfg: V8Config) -> float:
    if _resolution_lane(candidate) != "LOW_BASIS":
        return float("inf")
    if not cfg.low_basis_enabled:
        return 0.0
    total = sum(p.locked_capital for p in portfolio.positions if getattr(p, "resolution_lane", "STRICT_ARB") == "LOW_BASIS")
    market = sum(p.locked_capital for p in portfolio.positions if getattr(p, "resolution_lane", "STRICT_ARB") == "LOW_BASIS" and p.ticker == candidate.get("ticker"))
    event = sum(p.locked_capital for p in portfolio.positions if getattr(p, "resolution_lane", "STRICT_ARB") == "LOW_BASIS" and p.event_key == event_key)
    return max(0.0, min(
        portfolio.starting_bankroll * cfg.low_basis_total_fraction - total,
        portfolio.starting_bankroll * cfg.low_basis_market_fraction - market,
        portfolio.starting_bankroll * cfg.low_basis_event_fraction - event,
    ))


def _strict_yield_cap_limit(portfolio: V8Portfolio, candidate: dict, cfg: V8Config) -> float:
    if _resolution_lane(candidate) != "STRICT_ARB":
        return 0.0
    used = sum(
        p.locked_capital for p in portfolio.positions
        if bool(getattr(p, "strict_yield_sleeve", False))
    )
    return max(0.0, portfolio.starting_bankroll * cfg.strict_yield_sleeve_fraction - used)


def _basis_adjusted_signal(signal: dict, candidate: dict, cfg: V8Config) -> dict:
    row = dict(signal)
    qty = max(0, int(row.get("quantity") or 0))
    reserve_npc = _basis_reserve_per_contract(candidate, cfg)
    if reserve_npc <= 0 or qty <= 0:
        row.setdefault("gross_net_profit", float(row.get("net_profit") or 0.0))
        row.setdefault("gross_net_per_contract", float(row.get("net_per_contract") or 0.0))
        row["basis_risk_reserve"] = 0.0
        row["basis_risk_reserve_per_contract"] = 0.0
        return row
    gross = float(row.get("net_profit") or 0.0)
    gross_npc = float(row.get("net_per_contract") or 0.0)
    reserve = reserve_npc * qty
    row["gross_net_profit"] = gross
    row["gross_net_per_contract"] = gross_npc
    row["basis_risk_reserve"] = reserve
    row["basis_risk_reserve_per_contract"] = reserve_npc
    row["net_profit"] = gross - reserve
    row["net_per_contract"] = gross_npc - reserve_npc
    return row


def _economic_surplus(signal: dict, candidate: dict, pm: dict, cfg: V8Config) -> tuple[float, float, float]:
    capital = max(float(signal.get("capital") or 0.0), 1e-9)
    profit = float(signal.get("net_profit") or 0.0)
    _, days, annualized = _cross_settlement_metrics(candidate, pm, capital, profit, cfg)
    required_apr = _required_hold_apr(days, cfg)
    hurdle_dollars = capital * required_apr * days / 365.0
    return profit - hurdle_dollars, annualized, required_apr


def _select_economic_signal(signals: list[dict], candidate: dict, pm: dict, cfg: V8Config) -> dict | None:
    """Choose depth by economic surplus, not by largest affordable quantity.

    Positive-surplus sizes outrank sub-hurdle sizes.  If no size clears the
    duration hurdle, return the best annualized research size so the tightly
    capped sub-hurdle sleeve can still collect evidence.
    """
    if not signals:
        return None
    enriched = []
    for sig in signals:
        row = _basis_adjusted_signal(sig, candidate, cfg)
        required_edge = max(
            cfg.minimum_signal_net_per_contract,
            cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract,
        )
        if _resolution_lane(candidate) == "LOW_BASIS":
            required_edge = max(required_edge, cfg.low_basis_min_adjusted_npc)
        if float(row.get("net_per_contract") or 0.0) < required_edge or float(row.get("net_profit") or 0.0) < cfg.min_trade_net_dollars:
            continue
        surplus, annualized, required_apr = _economic_surplus(row, candidate, pm, cfg)
        row["sizing_economic_surplus"] = surplus
        row["sizing_annualized"] = annualized
        row["sizing_required_apr"] = required_apr
        enriched.append(row)
    above = [x for x in enriched if x["sizing_economic_surplus"] > 0]
    if above:
        return max(above, key=lambda x: (
            x["sizing_economic_surplus"] + cfg.sizing_profit_tiebreak_weight * max(0.0, float(x["net_profit"])),
            x["sizing_annualized"],
            x["net_profit"],
        ))
    return max(enriched, key=lambda x: (x["sizing_annualized"], x["net_per_contract"], x["net_profit"]))


def _optimized_fresh_signal(candidate: dict, pm: dict, series_info: dict, *, max_qty: int, capital_limit: float, cfg: V8Config):
    kalshi_side, poly_side = _strategy_sides(candidate["strategy"])
    tokens = parse_token_ids(pm)
    if len(tokens) != 2:
        return None, "invalid Polymarket token ids"
    poly_token = tokens[0] if poly_side == "yes" else tokens[1]

    kasks = _kalshi_taker_asks(candidate["ticker"], kalshi_side)
    kt = time.monotonic()
    pasks = _poly_asks(poly_token)
    pt = time.monotonic()
    skew = pt - kt
    if skew > cfg.max_quote_skew_seconds:
        return None, f"quote skew {skew:.3f}s above {cfg.max_quote_skew_seconds:.3f}s"
    if not kasks or not pasks:
        return None, "empty executable book"

    upper = max(0, min(int(max_qty), int(cfg.max_sizing_quantity)))
    signals = []
    # Exact integer evaluation is deliberate: fee rounding and depth steps can
    # make the economically best size sit between conventional size buckets.
    for qty in range(1, upper + 1):
        sig = _signal_from_levels(candidate, pm, series_info, qty, kasks, pasks, skew)
        if sig is None:
            break
        if float(sig["capital"]) <= float(capital_limit) + 1e-9:
            required_edge = max(
                cfg.minimum_signal_net_per_contract,
                cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract,
            )
            if float(sig["net_per_contract"]) >= required_edge and float(sig["net_profit"]) >= cfg.min_trade_net_dollars:
                signals.append(sig)
        elif qty == 1:
            break
    chosen = _select_economic_signal(signals, candidate, pm, cfg)
    if chosen is None:
        return None, "no positive integer quantity fits depth/capital"
    return chosen, "ok"


def _safe_optimized_signal(candidate, pm, series_info, *, max_qty, capital_limit, cfg: V8Config):
    if not cfg.optimized_quantity_sizing:
        return _safe_signal(candidate, pm, series_info, max_qty=max_qty, capital_limit=capital_limit, cfg=cfg)
    last = None
    for attempt in range(cfg.api_retries + 1):
        try:
            return _optimized_fresh_signal(candidate, pm, series_info, max_qty=max_qty, capital_limit=capital_limit, cfg=cfg)
        except Exception as exc:
            last = exc
            if attempt < cfg.api_retries:
                time.sleep(cfg.api_retry_backoff_seconds * (2 ** attempt))
    return None, f"API failure after retries: {type(last).__name__}: {last}"


@dataclass
class MakerProbe:
    probe_id: int
    candidate: dict
    pm: dict
    series_info: dict
    maker_side: str
    poly_side: str
    maker_price: float
    quantity: int
    queue_ahead: float
    created_ts: float
    reserved_capital: float
    expected_net: float
    expected_apr: float
    event_key: str = "unknown"
    duration_bucket: str = ">365d"
    cumulative_at_price: float = 0.0
    seen_trade_ids: set[str] = field(default_factory=set)
    status: str = "ACTIVE"
    last_poll_ts: float = 0.0


def _trade_created_ts(trade: dict) -> float | None:
    raw = trade.get("created_time")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _maker_trade_progress(probe: MakerProbe, trade: dict) -> tuple[bool, str]:
    """Update one paper maker probe from a subsequent public Kalshi print.

    A YES bid is consumed by a NO-side taker; a NO bid by a YES-side taker.
    A print through our price proves the resting order would have been reached.
    At the same price, displayed queue plus our own quantity must trade first.
    """
    if bool(trade.get("is_block_trade")):
        return False, "block trade ignored"
    tid = str(trade.get("trade_id") or "")
    if tid and tid in probe.seen_trade_ids:
        return False, "duplicate trade"
    tts = _trade_created_ts(trade)
    if tts is not None and tts + 1e-9 < probe.created_ts:
        return False, "pre-probe trade"
    if tid:
        probe.seen_trade_ids.add(tid)

    taker = str(trade.get("taker_outcome_side") or trade.get("taker_side") or "").lower()
    required_taker = "no" if probe.maker_side == "yes" else "yes"
    if taker != required_taker:
        return False, "wrong taker outcome side"
    book_side = str(trade.get("taker_book_side") or "").lower()
    required_book_side = "ask" if required_taker == "no" else "bid"
    if book_side and book_side != required_book_side:
        return False, "taker book side inconsistent with required aggressor"
    # Public Trade prices are quoted on Kalshi's canonical YES scale. A NO-side
    # maker bid at q is therefore a resting YES ask at 1-q.
    try:
        price = float(trade.get("yes_price_dollars"))
        qty = float(trade.get("count_fp") or 0.0)
    except (TypeError, ValueError):
        return False, "invalid trade payload"
    if qty <= 0:
        return False, "zero trade size"

    canonical_price = probe.maker_price if probe.maker_side == "yes" else 1.0 - probe.maker_price
    eps = 0.00005
    # YES maker = resting YES bid: a lower print proves our better bid was
    # consumed first. NO maker = resting YES ask: a higher print proves our
    # better (lower) ask was consumed first.
    through = (price < canonical_price - eps) if probe.maker_side == "yes" else (price > canonical_price + eps)
    if through:
        return True, "trade-through fill"
    if abs(price - canonical_price) <= eps:
        probe.cumulative_at_price += qty
        if probe.cumulative_at_price + 1e-9 >= probe.queue_ahead + probe.quantity:
            return True, "queue-cleared fill"
        return False, "queue progressing"
    return False, "trade did not reach maker price"


def _maker_candidate_ok(candidate: dict, pm: dict, cfg: V8Config) -> tuple[bool, str, float, float]:
    cert = candidate.get("equivalence_certificate") or {}
    if _resolution_lane(candidate) == "LOW_BASIS":
        return False, "maker probes restricted to strict arbitrage", 0.0, 0.0
    if not bool(cert.get("structured_complete")):
        return False, "incomplete payoff certificate", 0.0, 0.0
    fake = {
        "net_per_contract": float(candidate.get("net_per_contract") or 0.0),
        "net_profit": float(candidate.get("net_profit") or 0.0),
        "capital": float(candidate.get("capital") or 0.0),
    }
    ok, reason = _equivalence_edge_gate(candidate, fake, cfg)
    if not ok:
        return False, reason, 0.0, 0.0
    cap = max(float(candidate.get("capital") or 0.0), 1e-9)
    net = float(candidate.get("net_profit") or 0.0)
    _, days, apr = _cross_settlement_metrics(candidate, pm, cap, net, cfg)
    required_apr = _required_hold_apr(days, cfg)
    if float(candidate.get("net_per_contract") or 0.0) < cfg.maker_probe_min_expected_npc:
        return False, "maker expected edge below floor", apr, net
    if apr < max(cfg.maker_probe_min_expected_apr, required_apr):
        return False, "maker expected APR below final hurdle", apr, net
    return True, "ok", apr, net


def _build_maker_probes(passive: list[dict], by_question: dict, series_cache: dict, portfolio: V8Portfolio, cfg: V8Config, log: RunLoggerV8):
    if not cfg.maker_probe_enabled:
        return []
    probes: list[MakerProbe] = []
    total_limit = portfolio.starting_bankroll * cfg.maker_probe_total_fraction
    market_reserved: dict[str, float] = {}
    event_reserved: dict[str, float] = {}
    bucket_reserved: dict[str, float] = {}
    seen_keys: set[tuple[str, str]] = set()
    for candidate in sorted(passive, key=lambda x: float(x.get("execution_score") or 0.0), reverse=True):
        if len(probes) >= cfg.maker_probe_max_active or portfolio.maker_probe_reserved >= total_limit - 1e-9:
            break
        key = (str(candidate.get("ticker")), str(candidate.get("strategy")))
        if key in seen_keys:
            continue
        pm = by_question.get(candidate.get("poly_question"))
        if pm is None:
            continue
        ok, reason, expected_apr, expected_net = _maker_candidate_ok(candidate, pm, cfg)
        if not ok:
            continue
        qty0 = max(1, int(candidate.get("quantity") or 1))
        queue = max(0.0, float(candidate.get("queue_ahead") or 0.0))
        if queue / qty0 > cfg.maker_probe_max_queue_ratio:
            continue
        series = _series_ticker(candidate["ticker"])
        if series not in series_cache:
            try:
                series_cache[series] = get_series_info(series)
            except Exception:
                continue
        unit_cap = float(candidate.get("capital") or 0.0) / qty0
        reserve_unit = unit_cap + cfg.maker_probe_hedge_reserve_per_contract
        if reserve_unit <= 0:
            continue
        event_key = _event_key(candidate)
        _, days, _ = _cross_settlement_metrics(candidate, pm, max(float(candidate.get("capital") or 1.0), 1e-9), expected_net, cfg)
        bucket = _duration_bucket(days)
        per_probe_left = portfolio.starting_bankroll * cfg.maker_probe_max_market_fraction
        market_left = portfolio.starting_bankroll * cfg.max_market_fraction - market_reserved.get(key[0], 0.0)
        event_left = portfolio.starting_bankroll * cfg.max_event_fraction - event_reserved.get(event_key, 0.0)
        bucket_left = portfolio.starting_bankroll * _bucket_fraction(bucket, cfg) - bucket_reserved.get(bucket, 0.0)
        remaining_total = total_limit - portfolio.maker_probe_reserved
        max_reserved = max(0.0, min(
            per_probe_left, market_left, event_left, bucket_left, remaining_total, portfolio.deployable_cash
        ))
        qty = min(qty0, int(max_reserved // reserve_unit))
        if qty < 1:
            continue
        scale = qty / qty0
        reserved = qty * reserve_unit
        if not portfolio.reserve_maker_probe(reserved, ticker=key[0], event_key=event_key, bucket=bucket):
            continue
        maker_side, poly_side = _strategy_sides(candidate["strategy"])
        probe = MakerProbe(
            probe_id=len(probes) + 1, candidate=candidate, pm=pm, series_info=series_cache[series],
            maker_side=maker_side, poly_side=poly_side, maker_price=float(candidate["kalshi_price"]),
            quantity=qty, queue_ahead=queue, created_ts=time.time(), reserved_capital=reserved,
            expected_net=expected_net * scale, expected_apr=expected_apr, event_key=event_key, duration_bucket=bucket,
        )
        probes.append(probe); seen_keys.add(key)
        market_reserved[key[0]] = market_reserved.get(key[0], 0.0) + reserved
        event_reserved[event_key] = event_reserved.get(event_key, 0.0) + reserved
        bucket_reserved[bucket] = bucket_reserved.get(bucket, 0.0) + reserved
        log.append("maker_probes", {
            "timestamp": probe.created_ts, "probe_id": probe.probe_id, "action": "OPEN",
            "ticker": candidate.get("ticker"), "subject": candidate.get("subject"),
            "strategy": candidate.get("strategy"), "quantity": qty, "maker_price": probe.maker_price,
            "queue_ahead": queue, "reserved_capital": reserved, "expected_net": probe.expected_net,
            "expected_apr": expected_apr, "settlement_days": days, "duration_bucket": bucket,
            "reason": "strict public-trade fill watch",
        })
    return probes


def _maker_poly_hedge(probe: MakerProbe):
    tokens = parse_token_ids(probe.pm)
    if len(tokens) != 2:
        return None, "invalid Polymarket token ids"
    token = tokens[0] if probe.poly_side == "yes" else tokens[1]
    asks = _poly_asks(token)
    fill = consume_asks(asks, probe.quantity)
    if not fill.fully_filled:
        return None, "insufficient Polymarket hedge depth"
    kfee_obj = kalshi_fee(
        price=probe.maker_price, contracts=probe.quantity, fee_type=probe.series_info.get("fee_type"),
        fee_multiplier=probe.series_info.get("fee_multiplier") or 0, maker=True,
    )
    if kfee_obj is None:
        return None, "unknown Kalshi maker fee schedule"
    kfee = float(kfee_obj["cash_fee_upper"])
    pfee = float(polymarket_taker_fee(fill.average_price, probe.quantity, probe.pm))
    maker_cost = probe.quantity * probe.maker_price + kfee
    capital = maker_cost + fill.cost + pfee
    net = probe.quantity - capital
    return {
        "capital": capital, "net_profit": net, "net_per_contract": net / probe.quantity,
        "kalshi_fee": kfee, "poly_fee": pfee, "poly_avg": fill.average_price,
        "poly_worst": fill.worst_price, "maker_cost": maker_cost,
    }, "ok"


def _conservative_maker_leg_loss(probe: MakerProbe) -> float:
    kfee_obj = kalshi_fee(
        price=probe.maker_price, contracts=probe.quantity, fee_type=probe.series_info.get("fee_type"),
        fee_multiplier=probe.series_info.get("fee_multiplier") or 0, maker=True,
    )
    kfee = float(kfee_obj["cash_fee_upper"]) if kfee_obj is not None else 0.0
    return probe.quantity * probe.maker_price + kfee


def _fetch_probe_trades(probe: MakerProbe, max_pages: int = 5):
    trades = []
    cursor = None
    for _ in range(max_pages):
        page, cursor = get_market_trades(
            probe.candidate["ticker"], min_ts=max(0, int(probe.created_ts) - 1),
            limit=1000, cursor=cursor, is_block_trade=False,
        )
        trades.extend(page)
        if not cursor:
            return trades, False
    return trades, bool(cursor)


def _process_maker_probes(probes: list[MakerProbe], portfolio: V8Portfolio, cfg: V8Config, log: RunLoggerV8,
                          rejection_counts: Counter, trade_id: int, results: list[dict]):
    now = time.time()
    for probe in probes:
        if probe.status != "ACTIVE":
            continue
        if probe.last_poll_ts and now - probe.last_poll_ts < cfg.maker_probe_poll_seconds:
            continue
        probe.last_poll_ts = now
        try:
            trades, overflow = _fetch_probe_trades(probe)
        except Exception as exc:
            rejection_counts[f"maker trade-feed error: {type(exc).__name__}"] += 1
            continue
        if overflow:
            # Missing prints make queue inference unsafe. Keep the probe active
            # but never manufacture a fill from an incomplete trade window.
            rejection_counts["maker trade-feed pagination overflow"] += 1
            continue
        trades.sort(key=lambda t: (_trade_created_ts(t) or 0.0, str(t.get("trade_id") or "")))
        filled = False
        fill_reason = ""
        for trade in trades:
            did_fill, why = _maker_trade_progress(probe, trade)
            if did_fill:
                filled = True
                fill_reason = why
                break

        if filled:
            portfolio.release_maker_probe(probe.reserved_capital, ticker=str(probe.candidate.get("ticker")), event_key=probe.event_key, bucket=probe.duration_bucket)
            probe.status = "FILLED"
            try:
                hedge, hedge_reason = _maker_poly_hedge(probe)
            except Exception as exc:
                hedge, hedge_reason = None, f"hedge API error: {type(exc).__name__}: {exc}"
            if hedge is None or float(hedge["capital"]) > portfolio.available_cash + 1e-9:
                loss = min(portfolio.available_cash, _conservative_maker_leg_loss(probe))
                portfolio.available_cash -= loss
                portfolio.realized_pnl -= loss
                portfolio.residual_contracts += float(probe.quantity)
                portfolio._update_drawdown()
                row = {
                    "timestamp": time.time(), "probe_id": probe.probe_id, "ticker": probe.candidate.get("ticker"),
                    "subject": probe.candidate.get("subject"), "strategy": probe.candidate.get("strategy"),
                    "status": "MAKER_FILLED_HEDGE_FAILED", "fill_reason": fill_reason,
                    "hedge_reason": hedge_reason if hedge is None else "insufficient reserved/available cash after hedge move",
                    "quantity": probe.quantity, "maker_price": probe.maker_price, "conservative_loss": -loss,
                    "residual_unhedged": probe.quantity, "portfolio_equity": portfolio.equity,
                }
                log.append("maker_trades", row)
                print(f"MAKER HEDGE FAILURE {probe.candidate.get('subject')} | qty={probe.quantity} | "
                      f"conservative loss=${loss:.4f} | residual={probe.quantity}")
                rejection_counts["maker fill hedge failure"] += 1
                continue

            trade_id += 1
            capital = float(hedge["capital"]); net = float(hedge["net_profit"])
            settlement_ts, days, annualized = _cross_settlement_metrics(probe.candidate, probe.pm, capital, net, cfg)
            required_apr = _required_hold_apr(days, cfg)
            event_key = _event_key(probe.candidate)
            pos = V7Position(
                trade_id=trade_id, entry_time=time.time(), ticker=str(probe.candidate["ticker"]), event_key=event_key,
                subject=str(probe.candidate.get("subject")), topic=str(probe.candidate.get("topic")),
                strategy=str(probe.candidate.get("strategy")), route="kalshi_maker_then_poly_taker",
                quantity=probe.quantity, locked_capital=capital, hold_profit=net, residual_unhedged=0.0,
                kalshi_side=probe.maker_side, poly_side=probe.poly_side, settlement_ts=settlement_ts,
                settlement_days_at_entry=days, annualized_hold_return=annualized,
            )
            setattr(pos, "subhurdle", annualized < required_apr)
            portfolio.book(pos)
            row = {
                "timestamp": time.time(), "trade_id": trade_id, "probe_id": probe.probe_id,
                "ticker": probe.candidate.get("ticker"), "subject": probe.candidate.get("subject"),
                "topic": probe.candidate.get("topic"), "strategy": probe.candidate.get("strategy"),
                "route": "kalshi_maker_then_poly_taker", "status": "SIMULATED_MAKER_FILL_HEDGED",
                "fill_reason": fill_reason, "quantity": probe.quantity, "maker_price": probe.maker_price,
                "poly_avg_price": hedge["poly_avg"], "kalshi_fee": hedge["kalshi_fee"], "poly_fee": hedge["poly_fee"],
                "locked_capital": capital, "hold_profit": net, "net_per_contract": hedge["net_per_contract"],
                "settlement_ts": settlement_ts, "settlement_days": days, "annualized_hold_return": annualized,
                "required_hold_apr": required_apr, "duration_bucket": _duration_bucket(days),
                "portfolio_cash": portfolio.available_cash, "portfolio_locked": portfolio.locked_capital,
                "portfolio_locked_profit": portfolio.locked_profit, "portfolio_realized_pnl": portfolio.realized_pnl,
                "portfolio_equity": portfolio.equity, "kalshi_title": probe.candidate.get("kalshi_title"),
                "poly_question": probe.candidate.get("poly_question"), "match_source": probe.candidate.get("match_source"),
                "equivalence_score": probe.candidate.get("equivalence_score"),
                "equivalence_reasons": probe.candidate.get("equivalence_reasons"),
                "resolution_rule_status": (probe.candidate.get("equivalence_certificate") or {}).get("resolution_rule_status"),
                "latest_cross_settlement_ts": (probe.candidate.get("equivalence_certificate") or {}).get("latest_cross_settlement_ts"),
                "equivalence_certificate": repr(probe.candidate.get("equivalence_certificate") or {}),
            }
            results.append(row)
            log.append("maker_trades", row); log.append("trades", row); log.append("positions", row)
            print(f"MAKER FILL {probe.candidate.get('subject')} | {fill_reason} | qty={probe.quantity} | "
                  f"locked=${capital:.2f} | hold=${net:.4f} | {days:.0f}d | APR={annualized:.2%}")
            continue

        if now - probe.created_ts >= cfg.maker_probe_ttl_seconds:
            portfolio.release_maker_probe(probe.reserved_capital, ticker=str(probe.candidate.get("ticker")), event_key=probe.event_key, bucket=probe.duration_bucket)
            probe.status = "EXPIRED"
            log.append("maker_probes", {
                "timestamp": now, "probe_id": probe.probe_id, "action": "EXPIRE",
                "ticker": probe.candidate.get("ticker"), "subject": probe.candidate.get("subject"),
                "strategy": probe.candidate.get("strategy"), "quantity": probe.quantity,
                "maker_price": probe.maker_price, "queue_ahead": probe.queue_ahead,
                "reserved_capital": probe.reserved_capital, "expected_net": probe.expected_net,
                "expected_apr": probe.expected_apr, "reason": "TTL elapsed without proven public-trade fill",
            })
    return trade_id


def _capital_velocity(roi: float, days: float, cfg: V8Config) -> float:
    days = max(float(days), 1.0 / 24.0)
    time_multiplier = (365.0 / days) ** cfg.velocity_time_exponent
    return float(roi) * time_multiplier


def _allocation_score_v8(candidate: dict, signal: dict, pm: dict, cfg: V8Config):
    capital = max(float(signal["capital"]), 1e-9)
    profit = max(0.0, float(signal["net_profit"]))
    roi = profit / capital
    q = max(float(signal["quantity"]), 1.0)
    coverage = min(5.0, min(float(signal["kalshi_total_size"]), float(signal["poly_total_size"])) / q)
    depth_term = math.log1p(max(0.0, coverage))
    _, days, annualized = _cross_settlement_metrics(candidate, pm, capital, profit, cfg)
    velocity = _capital_velocity(roi, days, cfg)
    velocity_term = min(cfg.max_velocity_term, max(0.0, velocity))
    annualized_term = min(1.0, max(0.0, annualized))
    # Log-profit prevents a $1,000 bankroll from making raw dollars overwhelm
    # capital efficiency in the ranking.
    profit_term = math.log1p(profit)
    lock_penalty = math.log1p(days / 30.0)
    score = (
        cfg.capital_velocity_weight * velocity_term
        + cfg.roc_weight * roi
        + cfg.profit_weight * profit_term
        + cfg.depth_weight * depth_term
        + cfg.confirmation_weight * _confirmation_ratio(candidate)
        + cfg.annualized_score_weight * annualized_term
        - cfg.lock_penalty_weight * lock_penalty
    )
    return score, days, annualized, velocity, _required_hold_apr(days, cfg), _duration_bucket(days)


def _dynamic_unwind_capture(position: V7Position, cfg: V8Config, now: float) -> float:
    remaining_days = max(0.0, (position.settlement_ts - now) / 86400.0)
    horizon_scale = min(1.0, math.sqrt(remaining_days / 365.0)) if remaining_days > 0 else 0.0
    required = cfg.early_unwind_capture_fraction - cfg.long_lock_unwind_discount * horizon_scale
    return max(cfg.minimum_unwind_capture_fraction, min(1.0, required))


def _should_unwind_v8(position: V7Position, quote: dict, cfg: V8Config, now: float):
    exit_pnl = float(quote["pnl"])
    if exit_pnl < cfg.early_unwind_min_profit:
        return False, "exit profit below minimum"
    required_capture = _dynamic_unwind_capture(position, cfg, now)
    capture = exit_pnl / max(position.hold_profit, 1e-9)
    if capture < required_capture:
        return False, f"exit capture {capture:.1%} below dynamic {required_capture:.1%}"
    remaining_days = max(0.0, (position.settlement_ts - now) / 86400.0)
    opportunity_value = position.locked_capital * cfg.opportunity_cost_apr * remaining_days / 365.0
    if exit_pnl + opportunity_value + 1e-9 < position.hold_profit:
        return False, "holding dominates after opportunity-cost adjustment"
    return True, "capital-velocity early unwind"


def _equivalence_edge_gate(candidate: dict, signal: dict, cfg: V8Config) -> tuple[bool, str]:
    """Second-line defense against semantic false positives.

    V8.8 permits HIGH_CONFIDENCE matches for ordinary edges, but requires an
    EXACT semantic tier for unusually large apparent arbitrage. Absurd edges
    remain quarantined regardless of tier.
    """
    npc = float(signal.get("net_per_contract") or 0.0)
    capital = float(signal.get("capital") or 0.0)
    net = float(signal.get("net_profit") or 0.0)
    roc = net / capital if capital > 0 else 0.0
    source = str(candidate.get("match_source") or "")
    tier = source.split(":", 1)[0] if ":" in source else "EXACT"

    # V9.0 defense-in-depth: re-check payoff-defining geography/chamber at
    # execution time so a future discovery/matcher regression cannot turn a
    # cross-state or House-vs-Senate pair into paper P&L.
    ks = candidate.get("kalshi_signature") or {}
    ps = candidate.get("polymarket_signature") or {}
    ko, po = ks.get("office_scope"), ps.get("office_scope")
    if ko or po:
        if not ko or not po:
            return False, "execution audit: office/chamber missing on one venue"
        if ko != po:
            return False, "execution audit: office/chamber mismatch"
    kr, pr = ks.get("jurisdiction_region"), ps.get("jurisdiction_region")
    if kr or pr:
        if not kr or not pr:
            return False, "execution audit: jurisdiction missing on one venue"
        if kr != pr:
            return False, "execution audit: jurisdiction mismatch"
    kd, pd = ks.get("jurisdiction_district"), ps.get("jurisdiction_district")
    if kd or pd:
        if not kd or not pd:
            return False, "execution audit: district missing on one venue"
        if kd != pd:
            return False, "execution audit: district mismatch"

    # V11 defense-in-depth: an accepted headline match is not enough. The
    # matcher must have certified that public settlement rules are either exact
    # or materially compatible. REVIEW/missing rule status never contributes
    # to paper P&L, including maker probes.
    cert = candidate.get("equivalence_certificate") or {}
    rule_status = str(cert.get("resolution_rule_status") or "").upper()
    if rule_status not in {"EXACT", "COMPATIBLE", "LOW_BASIS"}:
        return False, "execution audit: resolution-rule certificate missing or unsafe"
    if rule_status == "LOW_BASIS":
        if not cfg.low_basis_enabled:
            return False, "execution audit: low-basis lane disabled"
        reserve = _basis_reserve_per_contract(candidate, cfg)
        if reserve <= 0.0:
            return False, "execution audit: low-basis reserve missing"
        if reserve > cfg.low_basis_max_reserve_per_contract + 1e-12:
            return False, "execution audit: low-basis reserve above cap"
        # This lane never gets the privileges of an EXACT match for extreme
        # edges, regardless of headline similarity.
        if npc >= cfg.extreme_edge_review_npc or roc >= cfg.extreme_roc_review:
            return False, "low-basis extreme edge quarantined"

    # Rule-sensitive political contracts also need a known conservative cross-
    # venue settlement clock. Unknown maximum payout dates are review-only.
    domain = str(ks.get("domain") or ps.get("domain") or "").lower()
    office = ko or po
    if (domain == "politics" or office) and not cert.get("latest_cross_settlement_ts"):
        return False, "execution audit: latest cross-venue settlement horizon unknown"

    if npc >= cfg.absolute_edge_quarantine_npc:
        return False, "extreme edge absolute quarantine"

    if tier == "HIGH_CONFIDENCE" and (npc >= cfg.high_confidence_max_npc or roc >= cfg.high_confidence_max_roc):
        return False, "high-confidence edge too large; exact match required"

    if npc < cfg.extreme_edge_review_npc and roc < cfg.extreme_roc_review:
        return True, f"normal edge ({tier})"

    if tier != "EXACT":
        return False, "extreme edge requires EXACT equivalence tier"
    if cfg.require_structured_certificate_for_extreme and not bool(cert.get("structured_complete")):
        return False, "extreme edge lacks complete equivalence certificate"
    if float(candidate.get("equivalence_score") or 0.0) < 0.91:
        return False, "extreme edge equivalence score below 0.91"
    prop = str(cert.get("proposition") or "")
    if prop == "binary_event":
        return False, "extreme generic-binary edge quarantined"
    metric = cert.get("metric")
    if metric and prop != "leader" and cert.get("threshold_low") is None:
        return False, "extreme stat edge missing threshold certificate"
    return True, "extreme edge exact certificate passed"

def _signal_row_v8(candidate, signal, eligible, reason, *, score=None, settlement_days=None,
                   annualized_return=None, capital_velocity=None, required_apr=None, duration_bucket=None):
    return {
        "timestamp": time.time(),
        "ticker": candidate.get("ticker"),
        "subject": candidate.get("subject"),
        "topic": candidate.get("topic"),
        "strategy": candidate.get("strategy"),
        "eligible": eligible,
        "reason": reason,
        "allocation_score": score,
        "quantity": None if signal is None else signal.get("quantity"),
        "capital": None if signal is None else signal.get("capital"),
        "net_profit": None if signal is None else signal.get("net_profit"),
        "net_per_contract": None if signal is None else signal.get("net_per_contract"),
        "quote_skew_seconds": None if signal is None else signal.get("fetch_skew_seconds"),
        "kalshi_total_size": None if signal is None else signal.get("kalshi_total_size"),
        "poly_total_size": None if signal is None else signal.get("poly_total_size"),
        "settlement_days": settlement_days,
        "annualized_hold_return": annualized_return,
        "capital_velocity": capital_velocity,
        "required_hold_apr": required_apr,
        "duration_bucket": duration_bucket,
    }


def _equity_row_v8(portfolio: V8Portfolio, cycle: int, elapsed_minutes: float):
    row = {
        "timestamp": time.time(),
        "cycle": cycle,
        "elapsed_minutes": elapsed_minutes,
        "available_cash": portfolio.available_cash,
        "locked_capital": portfolio.locked_capital,
        "locked_profit": portfolio.locked_profit,
        "realized_pnl": portfolio.realized_pnl,
        "equity": portfolio.equity,
        "open_positions": len(portfolio.positions),
        "closed_positions": len(portfolio.closed_positions),
        "residual_contracts": portfolio.residual_contracts,
        "max_drawdown": portfolio.max_drawdown, "subhurdle_capital": portfolio.subhurdle_capital,
        "maker_probe_reserved": portfolio.maker_probe_reserved,
        "capital_utilization": portfolio.locked_capital / portfolio.starting_bankroll,
    }
    for bucket in ("0-30d", "31-90d", "91-365d", ">365d"):
        row[f"bucket_{bucket.replace('>', 'gt').replace('-', '_')}"] = portfolio.bucket_capital.get(bucket, 0.0)
    return row


def _merge_confirmed_watch(watch: list, engine_config: V2Config, confirmation_config: ConfirmationConfig,
                           cfg: V8Config, series_cache: dict):
    """Refresh discovery and merge newly confirmed candidates into the watch list."""
    print("Refreshing confirmed opportunity universe...")
    _, confirmations, _ = run_v3(engine_config, confirmation_config)
    confirmed = [x for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    confirmed.sort(
        key=lambda x: (float(x.get("confirmation_priority", -999.0)), float(x.get("worst_net_profit", -999.0))),
        reverse=True,
    )
    by_question = _refresh_poly_map()
    existing = {(x[0].get("ticker"), x[0].get("strategy")) for x in watch}
    added = 0
    for c in confirmed:
        if len(watch) >= cfg.max_watch_universe:
            break
        key = (c.get("ticker"), c.get("strategy"))
        if key in existing:
            # Refresh the candidate and Polymarket metadata in place.
            for item in watch:
                if (item[0].get("ticker"), item[0].get("strategy")) == key:
                    pm = by_question.get(c.get("poly_question"))
                    if pm is not None:
                        item[0] = c
                        item[1] = pm
                    break
            continue
        pm = by_question.get(c.get("poly_question"))
        if pm is None:
            continue
        series = _series_ticker(c["ticker"])
        if series not in series_cache:
            series_cache[series] = get_series_info(series)
        watch.append([c, pm, series_cache[series]])
        existing.add(key)
        added += 1
    print(f"Universe refresh complete | watch={len(watch)} | new={added}")
    return by_question


def _counterfactual_allocate(d, bankroll: float, cfg: V8Config, policy: str):
    reserve = bankroll * cfg.reserve_cash_fraction
    cash = bankroll - reserve
    event_used: dict[str, float] = {}
    market_used: dict[str, float] = {}
    bucket_used: dict[str, float] = {}
    deployed = profit = weighted_days_num = weighted_apr_num = 0.0
    trades = 0

    if policy == "raw_profit":
        ranked = d.sort_values("net_profit", ascending=False)
    elif policy == "roc":
        ranked = d.assign(_metric=d.net_profit / d.capital).sort_values("_metric", ascending=False)
    elif policy == "duration_biased":
        ranked = d.assign(_metric=d.capital_velocity / (1.0 + d.settlement_days / 90.0)).sort_values("_metric", ascending=False)
    else:  # capital_velocity
        ranked = d.sort_values("capital_velocity", ascending=False)

    for _, r in ranked.iterrows():
        ticker = str(r["ticker"])
        event = str(r.get("topic", "unknown"))
        bucket = str(r.get("duration_bucket") or _duration_bucket(float(r.get("settlement_days", 730))))
        qty0 = max(float(r["quantity"]), 1.0)
        unit_cost = float(r["capital"]) / qty0
        unit_profit = float(r["net_profit"]) / qty0
        if unit_cost <= 0 or unit_profit <= 0:
            continue
        depth_qty = int(max(0.0, min(float(r.get("kalshi_total_size", 0) or 0), float(r.get("poly_total_size", 0) or 0))))
        if depth_qty < 1:
            continue
        bucket_left = bankroll * _bucket_fraction(bucket, cfg) - bucket_used.get(bucket, 0.0)
        cap = min(
            cash,
            bankroll * cfg.max_market_fraction - market_used.get(ticker, 0.0),
            bankroll * cfg.max_event_fraction - event_used.get(event, 0.0),
            bucket_left,
        )
        qty = min(depth_qty, int(max(0.0, cap // unit_cost)))
        if qty < 1:
            continue
        cost = qty * unit_cost
        pnl = qty * unit_profit
        days = float(r.get("settlement_days", 730) or 730)
        apr = float(r.get("annualized_hold_return", 0) or 0)
        cash -= cost
        deployed += cost
        profit += pnl
        weighted_days_num += cost * days
        weighted_apr_num += cost * apr
        trades += 1
        market_used[ticker] = market_used.get(ticker, 0.0) + cost
        event_used[event] = event_used.get(event, 0.0) + cost
        bucket_used[bucket] = bucket_used.get(bucket, 0.0) + cost

    return {
        "estimated_deployed": deployed,
        "estimated_locked_profit": profit,
        "estimated_return": profit / bankroll if bankroll else 0.0,
        "estimated_positions": trades,
        "weighted_settlement_days": weighted_days_num / deployed if deployed else 0.0,
        "weighted_hold_apr": weighted_apr_num / deployed if deployed else 0.0,
    }


def _analysis_outputs(log: RunLoggerV8, cfg: V8Config, actual_bankroll: float):
    path = log.paths["signals"]
    if not path.exists():
        return
    try:
        import pandas as pd
        d = pd.read_csv(path)
    except Exception:
        return
    d = d[(d["eligible"] == True) & d["capital"].notna() & (d["capital"] > 0) & d["net_profit"].notna()]
    if d.empty:
        return
    # One strongest observation per exact strategy prevents repeated polling
    # snapshots from manufacturing counterfactual capacity.
    d = d.sort_values("net_profit", ascending=False).drop_duplicates(["ticker", "strategy"])

    for policy in ("raw_profit", "roc", "capital_velocity", "duration_biased"):
        row = _counterfactual_allocate(d, actual_bankroll, cfg, policy)
        log.append("policy", {"timestamp": time.time(), "policy": policy, "bankroll": actual_bankroll, **row,
                              "method": "observed-book counterfactual"})

    for bankroll in cfg.capacity_bankrolls:
        row = _counterfactual_allocate(d, bankroll, cfg, "capital_velocity")
        log.append("capacity", {"timestamp": time.time(), "bankroll": bankroll, **row,
                                "method": "V8 capital-velocity observed-book counterfactual"})

    # Snapshot actual duration-bucket exposure for reporting.
    for bucket in ("0-30d", "31-90d", "91-365d", ">365d"):
        eligible = d[d["duration_bucket"] == bucket]
        log.append("duration", {
            "timestamp": time.time(),
            "duration_bucket": bucket,
            "eligible_unique_signals": len(eligible),
            "best_net_profit": float(eligible.net_profit.max()) if not eligible.empty else 0.0,
            "best_capital_velocity": float(eligible.capital_velocity.max()) if not eligible.empty else 0.0,
            "bucket_cap_fraction": _bucket_fraction(bucket, cfg),
        })


def run_v8(engine_config: V2Config, confirmation_config: ConfirmationConfig, cfg: V8Config):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLoggerV8(run_id)
    print(f"FINAL run id: {run_id}")
    print("Building V12.0 final rule-complete expanded-market universe...")
    _, confirmations, passive = run_v3(engine_config, confirmation_config)
    confirmed = [x for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    # Preserve V11's capital-efficiency confirmation ordering. Raw dollar
    # profit is only a tiebreaker; otherwise long-dated deep books crowd out
    # faster recyclable opportunities.
    confirmed.sort(
        key=lambda x: (
            float(x.get("confirmation_priority", -999.0)),
            float(x.get("worst_net_profit", -999.0)),
        ),
        reverse=True,
    )
    confirmed = confirmed[:cfg.max_watch_candidates]
    print(f"Confirmed watch candidates: {len(confirmed)}")

    portfolio = V8Portfolio(engine_config.bankroll, cfg.reserve_cash_fraction, cfg.max_market_fraction, cfg.max_event_fraction)
    by_question = _refresh_poly_map()
    series_cache: dict[str, dict] = {}
    watch: list[list] = []
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
    trade_id = 0
    maker_probes = _build_maker_probes(passive, by_question, series_cache, portfolio, cfg, log)
    if cfg.maker_probe_enabled:
        print(f"Strict maker probes: {len(maker_probes)} active | reserved=${portfolio.maker_probe_reserved:.2f}")
    api_failure_streak = 0
    started = time.monotonic()
    deadline = started + cfg.run_minutes * 60
    cycle = 0
    last_metadata_refresh = time.monotonic()
    last_universe_refresh = time.monotonic()
    last_unwind_check = 0.0
    last_fast_recycle_scan = 0.0
    fast_recycle_seen: set[str] = set()
    fast_recycle_profit = 0.0
    fast_cfg = FastRecycleConfig(
        enabled=cfg.fast_recycle_enabled,
        scan_seconds=cfg.fast_recycle_scan_seconds,
        max_markets_per_scan=cfg.fast_recycle_market_limit,
        max_capital_fraction_per_trade=cfg.fast_recycle_max_trade_fraction,
    )
    last_multi_outcome_scan = 0.0
    multi_outcome_seen: set[str] = set()
    multi_cfg = MultiOutcomeConfig(
        enabled=cfg.multi_outcome_enabled,
        max_settlement_days=cfg.multi_outcome_max_days,
        max_events_per_scan=cfg.multi_outcome_max_events,
        max_capital_fraction_per_event=cfg.multi_outcome_event_fraction,
        min_profit_dollars=cfg.min_trade_net_dollars,
        min_return_on_capital=cfg.multi_outcome_min_roc,
    )
    last_limitless_cross_scan = 0.0
    limitless_cross_seen: set[str] = set()
    limitless_cfg = LimitlessPolyConfig(
        enabled=cfg.limitless_cross_enabled,
        max_settlement_days=cfg.limitless_cross_max_days,
        max_capital_fraction_per_trade=cfg.limitless_cross_trade_fraction,
        min_profit_dollars=cfg.min_trade_net_dollars,
    )
    # Audit extra venues once. They do not affect P&L unless their payoff and
    # execution semantics are explicitly supported.
    for venue in audit_extra_venues():
        log.append("venues", venue.__dict__)
        print(f"Venue audit | {venue.venue}: {venue.mode} | markets={venue.markets} | {venue.note}")

    print(
        f"Watching {len(watch)} candidates for {cfg.run_minutes:g} minutes | "
        f"bankroll=${engine_config.bankroll:.2f} | velocity allocator ON"
    )

    while time.monotonic() < deadline:
        cycle += 1
        cycle_start = time.monotonic()
        now_wall = time.time()

        for p in portfolio.settle_due(now_wall):
            log.append("unwinds", {
                "timestamp": now_wall, "trade_id": p.trade_id, "ticker": p.ticker, "subject": p.subject,
                "action": "SETTLEMENT", "hold_profit": p.hold_profit, "exit_pnl": p.exit_pnl,
                "capital_released": p.exit_proceeds, "reason": p.exit_reason,
            })

        if cfg.maker_probe_enabled and maker_probes:
            trade_id = _process_maker_probes(
                maker_probes, portfolio, cfg, log, rejection_counts, trade_id, results
            )
            if portfolio.realized_pnl <= -abs(cfg.max_run_loss) or portfolio.residual_contracts > cfg.max_total_residual_contracts:
                print("Maker-probe risk circuit breaker reached; ending final paper run.")
                deadline = time.monotonic()
                break

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

        if cfg.universe_refresh_minutes > 0 and (time.monotonic() - last_universe_refresh) >= cfg.universe_refresh_minutes * 60:
            try:
                by_question = _merge_confirmed_watch(watch, engine_config, confirmation_config, cfg, series_cache)
                last_universe_refresh = time.monotonic()
                last_metadata_refresh = last_universe_refresh
            except Exception as exc:
                print(f"Universe refresh warning: {type(exc).__name__}: {exc}")

        # Fast-recycle lane. A Polymarket YES+NO complete set can be merged
        # back to collateral, so successful paper trades realize immediately
        # and do not consume a duration bucket. To avoid manufacturing P&L
        # from an unchanged snapshot, each market is booked at most once per
        # run unless a future implementation can prove replenishment.
        if cfg.fast_recycle_enabled and (time.monotonic() - last_fast_recycle_scan) >= cfg.fast_recycle_scan_seconds:
            last_fast_recycle_scan = time.monotonic()
            try:
                fast, fstats = scan_polymarket_complete_sets(
                    list(by_question.values()), portfolio.deployable_cash, portfolio.starting_bankroll, fast_cfg,
                    return_diagnostics=True,
                )
                near = fstats.get("best_near_miss") or {}
                print(
                    "SHORT DIAG merge | "
                    f"scanned={fstats.get('markets_scanned',0)} books={fstats.get('books_available',0)} "
                    f"depth={fstats.get('two_sided_depth',0)} raw+={fstats.get('raw_positive',0)} "
                    f"fee+={fstats.get('positive_after_fees',0)} buffer+={fstats.get('positive_after_buffer',0)} "
                    f"qualified={fstats.get('qualified',0)} "
                    f"best_net/set={float(near.get('net_per_set_after_buffer') or 0):+.5f}"
                )
                log.append("short_diag", {
                    "timestamp": time.time(), "lane": "complete_set_merge",
                    "scanned": fstats.get("markets_scanned",0), "stage1": fstats.get("books_available",0),
                    "stage2": fstats.get("two_sided_depth",0), "stage3": fstats.get("positive_after_buffer",0),
                    "raw_positive": fstats.get("raw_positive",0), "fee_positive": fstats.get("positive_after_fees",0),
                    "qualified": fstats.get("qualified",0),
                    "best_net": float(near.get("net_per_set_after_buffer") or 0),
                    "note": str(near.get("question") or ""),
                })
                for fr in fast:
                    fkey = str(fr.get("market_id") or fr.get("slug") or fr.get("question"))
                    if fkey in fast_recycle_seen:
                        continue
                    # Capital is committed and returned within the same paper
                    # transaction via complete-set merge; only net P&L changes cash.
                    portfolio.available_cash += float(fr["net_profit"])
                    portfolio.realized_pnl += float(fr["net_profit"])
                    portfolio._update_drawdown()
                    fast_recycle_profit += float(fr["net_profit"])
                    fast_recycle_seen.add(fkey)
                    log.append("fast_recycle", fr)
                    print(f"FAST MERGE {str(fr.get('question'))[:55]} | qty={fr['quantity']} | "
                          f"net=${fr['net_profit']:.4f} | ROC={fr['return_on_capital']:.2%} | capital recycled")
            except Exception as exc:
                rejection_counts[f"fast-recycle scan error: {type(exc).__name__}"] += 1

        # Short-duration exhaustive multi-outcome baskets. This lane can fill
        # the 0-30/31-90 day sleeves without weakening the long-duration caps.
        # A basket is booked at most once per event+direction per run so static
        # displayed depth cannot be repeatedly manufactured into paper P&L.
        if cfg.multi_outcome_enabled and (time.monotonic() - last_multi_outcome_scan) >= cfg.multi_outcome_scan_seconds:
            last_multi_outcome_scan = time.monotonic()
            try:
                baskets, bstats = scan_polymarket_multi_outcome(
                    list(by_question.values()), portfolio.deployable_cash, portfolio.starting_bankroll, multi_cfg,
                    return_diagnostics=True,
                )
                near = bstats.get("best_near_miss") or {}
                print(
                    "SHORT DIAG basket | "
                    f"groups={bstats.get('event_groups',0)} valid={bstats.get('valid_exhaustive_events',0)} "
                    f"books={bstats.get('book_complete_events',0)} raw+={bstats.get('raw_positive_sides',0)} "
                    f"fee+={bstats.get('positive_after_fees',0)} qualified={bstats.get('qualified',0)} "
                    f"best_net/set={float(near.get('net_per_set_after_buffer') or 0):+.5f}"
                )
                log.append("short_diag", {
                    "timestamp": time.time(), "lane": "multi_outcome_basket",
                    "scanned": bstats.get("event_groups",0), "stage1": bstats.get("valid_exhaustive_events",0),
                    "stage2": bstats.get("book_complete_events",0), "stage3": bstats.get("positive_after_fees",0),
                    "raw_positive": bstats.get("raw_positive_sides",0), "fee_positive": bstats.get("positive_after_fees",0),
                    "qualified": bstats.get("qualified",0),
                    "best_net": float(near.get("net_per_set_after_buffer") or 0),
                    "note": str(near.get("event_title") or ""),
                })
                for basket in baskets:
                    bkey = f"{basket['event_id']}:{basket['strategy']}"
                    if bkey in multi_outcome_seen:
                        continue
                    days = float(basket["settlement_days"])
                    ticker = f"POLYBASKET-{basket['event_id']}-{basket['strategy']}"
                    event_key = f"POLYEVENT-{basket['event_id']}"
                    cap = portfolio.max_capital_for_v8(ticker, event_key, days, cfg)
                    capital = float(basket["capital"])
                    if capital > cap + 1e-9:
                        rejection_counts["multi-outcome portfolio/duration cap"] += 1
                        continue
                    trade_id += 1
                    pos = V7Position(
                        trade_id=trade_id, entry_time=time.time(), ticker=ticker, event_key=event_key,
                        subject=str(basket.get("event_title") or event_key), topic="polymarket_multi_outcome",
                        strategy=str(basket["strategy"]), route="polymarket_basket",
                        quantity=int(basket["quantity"]), locked_capital=capital,
                        hold_profit=float(basket["net_profit"]), residual_unhedged=0.0,
                        kalshi_side="N/A", poly_side=str(basket["strategy"]),
                        settlement_ts=float(basket["settlement_ts"]),
                        settlement_days_at_entry=days,
                        annualized_hold_return=float(basket["annualized_return"]),
                    )
                    portfolio.book(pos)
                    multi_outcome_seen.add(bkey)
                    row = {**basket, "trade_id": trade_id, "ticker": ticker, "event_key": event_key,
                           "duration_bucket": _duration_bucket(days), "portfolio_cash": portfolio.available_cash,
                           "portfolio_locked": portfolio.locked_capital, "portfolio_equity": portfolio.equity}
                    log.append("multi_outcome", row)
                    log.append("positions", row)
                    print(
                        f"SHORT BASKET {str(basket.get('event_title'))[:48]} | {basket['strategy']} | "
                        f"qty={basket['quantity']} | locked=${capital:.2f} | hold=${basket['net_profit']:.4f} | "
                        f"{days:.1f}d | APR={basket['annualized_return']:.1%}"
                    )
            except Exception as exc:
                rejection_counts[f"multi-outcome scan error: {type(exc).__name__}"] += 1

        # Strict Limitless × Polymarket short-horizon cross-venue lane. Only
        # explicit matching threshold/deadline/oracle contracts can book P&L.
        if cfg.limitless_cross_enabled and (time.monotonic() - last_limitless_cross_scan) >= cfg.limitless_cross_scan_seconds:
            last_limitless_cross_scan = time.monotonic()
            try:
                lops, lstats = scan_limitless_polymarket(
                    list(by_question.values()), portfolio.deployable_cash, portfolio.starting_bankroll, limitless_cfg,
                    return_diagnostics=True,
                )
                near = lstats.get("best_near_miss") or {}
                print(
                    "SHORT DIAG limitless | "
                    f"markets={lstats.get('limitless_markets',0)} lumy={lstats.get('lumy_markets',0)} "
                    f"poly_sig={lstats.get('poly_signatures',0)} lim_pre={lstats.get('limitless_presignatures',0)} "
                    f"lim_sig={lstats.get('limitless_signatures',0)} "
                    f"enrich={lstats.get('detail_fetch_successes',0)}/{lstats.get('detail_fetch_attempts',0)} "
                    f"aot={lstats.get('asset_operator_threshold_matches',0)} source={lstats.get('source_matches',0)} "
                    f"deadline={lstats.get('deadline_matches',0)} books={lstats.get('book_pairs',0)} "
                    f"raw+={lstats.get('raw_positive',0)} lfee+={lstats.get('positive_after_limitless_fee',0)} "
                    f"allfee+={lstats.get('positive_after_fees',0)} qualified={lstats.get('qualified',0)} "
                    f"best_net/ct={float(near.get('net_after_buffer') or 0):+.5f}"
                )
                log.append("short_diag", {
                    "timestamp": time.time(), "lane": "limitless_cross",
                    "scanned": lstats.get("limitless_markets",0), "stage1": lstats.get("poly_signatures",0),
                    "stage2": lstats.get("source_matches",0), "stage3": lstats.get("deadline_matches",0),
                    "raw_positive": lstats.get("raw_positive",0), "fee_positive": lstats.get("positive_after_fees",0),
                    "limitless_presignatures": lstats.get("limitless_presignatures",0),
                    "limitless_signatures": lstats.get("limitless_signatures",0),
                    "detail_fetch_attempts": lstats.get("detail_fetch_attempts",0),
                    "detail_fetch_successes": lstats.get("detail_fetch_successes",0),
                    "positive_after_limitless_fee": lstats.get("positive_after_limitless_fee",0),
                    "qualified": lstats.get("qualified",0),
                    "best_net": float(near.get("net_after_buffer") or 0),
                    "note": str(near.get("strategy") or lstats.get("error") or ""),
                })
                for op in lops:
                    key=f"{op['limitless_slug']}:{op['poly_id']}:{op['strategy']}"
                    if key in limitless_cross_seen:
                        continue
                    days=float(op["settlement_days"]); ticker=f"LIMIPOLY-{op['limitless_slug']}-{op['strategy']}"
                    event_key=f"LIMIPOLY-{op['asset']}-{op['threshold']}-{int(op['settlement_ts'])}"
                    cap=portfolio.max_capital_for_v8(ticker,event_key,days,cfg)
                    capital=float(op["capital"])
                    if capital>cap+1e-9:
                        rejection_counts["limitless cross portfolio/duration cap"] += 1
                        continue
                    trade_id += 1
                    pos=V7Position(
                        trade_id=trade_id,entry_time=time.time(),ticker=ticker,event_key=event_key,
                        subject=f"{op['asset']} {op['operator']} {op['threshold']}",topic="limitless_polymarket_short",
                        strategy=str(op["strategy"]),route="limitless_polymarket",quantity=int(op["quantity"]),
                        locked_capital=capital,hold_profit=float(op["net_profit"]),residual_unhedged=0.0,
                        kalshi_side="N/A",poly_side=str(op["strategy"]),settlement_ts=float(op["settlement_ts"]),
                        settlement_days_at_entry=days,annualized_hold_return=float(op["annualized_return"]),
                    )
                    portfolio.book(pos); limitless_cross_seen.add(key)
                    row={**op,"trade_id":trade_id,"ticker":ticker,"event_key":event_key,
                         "duration_bucket":_duration_bucket(days),"portfolio_cash":portfolio.available_cash,
                         "portfolio_locked":portfolio.locked_capital,"portfolio_equity":portfolio.equity}
                    log.append("limitless_cross",row); log.append("positions",row)
                    print(f"SHORT CROSS {pos.subject} | {op['strategy']} | qty={op['quantity']} | "
                          f"locked=${capital:.2f} | hold=${op['net_profit']:.4f} | {days:.2f}d | "
                          f"APR={op['annualized_return']:.1%}")
            except Exception as exc:
                rejection_counts[f"limitless cross scan error: {type(exc).__name__}"] += 1

        # Capital-recycling scan.
        if cfg.early_unwind_enabled and (time.monotonic() - last_unwind_check) >= cfg.early_unwind_check_seconds:
            last_unwind_check = time.monotonic()
            candidate_by_ticker = {x[0]["ticker"]: x for x in watch}
            for probe in maker_probes:
                candidate_by_ticker.setdefault(
                    probe.candidate["ticker"], [probe.candidate, probe.pm, probe.series_info]
                )
            for pos in list(portfolio.positions):
                item = candidate_by_ticker.get(pos.ticker)
                if item is None:
                    continue
                _, pm, series_info = item
                try:
                    quote, reason = _early_unwind_quote(pos, pm, series_info)
                except Exception as exc:
                    quote, reason = None, f"exit API error: {type(exc).__name__}"
                if quote is None:
                    log.append("unwinds", {"timestamp": time.time(), "trade_id": pos.trade_id, "ticker": pos.ticker,
                        "subject": pos.subject, "action": "HOLD", "hold_profit": pos.hold_profit, "exit_pnl": None,
                        "capital_released": None, "reason": reason})
                    continue
                should, why = _should_unwind_v8(pos, quote, cfg, time.time())
                log.append("unwinds", {"timestamp": time.time(), "trade_id": pos.trade_id, "ticker": pos.ticker,
                    "subject": pos.subject, "action": "EXIT" if should else "HOLD", "hold_profit": pos.hold_profit,
                    "exit_pnl": quote["pnl"], "capital_released": quote["proceeds"], "reason": why})
                if should:
                    portfolio.close(pos, proceeds=quote["proceeds"], pnl=quote["pnl"], reason=why)
                    print(f"UNWIND {pos.subject} | realized=${quote['pnl']:.4f} | released=${quote['proceeds']:.2f}")

        proposals = []
        for candidate, pm, series_info in watch:
            ticker = candidate["ticker"]
            event_key = _event_key(candidate)
            _, rough_days, _ = _cross_settlement_metrics(candidate, pm, 1.0, 0.0, cfg)
            cap = portfolio.max_capital_for_v8(ticker, event_key, rough_days, cfg)
            cap = min(cap, _basis_cap_limit(portfolio, candidate, event_key, cfg))
            bucket = _duration_bucket(rough_days)
            if cap < 0.50:
                reason = "portfolio/event/duration capital limit"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, None, False, reason, settlement_days=rough_days, duration_bucket=bucket))
                continue

            signal, reason = _safe_optimized_signal(candidate, pm, series_info, max_qty=int(candidate["quantity"]), capital_limit=cap, cfg=cfg)
            if signal is None:
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, None, False, reason, settlement_days=rough_days, duration_bucket=bucket))
                if reason.startswith("API failure"):
                    api_failure_streak += 1
                continue
            api_failure_streak = 0

            required_edge = max(
                cfg.minimum_signal_net_per_contract,
                cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract,
            )
            if signal["net_per_contract"] < required_edge:
                reason = "edge lacks safety buffer"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, reason, settlement_days=rough_days, duration_bucket=bucket))
                continue
            if signal["net_profit"] < cfg.min_trade_net_dollars:
                reason = "profit dollars below minimum"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, reason, settlement_days=rough_days, duration_bucket=bucket))
                continue
            gate_ok, gate_reason = _equivalence_edge_gate(candidate, signal, cfg)
            if not gate_ok:
                rejection_counts[gate_reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, gate_reason, settlement_days=rough_days, duration_bucket=bucket))
                continue

            state = _book_state(signal)
            key = (ticker, candidate["strategy"])
            if cfg.require_material_book_change and not _materially_replenished(
                previous_executed_state.get(key), state, int(signal["quantity"]), cfg
            ):
                reason = "book not materially replenished/changed"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, reason, settlement_days=rough_days, duration_bucket=bucket))
                continue

            score, days, annualized, velocity, required_apr, bucket = _allocation_score_v8(candidate, signal, pm, cfg)
            if _resolution_lane(candidate) == "LOW_BASIS" and annualized < cfg.low_basis_min_adjusted_apr:
                reason = "low-basis risk-adjusted APR below floor"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, reason, score=score,
                    settlement_days=days, annualized_return=annualized, capital_velocity=velocity,
                    required_apr=required_apr, duration_bucket=bucket))
                continue
            if annualized < required_apr and cfg.hard_duration_hurdle:
                reason = "hold APR below duration hurdle"
                rejection_counts[reason] += 1
                log.append("signals", _signal_row_v8(candidate, signal, False, reason, score=score,
                    settlement_days=days, annualized_return=annualized, capital_velocity=velocity,
                    required_apr=required_apr, duration_bucket=bucket))
                continue
            # Soft duration penalty: sub-hurdle trades remain eligible only for
            # their duration bucket's tightly capped capital sleeve.
            if annualized < required_apr:
                score *= max(0.05, annualized / max(required_apr, 1e-9))

            proposals.append({
                "candidate": candidate, "pm": pm, "series": series_info, "signal": signal,
                "state": state, "score": score, "snapshot_time": time.monotonic(),
                "settlement_days": days, "annualized": annualized, "velocity": velocity,
                "required_apr": required_apr, "duration_bucket": bucket,
            })
            log.append("signals", _signal_row_v8(candidate, signal, True, "proposal", score=score,
                settlement_days=days, annualized_return=annualized, capital_velocity=velocity,
                required_apr=required_apr, duration_bucket=bucket))

        if proposals and cfg.allocation_window_seconds > 0:
            time.sleep(min(cfg.allocation_window_seconds, max(0.0, deadline - time.monotonic())))
        proposals.sort(key=lambda x: x["score"], reverse=True)

        for p in proposals:
            candidate, pm, series_info = p["candidate"], p["pm"], p["series"]
            ticker = candidate["ticker"]
            event_key = _event_key(candidate)
            proposal_age = time.monotonic() - p["snapshot_time"]
            if proposal_age > cfg.max_revalidation_age_seconds:
                rejection_counts["proposal too old even for fresh revalidation"] += 1
                continue
            # Aged ranking snapshots are safe to reconsider because the next
            # call obtains fresh executable depth on both venues. This recovers
            # persistent edges that V10/V11 unnecessarily dropped at 6s.

            is_subhurdle = p["annualized"] < p["required_apr"]
            strict_yield_eligible = (
                is_subhurdle
                and _resolution_lane(candidate) == "STRICT_ARB"
                and p["annualized"] >= cfg.strict_yield_min_apr
                and p["settlement_days"] <= cfg.strict_yield_max_days
            )
            if strict_yield_eligible:
                cap = portfolio.max_capital_for_v8(ticker, event_key, p["settlement_days"], cfg, subhurdle=False)
                cap = min(cap, _strict_yield_cap_limit(portfolio, candidate, cfg))
            else:
                cap = portfolio.max_capital_for_v8(ticker, event_key, p["settlement_days"], cfg, subhurdle=is_subhurdle)
            cap = min(cap, _basis_cap_limit(portfolio, candidate, event_key, cfg))
            if cap < 0.50:
                rejection_counts["portfolio/event/duration capital limit"] += 1
                continue
            signal, reason = _safe_optimized_signal(candidate, pm, series_info, max_qty=int(p["signal"]["quantity"]), capital_limit=cap, cfg=cfg)
            if signal is None:
                rejection_counts["failed pre-execution revalidation"] += 1
                continue

            required_edge = max(
                cfg.minimum_signal_net_per_contract,
                cfg.minimum_execution_net_per_contract + cfg.minimum_safety_buffer_per_contract,
            )
            if signal["net_per_contract"] < required_edge or signal["net_profit"] < cfg.min_trade_net_dollars:
                rejection_counts["edge disappeared on revalidation"] += 1
                continue
            gate_ok, gate_reason = _equivalence_edge_gate(candidate, signal, cfg)
            if not gate_ok:
                rejection_counts[gate_reason] += 1
                continue

            score, days, annualized, velocity, required_apr, bucket = _allocation_score_v8(candidate, signal, pm, cfg)
            if _resolution_lane(candidate) == "LOW_BASIS" and annualized < cfg.low_basis_min_adjusted_apr:
                rejection_counts["low-basis risk-adjusted APR below floor"] += 1
                continue
            if annualized < required_apr and cfg.hard_duration_hurdle:
                rejection_counts["hold APR below duration hurdle"] += 1
                continue
            if annualized < required_apr:
                score *= max(0.05, annualized / max(required_apr, 1e-9))

            route = _choose_route(signal, cfg)
            latency = _sample_latency(cfg, rng)
            expected_second = signal["poly_avg"] if route == "kalshi_first" else signal["kalshi_avg"]
            rr = simulate_selected_route(
                candidate=candidate, pm=pm, series_info=series_info, route=route,
                qty=int(signal["quantity"]), latency_seconds=latency,
                expected_second_price=expected_second, max_second_leg_move=cfg.max_second_leg_move,
            )

            for stress_latency in cfg.stress_latencies:
                sr = simulate_selected_route(
                    candidate=candidate, pm=pm, series_info=series_info, route=route,
                    qty=int(signal["quantity"]), latency_seconds=stress_latency,
                    expected_second_price=expected_second, max_second_leg_move=999.0,
                )
                log.append("latency", {
                    "timestamp": time.time(), "ticker": ticker, "subject": candidate.get("subject"),
                    "topic": candidate.get("topic"), "route": route, "quantity": int(signal["quantity"]),
                    "latency_seconds": stress_latency, "status": sr.status, "net_profit": sr.conservative_pnl,
                    "net_per_contract": sr.net_per_requested_contract, "residual_unhedged": sr.residual_unhedged,
                    "second_leg_price_move": sr.second_leg_price_move,
                })

            trade_id += 1
            if rr.status in ("ERROR", "FIRST_LEG_NO_FILL"):
                rejection_counts[rr.status] += 1
                continue
            if rr.locked_capital > portfolio.available_cash + 1e-9:
                rejection_counts["PORTFOLIO_REJECT_AFTER_MOVE"] += 1
                continue

            gross_pnl = float(rr.conservative_pnl)
            basis_reserve = _basis_reserve_per_contract(candidate, cfg) * int(signal["quantity"])
            accounted_pnl = gross_pnl - basis_reserve
            if accounted_pnl < cfg.min_trade_net_dollars:
                rejection_counts["post-latency profit below minimum after basis reserve"] += 1
                continue
            settlement_ts, days, annualized = _cross_settlement_metrics(candidate, pm, rr.locked_capital, accounted_pnl, cfg)
            required_apr = _required_hold_apr(days, cfg)
            velocity = _capital_velocity(accounted_pnl / max(rr.locked_capital, 1e-9), days, cfg)
            bucket = _duration_bucket(days)
            if _resolution_lane(candidate) == "LOW_BASIS" and annualized < cfg.low_basis_min_adjusted_apr:
                rejection_counts["post-latency low-basis APR below floor"] += 1
                continue
            if annualized < required_apr and cfg.hard_duration_hurdle:
                rejection_counts["post-latency APR below duration hurdle"] += 1
                continue

            kside, pside = _strategy_sides(candidate["strategy"])
            pos = V7Position(
                trade_id, time.time(), ticker, event_key, str(candidate.get("subject")),
                str(candidate.get("topic")), str(candidate.get("strategy")), route,
                int(signal["quantity"]), rr.locked_capital, accounted_pnl,
                rr.residual_unhedged, kside, pside, settlement_ts, days, annualized,
            )
            setattr(pos, "subhurdle", annualized < required_apr)
            setattr(pos, "resolution_lane", _resolution_lane(candidate))
            setattr(pos, "gross_hold_profit", gross_pnl)
            setattr(pos, "basis_risk_reserve", basis_reserve)
            setattr(pos, "strict_yield_sleeve", bool(
                _resolution_lane(candidate) == "STRICT_ARB"
                and annualized < required_apr
                and annualized >= cfg.strict_yield_min_apr
                and days <= cfg.strict_yield_max_days
            ))
            portfolio.book(pos)
            previous_executed_state[(ticker, candidate["strategy"])] = _book_state(signal)
            row = {
                "timestamp": time.time(), "trade_id": trade_id, "ticker": ticker,
                "subject": candidate.get("subject"), "topic": candidate.get("topic"),
                "strategy": candidate.get("strategy"), "route": route, "status": rr.status,
                "allocation_score": score, "capital_velocity": velocity, "duration_bucket": bucket,
                "required_hold_apr": required_apr, "quantity": int(signal["quantity"]),
                "latency_seconds": latency, "locked_capital": rr.locked_capital,
                "hold_profit": accounted_pnl, "gross_hold_profit": gross_pnl,
                "basis_risk_reserve": basis_reserve, "resolution_lane": _resolution_lane(candidate),
                "net_per_contract": accounted_pnl / max(int(signal["quantity"]), 1),
                "settlement_ts": settlement_ts, "settlement_days": days,
                "annualized_hold_return": annualized, "residual_unhedged": rr.residual_unhedged,
                "portfolio_cash": portfolio.available_cash, "portfolio_locked": portfolio.locked_capital,
                "portfolio_locked_profit": portfolio.locked_profit,
                "portfolio_realized_pnl": portfolio.realized_pnl, "portfolio_equity": portfolio.equity,
                "kalshi_title": candidate.get("kalshi_title"), "poly_question": candidate.get("poly_question"),
                "match_source": candidate.get("match_source"), "equivalence_score": candidate.get("equivalence_score"),
                "equivalence_reasons": candidate.get("equivalence_reasons"),
                "resolution_rule_status": (candidate.get("equivalence_certificate") or {}).get("resolution_rule_status"),
                "latest_cross_settlement_ts": (candidate.get("equivalence_certificate") or {}).get("latest_cross_settlement_ts"),
                "equivalence_certificate": repr(candidate.get("equivalence_certificate") or {}),
            }
            results.append(row)
            log.append("trades", row)
            log.append("positions", row)
            print(
                f"BOOK {candidate['subject']} | velocity={velocity:.4f} | score={score:.4f} | "
                f"qty={signal['quantity']} | locked=${rr.locked_capital:.2f} | hold=${accounted_pnl:.4f} | "
                f"{days:.0f}d | APR={annualized:.2%} | hurdle={required_apr:.2%} | {bucket}"
                + (f" | LOW_BASIS reserve=${basis_reserve:.4f}" if _resolution_lane(candidate) == "LOW_BASIS" else "")
                + (" | STRICT_YIELD SLEEVE" if bool(getattr(pos, "strict_yield_sleeve", False)) else "")
                + (" | SUB-HURDLE RESEARCH SLEEVE" if annualized < required_apr and not bool(getattr(pos, "strict_yield_sleeve", False)) else "")
            )

            if portfolio.realized_pnl <= -abs(cfg.max_run_loss) or portfolio.residual_contracts > cfg.max_total_residual_contracts:
                print("Risk circuit breaker reached; ending V12.0 Final.")
                deadline = time.monotonic()
                break

        if api_failure_streak >= cfg.max_consecutive_api_failures:
            print("API-health circuit breaker reached; ending V12.0 Final.")
            break

        elapsed = (time.monotonic() - started) / 60.0
        log.append("equity", _equity_row_v8(portfolio, cycle, elapsed))
        buckets = ", ".join(f"{k}:{v:.0f}" for k, v in portfolio.bucket_capital.items() if v > 0)
        if cycle == 1 or cycle % 6 == 0:
            top_now = rejection_counts.most_common(4)
            if top_now:
                print("Filter funnel (cumulative): " + " | ".join(f"{k}={v}" for k, v in top_now))

        print(
            f"Cycle {cycle} | {elapsed:.1f}m | proposals={len(proposals)} | open={len(portfolio.positions)} | "
            f"closed={len(portfolio.closed_positions)} | cash=${portfolio.available_cash:.2f} | "
            f"locked=${portfolio.locked_capital:.2f} | locked profit=${portfolio.locked_profit:.4f} | "
            f"realized=${portfolio.realized_pnl:.4f} | buckets=[{buckets}]"
        )
        sleep_for = cfg.poll_seconds - (time.monotonic() - cycle_start)
        if sleep_for > 0 and time.monotonic() < deadline:
            time.sleep(min(sleep_for, deadline - time.monotonic()))

    _analysis_outputs(log, cfg, engine_config.bankroll)
    return portfolio, results, log, rejection_counts


def print_summary(portfolio: V8Portfolio, results, log: RunLoggerV8, rejection_counts, run_minutes, cfg: V8Config):
    open_pos = portfolio.positions
    closed = portfolio.closed_positions
    hours = max(run_minutes / 60.0, 1e-9)
    total_open_cap = sum(p.locked_capital for p in open_pos)
    weighted_days = (
        sum(p.locked_capital * p.settlement_days_at_entry for p in open_pos) / max(total_open_cap, 1e-9)
        if open_pos else 0.0
    )
    weighted_apr = (
        sum(p.locked_capital * p.annualized_hold_return for p in open_pos) / max(total_open_cap, 1e-9)
        if open_pos else 0.0
    )
    weighted_velocity = (
        sum(
            p.locked_capital * _capital_velocity(p.hold_profit / max(p.locked_capital, 1e-9), p.settlement_days_at_entry, cfg)
            for p in open_pos
        ) / max(total_open_cap, 1e-9)
        if open_pos else 0.0
    )
    strict_pos = [p for p in open_pos if getattr(p, "resolution_lane", "STRICT_ARB") != "LOW_BASIS"]
    basis_pos = [p for p in open_pos if getattr(p, "resolution_lane", "STRICT_ARB") == "LOW_BASIS"]
    strict_cap = sum(p.locked_capital for p in strict_pos)
    basis_cap = sum(p.locked_capital for p in basis_pos)
    strict_profit = sum(p.hold_profit for p in strict_pos)
    basis_adjusted_profit = sum(p.hold_profit for p in basis_pos)
    basis_gross_profit = sum(float(getattr(p, "gross_hold_profit", p.hold_profit)) for p in basis_pos)
    basis_reserve = sum(float(getattr(p, "basis_risk_reserve", 0.0)) for p in basis_pos)
    strict_yield_capital = sum(p.locked_capital for p in open_pos if bool(getattr(p, "strict_yield_sleeve", False)))
    strict_yield_profit = sum(p.hold_profit for p in open_pos if bool(getattr(p, "strict_yield_sleeve", False)))

    print("\n" + "=" * 100)
    print("PAPER ENGINE V13.0 FINAL - STRICT+GRADED-BASIS PROFIT SUMMARY")
    print("=" * 100)
    print(f"Run ID:                         {log.run_id}")
    print(f"Starting bankroll:              ${portfolio.starting_bankroll:.2f}")
    print(f"Available / locked:             ${portfolio.available_cash:.2f} / ${portfolio.locked_capital:.2f}")
    print(f"Maker-probe cash reserved:      ${portfolio.maker_probe_reserved:.2f}")
    print(f"Risk-adjusted locked profit:     ${portfolio.locked_profit:.4f}")
    print(f"  Strict-arbitrage locked:       ${strict_profit:.4f} on ${strict_cap:.2f}")
    print(f"  Low-basis adjusted locked:     ${basis_adjusted_profit:.4f} on ${basis_cap:.2f}")
    print(f"  Low-basis gross spread:        ${basis_gross_profit:.4f}")
    print(f"  Basis-risk reserve deducted:   ${basis_reserve:.4f}")
    print(f"Realized early-exit/settle P&L: ${portfolio.realized_pnl:.4f}")
    print(f"Paper equity:                   ${portfolio.equity:.4f}")
    print(f"Open / closed positions:        {len(open_pos)} / {len(closed)}")
    print(f"Capital utilization:            {portfolio.locked_capital / portfolio.starting_bankroll:.2%}")
    print(f"Capital-weighted lock horizon:  {weighted_days:.1f} days")
    print(f"Capital-weighted hold APR:      {weighted_apr:.2%}")
    print(f"Presentation target APR:  {cfg.benchmark_target_apr:.2%}")
    print(f"APR gap to presentation target:     {(weighted_apr - cfg.benchmark_target_apr):+.2%}")
    print(f"Capital-weighted velocity:      {weighted_velocity:.4f}")
    print(f"Residual contracts:             {portfolio.residual_contracts:.4f}")
    print(f"Max drawdown:                   ${portfolio.max_drawdown:.4f}")
    print(f"New locked profit/hour sample:  ${sum(float(x.get('hold_profit', 0)) for x in results) / hours:.4f} (NOT reusable cash)")
    print(f"Realized fast/exit profit:       ${portfolio.realized_pnl:.4f} (reusable cash)")
    print(f"Strict-yield sleeve:             ${strict_yield_capital:.2f} capital | ${strict_yield_profit:.4f} locked profit")
    print(f"Other sub-hurdle research cap:   ${portfolio.starting_bankroll * cfg.subhurdle_total_fraction:.2f}")
    print("Duration-bucket capital:")
    for bucket in ("0-30d", "31-90d", "91-365d", ">365d"):
        amount = portfolio.bucket_capital.get(bucket, 0.0)
        print(f"  {bucket:8s} ${amount:8.2f}  ({amount / portfolio.starting_bankroll:.1%})")

    if rejection_counts:
        print("\nTop rejection reasons:")
        for reason, n in rejection_counts.most_common(12):
            print(f"  {n:5d}  {reason}")

    summary = {
        "timestamp": time.time(), "starting_bankroll": portfolio.starting_bankroll,
        "available_cash": portfolio.available_cash, "locked_capital": portfolio.locked_capital,
        "locked_profit": portfolio.locked_profit, "realized_pnl": portfolio.realized_pnl,
        "equity": portfolio.equity, "open_positions": len(open_pos), "closed_positions": len(closed),
        "capital_utilization": portfolio.locked_capital / portfolio.starting_bankroll,
        "weighted_settlement_days": weighted_days, "weighted_hold_apr": weighted_apr,
        "weighted_capital_velocity": weighted_velocity, "residual_contracts": portfolio.residual_contracts,
        "max_drawdown": portfolio.max_drawdown, "subhurdle_capital": portfolio.subhurdle_capital,
        "maker_probe_reserved": portfolio.maker_probe_reserved,
        "strict_locked_capital": strict_cap, "strict_locked_profit": strict_profit,
        "strict_yield_capital": strict_yield_capital, "strict_yield_profit": strict_yield_profit,
        "low_basis_locked_capital": basis_cap, "low_basis_adjusted_profit": basis_adjusted_profit,
        "low_basis_gross_profit": basis_gross_profit, "basis_risk_reserve": basis_reserve,
        "sample_new_locked_profit_per_hour": sum(float(x.get("hold_profit", 0)) for x in results) / hours,
    }
    for bucket in ("0-30d", "31-90d", "91-365d", ">365d"):
        summary[f"capital_{bucket}"] = portfolio.bucket_capital.get(bucket, 0.0)
    log.append("summary", summary)
    print(f"\nRun data: {log.root}")
    print("Paper only: V12.0 Final contains no order-placement call.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--minutes", type=float, default=480.0)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--min-edge", type=float, default=0.0025)
    ap.add_argument("--safety-buffer", type=float, default=0.0025)
    ap.add_argument("--min-profit", type=float, default=0.05)
    ap.add_argument("--confirmations", type=int, default=5)
    ap.add_argument("--confirm-delay", type=float, default=0.75)
    ap.add_argument("--confirm-top", type=int, default=100)
    ap.add_argument("--latency-mean", type=float, default=0.25)
    ap.add_argument("--latency-jitter", type=float, default=0.15)
    ap.add_argument("--max-skew", type=float, default=0.50)
    ap.add_argument("--event-cap", type=float, default=0.25)
    ap.add_argument("--market-cap", type=float, default=0.12)
    ap.add_argument("--reserve-cash", type=float, default=0.25)
    ap.add_argument("--metadata-refresh", type=float, default=30.0)
    ap.add_argument("--universe-refresh", type=float, default=60.0)
    ap.add_argument("--max-signal-age", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opportunity-cost-apr", type=float, default=0.12)
    ap.add_argument("--base-hold-apr", type=float, default=0.030)
    ap.add_argument("--lock-hurdle-apr", type=float, default=0.040)
    ap.add_argument("--hard-duration-hurdle", action="store_true")
    ap.add_argument("--subhurdle-cap", type=float, default=0.01)
    ap.add_argument("--unwind-check", type=float, default=60.0)
    ap.add_argument("--unwind-capture", type=float, default=0.75)
    ap.add_argument("--no-early-unwind", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-fast-recycle", action="store_true")
    ap.add_argument("--fast-scan", type=float, default=30.0)
    ap.add_argument("--fast-market-limit", type=int, default=3000)
    ap.add_argument("--no-multi-outcome", action="store_true")
    ap.add_argument("--basket-scan", type=float, default=60.0)
    ap.add_argument("--basket-max-days", type=float, default=45.0)
    ap.add_argument("--no-limitless-cross", action="store_true")
    ap.add_argument("--limitless-scan", type=float, default=120.0)
    ap.add_argument("--limitless-max-days", type=float, default=14.0)
    ap.add_argument("--no-low-basis", action="store_true")
    ap.add_argument("--low-basis-cap", type=float, default=0.05)
    ap.add_argument("--low-basis-min-apr", type=float, default=0.08)
    ap.add_argument("--no-maker-probes", action="store_true")
    ap.add_argument("--maker-max-active", type=int, default=10)
    ap.add_argument("--maker-total-cap", type=float, default=0.15)
    ap.add_argument("--maker-ttl", type=float, default=300.0)
    ap.add_argument("--no-optimized-sizing", action="store_true")
    a = ap.parse_args()

    ec = V2Config(
        bankroll=max(1.0, a.bankroll),
        min_net_per_contract=max(0.0, a.min_edge),
        max_quote_skew_seconds=max(0.01, a.max_skew),
        allow_taker_taker=True,
    )
    cc = ConfirmationConfig(
        samples=max(1, a.confirmations),
        delay_seconds=max(0.0, a.confirm_delay),
        max_candidates=max(1, a.confirm_top),
    )
    cfg = V8Config(
        run_minutes=max(0.0, a.minutes), poll_seconds=max(1.0, a.poll),
        max_watch_candidates=max(1, a.confirm_top), max_watch_universe=max(1, max(a.confirm_top, 180)),
        latency_mean_seconds=max(0.0, a.latency_mean), latency_jitter_seconds=max(0.0, a.latency_jitter),
        max_quote_skew_seconds=max(0.01, a.max_skew),
        minimum_execution_net_per_contract=max(0.0, a.min_edge),
        minimum_signal_net_per_contract=max(0.0, a.min_edge + a.safety_buffer),
        minimum_safety_buffer_per_contract=max(0.0, a.safety_buffer),
        min_trade_net_dollars=max(0.0, a.min_profit),
        max_event_fraction=min(1.0, max(0.01, a.event_cap)),
        max_market_fraction=min(1.0, max(0.01, a.market_cap)),
        reserve_cash_fraction=min(0.95, max(0.0, a.reserve_cash)),
        metadata_refresh_minutes=max(1.0, a.metadata_refresh),
        universe_refresh_minutes=max(0.0, a.universe_refresh),
        max_signal_age_seconds=max(0.25, a.max_signal_age), random_seed=a.seed,
        opportunity_cost_apr=max(0.0, a.opportunity_cost_apr),
        base_required_hold_apr=max(0.0, a.base_hold_apr),
        lock_horizon_hurdle_apr=max(0.0, a.lock_hurdle_apr),
        hard_duration_hurdle=bool(a.hard_duration_hurdle),
        subhurdle_total_fraction=min(0.20, max(0.0, a.subhurdle_cap)),
        early_unwind_check_seconds=max(5.0, a.unwind_check),
        early_unwind_capture_fraction=min(1.0, max(0.0, a.unwind_capture)),
        early_unwind_enabled=not a.no_early_unwind, auto_figures=not a.no_figures,
        fast_recycle_enabled=not a.no_fast_recycle,
        fast_recycle_scan_seconds=max(10.0, a.fast_scan),
        fast_recycle_market_limit=max(10, a.fast_market_limit),
        multi_outcome_enabled=not a.no_multi_outcome,
        multi_outcome_scan_seconds=max(15.0, a.basket_scan),
        multi_outcome_max_days=max(1.0, a.basket_max_days),
        limitless_cross_enabled=not a.no_limitless_cross,
        limitless_cross_scan_seconds=max(30.0, a.limitless_scan),
        limitless_cross_max_days=max(0.25, a.limitless_max_days),
        optimized_quantity_sizing=not a.no_optimized_sizing,
        low_basis_enabled=not a.no_low_basis,
        low_basis_total_fraction=min(0.10, max(0.0, a.low_basis_cap)),
        low_basis_min_adjusted_apr=max(0.0, a.low_basis_min_apr),
        maker_probe_enabled=not a.no_maker_probes,
        maker_probe_max_active=max(0, a.maker_max_active),
        maker_probe_total_fraction=min(0.30, max(0.0, a.maker_total_cap)),
        maker_probe_ttl_seconds=max(30.0, a.maker_ttl),
    )

    portfolio, results, log, rejects = run_v8(ec, cc, cfg)
    print_summary(portfolio, results, log, rejects, cfg.run_minutes, cfg)
    if cfg.auto_figures:
        try:
            from src.reporting.paper_v8_figures import generate_figures
            generate_figures(log.root)
        except Exception as exc:
            print(f"Figure generation warning: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
