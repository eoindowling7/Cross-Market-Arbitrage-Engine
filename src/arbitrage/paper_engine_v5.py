"""Portfolio-constrained cross-venue execution simulator (paper only).

V5 deliberately tries to make the apparent arbitrage harder to survive.
It never places orders.  It starts from V3-confirmed immediate (taker/taker)
opportunities, then applies:

- a real paper cash ledger; paired capital stays locked for the run
- per-market and per-event capital limits
- exact full-depth pricing and fee-aware signal gating
- one selected execution route per opportunity (the other route is not booked)
- randomized execution latency plus counterfactual latency stress tests
- partial-hedge accounting with conservative treatment of residual exposure
- no repeat trade until the executable book materially changes/replenishes
- quote-skew and minimum safety-buffer filters
- run-level loss / residual-exposure circuit breakers
- persistent signals, trades, equity, and stress-test logs

The simulator contains no order placement function and is safe for paper use.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.api.kalshi_client import get_market_orderbook, get_series_info
from src.api.polymarket_client import get_active_markets, get_orderbook, parse_token_ids
from src.arbitrage.exact_fees import kalshi_fee, polymarket_taker_fee
from src.arbitrage.execution_utils import consume_asks
from src.arbitrage.paper_engine_v2 import EngineConfig as V2Config, _kalshi_levels
from src.arbitrage.paper_engine_v3 import ConfirmationConfig, run_v3


SIGNAL_LOG = Path("data/paper_v5_signals.csv")
TRADE_LOG = Path("data/paper_v5_trades.csv")
EQUITY_LOG = Path("data/paper_v5_equity.csv")
STRESS_LOG = Path("data/paper_v5_latency_stress.csv")


@dataclass
class V5Config:
    run_minutes: float = 30.0
    poll_seconds: float = 10.0
    max_watch_candidates: int = 12

    # execution realism
    latency_mean_seconds: float = 0.25
    latency_jitter_seconds: float = 0.15
    latency_floor_seconds: float = 0.05
    latency_ceiling_seconds: float = 2.0
    stress_latencies: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00, 2.00)
    max_quote_skew_seconds: float = 0.50
    max_second_leg_move: float = 0.03

    # economics / safety
    minimum_signal_net_per_contract: float = 0.005
    minimum_execution_net_per_contract: float = 0.0025
    minimum_safety_buffer_per_contract: float = 0.0025
    max_run_loss: float = 5.0
    max_total_residual_contracts: float = 5.0

    # capital / concentration
    max_market_fraction: float = 0.25
    max_event_fraction: float = 0.35
    reserve_cash_fraction: float = 0.10
    hold_capital_until_resolution: bool = True

    # repeat / replenishment logic
    min_replenishment_contracts: float = 5.0
    min_replenishment_fraction: float = 0.50
    require_material_book_change: bool = True

    # route selection
    route_policy: str = "scarce_first"  # scarce_first | kalshi_first | poly_first
    random_seed: int = 42


@dataclass
class PaperPosition:
    trade_id: int
    timestamp: float
    ticker: str
    event_key: str
    subject: str
    strategy: str
    route: str
    quantity: int
    locked_capital: float
    conservative_pnl: float
    residual_unhedged: float


@dataclass
class PaperPortfolio:
    starting_bankroll: float
    reserve_cash_fraction: float
    max_market_fraction: float
    max_event_fraction: float
    available_cash: float = field(init=False)
    locked_capital: float = 0.0
    realized_pnl: float = 0.0
    residual_contracts: float = 0.0
    positions: list[PaperPosition] = field(default_factory=list)
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
        # Conservative paper equity: cash + locked principal + realized paper P&L.
        return self.available_cash + self.locked_capital + self.realized_pnl

    def max_capital_for(self, ticker: str, event_key: str) -> float:
        market_left = (
            self.starting_bankroll * self.max_market_fraction
            - self.market_capital.get(ticker, 0.0)
        )
        event_left = (
            self.starting_bankroll * self.max_event_fraction
            - self.event_capital.get(event_key, 0.0)
        )
        return max(0.0, min(self.deployable_cash, market_left, event_left))

    def book(self, position: PaperPosition) -> None:
        if position.locked_capital > self.available_cash + 1e-9:
            raise ValueError("attempted to lock more paper cash than available")
        self.available_cash -= position.locked_capital
        self.locked_capital += position.locked_capital
        self.realized_pnl += position.conservative_pnl
        self.residual_contracts += position.residual_unhedged
        self.positions.append(position)
        self.event_capital[position.event_key] = (
            self.event_capital.get(position.event_key, 0.0) + position.locked_capital
        )
        self.market_capital[position.ticker] = (
            self.market_capital.get(position.ticker, 0.0) + position.locked_capital
        )
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)


@dataclass
class ExecutionResult:
    status: str
    route: str
    quantity_requested: int
    first_leg_quantity: float
    second_leg_quantity: float
    hedged_quantity: float
    residual_unhedged: float
    first_leg_price: float | None
    second_leg_price: float | None
    first_leg_cost: float
    second_leg_cost: float
    kalshi_fee: float
    poly_fee: float
    conservative_pnl: float
    net_per_requested_contract: float
    locked_capital: float
    second_leg_price_move: float | None
    latency_seconds: float
    elapsed_seconds: float
    reason: str


def _series_ticker(ticker: str) -> str:
    return ticker.split("-")[0]


def _event_key(candidate: dict) -> str:
    # topic is intentionally conservative and already encodes the event family
    # for the validated universe (e.g. france_presidential_election).
    return str(candidate.get("topic") or candidate.get("ticker") or "unknown")


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _poly_asks(token_id: str) -> list[dict]:
    book = get_orderbook(token_id)
    return sorted(
        [
            {"price": float(x["price"]), "size": float(x["size"])}
            for x in book.get("asks", [])
            if float(x.get("size", 0)) > 0
        ],
        key=lambda x: x["price"],
    )


def _kalshi_taker_asks(ticker: str, kalshi_side: str) -> list[dict]:
    book = get_market_orderbook(ticker, depth=100)
    yes_bids = _kalshi_levels(book, "yes")
    no_bids = _kalshi_levels(book, "no")
    opposite = no_bids if kalshi_side == "yes" else yes_bids
    return sorted(
        ({"price": round(1.0 - x["price"], 4), "size": x["size"]} for x in opposite),
        key=lambda x: x["price"],
    )


def _strategy_sides(strategy: str) -> tuple[str, str]:
    if strategy.startswith("K YES"):
        return "yes", "no"
    if strategy.startswith("K NO"):
        return "no", "yes"
    raise ValueError(f"unsupported strategy: {strategy}")


def _best(levels: list[dict]) -> tuple[float | None, float]:
    if not levels:
        return None, 0.0
    return float(levels[0]["price"]), float(levels[0]["size"])


def _total_size(levels: list[dict]) -> float:
    return sum(float(x["size"]) for x in levels)


def _fee_pair(*, qty: int, kalshi_price: float, poly_price: float, series_info: dict, poly_market: dict) -> tuple[float, float]:
    kfee = kalshi_fee(
        price=kalshi_price,
        contracts=qty,
        fee_type=series_info.get("fee_type"),
        fee_multiplier=series_info.get("fee_multiplier") or 0,
        maker=False,
    )
    if kfee is None:
        raise ValueError("unknown Kalshi fee schedule")
    pfee = polymarket_taker_fee(poly_price, qty, poly_market)
    return float(kfee["cash_fee_upper"]), float(pfee)


def _signal_from_levels(candidate: dict, pm: dict, series_info: dict, qty: int, kasks: list[dict], pasks: list[dict], skew: float) -> dict | None:
    if qty < 1:
        return None
    kfill = consume_asks(kasks, qty)
    pfill = consume_asks(pasks, qty)
    if not kfill.fully_filled or not pfill.fully_filled:
        return None
    kfee, pfee = _fee_pair(
        qty=qty,
        kalshi_price=kfill.average_price,
        poly_price=pfill.average_price,
        series_info=series_info,
        poly_market=pm,
    )
    gross = qty - kfill.cost - pfill.cost
    net = gross - kfee - pfee
    capital = kfill.cost + pfill.cost + kfee + pfee
    kbest, kbest_size = _best(kasks)
    pbest, pbest_size = _best(pasks)
    return {
        "quantity": qty,
        "kalshi_best": kbest,
        "poly_best": pbest,
        "kalshi_best_size": kbest_size,
        "poly_best_size": pbest_size,
        "kalshi_total_size": _total_size(kasks),
        "poly_total_size": _total_size(pasks),
        "kalshi_avg": kfill.average_price,
        "poly_avg": pfill.average_price,
        "kalshi_worst": kfill.worst_price,
        "poly_worst": pfill.worst_price,
        "kalshi_fee": kfee,
        "poly_fee": pfee,
        "net_profit": net,
        "net_per_contract": net / qty,
        "capital": capital,
        "fetch_skew_seconds": skew,
    }


def _fresh_signal(candidate: dict, pm: dict, series_info: dict, max_qty: int, capital_limit: float, max_skew: float) -> tuple[dict | None, str]:
    """Fetch both books once and choose the largest safe quantity within capital."""
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
    if skew > max_skew:
        return None, f"quote skew {skew:.3f}s above {max_skew:.3f}s"
    if not kasks or not pasks:
        return None, "empty executable book"

    # Use candidate quantity as an upper bound, but shrink to fit the actual
    # portfolio and full book. Testing descending sizes avoids pretending a
    # top-of-book price applies to the whole order.
    max_qty = max(0, int(max_qty))
    for qty in range(max_qty, 0, -1):
        sig = _signal_from_levels(candidate, pm, series_info, qty, kasks, pasks, skew)
        if sig is None:
            continue
        if sig["capital"] <= capital_limit + 1e-9:
            return sig, "ok"
    return None, "no positive integer quantity fits depth/capital"


def _book_state(signal: dict) -> dict:
    return {
        "kalshi_best": round(float(signal["kalshi_best"]), 4),
        "poly_best": round(float(signal["poly_best"]), 4),
        "kalshi_best_size": float(signal["kalshi_best_size"]),
        "poly_best_size": float(signal["poly_best_size"]),
        "kalshi_total_size": float(signal["kalshi_total_size"]),
        "poly_total_size": float(signal["poly_total_size"]),
    }


def _materially_replenished(previous: dict | None, current: dict, qty: int, cfg: V5Config) -> bool:
    if previous is None:
        return True
    # A price change is a genuinely new executable state.
    if (
        current["kalshi_best"] != previous["kalshi_best"]
        or current["poly_best"] != previous["poly_best"]
    ):
        return True
    needed = max(cfg.min_replenishment_contracts, qty * cfg.min_replenishment_fraction)
    # Require actual observable depth replenishment, not merely a cooldown timer.
    k_growth = current["kalshi_total_size"] - previous["kalshi_total_size"]
    p_growth = current["poly_total_size"] - previous["poly_total_size"]
    return max(k_growth, p_growth) >= needed


def _sample_latency(cfg: V5Config, rng: random.Random) -> float:
    value = rng.gauss(cfg.latency_mean_seconds, cfg.latency_jitter_seconds)
    return min(cfg.latency_ceiling_seconds, max(cfg.latency_floor_seconds, value))


def _choose_route(signal: dict, cfg: V5Config) -> str:
    if cfg.route_policy in ("kalshi_first", "poly_first"):
        return cfg.route_policy
    # Secure the scarcer top-of-book leg first.  The larger book is then used
    # as the hedge.  This is a heuristic, but unlike V4 we book only one route.
    if signal["kalshi_best_size"] <= signal["poly_best_size"]:
        return "kalshi_first"
    return "poly_first"


def _partial_execution_pnl(
    *,
    route: str,
    requested: int,
    first_fill,
    second_fill,
    series_info: dict,
    pm: dict,
) -> tuple[float, float, float, float, float]:
    """Conservative P&L for full or partial hedge.

    Hedged contracts are worth $1 as a pair.  Any first-leg residual is marked
    to a worst-case terminal value of $0, so its entire acquisition cost is
    treated as a loss.  This intentionally makes partial hedges painful.
    """
    hedged = min(first_fill.quantity, second_fill.quantity)
    residual = max(0.0, first_fill.quantity - hedged)

    if route == "kalshi_first":
        kalshi_qty = first_fill.quantity
        poly_qty = second_fill.quantity
        kalshi_price = first_fill.average_price if first_fill.quantity else 0.0
        poly_price = second_fill.average_price if second_fill.quantity else 0.0
    else:
        poly_qty = first_fill.quantity
        kalshi_qty = second_fill.quantity
        poly_price = first_fill.average_price if first_fill.quantity else 0.0
        kalshi_price = second_fill.average_price if second_fill.quantity else 0.0

    if kalshi_qty > 0:
        kf = kalshi_fee(
            price=kalshi_price,
            contracts=kalshi_qty,
            fee_type=series_info.get("fee_type"),
            fee_multiplier=series_info.get("fee_multiplier") or 0,
            maker=False,
        )
        if kf is None:
            raise ValueError("unknown Kalshi fee schedule")
        kfee = float(kf["cash_fee_upper"])
    else:
        kfee = 0.0

    pfee = polymarket_taker_fee(poly_price, poly_qty, pm) if poly_qty > 0 else 0.0
    total_cost = first_fill.cost + second_fill.cost + kfee + pfee
    conservative_terminal_value = hedged  # unhedged first leg marked to $0
    pnl = conservative_terminal_value - total_cost
    return pnl, kfee, pfee, residual, total_cost


def simulate_selected_route(
    *,
    candidate: dict,
    pm: dict,
    series_info: dict,
    route: str,
    qty: int,
    latency_seconds: float,
    expected_second_price: float | None,
    max_second_leg_move: float,
) -> ExecutionResult:
    started = time.monotonic()
    kalshi_side, poly_side = _strategy_sides(candidate["strategy"])
    tokens = parse_token_ids(pm)
    if len(tokens) != 2:
        return ExecutionResult("ERROR", route, qty, 0, 0, 0, qty, None, None, 0, 0, 0, 0, -0.0, 0, 0, None, latency_seconds, 0, "invalid token ids")
    poly_token = tokens[0] if poly_side == "yes" else tokens[1]

    try:
        if route == "kalshi_first":
            first_levels = _kalshi_taker_asks(candidate["ticker"], kalshi_side)
            first_fill = consume_asks(first_levels, qty)
            if first_fill.quantity <= 0:
                return ExecutionResult("FIRST_LEG_NO_FILL", route, qty, 0, 0, 0, qty, None, None, 0, 0, 0, 0, 0, 0, 0, None, latency_seconds, time.monotonic()-started, "no Kalshi fill")
            time.sleep(latency_seconds)
            second_levels = _poly_asks(poly_token)
            second_fill = consume_asks(second_levels, first_fill.quantity)
            expected = expected_second_price
        else:
            first_levels = _poly_asks(poly_token)
            first_fill = consume_asks(first_levels, qty)
            if first_fill.quantity <= 0:
                return ExecutionResult("FIRST_LEG_NO_FILL", route, qty, 0, 0, 0, qty, None, None, 0, 0, 0, 0, 0, 0, 0, None, latency_seconds, time.monotonic()-started, "no Polymarket fill")
            time.sleep(latency_seconds)
            second_levels = _kalshi_taker_asks(candidate["ticker"], kalshi_side)
            second_fill = consume_asks(second_levels, first_fill.quantity)
            expected = expected_second_price

        pnl, kfee, pfee, residual, locked = _partial_execution_pnl(
            route=route,
            requested=qty,
            first_fill=first_fill,
            second_fill=second_fill,
            series_info=series_info,
            pm=pm,
        )
        hedged = min(first_fill.quantity, second_fill.quantity)
        second_price = second_fill.average_price if second_fill.quantity > 0 else None
        move = None if expected is None or second_price is None else second_price - expected

        if residual > 1e-9:
            status = "PARTIAL_HEDGE"
            reason = f"{residual:.4f} first-leg contracts remain unhedged; marked worst-case"
        elif pnl < 0:
            status = "SIMULATED_LOSS"
            reason = "fully hedged, but sequential execution was negative"
        elif move is not None and move > max_second_leg_move:
            status = "EXCESSIVE_HEDGE_MOVE"
            reason = f"hedge moved {move:.4f}, above limit {max_second_leg_move:.4f}"
        else:
            status = "SIMULATED_FILL"
            reason = "fully hedged"

        return ExecutionResult(
            status=status,
            route=route,
            quantity_requested=qty,
            first_leg_quantity=first_fill.quantity,
            second_leg_quantity=second_fill.quantity,
            hedged_quantity=hedged,
            residual_unhedged=residual,
            first_leg_price=first_fill.average_price if first_fill.quantity else None,
            second_leg_price=second_price,
            first_leg_cost=first_fill.cost,
            second_leg_cost=second_fill.cost,
            kalshi_fee=kfee,
            poly_fee=pfee,
            conservative_pnl=pnl,
            net_per_requested_contract=pnl / qty,
            locked_capital=locked,
            second_leg_price_move=move,
            latency_seconds=latency_seconds,
            elapsed_seconds=time.monotonic() - started,
            reason=reason,
        )
    except Exception as exc:
        return ExecutionResult("ERROR", route, qty, 0, 0, 0, qty, None, None, 0, 0, 0, 0, 0, 0, 0, None, latency_seconds, time.monotonic()-started, f"{type(exc).__name__}: {exc}")


def _stress_route(candidate: dict, pm: dict, series_info: dict, route: str, qty: int, signal: dict, latencies: tuple[float, ...]) -> list[dict]:
    """Counterfactual paper stress only; these results are never booked."""
    rows = []
    expected_second = signal["poly_avg"] if route == "kalshi_first" else signal["kalshi_avg"]
    for latency in latencies:
        rr = simulate_selected_route(
            candidate=candidate,
            pm=pm,
            series_info=series_info,
            route=route,
            qty=qty,
            latency_seconds=latency,
            expected_second_price=expected_second,
            max_second_leg_move=999.0,
        )
        rows.append({
            "timestamp": time.time(),
            "ticker": candidate["ticker"],
            "subject": candidate.get("subject"),
            "route": route,
            "quantity": qty,
            "latency_seconds": latency,
            "status": rr.status,
            "net_profit": rr.conservative_pnl,
            "net_per_contract": rr.net_per_requested_contract,
            "residual_unhedged": rr.residual_unhedged,
            "second_leg_price_move": rr.second_leg_price_move,
        })
    return rows


def _signal_log_row(candidate: dict, signal: dict | None, eligible: bool, reason: str, capital_limit: float) -> dict:
    return {
        "timestamp": time.time(),
        "ticker": candidate.get("ticker"),
        "subject": candidate.get("subject"),
        "topic": candidate.get("topic"),
        "strategy": candidate.get("strategy"),
        "quantity": None if signal is None else signal.get("quantity"),
        "capital_limit": capital_limit,
        "signal_capital": None if signal is None else signal.get("capital"),
        "signal_net": None if signal is None else signal.get("net_profit"),
        "signal_net_per_contract": None if signal is None else signal.get("net_per_contract"),
        "kalshi_best": None if signal is None else signal.get("kalshi_best"),
        "poly_best": None if signal is None else signal.get("poly_best"),
        "quote_skew_seconds": None if signal is None else signal.get("fetch_skew_seconds"),
        "eligible": eligible,
        "reason": reason,
    }


def _trade_log_row(trade_id: int, candidate: dict, signal: dict, rr: ExecutionResult, portfolio: PaperPortfolio) -> dict:
    return {
        "timestamp": time.time(),
        "trade_id": trade_id,
        "ticker": candidate.get("ticker"),
        "subject": candidate.get("subject"),
        "topic": candidate.get("topic"),
        "strategy": candidate.get("strategy"),
        "route": rr.route,
        "status": rr.status,
        "quantity_requested": rr.quantity_requested,
        "first_leg_quantity": rr.first_leg_quantity,
        "second_leg_quantity": rr.second_leg_quantity,
        "hedged_quantity": rr.hedged_quantity,
        "residual_unhedged": rr.residual_unhedged,
        "signal_net_per_contract": signal.get("net_per_contract"),
        "first_leg_price": rr.first_leg_price,
        "second_leg_price": rr.second_leg_price,
        "second_leg_price_move": rr.second_leg_price_move,
        "latency_seconds": rr.latency_seconds,
        "kalshi_fee": rr.kalshi_fee,
        "poly_fee": rr.poly_fee,
        "locked_capital": rr.locked_capital,
        "conservative_pnl": rr.conservative_pnl,
        "net_per_requested_contract": rr.net_per_requested_contract,
        "portfolio_available_cash": portfolio.available_cash,
        "portfolio_locked_capital": portfolio.locked_capital,
        "portfolio_realized_pnl": portfolio.realized_pnl,
        "portfolio_equity": portfolio.equity,
        "reason": rr.reason,
    }


def _equity_row(portfolio: PaperPortfolio, cycle: int, elapsed_minutes: float) -> dict:
    return {
        "timestamp": time.time(),
        "cycle": cycle,
        "elapsed_minutes": elapsed_minutes,
        "available_cash": portfolio.available_cash,
        "locked_capital": portfolio.locked_capital,
        "realized_pnl": portfolio.realized_pnl,
        "equity": portfolio.equity,
        "open_positions": len(portfolio.positions),
        "residual_contracts": portfolio.residual_contracts,
        "max_drawdown": portfolio.max_drawdown,
    }


def run_v5(engine_config: V2Config, confirmation_config: ConfirmationConfig, cfg: V5Config):
    print("Building and confirming immediate paper universe...")
    _, confirmations, _ = run_v3(engine_config, confirmation_config)
    confirmed = [x for x in confirmations if x.get("confirmation_status") == "CONFIRMED"]
    confirmed.sort(key=lambda x: x.get("worst_net_profit", -999), reverse=True)
    confirmed = confirmed[: cfg.max_watch_candidates]
    print(f"Confirmed watch candidates: {len(confirmed)}")
    if not confirmed:
        return PaperPortfolio(engine_config.bankroll, cfg.reserve_cash_fraction, cfg.max_market_fraction, cfg.max_event_fraction), []

    print("Refreshing Polymarket metadata once for V5 simulation...")
    polymarket = get_active_markets(limit=None)
    by_question = {str(m.get("question")): m for m in polymarket}
    series_cache: dict[str, dict] = {}
    watch = []
    for c in confirmed:
        pm = by_question.get(c.get("poly_question"))
        if pm is None:
            continue
        series = _series_ticker(c["ticker"])
        if series not in series_cache:
            series_cache[series] = get_series_info(series)
        watch.append((c, pm, series_cache[series]))

    portfolio = PaperPortfolio(
        starting_bankroll=engine_config.bankroll,
        reserve_cash_fraction=cfg.reserve_cash_fraction,
        max_market_fraction=cfg.max_market_fraction,
        max_event_fraction=cfg.max_event_fraction,
    )
    rng = random.Random(cfg.random_seed)
    previous_executed_state: dict[tuple, dict] = {}
    results: list[dict] = []
    trade_id = 0
    started = time.monotonic()
    deadline = started + max(0.0, cfg.run_minutes) * 60.0
    cycle = 0

    print(
        f"Watching {len(watch)} confirmed candidates for {cfg.run_minutes:g} minutes | "
        f"poll={cfg.poll_seconds:g}s | cash=${portfolio.starting_bankroll:.2f} | "
        f"reserve={cfg.reserve_cash_fraction:.0%} | event cap={cfg.max_event_fraction:.0%}"
    )

    while time.monotonic() < deadline:
        cycle += 1
        cycle_start = time.monotonic()
        eligible = 0

        # Highest confirmed edge first so scarce paper capital goes to the best
        # validated opportunities rather than dataframe order.
        for candidate, pm, series_info in watch:
            ticker = candidate["ticker"]
            event_key = _event_key(candidate)
            key = (ticker, candidate["strategy"])
            capital_limit = portfolio.max_capital_for(ticker, event_key)
            if capital_limit < 0.50:
                _append_csv(SIGNAL_LOG, _signal_log_row(candidate, None, False, "portfolio/event capital limit", capital_limit))
                continue

            signal, signal_reason = _fresh_signal(
                candidate,
                pm,
                series_info,
                max_qty=int(candidate["quantity"]),
                capital_limit=capital_limit,
                max_skew=cfg.max_quote_skew_seconds,
            )
            if signal is None:
                _append_csv(SIGNAL_LOG, _signal_log_row(candidate, None, False, signal_reason, capital_limit))
                continue

            # Require a real cushion above the execution minimum so a tiny price
            # move or rounding change does not turn the trade negative.
            required_signal_edge = (
                cfg.minimum_execution_net_per_contract
                + cfg.minimum_safety_buffer_per_contract
            )
            if signal["net_per_contract"] < max(cfg.minimum_signal_net_per_contract, required_signal_edge):
                _append_csv(SIGNAL_LOG, _signal_log_row(candidate, signal, False, "edge lacks safety buffer", capital_limit))
                continue

            state = _book_state(signal)
            if cfg.require_material_book_change and not _materially_replenished(
                previous_executed_state.get(key), state, int(signal["quantity"]), cfg
            ):
                _append_csv(SIGNAL_LOG, _signal_log_row(candidate, signal, False, "book not materially replenished/changed", capital_limit))
                continue

            route = _choose_route(signal, cfg)
            latency = _sample_latency(cfg, rng)
            expected_second = signal["poly_avg"] if route == "kalshi_first" else signal["kalshi_avg"]
            eligible += 1

            rr = simulate_selected_route(
                candidate=candidate,
                pm=pm,
                series_info=series_info,
                route=route,
                qty=int(signal["quantity"]),
                latency_seconds=latency,
                expected_second_price=expected_second,
                max_second_leg_move=cfg.max_second_leg_move,
            )

            # Counterfactual latency ladder is diagnostic only; never book it.
            for stress in _stress_route(
                candidate, pm, series_info, route, int(signal["quantity"]), signal, cfg.stress_latencies
            ):
                _append_csv(STRESS_LOG, stress)

            if rr.status == "FIRST_LEG_NO_FILL" or rr.status == "ERROR":
                row = _trade_log_row(trade_id + 1, candidate, signal, rr, portfolio)
                results.append(row)
                _append_csv(TRADE_LOG, row)
                previous_executed_state[key] = state
                continue

            # If actual sequential execution erodes below the hard minimum, we
            # still book the result because leg one already occurred in the
            # simulation.  This is exactly the adverse-selection risk we need to see.
            trade_id += 1
            position = PaperPosition(
                trade_id=trade_id,
                timestamp=time.time(),
                ticker=ticker,
                event_key=event_key,
                subject=str(candidate.get("subject")),
                strategy=str(candidate.get("strategy")),
                route=route,
                quantity=int(signal["quantity"]),
                locked_capital=rr.locked_capital,
                conservative_pnl=rr.conservative_pnl,
                residual_unhedged=rr.residual_unhedged,
            )

            # The pre-trade capital estimator is conservative but execution can
            # move.  If the actual required capital exceeds available cash, book
            # no imaginary position; record it as a portfolio rejection.
            if rr.locked_capital > portfolio.available_cash + 1e-9:
                rr.status = "PORTFOLIO_REJECT_AFTER_MOVE"
                rr.reason = "sequential price move pushed required capital above available cash"
            else:
                portfolio.book(position)
                previous_executed_state[key] = state

            row = _trade_log_row(trade_id, candidate, signal, rr, portfolio)
            results.append(row)
            _append_csv(TRADE_LOG, row)

            print(
                f"SIM {candidate['subject']} | {route} | qty={signal['quantity']} | "
                f"lat={latency:.3f}s | {rr.status} | net=${rr.conservative_pnl:.4f} | "
                f"cash=${portfolio.available_cash:.2f} | locked=${portfolio.locked_capital:.2f}"
            )

            if portfolio.realized_pnl <= -abs(cfg.max_run_loss):
                print("Run-loss circuit breaker reached; ending V5 simulation.")
                _append_csv(EQUITY_LOG, _equity_row(portfolio, cycle, (time.monotonic()-started)/60))
                return portfolio, results
            if portfolio.residual_contracts > cfg.max_total_residual_contracts:
                print("Residual-exposure circuit breaker reached; ending V5 simulation.")
                _append_csv(EQUITY_LOG, _equity_row(portfolio, cycle, (time.monotonic()-started)/60))
                return portfolio, results

        elapsed_minutes = (time.monotonic() - started) / 60.0
        _append_csv(EQUITY_LOG, _equity_row(portfolio, cycle, elapsed_minutes))
        print(
            f"Cycle {cycle} | elapsed={elapsed_minutes:.1f}m | eligible={eligible} | "
            f"trades={len(portfolio.positions)} | cash=${portfolio.available_cash:.2f} | "
            f"locked=${portfolio.locked_capital:.2f} | P&L=${portfolio.realized_pnl:.4f} | "
            f"residual={portfolio.residual_contracts:.2f}"
        )

        sleep_for = cfg.poll_seconds - (time.monotonic() - cycle_start)
        if sleep_for > 0 and time.monotonic() < deadline:
            time.sleep(min(sleep_for, max(0.0, deadline - time.monotonic())))

    return portfolio, results


def print_summary(portfolio: PaperPortfolio, results: list[dict], run_minutes: float) -> None:
    print("\n" + "=" * 90)
    print("PAPER ENGINE V5 - PORTFOLIO / EXECUTION SUMMARY")
    print("=" * 90)
    unique_markets = len({p.ticker for p in portfolio.positions})
    unique_events = len({p.event_key for p in portfolio.positions})
    wins = [p for p in portfolio.positions if p.conservative_pnl > 0]
    losses = [p for p in portfolio.positions if p.conservative_pnl <= 0]
    partials = [x for x in results if x.get("status") == "PARTIAL_HEDGE"]
    failures = [x for x in results if x.get("status") in ("ERROR", "FIRST_LEG_NO_FILL")]
    total = portfolio.realized_pnl
    return_pct = total / portfolio.starting_bankroll if portfolio.starting_bankroll else 0.0
    hours = max(run_minutes / 60.0, 1e-9)

    print(f"Starting bankroll:       ${portfolio.starting_bankroll:.2f}")
    print(f"Available cash:          ${portfolio.available_cash:.2f}")
    print(f"Locked capital:          ${portfolio.locked_capital:.2f}")
    print(f"Conservative paper P&L:  ${total:.4f}")
    print(f"Paper return:            {return_pct:.2%}")
    print(f"Profit/hour (sample):    ${total / hours:.4f}")
    print(f"Booked trades:           {len(portfolio.positions)}")
    print(f"Unique markets/events:   {unique_markets} / {unique_events}")
    print(f"Positive / nonpositive:  {len(wins)} / {len(losses)}")
    print(f"Partial hedges:          {len(partials)}")
    print(f"Execution failures:      {len(failures)}")
    print(f"Residual contracts:      {portfolio.residual_contracts:.4f}")
    print(f"Max drawdown:            ${portfolio.max_drawdown:.4f}")
    print(f"Capital utilization:     {portfolio.locked_capital / portfolio.starting_bankroll:.2%}")

    if portfolio.positions:
        print("\nBooked paper positions:")
        for p in portfolio.positions:
            print(
                f"#{p.trade_id:02d} {p.subject} | {p.route} | qty={p.quantity} | "
                f"locked=${p.locked_capital:.2f} | net=${p.conservative_pnl:.4f} | "
                f"residual={p.residual_unhedged:.2f}"
            )

    print("\nLatency stress data are counterfactual only and are never added to P&L.")
    print(f"Logs: {SIGNAL_LOG} | {TRADE_LOG} | {EQUITY_LOG} | {STRESS_LOG}")
    print("Paper only: V5 contains no order-placement call.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--min-edge", type=float, default=0.0025)
    parser.add_argument("--safety-buffer", type=float, default=0.0025)
    parser.add_argument("--confirmations", type=int, default=5)
    parser.add_argument("--confirm-delay", type=float, default=0.75)
    parser.add_argument("--confirm-top", type=int, default=12)
    parser.add_argument("--latency-mean", type=float, default=0.25)
    parser.add_argument("--latency-jitter", type=float, default=0.15)
    parser.add_argument("--max-skew", type=float, default=0.50)
    parser.add_argument("--event-cap", type=float, default=0.35)
    parser.add_argument("--market-cap", type=float, default=0.25)
    parser.add_argument("--reserve-cash", type=float, default=0.10)
    parser.add_argument("--max-run-loss", type=float, default=5.0)
    parser.add_argument("--max-residual", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine_config = V2Config(
        bankroll=max(1.0, args.bankroll),
        min_net_per_contract=max(0.0, args.min_edge),
        max_quote_skew_seconds=max(0.01, args.max_skew),
        allow_taker_taker=True,
    )
    confirmation_config = ConfirmationConfig(
        samples=max(1, args.confirmations),
        delay_seconds=max(0.0, args.confirm_delay),
        max_candidates=max(1, args.confirm_top),
    )
    cfg = V5Config(
        run_minutes=max(0.0, args.minutes),
        poll_seconds=max(1.0, args.poll),
        max_watch_candidates=max(1, args.confirm_top),
        latency_mean_seconds=max(0.0, args.latency_mean),
        latency_jitter_seconds=max(0.0, args.latency_jitter),
        max_quote_skew_seconds=max(0.01, args.max_skew),
        minimum_execution_net_per_contract=max(0.0, args.min_edge),
        minimum_signal_net_per_contract=max(0.0, args.min_edge + args.safety_buffer),
        minimum_safety_buffer_per_contract=max(0.0, args.safety_buffer),
        max_event_fraction=min(1.0, max(0.01, args.event_cap)),
        max_market_fraction=min(1.0, max(0.01, args.market_cap)),
        reserve_cash_fraction=min(0.95, max(0.0, args.reserve_cash)),
        max_run_loss=max(0.01, args.max_run_loss),
        max_total_residual_contracts=max(0.0, args.max_residual),
        random_seed=args.seed,
    )

    portfolio, results = run_v5(engine_config, confirmation_config, cfg)
    print_summary(portfolio, results, cfg.run_minutes)


if __name__ == "__main__":
    main()
