import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


D = Decimal


def dec(value):
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def floor_contracts(value):
    return int(dec(value).to_integral_value(rounding=ROUND_DOWN))


def parse_price_ranges(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    return value if isinstance(value, (list, tuple)) else []


def kalshi_tick_size(row, price):
    price = float(price)
    for item in parse_price_ranges(row.get("price_ranges")):
        try:
            start = float(item["start"])
            end = float(item["end"])
            step = float(item["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= price <= end:
            return step
    return 0.01


def choose_maker_price(best_bid, best_ask, tick_size, improve=True):
    if best_bid is None:
        return None
    best_bid = float(best_bid)
    tick_size = float(tick_size)
    if not improve or best_ask is None:
        return best_bid
    best_ask = float(best_ask)
    improved = round(best_bid + tick_size, 4)
    if improved < best_ask - 1e-12:
        return improved
    return best_bid


@dataclass
class DepthFill:
    quantity: float
    cost: float
    average_price: float
    worst_price: float
    fully_filled: bool


def consume_asks(levels, requested_quantity):
    """Consume ascending ask levels represented as {'price','size'} dicts."""
    requested = dec(requested_quantity)
    remaining = requested
    total_cost = D("0")
    worst = None

    normalized = sorted(
        (
            (dec(level["price"]), dec(level["size"]))
            for level in levels
            if dec(level.get("size", 0)) > 0
        ),
        key=lambda x: x[0],
    )

    for price, size in normalized:
        if remaining <= 0:
            break
        take = min(size, remaining)
        total_cost += take * price
        remaining -= take
        if take > 0:
            worst = price

    filled = requested - remaining
    avg = total_cost / filled if filled > 0 else D("0")
    return DepthFill(
        quantity=float(filled),
        cost=float(total_cost),
        average_price=float(avg),
        worst_price=float(worst) if worst is not None else 0.0,
        fully_filled=remaining <= D("0.00000001"),
    )
