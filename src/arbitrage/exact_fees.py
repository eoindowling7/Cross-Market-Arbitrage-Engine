import math


# ============================================================
# HELPERS
# ============================================================

def ceil_to_cent(value):
    """
    Round upward to the next cent.
    """

    if value <= 0:
        return 0.0

    return math.ceil(
        value * 100
    ) / 100


def ceil_to_centicent(value):
    """
    Round upward to the next $0.0001.
    Useful for examining the underlying trade fee
    before account-level cent alignment.
    """

    if value <= 0:
        return 0.0

    return math.ceil(
        value * 10000
    ) / 10000


# ============================================================
# KALSHI
# ============================================================

def kalshi_raw_quadratic_fee(
    price,
    contracts=1,
    rate=0.07,
    multiplier=1.0,
):
    """
    Raw quadratic Kalshi fee before cash-balance rounding.

        rate * multiplier * C * p * (1-p)
    """

    price = float(price)
    contracts = float(contracts)
    multiplier = float(multiplier)

    return (
        rate
        * multiplier
        * contracts
        * price
        * (1.0 - price)
    )


def kalshi_fee(
    price,
    contracts,
    fee_type,
    fee_multiplier=1.0,
    maker=False,
):
    """
    Estimate Kalshi fee using the series metadata.

    Supported:
        quadratic
        quadratic_with_maker_fees

    For maker orders:
        quadratic -> zero maker fee
        quadratic_with_maker_fees -> maker rate 0.0175

    For taker orders:
        quadratic / quadratic_with_maker_fees -> rate 0.07

    Returns both raw/economic fee and a conservative
    one-order cash fee rounded upward to the cent.
    """

    fee_type = str(
        fee_type or ""
    ).lower()

    fee_multiplier = float(
        fee_multiplier or 0
    )

    if fee_multiplier == 0:
        return {
            "raw_fee": 0.0,
            "trade_fee": 0.0,
            "cash_fee_upper": 0.0,
        }

    if maker:

        if fee_type == "quadratic":
            rate = 0.0

        elif fee_type == "quadratic_with_maker_fees":
            rate = 0.0175

        else:
            return None

    else:

        if fee_type in (
            "quadratic",
            "quadratic_with_maker_fees",
        ):
            rate = 0.07

        else:
            return None

    raw_fee = kalshi_raw_quadratic_fee(
        price=price,
        contracts=contracts,
        rate=rate,
        multiplier=fee_multiplier,
    )

    trade_fee = ceil_to_centicent(
        raw_fee
    )

    cash_fee_upper = ceil_to_cent(
        trade_fee
    )

    return {
        "raw_fee":
            raw_fee,

        "trade_fee":
            trade_fee,

        "cash_fee_upper":
            cash_fee_upper,
    }


# ============================================================
# POLYMARKET
# ============================================================

def get_polymarket_fee_rate(market):
    """Resolve the current market-specific Polymarket taker fee rate."""
    from src.arbitrage.polymarket_fee_policy import resolve_fee_rate
    return resolve_fee_rate(market)

def polymarket_taker_fee(
    price,
    contracts,
    market,
):
    """
    Current Polymarket fee model:

        C * feeRate * p * (1-p)

    Rounded to 5 decimals.
    """

    rate = get_polymarket_fee_rate(
        market
    )

    price = float(
        price
    )

    contracts = float(
        contracts
    )

    fee = (
        contracts
        * rate
        * price
        * (1.0 - price)
    )

    return round(
        fee,
        5
    )