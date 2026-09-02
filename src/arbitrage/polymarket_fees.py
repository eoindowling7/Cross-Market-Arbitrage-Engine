def polymarket_taker_fee(
    price,
    contracts=1,
    fee_rate=0.0,
):
    """
    Polymarket taker fee.

    Formula from current Polymarket documentation:

        fee = contracts * fee_rate * price * (1 - price)

    Fees are rounded to 5 decimal places.
    Makers pay zero.
    """

    if fee_rate is None:
        fee_rate = 0.0

    fee = (
        contracts
        * float(fee_rate)
        * float(price)
        * (1 - float(price))
    )

    return round(
        fee,
        5
    )


def get_polymarket_fee_rate(market):
    """Resolve the current market-specific Polymarket taker fee rate."""
    from src.arbitrage.polymarket_fee_policy import resolve_fee_rate
    return resolve_fee_rate(market)
