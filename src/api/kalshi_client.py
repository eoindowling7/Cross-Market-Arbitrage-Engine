import requests


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

def get_series_info(series_ticker):
    response = requests.get(
        f"{BASE_URL}/series/{series_ticker}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()["series"]



def get_market_details(ticker):
    """Return public metadata for one Kalshi market.

    Used lazily by the final matcher when event-list metadata does not contain
    enough rule text to prove cross-venue settlement equivalence.  Read-only;
    this endpoint never places or changes an order.
    """
    response = requests.get(
        f"{BASE_URL}/markets/{ticker}",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    market = payload.get("market") if isinstance(payload, dict) else None
    return market if isinstance(market, dict) else (payload if isinstance(payload, dict) else {})

def get_market_quotes(ticker, depth=20):
    """
    Fetch the Kalshi order book for a market.

    Returns the best YES and NO bid/ask prices,
    together with the available quantity at each price.
    """

    url = f"{BASE_URL}/markets/{ticker}/orderbook"

    response = requests.get(
        url,
        params={"depth": depth},
        timeout=30
    )

    response.raise_for_status()

    orderbook = response.json()["orderbook_fp"]

    yes_levels = orderbook.get("yes_dollars", [])
    no_levels = orderbook.get("no_dollars", [])

    # Best YES bid
    if yes_levels:
        best_yes = max(
            yes_levels,
            key=lambda x: float(x[0])
        )

        yes_bid = float(best_yes[0])
        yes_bid_size = float(best_yes[1])

    else:
        yes_bid = None
        yes_bid_size = 0.0

    # Best NO bid
    if no_levels:
        best_no = max(
            no_levels,
            key=lambda x: float(x[0])
        )

        no_bid = float(best_no[0])
        no_bid_size = float(best_no[1])

    else:
        no_bid = None
        no_bid_size = 0.0

    # Opposite-side bids imply asks
    yes_ask = (
        round(1 - no_bid, 4)
        if no_bid is not None
        else None
    )

    no_ask = (
        round(1 - yes_bid, 4)
        if yes_bid is not None
        else None
    )

    return {
        "ticker": ticker,

        "yes_bid": yes_bid,
        "yes_bid_size": yes_bid_size,
        "yes_ask": yes_ask,
        "yes_ask_size": no_bid_size,

        "no_bid": no_bid,
        "no_bid_size": no_bid_size,
        "no_ask": no_ask,
        "no_ask_size": yes_bid_size
    }

def get_market_orderbook(ticker, depth=100):
    """Return the raw fixed-point Kalshi order book as normalized dollar levels."""
    response = requests.get(
        f"{BASE_URL}/markets/{ticker}/orderbook",
        params={"depth": depth},
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json().get("orderbook_fp", {})
    return {
        "yes_dollars": [
            [float(price), float(size)]
            for price, size in raw.get("yes_dollars", [])
        ],
        "no_dollars": [
            [float(price), float(size)]
            for price, size in raw.get("no_dollars", [])
        ],
    }


def get_market_trades(ticker, *, min_ts=None, max_ts=None, limit=1000, cursor=None, is_block_trade=False):
    """Return recent public trades for one Kalshi market.

    The public REST trade feed is used by the final paper engine to validate
    hypothetical maker fills.  It never submits or modifies an order.
    """
    params = {"ticker": str(ticker), "limit": max(1, min(1000, int(limit)))}
    if min_ts is not None:
        params["min_ts"] = int(min_ts)
    if max_ts is not None:
        params["max_ts"] = int(max_ts)
    if cursor:
        params["cursor"] = str(cursor)
    if is_block_trade is not None:
        params["is_block_trade"] = str(bool(is_block_trade)).lower()
    response = requests.get(
        f"{BASE_URL}/markets/trades",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("trades", []) or [], payload.get("cursor") or ""
