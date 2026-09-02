import random
import time

import requests
import pandas as pd

from src.api.kalshi_client import BASE_URL


TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _retry_delay(response, attempt, base_delay=1.0):
    """Honor Retry-After when available, otherwise exponential backoff + jitter."""
    retry_after = None
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                retry_after = float(raw)
            except (TypeError, ValueError):
                retry_after = None
    if retry_after is not None:
        return max(0.0, min(retry_after, 60.0))
    return min(30.0, base_delay * (2 ** attempt)) + random.uniform(0.0, 0.25)


def get_open_markets(max_pages=None, limit=1000, *, retries=5, page_delay=0.20, session=None):
    """
    Download all currently open non-combo Kalshi markets.

    V28.4 hardens this pagination against Kalshi throttling:
    - retries the *same cursor* for 429/5xx/timeouts/connection failures,
    - honors Retry-After when present,
    - uses exponential backoff with jitter otherwise, and
    - gently paces successful pages to reduce repeated 429s.
    """

    all_markets = []
    cursor = None
    page = 0
    client = session or requests.Session()

    while True:
        page += 1

        if max_pages is not None and page > max_pages:
            break

        params = {
            "limit": limit,
            "status": "open",
            "mve_filter": "exclude",
        }
        if cursor:
            params["cursor"] = cursor

        response = None
        last_exc = None
        for attempt in range(max(0, int(retries)) + 1):
            try:
                response = client.get(f"{BASE_URL}/markets", params=params, timeout=30)
                if response.status_code in TRANSIENT_STATUSES:
                    if attempt >= retries:
                        response.raise_for_status()
                    delay = _retry_delay(response, attempt)
                    print(
                        f"Kalshi markets transient HTTP {response.status_code} on page {page} "
                        f"| retry {attempt + 1}/{retries} in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                last_exc = None
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt >= retries:
                    raise
                delay = min(30.0, 1.0 * (2 ** attempt)) + random.uniform(0.0, 0.25)
                print(
                    f"Kalshi markets transient {type(exc).__name__} on page {page} "
                    f"| retry {attempt + 1}/{retries} in {delay:.2f}s"
                )
                time.sleep(delay)
        else:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"Kalshi market page {page} failed after retries")

        data = response.json()
        page_markets = data.get("markets") or []
        all_markets.extend(page_markets)

        print(f"Page {page}: {len(page_markets)} markets | Total: {len(all_markets)}")

        cursor = data.get("cursor")
        if not cursor or len(page_markets) == 0:
            break

        if page_delay:
            time.sleep(max(0.0, float(page_delay)))

    markets = pd.DataFrame(all_markets)
    numeric_cols = [
        "yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp",
        "yes_ask_size_fp", "volume_fp", "volume_24h_fp",
    ]
    for col in numeric_cols:
        if col in markets.columns:
            markets[col] = pd.to_numeric(markets[col], errors="coerce")
    return markets
