"""Read-only Limitless Exchange market-data adapter.

V8.5 follows the current public API contract documented by Limitless:
* ``GET /markets/active`` accepts at most 25 rows per request, so larger
  discovery requests are paginated locally.
* order books are read-only snapshots for exact CLOB market slugs.
* no trading/authenticated methods are implemented here.
"""
from __future__ import annotations

import math
from typing import Any

import requests

BASE_URL = "https://api.limitless.exchange"
_PAGE_CAP = 25
_HEADERS = {"User-Agent": "kalshi-market-engine-paper/8.5"}


def _extract_rows(payload: Any) -> list[dict]:
    """Tolerate the small response-shape changes seen across API revisions."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "markets", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            for nested in ("data", "markets", "items", "results"):
                rows = value.get(nested)
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
    return []


def _total_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("totalMarketsCount", "totalCount", "total", "count"):
        try:
            value = payload.get(key)
            if value is not None:
                return int(value)
        except Exception:
            pass
    data = payload.get("data")
    if isinstance(data, dict):
        return _total_count(data)
    return None


def get_active_markets(
    limit: int = 500,
    max_pages: int | None = None,
    trade_type: str | None = None,
    automation_type: str | None = None,
    include_next_market: bool = False,
):
    """Return up to ``limit`` active markets using the official 25-row cap.

    Limitless rejects ``limit>25`` with HTTP 400.  Previous engine versions
    accidentally sent 100, which made the venue appear empty.  V8.5 always
    pages in chunks of 25 and stops on the documented ``totalMarketsCount``
    when available.
    """
    wanted = max(0, int(limit))
    if wanted == 0:
        return []
    page_size = min(_PAGE_CAP, wanted)
    if max_pages is None:
        max_pages = max(1, math.ceil(wanted / page_size) + 2)

    out: list[dict] = []
    seen: set[str] = set()
    total_count: int | None = None

    for page in range(1, int(max_pages) + 1):
        params: dict[str, Any] = {"page": page, "limit": page_size}
        if trade_type:
            params["tradeType"] = trade_type
        if automation_type:
            params["automationType"] = automation_type
        if include_next_market:
            params["includeNextMarket"] = "true"

        r = requests.get(
            f"{BASE_URL}/markets/active",
            params=params,
            headers=_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
        rows = _extract_rows(payload)
        if total_count is None:
            total_count = _total_count(payload)
        if not rows:
            break

        for row in rows:
            ident = str(row.get("slug") or row.get("id") or row.get("marketId") or id(row))
            if ident in seen:
                continue
            seen.add(ident)
            out.append(row)
            if len(out) >= wanted:
                return out[:wanted]

        if total_count is not None and len(out) >= total_count:
            break
        if len(rows) < page_size:
            break

    return out[:wanted]


def get_market_details(slug: str):
    """Fetch one exact market so deadline/oracle metadata can be validated."""
    r = requests.get(
        f"{BASE_URL}/markets/{slug}",
        headers=_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def get_orderbook(slug: str):
    r = requests.get(
        f"{BASE_URL}/markets/{slug}/orderbook",
        headers=_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload
