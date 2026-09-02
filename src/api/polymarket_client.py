import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests import exceptions as request_exceptions


GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# V8.6: Polymarket's legacy offset feed repeatedly ended around ~2,100 rows in
# our live tests.  The current official Gamma docs expose cursor/keyset
# pagination specifically for browsing the complete active market feed.  We use
# the old feed as a detailed baseline, enumerate the rest with keyset cursors,
# and hydrate only the additional market IDs.  A short-lived disk cache avoids
# doing thousands of detail requests multiple times during one engine startup.
EXPANDED_CACHE = Path("data/cache/polymarket_expanded_active.json")
DEFAULT_CACHE_TTL_SECONDS = 20 * 60
DEFAULT_MAX_HYDRATE = 20000
DEFAULT_HYDRATE_WORKERS = 24


def _flat_active_markets(limit=None):
    """Legacy detailed /markets offset feed.

    Kept as a baseline and fallback because its records include the rich Gamma
    metadata used by the matcher.  V8.6 no longer assumes this feed is the full
    active universe.
    """
    all_markets = []
    page_size = 100
    offset = 0

    while True:
        params = {"closed": "false", "limit": page_size, "offset": offset}
        response = requests.get(f"{GAMMA_URL}/markets", params=params, timeout=30)
        if response.status_code == 422:
            print(f"Reached end of Polymarket flat pagination at offset {offset}.")
            break
        response.raise_for_status()
        markets = response.json()
        if not markets:
            break
        all_markets.extend(markets)
        print(
            f"Polymarket flat offset {offset}: {len(markets)} markets | "
            f"total {len(all_markets)}"
        )
        if len(markets) < page_size:
            break
        if limit is not None and len(all_markets) >= limit:
            break
        offset += page_size

    return all_markets[:limit] if limit is not None else all_markets


def _request_keyset_page(after_cursor=None, page_size=100, retries=4):
    """Fetch one cursor page with bounded transient-error retries.

    V28.1 keeps the same cursor when a request fails.  An SSL EOF, timeout,
    connection reset, 429, or 5xx response is therefore retried in-place
    instead of destroying a nearly-complete 180k-market discovery pass.
    """
    sizes = []
    for size in (int(page_size), 50, 20):
        if size > 0 and size not in sizes:
            sizes.append(size)
    last = None
    transient_statuses = {429, 500, 502, 503, 504}
    transient_exceptions = (
        request_exceptions.SSLError,
        request_exceptions.Timeout,
        request_exceptions.ConnectionError,
    )
    for size in sizes:
        params = {"closed": "false", "limit": size}
        if after_cursor:
            params["after_cursor"] = after_cursor
        for attempt in range(max(1, int(retries))):
            try:
                response = requests.get(f"{GAMMA_URL}/markets/keyset", params=params, timeout=30)
                last = response
                if response.status_code in (400, 422) and size != sizes[-1]:
                    break
                if response.status_code in transient_statuses:
                    if attempt + 1 < retries:
                        delay = min(8.0, 0.75 * (2 ** attempt))
                        print(
                            f"Polymarket keyset transient HTTP {response.status_code}; "
                            f"retry {attempt + 1}/{retries - 1} in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload, None, size
                markets = payload.get("markets") or payload.get("items") or []
                cursor = payload.get("next_cursor") or payload.get("nextCursor")
                return markets, cursor, size
            except transient_exceptions as exc:
                if attempt + 1 >= retries:
                    raise
                delay = min(8.0, 0.75 * (2 ** attempt))
                print(
                    f"Polymarket keyset transient {type(exc).__name__}; "
                    f"retry {attempt + 1}/{retries - 1} in {delay:.1f}s"
                )
                time.sleep(delay)
        # 400/422 may indicate the requested page size is unsupported.
        if last is not None and last.status_code in (400, 422):
            continue
    if last is not None:
        last.raise_for_status()
    return [], None, page_size


def get_active_market_stubs_keyset(max_markets=None, page_size=100):
    """Enumerate active-market stubs using Gamma's cursor/keyset endpoint.

    This is the coverage path recommended by the current public Polymarket
    market-discovery documentation.  It has no dependency on trading auth.
    """
    out = []
    seen_ids = set()
    cursor = None
    page = 0
    seen_cursors = set()

    while True:
        rows, next_cursor, used_size = _request_keyset_page(cursor, page_size)
        page += 1
        for row in rows or []:
            market_id = str(row.get("id") or "")
            if not market_id or market_id in seen_ids:
                continue
            seen_ids.add(market_id)
            out.append(row)
            if max_markets is not None and len(out) >= max_markets:
                return out[:max_markets]

        if page == 1 or page % 10 == 0 or not next_cursor:
            print(
                f"Polymarket keyset page {page}: {len(rows or [])} markets | "
                f"unique {len(out)} | page_size={used_size}"
            )
        if not rows or not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return out


def get_market_by_id(market_id):
    response = requests.get(f"{GAMMA_URL}/markets/{market_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def _is_rich_market(row):
    return bool(
        row.get("clobTokenIds")
        and (row.get("endDate") or row.get("endDateIso"))
        and ("enableOrderBook" in row or "acceptingOrders" in row)
    )


def _market_is_usable(row):
    if not isinstance(row, dict):
        return False
    if row.get("closed") is True:
        return False
    if row.get("active") is False:
        return False
    # Discovery intentionally retains markets whose acceptingOrders field is
    # absent; detailed order-book checks later decide executability.
    return bool(row.get("id") and row.get("question"))


def _read_expanded_cache(ttl_seconds=None, *, allow_stale=False):
    """Read the last-known-good expanded universe.

    ``allow_stale=True`` is deliberately used only as a resilience fallback:
    an older 180k-market snapshot is safer than replacing it with the ~2.1k
    legacy flat feed after a transient Gamma failure.
    """
    try:
        if not EXPANDED_CACHE.exists():
            return None
        payload = json.loads(EXPANDED_CACHE.read_text(encoding="utf-8"))
        created = float(payload.get("created_at") or 0)
        if not allow_stale and ttl_seconds is not None:
            if time.time() - created > float(ttl_seconds):
                return None
        rows = payload.get("markets") or []
        return rows if isinstance(rows, list) and rows else None
    except Exception:
        return None


def _write_expanded_cache(markets, diagnostics):
    try:
        EXPANDED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = EXPANDED_CACHE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"created_at": time.time(), "diagnostics": diagnostics, "markets": markets},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(EXPANDED_CACHE)
        try:
            import csv
            report = Path("data/processed/polymarket_coverage_latest.csv")
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(diagnostics.keys()))
                writer.writeheader()
                writer.writerow(diagnostics)
        except Exception:
            pass
    except Exception as exc:
        print(f"Polymarket expanded-cache warning: {type(exc).__name__}: {exc}")


def get_active_markets_expanded(
    *,
    force_refresh=False,
    cache_ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
    max_hydrate=None,
    hydrate_workers=DEFAULT_HYDRATE_WORKERS,
):
    """Return a broad, rich active Polymarket universe.

    Pipeline:
      1. use the legacy detailed offset feed as a baseline;
      2. enumerate the complete public active feed with keyset pagination;
      3. dedupe by market ID;
      4. hydrate only IDs not already represented by a rich baseline record;
      5. cache the rich result briefly for repeated engine startup/refresh calls.

    The function fails safe: if keyset discovery is unavailable, it returns the
    detailed flat baseline instead of breaking the paper engine.
    """
    if os.getenv("POLYMARKET_DISCOVERY_MODE", "expanded").lower() == "flat":
        return _flat_active_markets(limit=None)

    # Always retain a last-known-good snapshot in memory for this refresh.
    # It may be older than the normal TTL, but it is used only when discovery
    # fails or produces an implausibly small replacement universe.
    stale_cache = _read_expanded_cache(cache_ttl_seconds, allow_stale=True)
    if not force_refresh:
        cached = _read_expanded_cache(cache_ttl_seconds)
        if cached:
            print(f"Polymarket expanded cache: {len(cached)} active markets")
            return cached

    baseline = _flat_active_markets(limit=None)
    by_id = {str(m.get("id")): dict(m) for m in baseline if m.get("id") is not None}

    try:
        stubs = get_active_market_stubs_keyset()
    except Exception as exc:
        if stale_cache and len(stale_cache) > len(baseline):
            print(
                "Polymarket keyset discovery warning: "
                f"{type(exc).__name__}: {exc} | preserving last-known-good "
                f"expanded cache ({len(stale_cache)} markets; flat={len(baseline)})"
            )
            return stale_cache
        print(
            "Polymarket keyset discovery warning: "
            f"{type(exc).__name__}: {exc} | no expanded cache available; using flat baseline"
        )
        return baseline

    keyset_ids = {str(m.get("id")) for m in stubs if m.get("id") is not None}
    additional = [m for m in stubs if str(m.get("id")) not in by_id]
    rich_from_keyset = 0
    to_hydrate = []
    for stub in additional:
        mid = str(stub.get("id"))
        if _is_rich_market(stub):
            by_id[mid] = dict(stub)
            rich_from_keyset += 1
        else:
            to_hydrate.append(stub)

    env_cap = os.getenv("POLYMARKET_MAX_HYDRATE")
    if max_hydrate is None:
        max_hydrate = int(env_cap) if env_cap else DEFAULT_MAX_HYDRATE
    if max_hydrate is not None:
        to_hydrate = to_hydrate[: max(0, int(max_hydrate))]

    print(
        "POLYMARKET COVERAGE | "
        f"flat={len(baseline)} keyset={len(keyset_ids)} "
        f"additional={len(additional)} hydrate={len(to_hydrate)}"
    )

    hydrated = 0
    hydrate_failures = 0
    workers = max(1, int(hydrate_workers))
    if to_hydrate:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(get_market_by_id, str(stub.get("id"))): stub for stub in to_hydrate}
            for idx, future in enumerate(as_completed(futures), start=1):
                stub = futures[future]
                mid = str(stub.get("id"))
                try:
                    detail = future.result()
                    # Merge the stub first because some list-level fields can be
                    # useful even if the detail endpoint omits them.
                    merged = {**stub, **detail}
                    if _market_is_usable(merged):
                        merged["_discovery_source"] = "keyset_hydrated"
                        by_id[mid] = merged
                        hydrated += 1
                except Exception:
                    hydrate_failures += 1
                if idx % 500 == 0 or idx == len(to_hydrate):
                    print(
                        f"Polymarket hydrate {idx}/{len(to_hydrate)} | "
                        f"success={hydrated} fail={hydrate_failures}"
                    )

    markets = [m for m in by_id.values() if _market_is_usable(m)]
    # Keep active, accepting-order markets first, then short-horizon and liquid
    # markets.  This improves the usefulness of capped downstream scans without
    # dropping the long tail from the matcher.
    def _sort_key(m):
        accepting = 1 if m.get("acceptingOrders") is True else 0
        active = 1 if m.get("active") is not False else 0
        try:
            end = str(m.get("endDate") or m.get("endDateIso") or "9999")
        except Exception:
            end = "9999"
        try:
            volume = float(m.get("volume24hr") or 0)
        except Exception:
            volume = 0.0
        return (-accepting, -active, end, -volume)

    markets.sort(key=_sort_key)

    # Catastrophic-downgrade guard.  A successful refresh should never replace
    # a known expanded universe with a tiny subset because of an upstream
    # pagination anomaly.  Preserve the old cache unless the new result is at
    # least half its size (and never accept a flat-sized ~2k replacement for a
    # six-figure cache).
    if stale_cache and len(stale_cache) >= 10000:
        min_safe = max(10000, int(len(stale_cache) * 0.50))
        if len(markets) < min_safe:
            print(
                "Polymarket refresh downgrade guard | "
                f"new={len(markets)} old={len(stale_cache)} minimum={min_safe} | "
                "preserving last-known-good expanded cache"
            )
            return stale_cache

    diagnostics = {
        "flat": len(baseline),
        "keyset": len(keyset_ids),
        "additional": len(additional),
        "rich_from_keyset": rich_from_keyset,
        "hydrated": hydrated,
        "hydrate_failures": hydrate_failures,
        "unique_rich_active": len(markets),
    }
    print(
        "POLYMARKET COVERAGE FINAL | "
        f"unique_active={len(markets)} gained_vs_flat={max(0, len(markets)-len(baseline))} "
        f"hydrated={hydrated} failures={hydrate_failures}"
    )
    _write_expanded_cache(markets, diagnostics)
    return markets


def get_active_markets(limit=None):
    """Fetch active Polymarket markets.

    Small explicit limits retain the lightweight legacy path.  Full-universe
    calls (``limit=None``), which are what the paper engine uses, now invoke the
    V8.6 keyset-expanded discovery pipeline.
    """
    if limit is not None:
        return _flat_active_markets(limit=limit)
    return get_active_markets_expanded()


def parse_token_ids(market):
    """
    Return Polymarket CLOB token IDs.

    For binary Yes/No markets, always normalize the return order to:

        [YES_token_id, NO_token_id]

    using the corresponding `outcomes` array rather than assuming the
    API always supplies the tokens in Yes/No order.

    For non-Yes/No or multi-outcome markets, preserve the API order.
    """
    token_ids = market.get("clobTokenIds")

    if isinstance(token_ids, str):
        try:
            token_ids = json.loads(token_ids)
        except Exception:
            token_ids = []

    if not isinstance(token_ids, list):
        return []

    outcomes = market.get("outcomes")

    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []

    if (
        isinstance(outcomes, list)
        and len(outcomes) == 2
        and len(token_ids) == 2
    ):
        normalized = [
            str(x).strip().lower()
            for x in outcomes
        ]

        if set(normalized) == {"yes", "no"}:
            yes_i = normalized.index("yes")
            no_i = normalized.index("no")

            return [
                token_ids[yes_i],
                token_ids[no_i],
            ]

    return token_ids


def get_orderbook(token_id):
    response = requests.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_best_ask(token_id):
    book = get_orderbook(token_id)
    asks = book.get("asks", [])
    if not asks:
        return None
    return min(float(level["price"]) for level in asks)


def get_best_bid(token_id):
    book = get_orderbook(token_id)
    bids = book.get("bids", [])
    if not bids:
        return None
    return max(float(level["price"]) for level in bids)


def get_best_ask_with_size(token_id):
    book = get_orderbook(token_id)
    asks = book.get("asks", [])
    if not asks:
        return None
    best = min(asks, key=lambda level: float(level["price"]))
    return {"price": float(best["price"]), "size": float(best["size"])}


def get_orderbooks(token_ids, chunk_size=100):
    """Fetch multiple CLOB books with Polymarket's public batch endpoint."""
    out = {}
    ids = [str(x) for x in token_ids if x is not None]
    for i in range(0, len(ids), max(1, int(chunk_size))):
        chunk = ids[i:i + max(1, int(chunk_size))]
        response = requests.post(
            f"{CLOB_URL}/books",
            json=[{"token_id": token_id} for token_id in chunk],
            timeout=30,
        )
        response.raise_for_status()
        books = response.json()
        if not isinstance(books, list):
            continue
        for book in books:
            asset = str(book.get("asset_id") or "")
            if asset:
                out[asset] = book
    return out


def get_event_by_id(event_id):
    response = requests.get(f"{GAMMA_URL}/events/{event_id}", timeout=30)
    response.raise_for_status()
    return response.json()
