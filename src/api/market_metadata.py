"""Metadata helpers for conservative cross-platform market matching.

Paper/research only.  The functions in this module only read public market
metadata.  They never place orders.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import requests

from src.api.kalshi_client import BASE_URL


def get_open_kalshi_events(*, with_nested_markets: bool = True, with_milestones: bool = True,
                           max_pages: int = 500) -> tuple[list[dict], list[dict]]:
    """Fetch open Kalshi events with rich metadata using cursor pagination.

    Kalshi documents a maximum page size of 200 and supports nested markets
    and milestones on the event listing endpoint.  A bounded max_pages guard
    prevents a malformed cursor from creating an infinite loop.
    """
    events: list[dict] = []
    milestones: list[dict] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(max_pages):
        params: dict[str, Any] = {
            "limit": 200,
            "status": "open",
            "with_nested_markets": str(bool(with_nested_markets)).lower(),
            "with_milestones": str(bool(with_milestones)).lower(),
        }
        if cursor:
            params["cursor"] = cursor

        response = requests.get(f"{BASE_URL}/events", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("events", []) or []
        events.extend(page)
        milestones.extend(payload.get("milestones", []) or [])

        next_cursor = payload.get("cursor") or ""
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return events, milestones


def build_kalshi_metadata_indexes(events: list[dict], milestones: list[dict] | None = None):
    """Return event- and market-level metadata keyed by ticker.

    Milestones are attached to events by their related/primary event ticker
    references.  This gives the matcher optional structured sports context
    without relying on fragile ticker parsing.
    """
    milestone_by_event: dict[str, list[dict]] = defaultdict(list)
    for milestone in milestones or []:
        refs = set(milestone.get("related_event_tickers", []) or [])
        refs.update(milestone.get("primary_event_tickers", []) or [])
        for ticker in refs:
            milestone_by_event[str(ticker)].append(milestone)

    event_index: dict[str, dict] = {}
    market_index: dict[str, dict] = {}

    for event in events:
        et = str(event.get("event_ticker") or "")
        if not et:
            continue
        event_copy = dict(event)
        event_copy["_milestones"] = milestone_by_event.get(et, [])
        event_index[et] = event_copy

        for market in event.get("markets", []) or []:
            ticker = str(market.get("ticker") or "")
            if not ticker:
                continue
            merged = dict(market)
            merged["_event"] = event_copy
            market_index[ticker] = merged

    return event_index, market_index
