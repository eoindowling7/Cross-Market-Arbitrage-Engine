"""Multi-venue discovery audit for Kalshi/Polymarket/Limitless/Opinion readiness.

This module intentionally separates *venue coverage* from *trade eligibility*.
Limitless is public/read-only here. Opinion is recorded as an optional venue
because its OpenAPI requires an API key. The core paper engine remains strict:
only strategies with verified payoff/conversion semantics can affect P&L.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.api.limitless_client import get_active_markets as get_limitless_markets

@dataclass
class VenueStatus:
    venue: str
    enabled: bool
    mode: str
    markets: int
    note: str


def audit_extra_venues(limitless_limit: int = 250):
    rows = []
    try:
        lm = get_limitless_markets(limitless_limit)
        rows.append(VenueStatus("limitless", True, "public discovery", len(lm),
                                "REST+WebSocket available; V8.5 uses documented 25-row pagination and strict short-horizon paper matching"))
    except Exception as exc:
        rows.append(VenueStatus("limitless", False, "public discovery", 0, f"{type(exc).__name__}: {exc}"))
    rows.append(VenueStatus("opinion", False, "optional adapter", 0,
                            "Official OpenAPI/CLOB supports books but requires API access key; not silently assumed"))
    return rows
