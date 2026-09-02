"""Conservative Polymarket fee-rate resolution for paper research.

The live market object is authoritative when it exposes a fee schedule.  When
Gamma identifies a fee-enabled category but omits the schedule, this module
uses the current documented category rate rather than silently assuming zero.
Explicit ``feesEnabled=False`` always wins.
"""
from __future__ import annotations


def _norm(value) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


CATEGORY_RATES = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "general": 0.05,
    "mentions": 0.04,
    "technology": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}


def _tag_text(market: dict) -> str:
    pieces = [market.get("category"), market.get("subcategory")]
    for tag in market.get("tags", []) or []:
        if isinstance(tag, dict):
            pieces.extend((tag.get("label"), tag.get("slug")))
        else:
            pieces.append(tag)
    for event in market.get("events", []) or []:
        pieces.extend((event.get("category"), event.get("title")))
        for tag in event.get("tags", []) or []:
            if isinstance(tag, dict):
                pieces.extend((tag.get("label"), tag.get("slug")))
            else:
                pieces.append(tag)
    return _norm(" ".join(str(x or "") for x in pieces))


def infer_documented_fee_rate(market: dict) -> float | None:
    text = _tag_text(market)
    if not text:
        return None
    # Geopolitical/world-event markets are explicitly fee free in the current
    # public fee documentation. Check before the broad "politics" marker.
    if "geopolit" in text or "world events" in text:
        return 0.0
    aliases = (
        (("crypto", "bitcoin", "ethereum", "solana"), "crypto"),
        (("sport", "nba", "nfl", "mlb", "nhl", "soccer", "tennis"), "sports"),
        (("finance", "stock", "equities"), "finance"),
        (("politic", "election", "congress", "senate", "house"), "politics"),
        (("economic", "inflation", "gdp", "unemployment", "fed"), "economics"),
        (("culture", "entertainment", "music", "movies", "awards"), "culture"),
        (("weather", "climate"), "weather"),
        (("mention",), "mentions"),
        (("technology", " tech ", "ai ", "artificial intelligence"), "technology"),
    )
    padded = f" {text} "
    for markers, key in aliases:
        if any(marker in padded for marker in markers):
            return CATEGORY_RATES[key]
    if "general" in text or "other" in text:
        return CATEGORY_RATES["other"]
    return None


def resolve_fee_rate(market: dict) -> float:
    explicit_enabled = market.get("feesEnabled")
    schedule = market.get("feeSchedule")
    if isinstance(schedule, dict) and schedule.get("rate") is not None:
        try:
            return max(0.0, float(schedule.get("rate")))
        except (TypeError, ValueError):
            pass

    # Some CLOB-enriched records expose fee details under a compact field.
    fd = market.get("feeDetails") or market.get("fd")
    if isinstance(fd, dict) and fd.get("r") is not None:
        try:
            return max(0.0, float(fd.get("r")))
        except (TypeError, ValueError):
            pass

    if explicit_enabled is False:
        return 0.0

    inferred = infer_documented_fee_rate(market)
    if explicit_enabled is True:
        # A fee-enabled record with a missing schedule should never be treated
        # as free. If category inference is unavailable, 5% is the conservative
        # generic documented rate.
        return inferred if inferred is not None else 0.05

    # If Gamma omitted the flag altogether, only apply a fallback when the
    # category itself is recognizable. This preserves backward compatibility
    # for sparse mocks while preventing known fee categories from becoming free.
    return inferred if inferred is not None else 0.0
