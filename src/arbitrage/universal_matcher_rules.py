"""V26 targeted-recall cross-platform matcher.

Design goals
------------
1. High recall *inside the right market family*.
2. Very low false-positive rate on payoff-defining semantics.
3. Bounded candidate count: never compare every vaguely related market.
4. Lazy expensive rule hydration only after structural reranking.

Architecture: FAMILY ROUTER -> SPARSE RETRIEVER -> STRUCTURAL RERANKER ->
PAYOFF VERIFIER -> RESOLUTION-RULE VERIFIER.

The implementation deliberately keeps a deterministic offline fallback rather
than requiring a transformer download at runtime.  This follows the same
retrieve/rerank pattern used by modern semantic search systems while keeping
paper-run reproducibility and protecting hard contract semantics.
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

# Optional semantic-similarity blocking pass (V19.2). Uses a local TF-IDF
# char n-gram vector space as a fast, offline, dependency-light proxy for a
# true embedding model -- no network access or model download required, in
# keeping with this codebase's existing "no transformer download at runtime"
# constraint. If scikit-learn is not installed, this pass is silently
# skipped and matching falls back to the pre-existing deterministic passes
# (i.e. this is additive: it can only add recall, never remove it).
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    import numpy as np

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SKLEARN_AVAILABLE = False

# Reuse the mature V9-V16 signature extraction and safety logic, then put a
# completely different candidate-generation architecture in front of it.
from src.arbitrage import universal_matcher_v16 as legacy

ContractSignature = legacy.ContractSignature
MatchAudit = legacy.MatchAudit
normalize_text = legacy.normalize_text
_entity_equivalent = legacy._entity_equivalent
_normalized_threshold = legacy._normalized_threshold
_thresholds_compatible = legacy._thresholds_compatible
_similarity = legacy._similarity
_hybrid_identity_similarity = legacy._hybrid_identity_similarity
_resolution_rule_comparison = legacy._resolution_rule_comparison
_basis_risk_assessment = legacy._basis_risk_assessment
_build_match = legacy._build_match
_needs_resolution_rule_hydration = legacy._needs_resolution_rule_hydration
_unwrap_market_detail = legacy._unwrap_market_detail


# V24 performance-only caches. These do not change matching decisions; they
# avoid repeating the same fuzzy-string work across overlapping retrieval
# routes and dense candidate families.
def _sym_pair(a: str | None, b: str | None) -> tuple[str, str]:
    aa, bb = str(a or ""), str(b or "")
    return (aa, bb) if aa <= bb else (bb, aa)


@lru_cache(maxsize=500_000)
def _cached_entity_equivalent_pair(a: str, b: str):
    return _entity_equivalent(a, b)


def _entity_equivalent_fast(a: str | None, b: str | None):
    aa, bb = _sym_pair(a, b)
    return _cached_entity_equivalent_pair(aa, bb)


@lru_cache(maxsize=750_000)
def _cached_hybrid_pair(a: str, b: str) -> float:
    return float(_hybrid_identity_similarity(a, b))


def _hybrid_identity_similarity_fast(a: str | None, b: str | None) -> float:
    aa, bb = _sym_pair(a, b)
    return _cached_hybrid_pair(aa, bb)


def _sanitize_year(value: str | None) -> str | None:
    """Undo the legacy YYYY-MM -> YYYY-20MM season false-positive.

    Polymarket slugs/end-date text can contain ISO dates such as 2026-08-27.
    The older season parser could interpret ``2026-08`` as season
    ``2026-2008``.  That is never a plausible season key and materially hurts
    cross-venue recall, especially for daily sports/weather markets.
    """
    if not value:
        return value
    m = re.fullmatch(r"(20\d{2})-(20(?:0[1-9]|1[0-2]))", str(value))
    if m:
        return m.group(1)
    return value


def _infer_us_region_from_text(text: str) -> str | None:
    t = f" {normalize_text(text)} "
    # Reuse the mature state dictionary from V9+, then accept postal
    # abbreviations only in clearly political US context.
    for name, abbr in sorted(legacy._US_STATES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", t):
            return f"US-{abbr}"
    us_context = bool(re.search(r"\b(?:u s|us|united states|house|senate|congress|midterm|governor|gubernatorial)\b", t))
    if us_context:
        allowed = set(legacy._US_STATES.values())
        for m in re.finditer(r"\b([a-z]{2})\b", t):
            abbr = m.group(1).upper()
            if abbr in allowed:
                return f"US-{abbr}"
    return None


def _clean_subject(value: str | None, *, domain: str, raw_text: str) -> str | None:
    s = normalize_text(value)
    if not s:
        return None
    # Common group labels are not entities.  Keeping them as subjects caused
    # unrelated sports markets to be compared as if "Game 2 Winner" were a
    # team/person name.
    if domain == "sports" and (
        re.fullmatch(r"(?:game|map|set)\s*\d+\s*(?:winner)?", s)
        or re.fullmatch(r"(?:first|second|1st|2nd)\s+half(?:\s+winner)?", s)
        or re.fullmatch(r"tie(?:\s+\d+(?:st|nd|rd|th)?\s+(?:inning|half))?", s)
        or s in {"winner", "match winner", "game winner", "map winner", "pro"}
    ):
        return None
    return value


def _correct_domain(sig: ContractSignature, raw_text: str) -> str:
    """Correct known legacy domain false positives with boundary-aware rules.

    The old heuristic searched for the substring ``nfl`` and therefore labeled
    every ``inflation`` market as sports.  That is catastrophic for family
    routing.  Structured metrics and word-boundary markers take precedence.
    """
    if sig.metric in {"inflation", "unemployment", "gdp", "interest_rate"}:
        return "economics"
    t = normalize_text(raw_text)
    if re.search(r"\b(?:bitcoin|btc|ethereum|eth|solana|crypto|cryptocurrency)\b", t):
        return "crypto"
    if re.search(r"\b(?:weather|rain|rainfall|temperature|precipitation|snow|hurricane|tornado)\b", t):
        return "weather"
    if sig.office_scope or re.search(r"\b(?:election|president|presidential|congress|senate|governor|gubernatorial|prime minister|mayor|midterm)\b", t):
        return "politics"
    sports_pat = r"\b(?:nfl|nba|nhl|mlb|ufc|soccer|football|baseball|basketball|tennis|cricket|golf|chess|valorant|dota)\b|\b(?:league of legends|counter[- ]?strike|honor of kings)\b"
    if re.search(sports_pat, t) or re.search(r"\bvs\.?\b", t):
        return "sports"
    return sig.domain


def _postprocess_signature(sig: ContractSignature | None, raw_text: str) -> ContractSignature | None:
    if sig is None:
        return None
    year = _sanitize_year(sig.year)
    domain = _correct_domain(sig, raw_text)
    subject = _clean_subject(sig.subject, domain=domain, raw_text=raw_text)
    region = sig.jurisdiction_region
    country = sig.jurisdiction_country
    if sig.domain == "politics" and not region:
        inferred = _infer_us_region_from_text(raw_text)
        if inferred:
            region = inferred
            country = country or "US"
    if year != sig.year or domain != sig.domain or subject != sig.subject or region != sig.jurisdiction_region or country != sig.jurisdiction_country:
        return replace(sig, domain=domain, year=year, subject=subject, jurisdiction_region=region, jurisdiction_country=country)
    return sig


def kalshi_signature(row, metadata: dict | None = None) -> ContractSignature | None:
    sig = legacy.kalshi_signature(row, metadata)
    raw = _raw_kalshi_text(row, metadata or {}) if '_raw_kalshi_text' in globals() else " ".join(str(x or "") for x in (row.get("title"), row.get("yes_sub_title"), row.get("subtitle")))
    return _postprocess_signature(sig, raw)


def polymarket_signature(market: dict) -> ContractSignature | None:
    sig = legacy.polymarket_signature(market)
    raw = _raw_poly_text(market) if '_raw_poly_text' in globals() else str(market.get("question") or "")
    return _postprocess_signature(sig, raw)


# ---------------------------------------------------------------------------
# Retrieval vocabulary
# ---------------------------------------------------------------------------
_RETRIEVAL_STOP = set(legacy.STOPWORDS) | {
    "market", "markets", "event", "events", "resolve", "resolves", "resolution",
    "result", "results", "official", "according", "based", "contract", "contracts",
    "exactly", "least", "most", "more", "less", "than", "above", "below", "under",
    "over", "win", "wins", "winner", "winning", "yes", "no", "price", "value",
    "number", "how", "many", "next", "during", "before", "after", "end", "date",
}

_ASSET_ALIASES = {
    "btc": "bitcoin", "xbt": "bitcoin", "eth": "ethereum", "sol": "solana",
    "brent crude": "brent", "wti crude": "wti", "crude oil": "oil",
    "gold price": "gold", "silver price": "silver", "platinum price": "platinum",
}

_GENERIC_ENTITY = {
    "democrats", "republicans", "democratic party", "republican party",
    "yes", "no", "other", "field", "candidate", "team", "player",
}


def _canon_asset(text: str) -> str:
    t = normalize_text(text)
    for src, dst in _ASSET_ALIASES.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    return t


def _doc_tokens(*parts: Any) -> set[str]:
    text = _canon_asset(" ".join(str(p or "") for p in parts))
    toks = set()
    for tok in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text):
        if len(tok) < 3 or tok in _RETRIEVAL_STOP:
            continue
        if re.fullmatch(r"20\d{2}", tok):
            continue
        toks.add(tok)
    return toks


def _raw_poly_text(pm: dict) -> str:
    events = pm.get("events") or []
    event = events[0] if events else {}
    tags = []
    for x in pm.get("tags") or []:
        if isinstance(x, dict):
            tags.extend([x.get("label"), x.get("slug")])
        else:
            tags.append(x)
    return " ".join(str(x or "") for x in (
        pm.get("question"), pm.get("groupItemTitle"), pm.get("groupItemRange"),
        event.get("title"), event.get("slug"), pm.get("category"), *tags,
    ))


def _raw_kalshi_text(row, md: dict) -> str:
    event = (md or {}).get("_event") or {}
    return " ".join(str(x or "") for x in (
        (md or {}).get("title") or row.get("title"),
        (md or {}).get("yes_sub_title") or row.get("yes_sub_title"),
        (md or {}).get("subtitle") or row.get("subtitle"),
        event.get("title"), event.get("sub_title"), event.get("category"),
    ))


_SPORT_KIND_PATTERNS = [
    ("esports", r"\b(?:counter[- ]?strike|cs2|valorant|dota|league of legends|\blol\b|honor of kings|kpl|lck|lpl|lec|cblol|vct|bo[1357])\b"),
    ("table_tennis", r"\b(?:table tennis|tt-series|elite series tt|setka cup)\b"),
    ("tennis", r"\b(?:atp|wta|tennis|wimbledon|us open|australian open|french open|roland garros)\b"),
    ("cricket", r"\b(?:cricket|t20|odi|innings|espncricinfo|wicket|sixes)\b"),
    ("baseball", r"\b(?:mlb|baseball|inning|world series)\b"),
    ("basketball", r"\b(?:nba|wnba|basketball|ncaa basketball)\b"),
    ("american_football", r"\b(?:nfl|touchdown|rushing yards|receiving yards|passing yards|super bowl)\b"),
    ("soccer", r"\b(?:soccer|football|premier league|epl|champions league|europa league|liga|serie a|bundesliga|mls|fc\b)"),
    ("golf", r"\b(?:pga|dp world tour|golf|masters tournament|ryder cup|tour championship)\b"),
    ("combat", r"\b(?:ufc|mma|boxing|bout|fight)\b"),
    ("chess", r"\b(?:chess|titled tuesday|grandmaster)\b"),
]


def _sports_kind(raw_text: str, sig: ContractSignature | None = None) -> str:
    t = normalize_text(raw_text)
    if sig and sig.competition:
        comp = str(sig.competition)
        if comp in {"nba"}: return "basketball"
        if comp in {"nfl", "super_bowl"}: return "american_football"
        if comp in {"mlb"}: return "baseball"
        if comp in {"ufc"}: return "combat"
        if comp in {"wimbledon", "australian_open", "french_open", "us_open"}: return "tennis"
        if comp.startswith("lol_"): return "esports"
        if comp in {"premier_league", "uefa_champions_league", "uefa_europa_league", "uefa_conference_league", "fifa_world_cup"}: return "soccer"
        if comp in {"golf_masters", "pga_championship", "ryder_cup"}: return "golf"
        if comp == "chess_olympiad": return "chess"
    for kind, pat in _SPORT_KIND_PATTERNS:
        if re.search(pat, t):
            return kind
    return "generic"


def _sports_scope(raw_text: str, sig: ContractSignature | None = None) -> str:
    t = normalize_text(raw_text)
    if re.search(r"\b(?:game|map)\s*\d+\b", t): return "subgame"
    if re.search(r"\bset\s*\d+\b", t): return "set"
    if re.search(r"\b\d+(?:st|nd|rd|th)?\s+inning\b", t): return "inning"
    if re.search(r"\b(?:first|second|1st|2nd)\s+half\b", t): return "half"
    if re.search(r"\bexact match score|\bset score\b", t): return "scoreline"
    # Stage is taken from the retrieval text, not the signature field: venue
    # rule boilerplate such as "official final result" can otherwise create a
    # false stage=final.
    if re.search(r"\bplayoffs?\b", t): return "playoffs"
    if re.search(r"\bqualifying\b", t): return "qualifying"
    if re.search(r"\bgroup stage\b", t): return "group_stage"
    if re.search(r"\bfinals\b", t): return "finals"
    if re.search(r"\bfinal\b", t) and not re.search(r"\bfinal result\b", t): return "final"
    return "event"


def _calendar_day(ts: float | None) -> int | None:
    if ts is None:
        return None
    try:
        return int(float(ts) // 86400)
    except Exception:
        return None


def _event_anchor_tokens(sig: ContractSignature, raw_text: str, family: str) -> set[str]:
    """Payoff-relevant tokens used only for retrieval/guarding.

    This deliberately strips generic market words and sport/category labels so
    an entity/event token must do the work.  The final verifier remains the
    authority on equivalence.
    """
    toks = _doc_tokens(raw_text, sig.event_identity, sig.context, sig.subject)
    generic = {
        "sports","sport","gaming","game","games","winner","wins","match","matches",
        "league","series","tournament","playoffs","regular","season","group","stage",
        "weather","rain","rainfall","temperature","degrees","high","low","daily",
        "election","elections","presidential","house","senate","seat","seats",
        "price","market","value","rate","percent","percentage",
    }
    return {t for t in toks if t not in generic}


def _anchor_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _semantic_aliases(sig: ContractSignature, raw_text: str, primary: str) -> list[str]:
    """Compatible retrieval routes that recover category/metadata asymmetry.

    Aliases never bypass the final family/payoff verifier.  Sports and weather
    intentionally do not get a global winner/binary alias because calibration
    showed those domains were the main source of noisy false candidates.
    """
    out = [primary]
    if sig.domain in {"sports", "weather"}:
        return out
    core = sig.metric or sig.office_scope or sig.competition
    if core:
        out.append(f"semantic:{sig.proposition}:{core}")
    if sig.proposition in {"above_threshold", "below_threshold", "range", "exact_count"}:
        out.append(f"numeric:{sig.proposition}:{sig.metric or 'generic'}")
    if sig.domain == "politics" and sig.office_scope:
        out.append(f"politics_semantic:{sig.office_scope}:{sig.proposition}:{sig.metric or 'generic'}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Family router
# ---------------------------------------------------------------------------

def _family(sig: ContractSignature, raw_text: str = "") -> str:
    """Route contracts into payoff families before retrieval.

    This is intentionally coarser than equivalence.  It increases recall while
    preventing nonsense comparisons such as commodity thresholds vs shipping
    counts or generic political winner markets vs sports props.
    """
    d = sig.domain or "other"
    p = sig.proposition or "unknown"
    m = sig.metric
    office = sig.office_scope

    if d == "politics":
        if m in {"seats", "votes"}:
            return f"politics_count:{office or 'generic'}:{m}:{p}"
        if office:
            return f"politics_office:{office}:{p}"
        return f"politics_generic:{p}"

    if d == "sports":
        stat_metrics = {
            "rushing_touchdowns", "receiving_touchdowns", "passing_touchdowns",
            "rushing_yards", "receiving_yards", "passing_yards", "receptions",
            "sacks", "interceptions", "home_runs", "goals", "assists", "points",
        }
        kind = _sports_kind(raw_text, sig)
        scope = _sports_scope(raw_text, sig)
        if m in stat_metrics:
            return f"sports_stat:{kind}:{m}:{p}"
        if sig.competition:
            return f"sports_comp:{kind}:{sig.competition}:{scope}:{p}"
        return f"sports_event:{kind}:{scope}:{p}"

    if d in {"crypto", "economics", "weather"}:
        return f"{d}:{m or 'generic'}:{p}"

    # Asset/commodity price and numeric threshold contracts often arrive with
    # category=Other.  Route them by semantics rather than leaving them in one
    # gigantic generic bucket.
    rt = normalize_text(raw_text)
    if p in {"above_threshold", "below_threshold", "range", "exact_count"}:
        asset_words = ("bitcoin", "ethereum", "solana", "gold", "silver", "platinum", "oil", "brent", "wti")
        if any(re.search(rf"\b{w}\b", rt) for w in asset_words):
            return f"asset_threshold:{p}"
        if m:
            return f"numeric:{m}:{p}"
        return f"numeric_generic:{p}"

    if p in {"winner", "primary_winner", "participation", "qualify", "relegation", "make_playoffs"}:
        return f"entity_event:{d}:{p}"

    return f"generic:{d}:{p}"


def _entity_sensitive(family: str, sig: ContractSignature) -> bool:
    if family.startswith(("politics_office:", "sports_stat:", "sports_comp:", "sports_event:", "entity_event:")):
        return True
    # Party is payoff-defining in seat/vote count markets.
    if family.startswith("politics_count:"):
        return True
    return False


def _subject_key(subject: str | None) -> str | None:
    c = legacy._canonical_entity(subject)
    if not c or c in _GENERIC_ENTITY:
        return c or None
    return c


def _threshold_key(sig: ContractSignature) -> tuple | None:
    op, low, high = _normalized_threshold(sig)
    if low is None and high is None:
        return None
    return (op, None if low is None else round(float(low), 6), None if high is None else round(float(high), 6))


def _route_keys(sig: ContractSignature, family: str) -> list[tuple]:
    keys: list[tuple] = [("family", family)]
    subj = _subject_key(sig.subject)
    if subj:
        keys.append(("subject", family, subj))
    if sig.jurisdiction_region:
        keys.append(("region", family, sig.jurisdiction_region))
    if sig.jurisdiction_district:
        keys.append(("district", family, sig.jurisdiction_district))
    if sig.competition:
        keys.append(("competition", family, sig.competition))
    if sig.office_scope:
        keys.append(("office", family, sig.office_scope))
    if sig.year:
        keys.append(("year", family, sig.year))
    tk = _threshold_key(sig)
    if tk:
        keys.append(("threshold", family, tk))
    return keys


def _families_compatible(a: ContractSignature, araw: str, b: ContractSignature, braw: str) -> bool:
    af = _family(a, araw)
    bf = _family(b, braw)
    if af == bf:
        return True
    # Metadata/category asymmetry is common outside sports/weather.  Allow the
    # semantic aliases to bridge it, but only when proposition and critical
    # structured semantics are compatible.
    if a.domain in {"sports", "weather"} or b.domain in {"sports", "weather"}:
        return False
    if a.proposition != b.proposition:
        return False
    if a.metric and b.metric and a.metric != b.metric:
        return False
    if a.office_scope and b.office_scope and a.office_scope != b.office_scope:
        return False
    if a.competition and b.competition and a.competition != b.competition:
        return False
    return bool(set(_semantic_aliases(a, araw, af)) & set(_semantic_aliases(b, braw, bf)))


def _families_compatible_fast(
    a: ContractSignature, af: str, a_routes: frozenset[str],
    b: ContractSignature, bf: str, b_routes: frozenset[str],
) -> bool:
    if af == bf:
        return True
    if a.domain in {"sports", "weather"} or b.domain in {"sports", "weather"}:
        return False
    if a.proposition != b.proposition:
        return False
    if a.metric and b.metric and a.metric != b.metric:
        return False
    if a.office_scope and b.office_scope and a.office_scope != b.office_scope:
        return False
    if a.competition and b.competition and a.competition != b.competition:
        return False
    return bool(a_routes & b_routes)


def _weather_location_tokens(sig: ContractSignature, raw: str) -> set[str]:
    toks = _event_anchor_tokens(sig, raw, _family(sig, raw))
    # Remove measurement/time vocabulary; remaining rare text is primarily the
    # location/station identity.
    discard = {
        "rain", "rains", "rainfall", "temperature", "degree", "degrees",
        "fahrenheit", "celsius", "weather", "daily", "high", "low", "maximum",
        "minimum", "inch", "inches", "mm", "precipitation", "airport",
        "august", "september", "october", "november", "december", "january",
        "february", "march", "april", "may", "june", "july",
    }
    return {x for x in toks if x not in discard and not x.isdigit()}


def _family_anchor_guard(a: ContractSignature, araw: str, aa: set[str], b: ContractSignature, braw: str, ba: set[str]) -> bool:
    """Calibration-derived early guard for noisy families.

    It is intentionally a retrieval guard, not an acceptance rule.  Its job is
    to keep obviously unrelated sports/weather pairs away from expensive rule
    comparison while preserving candidates with a genuine event anchor.
    """
    fam = _family(a, araw)
    if a.domain == "sports" or b.domain == "sports":
        if _sports_kind(araw, a) != _sports_kind(braw, b):
            return False
        if _sports_scope(araw, a) != _sports_scope(braw, b):
            # Whole-event vs game/map/set/inning/half is payoff-defining.
            return False
        # Exact/alias entity is the strongest anchor when both sides expose it.
        if a.subject and b.subject:
            ok, _ = _entity_equivalent_fast(a.subject, b.subject)
            if ok:
                return True
        overlap = _anchor_overlap(aa, ba)
        ident = _hybrid_identity_similarity_fast(a.event_identity or a.context, b.event_identity or b.context)
        # Calibration showed unrelated winner markets surviving on generic
        # terms. Require at least one substantial event anchor instead.
        return overlap >= 0.34 or ident >= 0.68

    if a.domain == "weather" or b.domain == "weather":
        ad, bd = _calendar_day(a.end_ts), _calendar_day(b.end_ts)
        if ad is not None and bd is not None and abs(ad - bd) > 1:
            return False
        aloc = _weather_location_tokens(a, araw)
        bloc = _weather_location_tokens(b, braw)
        if aloc and bloc and not (aloc & bloc):
            return False
        return bool(aloc & bloc) or _hybrid_identity_similarity_fast(a.event_identity or a.context, b.event_identity or b.context) >= 0.78

    return True


def _family_anchor_guard_precomputed(
    a: ContractSignature, a_anchors: set[str],
    a_sports_kind: str | None, a_sports_scope: str | None,
    a_weather_locations: frozenset[str], a_day: int | None,
    b: ContractSignature, b_entry: "_PolyEntry",
) -> bool:
    if a.domain == "sports" or b.domain == "sports":
        if a_sports_kind != b_entry.sports_kind:
            return False
        if a_sports_scope != b_entry.sports_scope:
            return False
        if a.subject and b.subject:
            ok, _ = _entity_equivalent_fast(a.subject, b.subject)
            if ok:
                return True
        overlap = _anchor_overlap(a_anchors, b_entry.anchors)
        ident = _hybrid_identity_similarity_fast(
            a.event_identity or a.context, b.event_identity or b.context
        )
        return overlap >= 0.34 or ident >= 0.68
    if a.domain == "weather" or b.domain == "weather":
        bd = b_entry.calendar_day
        if a_day is not None and bd is not None and abs(a_day - bd) > 1:
            return False
        bloc = b_entry.weather_locations
        if a_weather_locations and bloc and not (a_weather_locations & bloc):
            return False
        return bool(a_weather_locations & bloc) or _hybrid_identity_similarity_fast(
            a.event_identity or a.context, b.event_identity or b.context
        ) >= 0.78
    return True


@dataclass
class _PolyEntry:
    market: dict
    sig: ContractSignature
    family: str
    tokens: set[str]
    raw_text: str
    anchors: set[str]
    route_families: tuple[str, ...]
    route_family_set: frozenset[str]
    sports_kind: str | None = None
    sports_scope: str | None = None
    weather_locations: frozenset[str] = frozenset()
    calendar_day: int | None = None


@dataclass
class _Candidate:
    market: dict
    sig: ContractSignature
    retrieval_score: float
    lexical_score: float
    retrieval_sources: tuple[str, ...] = ()

    @property
    def strong_structural_block(self) -> bool:
        src = set(self.retrieval_sources)
        return (
            "exact_text" in src
            or {"subject", "juris_office"}.issubset(src)
            or {"subject", "metric_threshold"}.issubset(src)
            or "politics_struct" in src
            or "numeric_struct" in src
            or "v84_subject_key" in src
            or "v84_threshold_key" in src
        )


# Compact bitmasks replace a per-candidate Python set in the hottest retrieval
# loop. The final _Candidate receives the exact same source-name tuple.
_SOURCE_NAMES = (
    "exact_text", "v84_subject_key", "v84_threshold_key", "v84_context_key",
    "subject", "juris_office", "metric_threshold", "politics_struct",
    "numeric_struct", "family_route", "family_token", "domain_token",
    "semantic_similarity", "tiny_family",
)
_SOURCE_BITS = {name: 1 << i for i, name in enumerate(_SOURCE_NAMES)}

def _source_tuple(mask: int) -> tuple[str, ...]:
    return tuple(name for i, name in enumerate(_SOURCE_NAMES) if mask & (1 << i))

def _has_source(mask: int, name: str) -> bool:
    return bool(mask & _SOURCE_BITS[name])


class CandidateRetriever:
    """V19 multi-pass blocking retriever.

    Record-linkage systems normally recover recall with several alternative
    blocking schemes, then apply the expensive matcher only to the union of
    those blocks.  V18 effectively had one family-centred route; a parser miss
    in that route could make a genuine pair unreachable.  V19 adds independent
    subject, jurisdiction/office, threshold/metric, exact-text and broad-domain
    rare-token passes while keeping sports/weather on strict anchor-only routes.
    The final payoff verifier is unchanged.
    """

    def __init__(self, polymarket_markets: list[dict], *, top_k: int = 250):
        self.top_k = int(top_k)
        self.stats: Counter = Counter()
        self.entries: dict[str, _PolyEntry] = {}
        self.route_index: dict[tuple, set[str]] = defaultdict(set)
        self.token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.domain_token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.subject_index: dict[str, set[str]] = defaultdict(set)
        self.juris_office_index: dict[tuple, set[str]] = defaultdict(set)
        self.metric_threshold_index: dict[tuple, set[str]] = defaultdict(set)
        self.politics_struct_index: dict[tuple, set[str]] = defaultdict(set)
        self.numeric_struct_index: dict[tuple, set[str]] = defaultdict(set)
        self.exact_text_index: dict[str, set[str]] = defaultdict(set)
        # V20: restore V8.4-style broad blocking as an additional recovery route.
        # These keys are candidate-generation only; they do NOT bypass the modern verifier.
        self.v84_subject_index: dict[tuple, set[str]] = defaultdict(set)
        self.v84_threshold_index: dict[tuple, set[str]] = defaultdict(set)
        self.v84_context_index: dict[tuple, set[str]] = defaultdict(set)
        self.family_members: dict[str, set[str]] = defaultdict(set)
        token_df: Counter = Counter()
        domain_df: Counter = Counter()
        staged = []

        def text_key(raw: str) -> str:
            # Word-set key is deliberately insensitive to venue boilerplate order.
            toks = sorted(_doc_tokens(raw))
            return " ".join(toks)

        def v84_keys(sig: ContractSignature):
            """Faithful V8.4-style broad keys, split by confidence/source.

            V8.4 preferred proposition+subject+year, then proposition+threshold+unit+year,
            then a small context-token key.  V20 restores all three only as blocking
            routes; modern contradiction/rule checks remain downstream.
            """
            subj = _subject_key(sig.subject)
            if subj:
                return ("subject", (sig.proposition, subj, sig.year))
            tk = _threshold_key(sig)
            if tk is not None:
                return ("threshold", (sig.proposition, tk, sig.unit, sig.year))
            core = tuple(sorted(_doc_tokens(sig.context or sig.event_identity or ""))[:5])
            if core:
                return ("context", (sig.proposition, core, sig.year))
            return (None, None)

        for pm in polymarket_markets:
            sig = polymarket_signature(pm)
            if sig is None:
                continue
            pid = str(pm.get("id") or pm.get("conditionId") or "")
            if not pid:
                continue
            raw = _raw_poly_text(pm)
            fam = _family(sig, raw)
            toks = _doc_tokens(raw, sig.event_identity, sig.context, sig.subject)
            anchors = _event_anchor_tokens(sig, raw, fam)
            route_families = tuple(_semantic_aliases(sig, raw, fam))
            staged.append((pid, pm, sig, fam, toks, raw, anchors, route_families))
            for rfam in route_families:
                token_df.update({(rfam, t) for t in toks})
            domain_df.update({(sig.domain or "other", t) for t in toks})

        family_sizes = Counter(rfam for _, _, _, _, _, _, _, rfams in staged for rfam in rfams)
        domain_sizes = Counter((sig.domain or "other") for _, _, sig, _, _, _, _, _ in staged)

        for pid, pm, sig, fam, toks, raw, anchors, route_families in staged:
            self.entries[pid] = _PolyEntry(
                pm, sig, fam, toks, raw, anchors, route_families, frozenset(route_families),
                _sports_kind(raw, sig) if sig.domain == "sports" else None,
                _sports_scope(raw, sig) if sig.domain == "sports" else None,
                frozenset(_weather_location_tokens(sig, raw)) if sig.domain == "weather" else frozenset(),
                _calendar_day(sig.end_ts),
            )
            for rfam in route_families:
                self.family_members[rfam].add(pid)
                for key in _route_keys(sig, rfam):
                    self.route_index[key].add(pid)
                max_df = max(80, int(0.08 * max(1, family_sizes[rfam])))
                for tok in toks:
                    if token_df[(rfam, tok)] <= max_df:
                        self.token_index[(rfam, tok)].add(pid)

            subj = _subject_key(sig.subject)
            if subj:
                self.subject_index[subj].add(pid)
            if sig.jurisdiction_region or sig.jurisdiction_district or sig.office_scope:
                self.juris_office_index[(sig.jurisdiction_region, sig.jurisdiction_district, sig.office_scope, sig.year)].add(pid)
            tk = _threshold_key(sig)
            if tk is not None:
                self.metric_threshold_index[(sig.metric, tk, sig.year)].add(pid)
            # High-information compound keys.  These are the preferred V19
            # blocks because they represent payoff-defining structure rather
            # than generic semantic similarity.
            if sig.domain == "politics" and (sig.jurisdiction_region or sig.jurisdiction_country) and sig.office_scope:
                self.politics_struct_index[(
                    sig.jurisdiction_country, sig.jurisdiction_region, sig.jurisdiction_district,
                    sig.office_scope, sig.year, subj,
                )].add(pid)
            if sig.domain in {"economics", "crypto", "finance"} and tk is not None and sig.metric:
                self.numeric_struct_index[(sig.domain, sig.metric, tk, sig.year, subj)].add(pid)
            tk_text = text_key(raw)
            if tk_text:
                self.exact_text_index[tk_text].add(pid)

            v84_kind, v84_key = v84_keys(sig)
            if v84_kind == "subject":
                self.v84_subject_index[v84_key].add(pid)
            elif v84_kind == "threshold":
                self.v84_threshold_index[v84_key].add(pid)
            elif v84_kind == "context":
                self.v84_context_index[v84_key].add(pid)

            # Broad-domain pass remains conservative for sports/weather; those
            # those were the noisy V17/V18 calibration families.
            dom = sig.domain or "other"
            if dom not in {"sports", "weather"}:
                max_df = max(60, int(0.03 * max(1, domain_sizes[dom])))
                for tok in toks:
                    if domain_df[(dom, tok)] <= max_df:
                        self.domain_token_index[(dom, tok)].add(pid)

        self.family_sizes = dict(family_sizes)
        self.domain_sizes = dict(domain_sizes)
        self.token_df = token_df
        self.domain_df = domain_df

        # Semantic (TF-IDF char n-gram) index. This is the recall backstop
        # for pairs whose wording differs enough that no deterministic block
        # above shares a key -- e.g. Kalshi's "Fed cuts rates in March?" vs
        # Polymarket's "Federal Reserve interest rate decision - March",
        # which have zero exact-token overlap in the structural sense but
        # are obviously the same contract to a human. Character n-grams
        # (rather than word n-grams) also tolerate abbreviations, plurals,
        # and minor phrasing differences without any hand-written regex.
        self._semantic_vectorizer = None
        self._semantic_matrix = None
        self._semantic_pids: list[str] = []
        if _SKLEARN_AVAILABLE and staged:
            pids = [pid for pid, *_ in staged]
            texts = [raw for _, _, _, _, _, raw, _, _ in staged]
            try:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5),
                    min_df=1, max_df=0.9, sublinear_tf=True,
                )
                matrix = vectorizer.fit_transform(texts)
                self._semantic_vectorizer = vectorizer
                self._semantic_matrix = matrix
                self._semantic_pids = pids
            except Exception:
                self._semantic_vectorizer = None
                self._semantic_matrix = None
                self._semantic_pids = []

    def retrieve(self, sig: ContractSignature, raw_text: str) -> list[_Candidate]:
        fam = _family(sig, raw_text)
        route_families = _semantic_aliases(sig, raw_text, fam)
        qtokens = _doc_tokens(raw_text, sig.event_identity, sig.context, sig.subject)
        qanchors = _event_anchor_tokens(sig, raw_text, fam)
        qroute_set = frozenset(route_families)
        q_sports_kind = _sports_kind(raw_text, sig) if sig.domain == "sports" else None
        q_sports_scope = _sports_scope(raw_text, sig) if sig.domain == "sports" else None
        q_weather_locations = frozenset(_weather_location_tokens(sig, raw_text)) if sig.domain == "weather" else frozenset()
        q_day = _calendar_day(sig.end_ts)
        votes: Counter[str] = Counter()
        source_hits: Counter[str] = Counter()
        candidate_sources: dict[str, int] = {}

        def add(ids, weight: float, source: str):
            bit = _SOURCE_BITS[source]
            vget = votes.get
            sget = candidate_sources.get
            n = 0
            for pid in ids:
                votes[pid] = vget(pid, 0.0) + weight
                candidate_sources[pid] = sget(pid, 0) | bit
                n += 1
            if n:
                source_hits[source] += n

        # Pass 1: exact normalized text. Extremely high precision and cheap.
        text_key = " ".join(sorted(_doc_tokens(raw_text)))
        if text_key:
            add(self.exact_text_index.get(text_key, ()), 30.0, "exact_text")

        # Pass 1b: restore V8.4 broad blocking across the entire Kalshi universe.
        # Unlike V8.4, these are retrieval sources only; they cannot auto-accept a pair.
        subj = _subject_key(sig.subject)
        if subj:
            add(self.v84_subject_index.get((sig.proposition, subj, sig.year), ()), 24.0, "v84_subject_key")
        tk = _threshold_key(sig)
        if tk is not None:
            add(self.v84_threshold_index.get((sig.proposition, tk, sig.unit, sig.year), ()), 22.0, "v84_threshold_key")
        core = tuple(sorted(_doc_tokens(sig.context or sig.event_identity or ""))[:5])
        if core:
            add(self.v84_context_index.get((sig.proposition, core, sig.year), ()), 12.0, "v84_context_key")

        # Pass 2: canonical subject/entity, independent of family parser.
        subj = _subject_key(sig.subject)
        if subj:
            add(self.subject_index.get(subj, ()), 14.0, "subject")

        # Pass 3: jurisdiction/office/year block for politics.
        jkey = (sig.jurisdiction_region, sig.jurisdiction_district, sig.office_scope, sig.year)
        if any(x is not None for x in jkey[:3]):
            add(self.juris_office_index.get(jkey, ()), 16.0, "juris_office")

        # Pass 4: metric + exact threshold + year for finance/economics/numeric contracts.
        tk = _threshold_key(sig)
        if tk is not None:
            add(self.metric_threshold_index.get((sig.metric, tk, sig.year), ()), 15.0, "metric_threshold")

        # Pass 4b: compound deterministic keys.  These are intentionally
        # redundant with the simpler blocks: multi-pass blocking works because
        # a true pair can be recovered even when one parser field is missing.
        if sig.domain == "politics" and (sig.jurisdiction_region or sig.jurisdiction_country) and sig.office_scope:
            pkey = (sig.jurisdiction_country, sig.jurisdiction_region, sig.jurisdiction_district, sig.office_scope, sig.year, subj)
            add(self.politics_struct_index.get(pkey, ()), 28.0, "politics_struct")
        if sig.domain in {"economics", "crypto", "finance"} and tk is not None and sig.metric:
            nkey = (sig.domain, sig.metric, tk, sig.year, subj)
            add(self.numeric_struct_index.get(nkey, ()), 28.0, "numeric_struct")

        # Pass 5: V18 family-specific postings and rare-token voting.
        route_weights = {
            "subject": 9.0, "district": 12.0, "region": 10.0,
            "threshold": 8.0, "competition": 7.0, "office": 6.0,
            "year": 3.0, "family": 0.10,
        }
        for rfam in route_families:
            for key in _route_keys(sig, rfam):
                label = key[0]
                ids = self.route_index.get(key, ())
                if label == "family" and len(ids) > 400:
                    continue
                add(ids, route_weights.get(label, 1.0), "family_route")

            fam_n = max(1, self.family_sizes.get(rfam, 1))
            for tok in qtokens:
                ids = self.token_index.get((rfam, tok), ())
                if not ids:
                    continue
                df = max(1, len(ids))
                idf = math.log1p(fam_n / df)
                add(ids, min(5.0, 1.2 * idf), "family_token")

        # Pass 6: broad-domain rare-token blocking recovers genuine pairs when
        # proposition/family extraction differs between venues.  No sports or
        # weather here; those categories remain anchor-only.
        dom = sig.domain or "other"
        if dom not in {"sports", "weather"}:
            dom_n = max(1, self.domain_sizes.get(dom, 1))
            for tok in qtokens:
                ids = self.domain_token_index.get((dom, tok), ())
                if not ids:
                    continue
                df = max(1, len(ids))
                idf = math.log1p(dom_n / df)
                add(ids, min(4.0, idf), "domain_token")

        # Pass 7: semantic TF-IDF similarity blocking. This is the recall
        # backstop when every deterministic pass above misses because the
        # two venues simply worded the question differently (the dominant
        # real-world failure mode -- Kalshi and Polymarket titles are
        # written independently and rarely share exact phrasing). It is
        # NOT added to strong_structural_block, so it never bypasses the
        # family-router/anchor/contradiction safety checks below or the
        # hard payoff verifier downstream -- it only ever proposes more
        # candidates for the existing safety machinery to accept or reject.
        if self._semantic_matrix is not None and self._semantic_vectorizer is not None:
            try:
                qvec = self._semantic_vectorizer.transform([raw_text])
                sims = self._semantic_matrix.dot(qvec.T).toarray().ravel()
                if sims.size:
                    top_n = min(25, sims.size)
                    top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
                    # NOTE on threshold: TF-IDF n-gram similarity is a lexical
                    # (surface-form) signal, not a true semantic one. Empirically
                    # it scores near-duplicate wording (typos, reordering,
                    # abbreviations, shared boilerplate) reliably above ~0.5,
                    # but it can score two DIFFERENT contracts that happen to
                    # share generic domain vocabulary (e.g. "interest rates in
                    # March") comparably to two genuinely equivalent contracts
                    # phrased with completely different words (e.g. "Fed cuts"
                    # vs "Federal Reserve lowers the target rate"). The 0.5
                    # floor keeps this pass a safe near-duplicate catcher; it
                    # will NOT recover paraphrase-level cross-venue rewordings.
                    # For that you need real embeddings (see note below).
                    hits = {
                        self._semantic_pids[i]: float(sims[i])
                        for i in top_idx if sims[i] >= 0.50
                    }
                    if hits:
                        for pid, sim in hits.items():
                            votes[pid] += min(8.0, 12.0 * sim)
                            candidate_sources[pid] = candidate_sources.get(pid, 0) | _SOURCE_BITS["semantic_similarity"]
                        source_hits["semantic_similarity"] += len(hits)
            except Exception:
                pass

        # Sports/weather are deliberately shaved down to candidates possessing
        # an actual same-event anchor; no broad fallback is allowed.
        if not votes:
            ids = set()
            for rfam in route_families:
                members = self.family_members.get(rfam, set())
                if len(members) <= 80:
                    ids.update(members)
            if not ids:
                return []
            add(ids, 0.1, "tiny_family")

        for k, v in source_hits.items():
            self.stats[f"block_{k}_hits"] += int(v)

        ranked: list[_Candidate] = []
        # Larger union than V18, but still bounded before the expensive verifier.
        shortlist = votes.most_common(800)
        self.stats["retrieval_vote_shortlist"] += len(shortlist)
        for pid, retrieval in shortlist:
            ent = self.entries[pid]
            source_mask = candidate_sources.get(pid, 0)
            strong_block = (
                _has_source(source_mask, "exact_text")
                or (_has_source(source_mask, "subject") and _has_source(source_mask, "juris_office"))
                or (_has_source(source_mask, "subject") and _has_source(source_mask, "metric_threshold"))
                or _has_source(source_mask, "politics_struct")
                or _has_source(source_mask, "numeric_struct")
                or _has_source(source_mask, "v84_subject_key")
                or _has_source(source_mask, "v84_threshold_key")
            )
            # A strong deterministic block is allowed to survive a family-router
            # disagreement; the downstream payoff verifier still has veto power.
            # This directly addresses V18's failure mode where parser asymmetry
            # made a genuine pair unreachable.
            # V21 RECALL-FIRST: family routing and anchor guards are now soft
            # ranking signals, not hard gates.  V17-V20 repeatedly showed that
            # parser asymmetry can make a genuine cross-venue pair disagree on
            # family/event anchors before the expensive verifier sees it.
            family_ok = _families_compatible_fast(sig, fam, qroute_set, ent.sig, ent.family, ent.route_family_set)
            anchor_ok = _family_anchor_guard_precomputed(sig, qanchors, q_sports_kind, q_sports_scope, q_weather_locations, q_day, ent.sig, ent)
            soft_retrieval = float(retrieval)
            if not family_ok:
                self.stats["family_incompatible_soft_kept"] += 1
                soft_retrieval -= 2.0
            if not anchor_ok:
                self.stats["family_anchor_soft_kept"] += 1
                soft_retrieval -= 2.0

            quick = _quick_pair_score(sig, ent.sig, qtokens, ent.tokens, fam)
            # V21 retains weak/partially contradictory parser outputs for the
            # full verifier. Only extremely poor candidates are dropped here.
            if quick < -4.0 and not strong_block:
                self.stats["quick_low_score_pruned"] += 1
                continue
            ranked.append(_Candidate(ent.market, ent.sig, soft_retrieval + quick, quick, _source_tuple(source_mask)))
        ranked.sort(key=lambda c: (c.retrieval_score, c.lexical_score), reverse=True)

        # V21.1 efficiency pass: preserve recall by reserving every strong
        # deterministic/V8.4-style block that survived the cheap score, then
        # fill the remaining verifier budget with the highest-ranked generic
        # candidates.  This is intentionally ranking, not a stricter verifier.
        # UNKNOWN/missing metadata is never rejected here.
        strong = [c for c in ranked if c.strong_structural_block]
        generic = [c for c in ranked if not c.strong_structural_block]
        strong_cap = min(40, self.top_k)
        generic_cap = max(0, self.top_k - min(len(strong), strong_cap))
        selected = strong[:strong_cap] + generic[:generic_cap]
        selected.sort(key=lambda c: (c.retrieval_score, c.lexical_score), reverse=True)
        self.stats["efficient_strong_reserved"] += min(len(strong), strong_cap)
        self.stats["efficient_generic_selected"] += min(len(generic), generic_cap)
        self.stats["efficient_verifier_pruned"] += max(0, len(ranked) - len(selected))
        return selected


def _quick_pair_score(a: ContractSignature, b: ContractSignature, at: set[str], bt: set[str], family: str) -> float:
    """Recall-first cheap reranker.

    V21 deliberately stops treating parser disagreements as candidate-generation
    vetoes.  They become penalties here and are reconsidered by the full
    equivalence logic.  This is the main reversal of the V17-V20 recall collapse.
    """
    score = 0.0

    # Parser disagreements are penalties, not retrieval-time rejections.
    if a.proposition and b.proposition and a.proposition != b.proposition:
        score -= 2.0
    elif a.proposition == b.proposition and a.proposition:
        score += 1.0
    if a.domain and b.domain and a.domain != b.domain and "other" not in (a.domain, b.domain):
        score -= 1.5
    elif a.domain == b.domain and a.domain:
        score += 0.75
    if a.metric and b.metric:
        score += 2.0 if a.metric == b.metric else -2.0
    if a.office_scope and b.office_scope:
        score += 2.0 if a.office_scope == b.office_scope else -2.0
    if a.jurisdiction_region and b.jurisdiction_region:
        score += 4.0 if a.jurisdiction_region == b.jurisdiction_region else -3.0
    if a.jurisdiction_district and b.jurisdiction_district:
        score += 5.0 if a.jurisdiction_district == b.jurisdiction_district else -3.0
    if a.year and b.year:
        score += 1.5 if a.year == b.year else -1.5
    if a.competition and b.competition:
        score += 2.0 if a.competition == b.competition else -2.0

    ak, bk = _threshold_key(a), _threshold_key(b)
    if ak is not None and bk is not None:
        score += 3.0 if ak == bk else -3.0

    if a.subject and b.subject:
        ok, es = _entity_equivalent_fast(a.subject, b.subject)
        score += 3.5 * es if ok else -2.5
    elif _entity_sensitive(family, a):
        score -= 0.5

    inter = len(at & bt)
    union = max(1, len(at | bt))
    score += 5.0 * inter / union
    score += 3.0 * _hybrid_identity_similarity_fast(a.event_identity or a.context, b.event_identity or b.context)
    return score


def _anchor_similarity(a: ContractSignature, b: ContractSignature) -> float:
    return _hybrid_identity_similarity_fast(a.event_identity or a.context, b.event_identity or b.context)


def _clear_nonentity_subject(sig: ContractSignature, family: str) -> ContractSignature:
    """V16 often treated structured labels as subjects on only one venue.

    For non-entity market families, subject asymmetry is not payoff evidence;
    event identity/metric/threshold carry the semantics instead. Entity-driven
    markets (candidate/team/player/party) keep the hard subject requirement.
    """
    if _entity_sensitive(family, sig):
        return sig
    return replace(sig, subject=None)


def evaluate_equivalence(left: ContractSignature, right: ContractSignature, *, require_rule_completeness: bool = False) -> tuple[str, float, list[str]]:
    lraw = left.event_identity or left.context or ""
    rraw = right.event_identity or right.context or ""
    family = _family(left, lraw)
    right_family = _family(right, rraw)
    if not _families_compatible(left, lraw, right, rraw):
        return "REJECT", 0.0, [f"market family differs ({family} vs {right_family})"]

    # Use the more specific family for entity sensitivity.  If metadata differs
    # only by venue category, final payoff checks still govern acceptance.
    effective_family = family if family == right_family else (family if left.domain != "other" else right_family)
    left2 = _clear_nonentity_subject(left, effective_family)
    right2 = _clear_nonentity_subject(right, effective_family)
    verdict, score, reasons = legacy.evaluate_equivalence(
        left2, right2, require_rule_completeness=require_rule_completeness
    )

    # Recover a common public-metadata asymmetry: one venue leaves an entity
    # blank even though event identity is near-identical and every structured
    # payoff field agrees. This is allowed only outside entity-sensitive
    # families; entity-sensitive markets remain REVIEW/REJECT.
    if verdict == "REVIEW" and reasons == ["subject missing on one venue"] and not _entity_sensitive(effective_family, left):
        sim = _anchor_similarity(left2, right2)
        if sim >= 0.82 and _thresholds_compatible(left2, right2):
            return "HIGH_CONFIDENCE", max(0.76, score), [
                "non-entity family; one-sided label ignored",
                f"event identity strong ({sim:.2f})",
            ]

    return verdict, score, reasons


def _v20_hard_contradiction(left: ContractSignature, right: ContractSignature) -> str | None:
    """Return an explicit payoff contradiction, never a missing-field complaint.

    V20 adopts V8.4's tolerance principle: UNKNOWN is not the same as
    CONTRADICTION.  Only fields present on both venues can contradict one
    another.  Threshold/operator incompatibility remains hard even when it is
    inferred from structured numeric fields.
    """
    if left.proposition and right.proposition and left.proposition != right.proposition:
        return f"proposition differs ({left.proposition} vs {right.proposition})"
    if left.metric and right.metric and left.metric != right.metric:
        return f"metric differs ({left.metric} vs {right.metric})"
    if left.rank_semantics and right.rank_semantics and left.rank_semantics != right.rank_semantics:
        return f"rank semantics differ ({left.rank_semantics} vs {right.rank_semantics})"
    if left.office_scope and right.office_scope and left.office_scope != right.office_scope:
        return f"office/chamber differs ({left.office_scope} vs {right.office_scope})"
    if left.jurisdiction_country and right.jurisdiction_country and left.jurisdiction_country != right.jurisdiction_country:
        return f"country differs ({left.jurisdiction_country} vs {right.jurisdiction_country})"
    if left.jurisdiction_region and right.jurisdiction_region and left.jurisdiction_region != right.jurisdiction_region:
        return f"jurisdiction differs ({left.jurisdiction_region} vs {right.jurisdiction_region})"
    if left.jurisdiction_district and right.jurisdiction_district and left.jurisdiction_district != right.jurisdiction_district:
        return f"district differs ({left.jurisdiction_district} vs {right.jurisdiction_district})"
    if left.year and right.year and left.year != right.year:
        return f"year/season differs ({left.year} vs {right.year})"
    if left.competition and right.competition and left.competition != right.competition:
        return f"competition differs ({left.competition} vs {right.competition})"
    if left.stage and right.stage and left.stage != right.stage:
        return f"stage differs ({left.stage} vs {right.stage})"
    if left.period_scope and right.period_scope and left.period_scope != right.period_scope:
        return f"period scope differs ({left.period_scope} vs {right.period_scope})"
    if left.gender_scope and right.gender_scope and left.gender_scope != right.gender_scope:
        return f"gender scope differs ({left.gender_scope} vs {right.gender_scope})"
    if left.unit and right.unit and left.unit != right.unit:
        return f"unit differs ({left.unit} vs {right.unit})"
    if not _thresholds_compatible(left, right):
        return "threshold semantics differ"
    if left.subject and right.subject:
        ok, _ = _entity_equivalent_fast(left.subject, right.subject)
        if not ok:
            return "subject differs"
    conflict, why = legacy._flag_conflict(left.danger_flags, right.danger_flags)
    if conflict:
        return why or "resolution qualifier conflict"
    return None


def _v20_feature_tolerant_recovery(
    verdict: str, score: float, reasons: list[str],
    left: ContractSignature, right: ContractSignature, cand: _Candidate,
    *, require_rule_completeness: bool,
) -> tuple[str, float, list[str]]:
    """Recover high-confidence pairs lost only to one-sided parser omissions.

    This is intentionally NOT a generic threshold relaxation.  It implements
    a three-state view of fields: MATCH / CONTRADICTION / UNKNOWN.  Explicit
    contradictions always reject; UNKNOWN can be tolerated only when several
    independent anchors identify the same event and settlement rules are not
    incompatible.
    """
    if verdict in {"EXACT", "HIGH_CONFIDENCE"}:
        return verdict, score, reasons

    contradiction = _v20_hard_contradiction(left, right)
    if contradiction:
        return "REJECT", 0.0, [contradiction]

    reason_text = " ".join(reasons or []).lower()
    recoverable_missing = any(x in reason_text for x in (
        "missing on one venue",
        "missing jurisdiction country",
        "metric differs",          # only one-sided survives hard-contradiction scan
        "rank semantics differ",   # only one-sided survives hard-contradiction scan
    ))
    if not recoverable_missing:
        return verdict, score, reasons

    identity = _anchor_similarity(left, right)
    family = _family(left, left.event_identity or left.context or "")
    right_family = _family(right, right.event_identity or right.context or "")

    # Stronger identity requirements in historically noisy families.
    identity_floor = 0.90 if (family in {"sports", "weather"} or right_family in {"sports", "weather"}) else 0.78
    if "v84_context_key" in cand.retrieval_sources:
        identity_floor = max(identity_floor, 0.90)
    if identity < identity_floor:
        return verdict, score, reasons

    evidence = 1.0  # proposition agreement / same candidate route
    evidence_reasons = [f"event identity strong ({identity:.2f})"]
    if left.subject and right.subject:
        ok, es = _entity_equivalent_fast(left.subject, right.subject)
        if ok:
            evidence += 2.0; evidence_reasons.append(f"entity compatible ({es:.2f})")
    if left.metric and right.metric and left.metric == right.metric:
        evidence += 1.0; evidence_reasons.append("metric exact")
    if _threshold_key(left) is not None and _threshold_key(left) == _threshold_key(right):
        evidence += 1.5; evidence_reasons.append("threshold/operator exact")
    if left.year and right.year and left.year == right.year:
        evidence += 1.0; evidence_reasons.append("year exact")
    if left.competition and right.competition and left.competition == right.competition:
        evidence += 1.0; evidence_reasons.append("competition exact")
    if left.office_scope and right.office_scope and left.office_scope == right.office_scope:
        evidence += 1.0; evidence_reasons.append("office exact")
    if left.jurisdiction_region and right.jurisdiction_region and left.jurisdiction_region == right.jurisdiction_region:
        evidence += 1.5; evidence_reasons.append("jurisdiction exact")
    if left.jurisdiction_district and right.jurisdiction_district and left.jurisdiction_district == right.jurisdiction_district:
        evidence += 1.5; evidence_reasons.append("district exact")
    if cand.strong_structural_block:
        evidence += 1.0; evidence_reasons.append("strong deterministic/V8.4 block")

    # Generic/entity-sensitive contracts need more corroboration; exact text is
    # allowed to compensate for parser omissions.
    required = 4.0
    if family in {"sports", "weather"} or right_family in {"sports", "weather"}:
        required = 5.0
    if not (left.subject and right.subject) and _entity_sensitive(family, left):
        required = max(required, 5.0)
    if "exact_text" in cand.retrieval_sources:
        evidence += 2.0

    if evidence < required:
        return verdict, score, reasons

    # Settlement semantics still have veto power. Sparse but non-sensitive
    # rules may enter LOW_BASIS through the existing basis-risk machinery; an
    # unresolved sensitive rule comparison is never auto-promoted.
    rule_status, rule_reasons = legacy._resolution_rule_comparison(
        left, right, force_strict=require_rule_completeness
    )
    if rule_status == "REVIEW":
        return verdict, score, reasons

    recovered_score = min(0.90, 0.76 + 0.025 * min(5.0, evidence - required) + 0.10 * max(0.0, identity - identity_floor))
    recovery_reasons = [
        "V20 feature-tolerant recovery: one-sided metadata treated as UNKNOWN, not contradiction",
        *evidence_reasons,
        *list(rule_reasons or []),
    ]
    if rule_status == "LOW_BASIS":
        recovery_reasons.append("low-basis rule lane retained; execution reserve applies")
    return "HIGH_CONFIDENCE", max(0.76, recovered_score), recovery_reasons



def _v26_text_similarity(a: str | None, b: str | None) -> float:
    """Small title/event similarity helper used only by the V26 relaxed gate."""
    aa = normalize_text(a or "")
    bb = normalize_text(b or "")
    if not aa or not bb:
        return 0.0
    stop = {"will", "the", "a", "an", "in", "on", "at", "to", "of", "for", "be", "is", "and", "or", "by", "from"}
    ta = {t for t in re.findall(r"[a-z0-9]+", aa) if len(t) > 1 and t not in stop}
    tb = {t for t in re.findall(r"[a-z0-9]+", bb) if len(t) > 1 and t not in stop}
    jac = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, aa, bb).ratio()
    return 0.72 * jac + 0.28 * seq


def _v26_relaxed_gate(
    left: ContractSignature, right: ContractSignature, cand: _Candidate,
    *, kalshi_title: str = "", poly_question: str = "", original_verdict: str = "",
    original_reasons: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Targeted precision gate learned from the V25 13,169-pair audit.

    This gate applies ONLY to V21 relaxed-recall promotions. Strict modern
    matches are untouched. It preserves the V8.4 idea of simple, strong
    candidate keys while preventing generic/missing-field matches from being
    promoted merely because they accumulated weak retrieval votes.
    """
    reasons = [str(x) for x in (original_reasons or [])]
    reason_text = " ".join(reasons).lower()

    # V25 audit: 1,525 accepted pairs crossed domains. A relaxed pair must not.
    if left.domain and right.domain and left.domain != right.domain:
        return False, [f"V26 gate: cross-domain relaxed pair ({left.domain} vs {right.domain})"]

    # V25 audit: recovering an explicit family rejection produced large clusters.
    if original_verdict == "REJECT" and "market family differs" in reason_text:
        return False, ["V26 gate: family-difference REJECT cannot be relaxed"]

    # Explicit structured contradictions remain absolute vetoes.
    contradiction = _v20_hard_contradiction(left, right)
    if contradiction:
        return False, [f"V26 gate: {contradiction}"]

    if left.proposition and right.proposition and left.proposition != right.proposition:
        return False, ["V26 gate: proposition differs"]

    title_sim = _v26_text_similarity(kalshi_title, poly_question)
    identity = _anchor_similarity(left, right)

    # Raw-title scope guard for parser misses found in the V25 audit.  Round 2
    # and Round 3 are different contracts even when player/tournament match.
    def _round_no(text: str) -> int | None:
        t = normalize_text(text)
        m = re.search(r"\bround\s*([1-9])\b", t)
        if m:
            return int(m.group(1))
        words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        for w, n in words.items():
            if re.search(rf"\b{w}\s+round\b", t):
                return n
        return None

    kr, pr = _round_no(kalshi_title), _round_no(poly_question)
    if kr is not None and pr is not None and kr != pr:
        return False, [f"V26 scope gate: round differs ({kr} vs {pr})"]
    # A round-specific finish/leader contract is not equivalent to an overall
    # tournament finish/leader contract when only one side names a round.
    scope_words = ("finish", "lead", "leader", "top ")
    if (kr is None) != (pr is None) and any(w in normalize_text(kalshi_title + " " + poly_question) for w in scope_words):
        return False, ["V26 scope gate: round-specific vs overall contract"]

    # Superlative margin markets are not ordinary numeric margin thresholds.
    kt, pt = normalize_text(kalshi_title), normalize_text(poly_question)
    k_super = bool(re.search(r"\b(?:smallest|largest|closest|widest)\s+margin\b", kt))
    p_super = bool(re.search(r"\b(?:smallest|largest|closest|widest)\s+margin\b", pt))
    if k_super != p_super:
        return False, ["V26 scope gate: superlative-margin vs threshold/event contract"]
    src = set(cand.retrieval_sources)
    v84_subject = "v84_subject_key" in src
    v84_threshold = "v84_threshold_key" in src

    subjects_both = bool(left.subject and right.subject)
    subjects_match = False
    if subjects_both:
        subjects_match, _ = _entity_equivalent_fast(left.subject, right.subject)

    domain = left.domain or right.domain or "other"

    # Sports was the largest V25 pool. Require the same participant/entity,
    # plus a real event/tournament/title anchor. Missing-subject sports survive
    # only for near-identical event text (e.g. attendance-style contracts).
    if domain == "sports":
        if subjects_both and not subjects_match:
            return False, ["V26 sports gate: participant/team differs"]
        if not subjects_both:
            if not (identity >= 0.72 and title_sim >= 0.68):
                return False, ["V26 sports gate: missing participant without near-identical event"]
        event_ok = (
            identity >= 0.35
            or title_sim >= 0.45
            or bool(left.competition and right.competition and left.competition == right.competition)
        )
        if not event_ok:
            return False, ["V26 sports gate: event/tournament identity too weak"]
        return True, [f"V26 sports gate passed (title={title_sim:.2f}, event={identity:.2f})"]

    # Politics can legitimately lack a person subject (seat-count contracts).
    # In that case require explicit same jurisdiction + office + threshold.
    if domain == "politics":
        if subjects_both and subjects_match and (title_sim >= 0.35 or identity >= 0.35 or v84_subject):
            return True, ["V26 politics gate: same subject plus event/title anchor"]
        region_ok = bool(
            (left.jurisdiction_district and right.jurisdiction_district and left.jurisdiction_district == right.jurisdiction_district)
            or (left.jurisdiction_region and right.jurisdiction_region and left.jurisdiction_region == right.jurisdiction_region)
        )
        office_ok = bool(left.office_scope and right.office_scope and left.office_scope == right.office_scope)
        threshold_ok = _thresholds_compatible(left, right) and (
            _threshold_key(left) is not None or _threshold_key(right) is not None
        )
        if region_ok and office_ok and threshold_ok:
            return True, ["V26 politics gate: jurisdiction+office+threshold exact"]
        return False, ["V26 politics gate: insufficient jurisdiction/office/entity proof"]

    # Numeric domains: V8.4's threshold key worked because it was combined
    # with proposition/unit/year. V26 adds same-domain metric/entity identity.
    if domain in {"economics", "finance", "crypto"}:
        threshold_ok = _thresholds_compatible(left, right) and (
            _threshold_key(left) is not None or _threshold_key(right) is not None
        )
        metric_ok = bool(left.metric and right.metric and left.metric == right.metric)
        entity_ok = subjects_both and subjects_match
        if threshold_ok and (metric_ok or (entity_ok and title_sim >= 0.45)):
            return True, ["V26 numeric gate: threshold plus metric/entity identity"]
        if v84_threshold and threshold_ok and title_sim >= 0.62:
            return True, ["V26 V8.4 threshold rescue: exact key plus strong title"]
        return False, ["V26 numeric gate: threshold alone is not sufficient"]

    # Other/entertainment/weather/technology: prefer same entity, or an almost
    # identical title/event. This removes generic token-only promotions.
    if subjects_both and subjects_match and (title_sim >= 0.50 or identity >= 0.50 or v84_subject):
        return True, ["V26 generic gate: same entity plus strong anchor"]
    if title_sim >= 0.76 and identity >= 0.55:
        return True, ["V26 generic gate: near-identical title and event"]
    return False, ["V26 generic gate: insufficient payoff identity"]

def _v21_recall_first_recovery(
    verdict: str, score: float, reasons: list[str],
    left: ContractSignature, right: ContractSignature, cand: _Candidate,
    *, kalshi_title: str = "", poly_question: str = "",
) -> tuple[str, float, list[str], bool]:
    """Aggressive recall lane for the presentation/paper engine.

    The modern verifier remains the primary classifier.  If it refuses a pair
    because proof is incomplete rather than because the two contracts contain
    an explicit structured contradiction, V21 promotes the pair into a clearly
    labelled RELAXED_RECALL/LOW_BASIS lane. This intentionally accepts more
    false positives in exchange for restoring V8.4-like match recall.
    """
    if verdict in {"EXACT", "HIGH_CONFIDENCE"}:
        return verdict, score, reasons, False

    # Keep only one final hard safety rail: do not deliberately promote an
    # explicit, simultaneously-observed payoff contradiction.  Everything
    # else (missing fields, parser family disagreement, incomplete rules, weak
    # identity) is allowed to compete in the relaxed recall lane.
    contradiction = _v20_hard_contradiction(left, right)
    if contradiction:
        return verdict, score, reasons, False

    gate_ok, gate_notes = _v26_relaxed_gate(
        left, right, cand, kalshi_title=kalshi_title, poly_question=poly_question,
        original_verdict=verdict, original_reasons=reasons,
    )
    if not gate_ok:
        return verdict, score, [*(reasons or []), *gate_notes], False

    identity = _anchor_similarity(left, right)
    src = set(cand.retrieval_sources)
    evidence = 0.0
    evidence_notes: list[str] = []

    if cand.strong_structural_block:
        evidence += 2.0; evidence_notes.append("strong/V8.4 structural block")
    if "semantic_similarity" in src:
        evidence += 0.75; evidence_notes.append("semantic text recovery")
    if "subject" in src or "v84_subject_key" in src:
        evidence += 1.0; evidence_notes.append("subject retrieval")
    if "juris_office" in src or "politics_struct" in src:
        evidence += 1.0; evidence_notes.append("jurisdiction/office retrieval")
    if "metric_threshold" in src or "v84_threshold_key" in src or "numeric_struct" in src:
        evidence += 1.0; evidence_notes.append("numeric/threshold retrieval")
    if "family_token" in src or "domain_token" in src:
        evidence += 0.35
    if left.proposition and right.proposition and left.proposition == right.proposition:
        evidence += 0.5
    if left.domain and right.domain and left.domain == right.domain:
        evidence += 0.5
    if identity >= 0.45:
        evidence += 0.75; evidence_notes.append(f"event identity {identity:.2f}")
    if identity >= 0.65:
        evidence += 0.75
    if cand.lexical_score >= 0.5:
        evidence += 0.5
    if cand.lexical_score >= 2.0:
        evidence += 0.75

    # REVIEW requires only one reasonably strong corroborating path. REJECT can
    # still be recovered, but needs more independent evidence. This is
    # intentionally much looser than V20.
    required = 1.50 if verdict == "REVIEW" else 2.75
    if evidence < required:
        return verdict, score, reasons, False

    relaxed_score = max(0.55, min(0.79, 0.55 + 0.035 * evidence + 0.08 * identity))
    return "HIGH_CONFIDENCE", relaxed_score, [
        "V21 RELAXED_RECALL: promoted despite incomplete modern equivalence proof",
        f"original verdict={verdict}",
        f"recall evidence={evidence:.2f}",
        *evidence_notes,
        *gate_notes,
        *(reasons or [])[:3],
    ], True


class BoundedAuditList(list):
    """Memory-bounded audit plus full funnel counters."""
    def __init__(self, max_rows: int = 25000):
        super().__init__()
        self.max_rows = max_rows
        self.verdict_counts = Counter()
        self.reason_counts = Counter()
        self.stage_counts = Counter()
        self.total_seen = 0
        self._review_kept = 0
        self._reject_kept = 0

    def bump(self, stage: str, n: int = 1):
        self.stage_counts[stage] += int(n)

    def record(self, audit: MatchAudit):
        self.total_seen += 1
        self.verdict_counts[audit.verdict] += 1
        for reason in audit.reasons or ["unknown"]:
            self.reason_counts[reason] += 1

        # Accepted rows are always retained. Non-accepted rows are bounded to
        # max_rows. Early rejects may fill the sample, but later REVIEW rows
        # replace sampled rejects so diagnostic evidence is not lost.
        if audit.verdict in {"EXACT", "HIGH_CONFIDENCE"}:
            self.append(audit)
            return
        nonaccepted = self._review_kept + self._reject_kept
        if nonaccepted < self.max_rows:
            self.append(audit)
            if audit.verdict == "REVIEW": self._review_kept += 1
            else: self._reject_kept += 1
            return
        if audit.verdict == "REVIEW" and self._review_kept < min(10000, self.max_rows):
            for i in range(len(self) - 1, -1, -1):
                if self[i].verdict == "REJECT":
                    self[i] = audit
                    self._reject_kept -= 1
                    self._review_kept += 1
                    break


def find_universal_matches(
    kalshi_markets,
    polymarket_markets: list[dict],
    kalshi_market_metadata: dict[str, dict] | None = None,
    *,
    include_legacy: bool = True,
    kalshi_detail_fetcher=None,
    polymarket_detail_fetcher=None,
):
    """V26 targeted-recall matcher: V24 retrieval + V25 pricing-compatible precision gate."""
    metadata = kalshi_market_metadata or {}
    audits = BoundedAuditList(max_rows=25000)
    matches: list[dict] = []
    retriever = CandidateRetriever(polymarket_markets, top_k=80)
    audits.bump("polymarket_signatures", len(retriever.entries))

    hydrated_kmeta: dict[str, dict] = {}
    hydrated_ksig: dict[str, ContractSignature] = {}
    hydrated_pmarket: dict[str, dict] = {}
    hydrated_psig: dict[str, ContractSignature] = {}
    hydration_counts = Counter()

    # Rules are strict only for the families that the existing resolution-rule
    # module itself classifies as sensitive. The pair verifier handles that.
    production_mode = bool(kalshi_detail_fetcher is not None and polymarket_detail_fetcher is not None)

    seen_pairs: set[tuple[str, str]] = set()
    ksig_count = 0

    for _, row in kalshi_markets.iterrows():
        ticker = str(row.get("ticker") or "")
        md = metadata.get(ticker) or {}
        ksig = kalshi_signature(row, md)
        if ksig is None:
            continue
        ksig_count += 1
        if ksig_count % 5000 == 0:
            print(
                f"V26 matcher progress {ksig_count}/{len(kalshi_markets)} | "
                f"candidates={int(audits.stage_counts.get('retrieved_topk_pairs', 0))} | "
                f"verified={int(audits.stage_counts.get('unique_pairs_verified', 0))} | "
                f"accepted={int(audits.stage_counts.get('accepted_pairs', 0))}",
                flush=True,
            )
        raw_k = _raw_kalshi_text(row, md)

        # V20 deliberately restores V8.4's broad family coverage. No Kalshi
        # signature is discarded merely because it belongs to sports, weather,
        # entertainment, technology, or a generic event family. Precision is
        # enforced downstream by explicit contradiction and rule checks.
        candidates = retriever.retrieve(ksig, raw_k)
        audits.bump("retrieved_topk_pairs", len(candidates))
        if candidates:
            audits.bump("kalshi_with_candidates")

        for cand in candidates:
            pm = cand.market
            pid = str(pm.get("id") or pm.get("conditionId") or "")
            pair_key = (ticker, pid)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            audits.bump("unique_pairs_verified")

            current_ksig = hydrated_ksig.get(ticker, ksig)
            current_pm = hydrated_pmarket.get(pid, pm)
            current_psig = hydrated_psig.get(pid, cand.sig)

            verdict, score, reasons = evaluate_equivalence(
                current_ksig, current_psig, require_rule_completeness=production_mode
            )

            # Preserve the original REVIEW long enough to hydrate public rule
            # text. Feature-tolerant promotion happens after hydration below.

            # Only structurally plausible pairs are allowed to trigger detail
            # API calls. This prevents rule hydration from dominating runtime.
            if verdict == "REVIEW" and (cand.lexical_score >= 2.0 or cand.strong_structural_block) and _needs_resolution_rule_hydration(
                verdict, reasons, current_ksig, current_psig
            ):
                audits.bump("rule_hydration_candidates")
                changed = False
                if not current_ksig.rule_text_present and kalshi_detail_fetcher is not None:
                    if ticker not in hydrated_kmeta:
                        try:
                            detail = _unwrap_market_detail(kalshi_detail_fetcher(ticker))
                            base = dict(md)
                            event_ctx = base.get("_event")
                            merged = {**base, **detail}
                            if event_ctx is not None:
                                merged["_event"] = event_ctx
                            hydrated_kmeta[ticker] = merged
                            hydration_counts["kalshi_success"] += 1
                        except Exception:
                            hydrated_kmeta[ticker] = dict(md)
                            hydration_counts["kalshi_failure"] += 1
                    ns = kalshi_signature(row, hydrated_kmeta[ticker])
                    if ns is not None:
                        current_ksig = ns; hydrated_ksig[ticker] = ns; changed = True

                if not current_psig.rule_text_present and polymarket_detail_fetcher is not None:
                    if pid not in hydrated_pmarket:
                        try:
                            detail = _unwrap_market_detail(polymarket_detail_fetcher(pid))
                            hydrated_pmarket[pid] = {**pm, **detail}
                            hydration_counts["polymarket_success"] += 1
                        except Exception:
                            hydrated_pmarket[pid] = pm
                            hydration_counts["polymarket_failure"] += 1
                    current_pm = hydrated_pmarket[pid]
                    ns = polymarket_signature(current_pm)
                    if ns is not None:
                        current_psig = ns; hydrated_psig[pid] = ns; changed = True

                if changed:
                    verdict, score, reasons = evaluate_equivalence(
                        current_ksig, current_psig, require_rule_completeness=production_mode
                    )

            verdict, score, reasons = _v20_feature_tolerant_recovery(
                verdict, score, reasons, current_ksig, current_psig, cand,
                require_rule_completeness=production_mode,
            )
            if verdict == "HIGH_CONFIDENCE" and reasons and str(reasons[0]).startswith("V20 feature-tolerant"):
                audits.bump("feature_tolerant_recovered")

            verdict, score, reasons, relaxed_recall = _v21_recall_first_recovery(
                verdict, score, reasons, current_ksig, current_psig, cand,
                kalshi_title=str(row.get("title") or ""),
                poly_question=str(current_pm.get("question") or ""),
            )
            if relaxed_recall:
                audits.bump("relaxed_recall_recovered")

            audit = MatchAudit(
                verdict, score, reasons, ticker, str(row.get("title") or ""),
                str(current_pm.get("question") or ""), asdict(current_ksig), asdict(current_psig)
            )
            audits.record(audit)
            if verdict in {"EXACT", "HIGH_CONFIDENCE"}:
                audits.bump("accepted_pairs")
                match = _build_match(
                    ticker, row, current_pm, current_ksig, current_psig, score, reasons,
                    f"{verdict}:v21_recall_first",
                    strict_rules=production_mode and not relaxed_recall,
                )
                if relaxed_recall:
                    cert = match.setdefault("equivalence_certificate", {})
                    cert["resolution_lane"] = "LOW_BASIS"
                    cert["match_verdict"] = "RELAXED_RECALL"
                    cert["relaxed_recall"] = True
                    # Conservative paper reserve for uncertainty; this lane is
                    # intentionally exploratory and separately reported.
                    cert["basis_risk_reserve_per_contract"] = max(
                        float(cert.get("basis_risk_reserve_per_contract") or 0.0), 0.03
                    )
                    match["match_source"] = "RELAXED_RECALL:v21"
                matches.append(match)

    audits.bump("kalshi_signatures", ksig_count)
    for k, v in hydration_counts.items():
        audits.stage_counts[f"hydration_{k}"] += int(v)
    for k, v in retriever.stats.items():
        audits.stage_counts[f"retriever_{k}"] += int(v)

    # V20 restores V8.4-style broad candidate generation inside CandidateRetriever.
    # Legacy pairs are never auto-accepted; every returned match passed the modern verifier.
    return matches, audits
