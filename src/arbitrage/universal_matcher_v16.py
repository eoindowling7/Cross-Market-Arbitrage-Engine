"""V16.0 hybrid retrieve-rerank high-recall / high-precision cross-platform contract matcher.

The V8.6 expanded Polymarket universe materially improved coverage but exposed
an important failure mode: a shared person/team and related event were enough
for broad candidates such as player-stat vs season-leader, tournament-entry vs
tournament-winner, and top-half vs champion markets to reach the execution
engine.  V8.8 keeps the V8.7 false-positive protections, but recovers legitimate
pairs by separating critical payoff semantics from optional metadata and by
using an audited HIGH_CONFIDENCE tier when one venue omits non-critical fields.

Only contracts with the same payoff proposition, subject/entity, metric,
threshold/rank semantics, scope, season, competition/event identity and
material resolution qualifiers can be labelled EXACT.  Unclear pairs are sent
to REVIEW and never enter paper P&L.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any
from collections import Counter

from src.arbitrage.structural_matcher import find_structural_matches as find_legacy_matches
from src.arbitrage.resolution_equivalence import (
    ResolutionProfile,
    build_resolution_profile,
    compare_resolution_profiles,
    assess_low_basis_risk,
    rule_sensitive_family,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for", "from",
    "in", "is", "of", "on", "or", "the", "to", "will", "who", "yes", "no",
    "market", "season", "2025", "2026", "2027", "2028", "2029", "2030",
}

DOMAIN_ALIASES = {
    "sports": "sports", "sport": "sports",
    "politics": "politics", "political": "politics", "elections": "politics",
    "economics": "economics", "economy": "economics", "economic": "economics",
    "crypto": "crypto", "cryptocurrency": "crypto",
    "technology": "technology", "tech": "technology", "business": "business",
    "entertainment": "entertainment", "culture": "entertainment",
    "science": "science", "weather": "weather", "climate": "weather",
}

COMPETITION_PATTERNS = [
    (r"\b(?:uefa\s+)?champions league\b", "uefa_champions_league"),
    (r"\b(?:english\s+)?premier league\b|\bepl\b", "premier_league"),
    (r"\beuropa league\b", "uefa_europa_league"),
    (r"\bconference league\b", "uefa_conference_league"),
    (r"\bfifa world cup\b|\bworld cup\b", "fifa_world_cup"),
    (r"\bballon d or\b", "ballon_dor"),
    (r"\bsuper bowl\b", "super_bowl"),
    (r"\bnba\b", "nba"), (r"\bnfl\b|\bpro football\b", "nfl"),
    (r"\bnhl\b", "nhl"), (r"\bmlb\b|\bworld series\b", "mlb"),
    (r"\bformula 1\b|\bformula one\b|\bf1\b", "formula_1"),
    (r"\bufc\b", "ufc"),
    (r"\blck\b", "lol_lck"), (r"\blpl\b", "lol_lpl"),
    (r"\b(?:league of legends|lol) worlds\b|\bworlds\s+20\d{2}\b", "lol_worlds"),
    (r"\bwimbledon\b", "wimbledon"), (r"\baustralian open\b", "australian_open"),
    (r"\bfrench open\b|\broland garros\b", "french_open"),
    (r"\bus open\b", "us_open"),
    (r"\bchess olympiad\b", "chess_olympiad"),
    (r"\bmasters tournament\b|\bthe masters\b", "golf_masters"),
    (r"\bpga championship\b", "pga_championship"),
    (r"\bryder cup\b", "ryder_cup"),
]

DANGER_FLAGS = {
    "interim": ("interim",), "first_round": ("first round", "1st round"),
    "second_round": ("second round", "2nd round"), "runoff": ("runoff", "run off"),
    "primary": (" primary", "primary "), "regular_season": ("regular season",),
    "playoffs": ("playoff", "playoffs"), "finals": ("finals",),
    "qualifying": ("qualifying", "qualify for"), "popular_vote": ("popular vote",),
    "electoral_college": ("electoral college",), "overtime": ("overtime", "extra time"),
    "penalties": ("penalt",), "aggregate": ("aggregate", "two leg", "2 leg"),
}

# Metrics where a missing or mismatched value makes two contracts unsafe.
METRIC_PATTERNS = [
    (r"\brushing\s+(?:touchdowns?|tds?)\b", "rushing_touchdowns"),
    (r"\breceiving\s+(?:touchdowns?|tds?)\b", "receiving_touchdowns"),
    (r"\bpassing\s+(?:touchdowns?|tds?)\b", "passing_touchdowns"),
    (r"\brushing\s+yards?\b", "rushing_yards"),
    (r"\breceiving\s+yards?\b", "receiving_yards"),
    (r"\bpassing\s+yards?\b", "passing_yards"),
    (r"\breceptions?\b", "receptions"),
    (r"\bsacks?\b", "sacks"),
    (r"\binterceptions?\b", "interceptions"),
    (r"\bhome\s+runs?\b", "home_runs"),
    (r"\bgoals?\b", "goals"), (r"\bassists?\b", "assists"),
    (r"\bpoints?\b", "points"), (r"\bseats?\b", "seats"),
    (r"\bvotes?\b", "votes"),
    (r"\b(?:google\s+)?search(?:ed|es)?\b|\bmost searched\b", "search_rank"),
    (r"\bmarket cap\b|\bmarket capitalization\b", "market_cap"),
    (r"\binflation\b|\bcpi\b", "inflation"),
    (r"\bunemployment\b", "unemployment"),
    (r"\bgdp\b", "gdp"),
    (r"\bfed(?:eral reserve)?\s+(?:rate|funds)|\binterest rate\b", "interest_rate"),
    (r"\btemperature\b|\bdegrees?\b", "temperature"),
]


@dataclass(frozen=True)
class ContractSignature:
    domain: str
    proposition: str
    subject: str | None
    competition: str | None
    context: str
    year: str | None
    stage: str | None
    threshold_op: str | None
    threshold_low: float | None
    threshold_high: float | None
    unit: str | None
    end_ts: float | None
    danger_flags: tuple[str, ...]
    # V8.8 semantic fields (defaults preserve compatibility with old tests/code).
    metric: str | None = None
    rank_semantics: str | None = None
    period_scope: str | None = None
    gender_scope: str | None = None
    sports_market_type: str | None = None
    resolution_source: str | None = None
    event_identity: str | None = None
    structured_complete: bool = False
    jurisdiction_country: str | None = None
    jurisdiction_region: str | None = None
    jurisdiction_district: str | None = None
    office_scope: str | None = None
    resolution_basis: str | None = None
    resolution_flags: tuple[str, ...] = ()
    resolution_rule_source_family: str | None = None
    resolution_rule_deadline_ts: float | None = None
    rule_text_present: bool = False
    rule_material_coverage: float = 0.0
    latest_settlement_ts: float | None = None


@dataclass
class MatchAudit:
    verdict: str
    score: float
    reasons: list[str]
    kalshi_ticker: str
    kalshi_title: str
    polymarket_question: str
    kalshi_signature: dict
    polymarket_signature: dict


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9%$\.\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {x for x in normalize_text(text).split() if x not in STOPWORDS and len(x) > 1}


def _parse_ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        text = str(value)
        if len(text) == 10:
            text += "T23:59:59+00:00"
        else:
            text = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _extract_year(text: str) -> str | None:
    m = re.search(r"\b(20\d{2})(?:\s*[-/]\s*(?:20)?\d{2})?\b", normalize_text(text))
    return m.group(1) if m else None


def _season_key(text: str) -> str | None:
    t = normalize_text(text)
    m = re.search(r"\b(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})\b", t)
    if m:
        end = m.group(2)
        if len(end) == 2:
            end = m.group(1)[:2] + end
        return f"{m.group(1)}-{end}"
    return _extract_year(t)


def _domain_from_metadata(category: str, fee_type: str, tags: list[str], text: str) -> str:
    candidates = [normalize_text(category), normalize_text(fee_type), *[normalize_text(x) for x in tags]]
    for c in candidates:
        for k, v in DOMAIN_ALIASES.items():
            if k and k in c:
                return v
    t = normalize_text(text)
    sports_markers = (" vs ", "championship", "league", "tournament", "super bowl", "ufc", "nba", "nfl", "nhl", "mlb", "world cup", "ballon d or", "grand slam", "touchdown", "yards", "chess olympiad")
    if any(x in f" {t} " for x in sports_markers):
        return "sports"
    if any(x in t for x in ("president", "election", "nominee", "congress", "senate", "governor", "prime minister")):
        return "politics"
    if any(x in t for x in ("inflation", "gdp", "unemployment", "interest rate", "fed ", "cpi", "jobs report")):
        return "economics"
    if any(x in t for x in ("bitcoin", "ethereum", "crypto", "solana")):
        return "crypto"
    if any(x in t for x in ("movie", "oscar", "grammy", "emmy", "box office", "album", "actor", "actress", "spotify", "google search")):
        return "entertainment"
    return "other"


def _competition(text: str) -> str | None:
    t = normalize_text(text)
    for pattern, key in COMPETITION_PATTERNS:
        if re.search(pattern, t):
            return key
    return None


def _stage(text: str) -> str | None:
    t = normalize_text(text)
    ordered = [
        ("first_round", ("first round", "1st round")), ("second_round", ("second round", "2nd round")),
        ("runoff", ("runoff", "run off")), ("primary", (" primary", "primary ")),
        ("regular_season", ("regular season",)), ("playoffs", ("playoff",)),
        ("finals", ("finals",)), ("final", (" final", "final ")),
        ("qualifying", ("qualifying", "qualify for")), ("group_stage", ("group stage",)),
    ]
    padded = f" {t} "
    for name, terms in ordered:
        if any(x in padded for x in terms):
            return name
    return None


def _proposition(text: str) -> str | None:
    t = f" {normalize_text(text)} "
    # Participation/appearance must be checked before generic winner words.
    if re.search(r"\b(compete|participate|appear|enter|play in|make the field|make the ballot|run for|running for)\b", t):
        return "participation"
    if "primary" in t and re.search(r"\b(win|winner|nominee)\b", t):
        return "primary_winner"
    if "top half" in t:
        return "top_half"
    if re.search(r"\btop\s+\d+\b", t) or re.search(r"\b(?:finish|rank|be)\s+#?\d+\b", t):
        return "rank_threshold"
    if re.search(r"\b(?:leader|lead the|most rushing|most receiving|most passing|most goals|most points|most searched)\b", t):
        return "leader"
    if re.search(r"\bexactly\s+\d", t):
        return "exact_count"
    if re.search(r"\bbetween\s+\$?\d", t):
        return "range"
    if any(x in t for x in (" at least ", " more than ", " greater than ", " above ", " over ")):
        return "above_threshold"
    if any(x in t for x in (" at most ", " less than ", " fewer than ", " below ", " under ")):
        return "below_threshold"
    if "relegat" in t:
        return "relegation"
    if "qualif" in t:
        return "qualify"
    if "make the playoffs" in t or "reach the playoffs" in t:
        return "make_playoffs"
    if re.search(r"\b(nominee|nomination)\b", t):
        return "winner"
    if re.search(r"\b(win|winner|wins|champion|championship|most seats)\b", t):
        return "winner"
    if t.strip().startswith(("will ", "is ", "does ")):
        return "binary_event"
    return None


def _metric(text: str) -> str | None:
    t = normalize_text(text)
    for pat, key in METRIC_PATTERNS:
        if re.search(pat, t):
            return key
    return None


def _rank_semantics(text: str, proposition: str | None) -> str | None:
    t = normalize_text(text)
    if "top half" in t:
        return "top_half"
    m = re.search(r"\btop\s+(\d+)\b", t)
    if m:
        return f"top_{int(m.group(1))}"
    m = re.search(r"\b(?:rank|be|finish)\s+#?(\d+)\b", t)
    if m:
        return f"exact_rank_{int(m.group(1))}"
    if proposition == "leader" or "most searched" in t:
        return "leader"
    return None


def _threshold(text: str, proposition: str | None):
    t = normalize_text(text)
    if proposition == "range":
        m = re.search(r"between\s+\$?([0-9]+(?:\.[0-9]+)?)\s*%?\s+and\s+\$?([0-9]+(?:\.[0-9]+)?)", t)
        if m:
            return "range", float(m.group(1)), float(m.group(2))
    if proposition == "exact_count":
        m = re.search(r"\bexactly\s+\$?([0-9]+(?:\.[0-9]+)?)", t)
        if m:
            return "eq", float(m.group(1)), None
    patterns = [
        ("ge", r"(?:at least|greater than or equal to)\s+\$?([0-9]+(?:\.[0-9]+)?)"),
        ("gt", r"(?:more than|greater than|above|over)\s+\$?([0-9]+(?:\.[0-9]+)?)"),
        ("le", r"(?:at most|less than or equal to)\s+\$?([0-9]+(?:\.[0-9]+)?)"),
        ("lt", r"(?:less than|fewer than|below|under)\s+\$?([0-9]+(?:\.[0-9]+)?)"),
    ]
    for op, pat in patterns:
        m = re.search(pat, t)
        if m:
            return op, float(m.group(1)), None
    return None, None, None


def _fallback_metric_threshold(text: str, metric: str | None, op, low, high):
    """Infer a numeric stat threshold only when the metric is explicit.

    This is designed for grouped markets whose event title is e.g. "6 rushing
    touchdowns" while the child market title is simply a player name.
    """
    if metric is None or low is not None:
        return op, low, high
    t = normalize_text(text)
    # Avoid years, rankings and unrelated IDs by requiring the number near a metric phrase.
    metric_words = {
        "rushing_touchdowns": r"rushing\s+(?:touchdowns?|tds?)",
        "receiving_touchdowns": r"receiving\s+(?:touchdowns?|tds?)",
        "passing_touchdowns": r"passing\s+(?:touchdowns?|tds?)",
        "rushing_yards": r"rushing\s+yards?", "receiving_yards": r"receiving\s+yards?",
        "passing_yards": r"passing\s+yards?", "seats": r"seats?",
    }
    pat = metric_words.get(metric)
    if not pat:
        return op, low, high
    m = re.search(rf"\b([0-9]+(?:\.[0-9]+)?)\s+{pat}\b", t)
    if m:
        return "threshold_unspecified", float(m.group(1)), None
    return op, low, high


def _unit(text: str) -> str | None:
    t = normalize_text(text)
    units = [
        (("%", "percent", "percentage"), "percent"), ((" goal", " goals"), "goals"),
        ((" point", " points"), "points"), ((" game", " games"), "games"),
        ((" map", " maps"), "maps"), ((" set", " sets"), "sets"),
        ((" seat", " seats"), "seats"), ((" vote", " votes"), "votes"),
        ((" touchdown", " touchdowns", " td ", " tds "), "touchdowns"),
        ((" yard", " yards"), "yards"), ((" degree", " degrees"), "degrees"),
        (("$", "usd", "dollar"), "usd"),
    ]
    for markers, key in units:
        if any(m in f" {t} " for m in markers):
            return key
    return None


def _period_scope(text: str) -> str | None:
    t = normalize_text(text)
    if "regular season" in t:
        return "regular_season"
    if "postseason" in t or "playoffs" in t:
        return "postseason"
    if "including playoffs" in t:
        return "including_playoffs"
    if "calendar year" in t:
        return "calendar_year"
    return None


def _gender_scope(text: str) -> str | None:
    t = normalize_text(text)
    if re.search(r"\b(women|women s|womens|female|ladies)\b", t):
        return "women"
    if re.search(r"\b(men|men s|mens|male)\b", t):
        return "men"
    return None


def _danger_flags(text: str) -> tuple[str, ...]:
    padded = f" {normalize_text(text)} "
    found = []
    for flag, terms in DANGER_FLAGS.items():
        if any(term in padded for term in terms):
            found.append(flag)
    if "interim" in padded:
        if re.search(r"interim.{0,45}(?:will not|do not|does not|not count|excluded|exclude)", padded):
            found.append("interim_excluded")
        elif re.search(r"interim.{0,45}(?:will count|count as|included|include)", padded):
            found.append("interim_included")
        else:
            found.append("interim_unspecified")
    if "overtime" in padded or "extra time" in padded:
        if re.search(r"(?:overtime|extra time).{0,45}(?:will not|does not|not count|excluded|exclude)", padded):
            found.append("overtime_excluded")
        elif re.search(r"(?:overtime|extra time).{0,45}(?:will count|included|include)", padded):
            found.append("overtime_included")
    return tuple(sorted(set(found)))


def _meaningful_subject(value: str | None) -> str | None:
    s = normalize_text(value)
    if not s or s in ("nan", "none", "yes", "no"):
        return None
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        return None
    if len(s) > 100 or len(s.split()) > 10:
        return None
    return s


def _extract_subject(question: str, explicit: str | None = None) -> str | None:
    e = _meaningful_subject(explicit)
    if e:
        return e
    q = str(question or "").strip()
    patterns = [
        r"^Will\s+(.+?)\s+(?:win|become|be|qualify|make|reach|retire|resign|leave|lose|compete|participate|record|have|get|finish)\b",
        r"^Does\s+(.+?)\s+\b", r"^Is\s+(.+?)\s+\b",
    ]
    for p in patterns:
        m = re.search(p, q, flags=re.I)
        if m:
            s = _meaningful_subject(m.group(1))
            if s:
                return s
    return None


def _context_key(event_title: str, question: str, subject: str | None) -> str:
    base = normalize_text(event_title) or normalize_text(question)
    if subject:
        base = re.sub(rf"\b{re.escape(subject)}\b", " ", base)
    base = re.sub(r"\b20\d{2}(?:\s*[-/]\s*(?:20)?\d{2})?\b", " ", base)
    generic = r"\b(?:will|who|be|become|the|in|market|yes|no)\b"
    base = re.sub(generic, " ", base)
    return re.sub(r"\s+", " ", base).strip()


def _event_identity(text: str, subject: str | None, metric: str | None) -> str:
    t = normalize_text(text)
    if subject:
        t = re.sub(rf"\b{re.escape(subject)}\b", " ", t)
    if metric:
        # Metric equality is checked separately; removing its vocabulary makes
        # identity focus on league/event/jurisdiction/season.
        for pat, key in METRIC_PATTERNS:
            if key == metric:
                t = re.sub(pat, " ", t)
    t = re.sub(r"\b20\d{2}(?:\s*[-/]\s*(?:20)?\d{2})?\b", " ", t)
    t = re.sub(r"\b(?:will|who|win|winner|wins|be|become|exactly|at least|more than|above|over|under|below|top|most|the|in|market|yes|no)\b", " ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()



_US_STATES = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
    "connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
    "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
    "maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
    "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
    "west virginia":"WV","wisconsin":"WI","wyoming":"WY","district of columbia":"DC","washington dc":"DC",
}

_COUNTRIES = {
    "united states":"US", "u s":"US", "usa":"US", "canada":"CA", "united kingdom":"GB", "uk":"GB",
    "france":"FR", "germany":"DE", "italy":"IT", "spain":"ES", "brazil":"BR", "argentina":"AR",
    "australia":"AU", "india":"IN", "china":"CN", "japan":"JP", "mexico":"MX", "ireland":"IE",
}

def _office_scope(text: str) -> str | None:
    t = normalize_text(text)
    if re.search(r"\b(?:u s |us |united states )?house(?: of representatives)?\b", t):
        return "us_house"
    if re.search(r"\b(?:u s |us |united states )?senate\b", t):
        return "us_senate"
    if re.search(r"\bgovernor|gubernatorial\b", t):
        return "governor"
    if re.search(r"\bpresident|presidential\b", t):
        return "president"
    if re.search(r"\bprime minister\b", t):
        return "prime_minister"
    if re.search(r"\bmayor|mayoral\b", t):
        return "mayor"
    return None

def _jurisdiction(text: str, domain: str | None = None, office_scope: str | None = None):
    """Extract payoff-defining geography conservatively.

    US states are only treated as US jurisdictions when the surrounding
    contract is recognisably US political/electoral. This avoids interpreting
    the country Georgia as the US state in unrelated markets.
    """
    t = f" {normalize_text(text)} "
    us_context = bool(office_scope in {"us_house", "us_senate"} or re.search(r"\b(?:u s|us|united states|midterm|congress)\b", t))
    region = None
    if us_context:
        # Longest names first so "carolina" style fragments cannot win.
        for name, abbr in sorted(_US_STATES.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(name)}\b", t):
                region = f"US-{abbr}"
                break
        # Congressional district forms: NH-01, CA-12, etc.
        m = re.search(r"\b([a-z]{2})[- ]?(\d{1,2})\b", t)
        district = f"US-{m.group(1).upper()}-{int(m.group(2)):02d}" if m and m.group(1).upper() in set(_US_STATES.values()) else None
        return "US", region, district
    country = None
    for name, code in sorted(_COUNTRIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", t):
            country = code
            break
    return country, None, None

def _location_sensitive(sig: ContractSignature) -> bool:
    return bool(sig.jurisdiction_region or sig.jurisdiction_district or (sig.domain == "politics" and sig.metric in {"seats", "votes"}))

def _poly_tags(market: dict) -> list[str]:
    tags: list[str] = []
    for obj in market.get("tags", []) or []:
        if isinstance(obj, dict):
            tags.extend([str(obj.get("label") or ""), str(obj.get("slug") or "")])
        else:
            tags.append(str(obj))
    for event in market.get("events", []) or []:
        for obj in event.get("tags", []) or []:
            if isinstance(obj, dict):
                tags.extend([str(obj.get("label") or ""), str(obj.get("slug") or "")])
            else:
                tags.append(str(obj))
    return [x for x in tags if x]


def _source_key(value: Any) -> str | None:
    t = normalize_text(value).replace(".", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # Keep informative provider/authority tokens; URLs and boilerplate vanish.
    toks = [x for x in t.split() if len(x) > 2 and x not in {"http", "https", "www", "com", "org", "source", "official"}]
    return " ".join(toks[:12]) or None


def _structured_kalshi_threshold(md: dict, op, low, high):
    if low is not None:
        return op, low, high
    strike_type = normalize_text(md.get("strike_type"))
    floor = md.get("floor_strike")
    cap = md.get("cap_strike")
    try:
        floor = float(floor) if floor is not None else None
    except Exception:
        floor = None
    try:
        cap = float(cap) if cap is not None else None
    except Exception:
        cap = None
    if floor is not None and cap is not None and abs(floor - cap) > 1e-12:
        return "range_structured", floor, cap
    if floor is not None:
        if "greater" in strike_type:
            return "ge_structured", floor, None
        if "less" in strike_type:
            return "le_structured", floor, None
        return "structured", floor, None
    return op, low, high


def kalshi_signature(row, metadata: dict | None = None) -> ContractSignature | None:
    md = metadata or {}
    event = md.get("_event", {}) or {}
    title = str(md.get("title") or row.get("title") or "")
    subtitle = str(md.get("subtitle") or row.get("subtitle") or "")
    yes_sub = str(md.get("yes_sub_title") or row.get("yes_sub_title") or "").strip()
    event_title = str(event.get("title") or "")
    event_subtitle = str(event.get("sub_title") or "")
    rules = " ".join(str(x or "") for x in (md.get("rules_primary"), md.get("rules_secondary"), md.get("early_close_condition")))
    milestone_text = " ".join(f"{m.get('category','')} {m.get('type','')} {m.get('title','')}" for m in event.get("_milestones", []) or [])
    combined = " ".join((title, subtitle, yes_sub, event_title, event_subtitle, rules, milestone_text))
    semantic_text = " ".join((title, subtitle, event_title, event_subtitle))
    proposition = _proposition(semantic_text)
    if proposition is None:
        return None
    subject = _extract_subject(title, yes_sub)
    metric = _metric(semantic_text)
    op, low, high = _threshold(semantic_text, proposition)
    op, low, high = _fallback_metric_threshold(semantic_text, metric, op, low, high)
    op, low, high = _structured_kalshi_threshold(md, op, low, high)
    category = str(event.get("category") or row.get("category") or "")
    domain = _domain_from_metadata(category, "", [], combined)
    competition = _competition(event_title + " " + title + " " + milestone_text)
    year = _season_key(event_title + " " + title + " " + subtitle)
    stage = _stage(title + " " + subtitle + " " + rules)
    context = _context_key(event_title or title, title, subject)
    identity = _event_identity(event_title or title, subject, metric)
    end_ts = _parse_ts(md.get("expected_expiration_time") or md.get("expiration_time") or md.get("close_time"))
    settlement_sources = event.get("settlement_sources", []) or []
    source_text = " ".join(str(x.get("name") or "") + " " + str(x.get("url") or "") for x in settlement_sources if isinstance(x, dict))
    rule_profile = build_resolution_profile(" ".join((rules, source_text)), market_year=year)
    known_ts = [
        _parse_ts(md.get("latest_expiration_time")),
        _parse_ts(md.get("settlement_ts")),
        _parse_ts(md.get("expiration_time")),
        _parse_ts(md.get("expected_expiration_time")),
        _parse_ts(md.get("close_time")),
        rule_profile.explicit_deadline_ts,
    ]
    known_ts = [x for x in known_ts if x is not None]
    latest_settlement_ts = max(known_ts) if known_ts else end_ts
    try:
        if latest_settlement_ts and md.get("settlement_timer_seconds"):
            latest_settlement_ts += max(0.0, float(md.get("settlement_timer_seconds") or 0.0))
    except (TypeError, ValueError):
        pass
    office = _office_scope(semantic_text + " " + event_title)
    j_country, j_region, j_district = _jurisdiction(semantic_text + " " + event_title + " " + event_subtitle, domain, office)
    structured_complete = bool(subject and proposition and (competition or identity) and (metric is None or proposition in ("winner", "leader") or low is not None))
    # State/district election/count contracts are not structurally complete without geography.
    if domain == "politics" and metric in {"seats", "votes"} and office in {"us_house", "us_senate", "governor"} and not (j_region or j_district):
        structured_complete = False
    return ContractSignature(
        domain, proposition, subject, competition, context, year, stage, op, low, high, _unit(semantic_text), end_ts,
        _danger_flags(title + " " + rules), metric, _rank_semantics(semantic_text, proposition), _period_scope(semantic_text + " " + rules),
        _gender_scope(semantic_text + " " + rules), None, _source_key(source_text), identity, structured_complete,
        j_country, j_region, j_district, office,
        rule_profile.basis, rule_profile.flags, rule_profile.source_family, rule_profile.explicit_deadline_ts,
        rule_profile.text_present, rule_profile.material_coverage, latest_settlement_ts,
    )


def polymarket_signature(market: dict) -> ContractSignature | None:
    question = str(market.get("question") or "")
    events = market.get("events", []) or []
    event = events[0] if events else {}
    event_title = str(event.get("title") or "")
    event_slug = str(event.get("slug") or "")
    description = " ".join(str(x or "") for x in (market.get("description"), event.get("description")))
    semantic_text = " ".join((question, event_title, str(market.get("groupItemRange") or ""), str(market.get("groupItemThreshold") or "")))
    combined = " ".join((semantic_text, event_slug, description))
    proposition = _proposition(semantic_text)
    if proposition is None:
        return None
    subject = _extract_subject(question, str(market.get("groupItemTitle") or "").strip())
    metric = _metric(semantic_text)
    op, low, high = _threshold(semantic_text, proposition)
    op, low, high = _fallback_metric_threshold(semantic_text, metric, op, low, high)
    # groupItemThreshold is structured metadata and can rescue grouped stat/count markets.
    if low is None:
        try:
            raw_thr = str(market.get("groupItemThreshold") or "").strip()
            m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", raw_thr)
            if m:
                low = float(m.group(0)); op = op or "threshold_unspecified"
        except Exception:
            pass
    tags = _poly_tags(market)
    domain = _domain_from_metadata(str(market.get("category") or ""), str(market.get("feeType") or ""), tags, combined)
    competition = _competition(event_title + " " + event_slug + " " + question + " " + " ".join(tags))
    year = _season_key(event_title + " " + question + " " + event_slug)
    stage = _stage(question + " " + event_title)
    context = _context_key(event_title or question, question, subject)
    identity = _event_identity(event_title or question, subject, metric)
    end_ts = _parse_ts(market.get("endDate") or market.get("endDateIso"))
    sports_type = normalize_text(market.get("sportsMarketType")) or None
    source = market.get("resolutionSource") or event.get("resolutionSource")
    rule_profile = build_resolution_profile(" ".join((description, str(source or ""))), market_year=year)
    known_ts = [
        end_ts,
        _parse_ts(event.get("endDate") or event.get("endDateIso")),
        _parse_ts(market.get("upperBoundDate")),
        rule_profile.explicit_deadline_ts,
    ]
    known_ts = [x for x in known_ts if x is not None]
    latest_settlement_ts = max(known_ts) if known_ts else end_ts
    office = _office_scope(semantic_text + " " + event_title)
    j_country, j_region, j_district = _jurisdiction(semantic_text + " " + event_title + " " + event_slug, domain, office)
    structured_complete = bool(subject and proposition and (competition or identity) and (metric is None or proposition in ("winner", "leader") or low is not None))
    if domain == "politics" and metric in {"seats", "votes"} and office in {"us_house", "us_senate", "governor"} and not (j_region or j_district):
        structured_complete = False
    return ContractSignature(
        domain, proposition, subject, competition, context, year, stage, op, low, high, _unit(semantic_text), end_ts,
        _danger_flags(question + " " + event_title + " " + description), metric, _rank_semantics(semantic_text, proposition),
        _period_scope(semantic_text + " " + description), _gender_scope(semantic_text + " " + description), sports_type,
        _source_key(source), identity, structured_complete, j_country, j_region, j_district, office,
        rule_profile.basis, rule_profile.flags, rule_profile.source_family, rule_profile.explicit_deadline_ts,
        rule_profile.text_present, rule_profile.material_coverage, latest_settlement_ts,
    )


def _similarity(a: str | None, b: str | None) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, a, b).ratio()
    return 0.70 * jaccard + 0.30 * seq


_ENTITY_GENERIC = {"fc", "afc", "cf", "the", "team", "club", "jr", "sr", "ii", "iii", "iv"}
_ENTITY_ALIAS_PHRASES = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "psg": "paris saint germain",
    "inter milan": "internazionale",
    "inter": "internazionale",
    "ny": "new york",
    "la": "los angeles",
    "jd vance": "j d vance",
}

def _canonical_entity(value: str | None) -> str:
    t = normalize_text(value).replace(".", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    for src, dst in _ENTITY_ALIAS_PHRASES.items():
        if t == src:
            t = dst
            break
    # Ignore middle initials and harmless club/person suffixes, but never drop
    # meaningful first/last/team-name tokens.
    toks = [x for x in t.split() if x not in _ENTITY_GENERIC]
    # Collapse consecutive initials ("J. D." -> "jd") before treating a
    # single middle initial as optional.
    collapsed=[]; i=0
    while i < len(toks):
        if len(toks[i]) == 1 and i + 1 < len(toks) and len(toks[i+1]) == 1:
            collapsed.append(toks[i] + toks[i+1]); i += 2
        else:
            collapsed.append(toks[i]); i += 1
    toks = collapsed
    if len(toks) >= 3:
        toks = [x for i, x in enumerate(toks) if not (len(x) == 1 and 0 < i < len(toks)-1)]
    return " ".join(toks)

def _entity_equivalent(a: str | None, b: str | None) -> tuple[bool, float]:
    ca, cb = _canonical_entity(a), _canonical_entity(b)
    if not ca or not cb:
        return False, 0.0
    if ca == cb:
        return True, 1.0
    ta, tb = set(ca.split()), set(cb.split())
    # Token-identical after safe normalization is strong evidence.
    if ta == tb and len(ta) >= 2:
        return True, 0.98
    # Conservative near-alias: all but one token agree, strings are highly
    # similar, and both sides retain at least two discriminative tokens.
    inter = len(ta & tb); union = len(ta | tb)
    j = inter / max(1, union)
    seq = SequenceMatcher(None, ca, cb).ratio()
    if min(len(ta), len(tb)) >= 2 and j >= 0.80 and seq >= 0.88:
        return True, 0.92
    return False, 0.70*j + 0.30*seq

def _hybrid_identity_similarity(a: str | None, b: str | None) -> float:
    """Fast lexical reranker used after blocking.

    Mirrors the retrieve->rerank architecture used by modern semantic-search
    systems: blocking is the retriever; this richer pair score is the cheap
    reranker. Hard payoff contradictions still decide acceptance.
    """
    aa, bb = normalize_text(a), normalize_text(b)
    if not aa or not bb:
        return 0.0
    base = _similarity(aa, bb)
    def grams(x):
        z = f"  {x}  "
        return {z[i:i+3] for i in range(max(0, len(z)-2))}
    ga, gb = grams(aa), grams(bb)
    char = len(ga & gb) / max(1, len(ga | gb))
    # Reward containment for venue boilerplate differences.
    contain = 1.0 if (aa in bb or bb in aa) and min(len(aa),len(bb)) >= 10 else 0.0
    enriched = min(1.0, 0.62*base + 0.28*char + 0.10*contain)
    # The reranker may add synonym/boilerplate evidence but must never make a
    # formerly strong structured identity look weaker than the legacy score.
    return max(base, enriched)


def _normalized_threshold(sig: ContractSignature):
    op = sig.threshold_op
    # Structured and textual variants with the same economic direction can agree.
    aliases = {"ge_structured": "ge", "le_structured": "le", "structured": "unspecified", "threshold_unspecified": "unspecified"}
    return aliases.get(op, op), sig.threshold_low, sig.threshold_high


def _thresholds_compatible(a: ContractSignature, b: ContractSignature) -> bool:
    ao, al, ah = _normalized_threshold(a)
    bo, bl, bh = _normalized_threshold(b)
    if al is None and bl is None and ah is None and bh is None:
        return True
    if (al is None) != (bl is None):
        return False
    if al is not None and not math.isclose(float(al), float(bl), rel_tol=0, abs_tol=1e-9):
        return False
    if (ah is None) != (bh is None):
        return False
    if ah is not None and not math.isclose(float(ah), float(bh), rel_tol=0, abs_tol=1e-9):
        return False
    # If one operator is unspecified, do not promote to EXACT: caller handles as REVIEW.
    if ao == bo:
        return True
    if "unspecified" in (ao, bo):
        return True
    return False


def _source_compatible(a: str | None, b: str | None) -> tuple[bool, bool]:
    """Return (compatible, explicit_mismatch).

    Resolution sources can legitimately differ while referring to the same
    objective result, so source absence is not a rejection. But when both are
    explicit and lexically disjoint, the pair is REVIEW rather than EXACT.
    """
    if not a or not b:
        return True, False
    sim = _similarity(a, b)
    if sim < 0.12:
        return False, True
    return True, False


def _flag_conflict(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[bool, str | None]:
    """Detect only *material* resolution-qualifier conflicts.

    V8.7 treated any missing flag as a REVIEW, which killed recall when one
    venue's description simply omitted words such as "regular season".  V8.8
    rejects contradictory qualifiers, while treating one-sided omissions as
    confidence penalties.
    """
    A, B = set(a), set(b)
    incompatible = [
        ({"first_round"}, {"second_round"}),
        ({"regular_season"}, {"playoffs", "finals"}),
        ({"popular_vote"}, {"electoral_college"}),
        ({"interim_included"}, {"interim_excluded"}),
        ({"overtime_included"}, {"overtime_excluded"}),
    ]
    for left, right in incompatible:
        if (A & left and B & right) or (A & right and B & left):
            return True, f"material qualifier conflict ({sorted(A)} vs {sorted(B)})"
    return False, None


def _resolution_profile_from_signature(sig: ContractSignature) -> ResolutionProfile:
    return ResolutionProfile(
        sig.resolution_basis,
        tuple(sig.resolution_flags or ()),
        sig.resolution_rule_source_family,
        sig.resolution_rule_deadline_ts,
        bool(sig.rule_text_present),
        float(sig.rule_material_coverage or 0.0),
    )


def _resolution_rule_comparison(left: ContractSignature, right: ContractSignature, *, force_strict: bool = False):
    family_sensitive = rule_sensitive_family(
        domain=left.domain if left.domain != "other" else right.domain,
        proposition=left.proposition,
        metric=left.metric or right.metric,
        office_scope=left.office_scope or right.office_scope,
    )
    # V16: production detail fetchers no longer make *every* market family
    # political-grade rule-sensitive. That V11/V15 behavior collapsed recall
    # to zero because ordinary objective contracts often expose sparse public
    # rule fingerprints. Strict mode still applies to genuinely rule-sensitive
    # families; other families are protected by hard semantic contradictions,
    # explicit source checks, and conservative settlement timing.
    sensitive = family_sensitive
    lp = _resolution_profile_from_signature(left)
    rp = _resolution_profile_from_signature(right)
    status, reasons = compare_resolution_profiles(lp, rp, rule_sensitive=sensitive)
    if status == "REVIEW":
        basis = assess_low_basis_risk(lp, rp, rule_sensitive=sensitive)
        if basis.status == "LOW_BASIS":
            return "LOW_BASIS", list(basis.reasons)
    return status, reasons


def _basis_risk_assessment(left: ContractSignature, right: ContractSignature, *, force_strict: bool = False):
    family_sensitive = rule_sensitive_family(
        domain=left.domain if left.domain != "other" else right.domain,
        proposition=left.proposition,
        metric=left.metric or right.metric,
        office_scope=left.office_scope or right.office_scope,
    )
    # V16: production detail fetchers no longer make *every* market family
    # political-grade rule-sensitive. That V11/V15 behavior collapsed recall
    # to zero because ordinary objective contracts often expose sparse public
    # rule fingerprints. Strict mode still applies to genuinely rule-sensitive
    # families; other families are protected by hard semantic contradictions,
    # explicit source checks, and conservative settlement timing.
    sensitive = family_sensitive
    return assess_low_basis_risk(
        _resolution_profile_from_signature(left),
        _resolution_profile_from_signature(right),
        rule_sensitive=sensitive,
    )


def _semantic_completeness(sig: ContractSignature) -> float:
    fields = [
        sig.proposition, sig.subject, sig.event_identity or sig.context,
        sig.year or sig.competition,
    ]
    score = sum(bool(x) for x in fields) / len(fields)
    if sig.metric:
        score += 0.10
    if sig.threshold_low is not None:
        score += 0.10
    if sig.period_scope:
        score += 0.05
    if sig.stage:
        score += 0.05
    if sig.jurisdiction_region or sig.jurisdiction_district:
        score += 0.08
    if sig.office_scope:
        score += 0.05
    return min(1.0, score)


def evaluate_equivalence(left: ContractSignature, right: ContractSignature, *, require_rule_completeness: bool = False) -> tuple[str, float, list[str]]:
    """Classify a pair as EXACT, HIGH_CONFIDENCE, REVIEW or REJECT.

    Critical payoff semantics (entity, proposition, metric/rank, threshold,
    competition and explicit contradictory scope) remain hard gates. Optional
    metadata omissions reduce confidence rather than automatically rejecting a
    pair. This recovers true contracts whose public metadata differ in
    completeness while preserving the V8.6 false-positive regressions.
    """
    reasons: list[str] = []
    penalties = 0.0

    # 1) Payoff proposition is critical.
    if left.proposition != right.proposition:
        return "REJECT", 0.0, [f"proposition differs ({left.proposition} vs {right.proposition})"]
    reasons.append(f"proposition exact ({left.proposition})")

    # 2) Domain mismatch is only hard when both are confidently non-generic.
    if left.domain != right.domain and "other" not in (left.domain, right.domain):
        return "REJECT", 0.0, [f"domain differs ({left.domain} vs {right.domain})"]

    # V9.0: payoff-defining geography and political office are hard semantics.
    if left.office_scope and right.office_scope and left.office_scope != right.office_scope:
        return "REJECT", 0.0, [f"office/chamber differs ({left.office_scope} vs {right.office_scope})"]
    if left.office_scope or right.office_scope:
        if not (left.office_scope and right.office_scope):
            return "REVIEW", 0.45, ["office/chamber missing on one venue"]
        reasons.append(f"office/chamber exact ({left.office_scope})")

    if left.jurisdiction_country and right.jurisdiction_country and left.jurisdiction_country != right.jurisdiction_country:
        return "REJECT", 0.0, [f"country differs ({left.jurisdiction_country} vs {right.jurisdiction_country})"]
    if left.jurisdiction_region or right.jurisdiction_region:
        if not left.jurisdiction_region or not right.jurisdiction_region:
            return "REVIEW", 0.40, ["jurisdiction region missing on one venue"]
        if left.jurisdiction_region != right.jurisdiction_region:
            return "REJECT", 0.0, [f"jurisdiction differs ({left.jurisdiction_region} vs {right.jurisdiction_region})"]
        reasons.append(f"jurisdiction exact ({left.jurisdiction_region})")
    if left.jurisdiction_district or right.jurisdiction_district:
        if not left.jurisdiction_district or not right.jurisdiction_district:
            return "REVIEW", 0.40, ["district missing on one venue"]
        if left.jurisdiction_district != right.jurisdiction_district:
            return "REJECT", 0.0, [f"district differs ({left.jurisdiction_district} vs {right.jurisdiction_district})"]
        reasons.append(f"district exact ({left.jurisdiction_district})")

    # A state/district seat/vote market cannot be HIGH_CONFIDENCE if geography
    # was not successfully extracted on both venues.
    if (left.domain == "politics" or right.domain == "politics") and (left.metric in {"seats", "votes"} or right.metric in {"seats", "votes"}):
        if not (left.jurisdiction_country and right.jurisdiction_country):
            return "REVIEW", 0.40, ["political seat/vote contract missing jurisdiction country"]

    # 3) Entity/subject is critical whenever either contract is entity-specific.
    if left.subject or right.subject:
        if not left.subject or not right.subject:
            return "REVIEW", 0.45, ["subject missing on one venue"]
        entity_ok, entity_score = _entity_equivalent(left.subject, right.subject)
        if not entity_ok:
            return "REJECT", entity_score, ["subject differs"]
        if entity_score >= 0.99:
            reasons.append("subject exact")
        else:
            penalties += 0.03
            reasons.append(f"subject alias-equivalent ({entity_score:.2f})")

    # 4) Metric and rank semantics are critical. These are the main guards
    # against rushing-TD vs yards-leader, winner vs top-half, etc.
    if left.metric != right.metric and (left.metric or right.metric):
        return "REJECT", 0.0, [f"metric differs ({left.metric} vs {right.metric})"]
    if left.metric:
        reasons.append(f"metric exact ({left.metric})")
    if left.rank_semantics != right.rank_semantics and (left.rank_semantics or right.rank_semantics):
        return "REJECT", 0.0, [f"rank semantics differ ({left.rank_semantics} vs {right.rank_semantics})"]
    if left.rank_semantics:
        reasons.append(f"rank exact ({left.rank_semantics})")

    # 5) Numeric thresholds are critical for stat/count/range markets.
    if not _thresholds_compatible(left, right):
        return "REJECT", 0.0, ["threshold semantics differ"]
    lo, ll, lh = _normalized_threshold(left)
    ro, rl, rh = _normalized_threshold(right)
    if ll is not None or rl is not None:
        if "unspecified" in (lo, ro) and lo != ro:
            penalties += 0.08
            reasons.append("threshold value exact; comparator omitted on one venue")
        else:
            reasons.append("threshold exact")

    stat_metrics = {"rushing_touchdowns", "receiving_touchdowns", "passing_touchdowns", "rushing_yards", "receiving_yards", "passing_yards", "receptions", "sacks", "interceptions", "home_runs", "goals", "assists", "points"}
    if left.metric in stat_metrics and left.proposition != "leader":
        if left.threshold_low is None or right.threshold_low is None:
            return "REVIEW", 0.55, ["player/team stat contract missing explicit threshold"]

    if left.unit and right.unit and left.unit != right.unit:
        return "REJECT", 0.0, [f"unit differs ({left.unit} vs {right.unit})"]

    # 6) Time/competition identity.
    if left.year and right.year and left.year != right.year:
        return "REJECT", 0.0, [f"year/season differs ({left.year} vs {right.year})"]
    if left.year and right.year:
        reasons.append("year/season exact")
    else:
        penalties += 0.04

    if left.competition and right.competition and left.competition != right.competition:
        return "REJECT", 0.0, [f"competition differs ({left.competition} vs {right.competition})"]
    if left.competition and right.competition:
        reasons.append("competition exact")
    elif left.domain == "sports" or right.domain == "sports":
        penalties += 0.06

    # 7) Scope conflicts are hard only when both sides explicitly disagree.
    if left.stage and right.stage and left.stage != right.stage:
        return "REJECT", 0.0, [f"stage differs ({left.stage} vs {right.stage})"]
    if left.stage and right.stage:
        reasons.append("stage exact")
    elif left.stage or right.stage:
        penalties += 0.05
        reasons.append("stage omitted on one venue")

    if left.period_scope and right.period_scope and left.period_scope != right.period_scope:
        return "REJECT", 0.0, [f"period scope differs ({left.period_scope} vs {right.period_scope})"]
    if left.period_scope and right.period_scope:
        reasons.append("period scope exact")
    elif left.period_scope or right.period_scope:
        penalties += 0.05
        reasons.append("period scope omitted on one venue")

    if left.gender_scope and right.gender_scope and left.gender_scope != right.gender_scope:
        return "REJECT", 0.0, [f"gender scope differs ({left.gender_scope} vs {right.gender_scope})"]
    if left.gender_scope and right.gender_scope:
        reasons.append("gender scope exact")
    elif left.gender_scope or right.gender_scope:
        penalties += 0.08
        reasons.append("gender scope omitted on one venue")

    if right.sports_market_type in {"spreads", "spread", "totals", "total"} and left.proposition == "winner":
        return "REJECT", 0.0, [f"Polymarket sports market type incompatible ({right.sports_market_type})"]

    conflict, conflict_reason = _flag_conflict(left.danger_flags, right.danger_flags)
    if conflict:
        return "REJECT", 0.0, [conflict_reason or "resolution qualifier conflict"]
    if set(left.danger_flags) != set(right.danger_flags):
        penalties += 0.05
        reasons.append("non-conflicting qualifier metadata differs")

    # V11.0: question/title equivalence is not enough. Public settlement-rule
    # text must not contain a payout-changing basis-risk mismatch. This blocks
    # contracts such as election-result vs take-office and hard fallback-to-
    # Other rules, while retaining objective markets whose rule fingerprints
    # are compatible.
    rule_status, rule_reasons = _resolution_rule_comparison(left, right, force_strict=require_rule_completeness)
    low_basis = rule_status == "LOW_BASIS"
    if rule_status == "REVIEW":
        return "REVIEW", 0.50, rule_reasons
    reasons.extend(rule_reasons)
    if rule_status == "COMPATIBLE":
        penalties += 0.03
    elif low_basis:
        # V12 keeps these separate from guaranteed arbitrage and execution
        # later subtracts an explicit per-contract basis-risk reserve.
        penalties += 0.10
        reasons.append("V13 low-basis presentation lane")

    # Resolution timestamps on prediction venues often include administrative
    # settlement lag. Large gaps are unsafe, moderate gaps reduce confidence.
    if left.end_ts and right.end_ts:
        gap_days = abs(left.end_ts - right.end_ts) / 86400.0
        if gap_days > 120.0:
            # Administrative payout timing alone is not a payoff mismatch. V16
            # keeps the pair eligible but penalizes it heavily; the allocator
            # uses the later cross-venue settlement timestamp for APR/capital
            # lock calculations. Payout-changing deadline/fallback clauses are
            # still rejected by the resolution-rule layer above.
            penalties += 0.14
            reasons.append(f"long administrative resolution-horizon gap ({gap_days:.0f}d); later horizon used")
        elif gap_days > 30.0:
            penalties += 0.08
            reasons.append(f"resolution horizon differs ({gap_days:.0f}d); later horizon used")
        else:
            reasons.append("resolution horizon compatible")
    else:
        penalties += 0.04

    # Event identity is supporting evidence, not the sole semantic gate. The
    # threshold is lower when exact structured semantics already pin the payoff.
    identity_score = _hybrid_identity_similarity(left.event_identity or left.context, right.event_identity or right.context)
    if left.competition and right.competition and left.competition == right.competition:
        if left.metric and right.metric and left.metric == right.metric and left.subject == right.subject and _thresholds_compatible(left, right):
            floor = 0.35
        else:
            floor = 0.42 if left.proposition in {"winner", "participation", "primary_winner"} else 0.52
    elif left.metric and right.metric and left.metric == right.metric:
        # Exact entity+metric+threshold/year is already highly specific; event
        # titles often differ only by group-label boilerplate.
        floor = 0.35 if (left.subject == right.subject and _thresholds_compatible(left, right) and left.year == right.year) else 0.50
    else:
        if left.subject and right.subject and left.subject == right.subject and left.year and left.year == right.year and left.proposition == right.proposition:
            floor = 0.55
        else:
            floor = 0.62
    if identity_score < floor:
        return "REJECT", identity_score, [f"event identity too different ({identity_score:.2f})"]
    if identity_score < floor + 0.12:
        penalties += 0.10
        reasons.append(f"event identity moderate ({identity_score:.2f})")
    else:
        reasons.append(f"event identity strong ({identity_score:.2f})")

    compatible_source, explicit_source_mismatch = _source_compatible(left.resolution_source, right.resolution_source)
    if explicit_source_mismatch:
        return "REVIEW", min(identity_score, 0.70), ["explicit resolution sources differ"]
    if left.resolution_source and right.resolution_source and compatible_source:
        reasons.append("resolution source compatible")
    elif left.resolution_source or right.resolution_source:
        penalties += 0.03

    # Generic binary events remain intentionally hard. We do not want broad
    # entity/context matching to manufacture hedges.
    if left.proposition == "binary_event":
        if not (left.structured_complete and right.structured_complete):
            return "REVIEW", min(identity_score, 0.68), ["generic binary event lacks complete structured semantics"]
        if identity_score < 0.90:
            return "REVIEW", identity_score, ["generic binary event requires near-identical event identity"]

    completeness = min(_semantic_completeness(left), _semantic_completeness(right))
    base = 0.72 + 0.18 * identity_score + 0.10 * completeness
    score = max(0.0, min(1.0, base - penalties))

    # EXACT requires very complete evidence. HIGH_CONFIDENCE allows harmless
    # one-sided omissions but never a critical semantic conflict.
    if score >= 0.91 and penalties <= 0.08 and not low_basis:
        return "EXACT", score, reasons
    if score >= 0.76 and completeness >= 0.70:
        return "HIGH_CONFIDENCE", score, reasons
    # A low-basis pair already passed all hard payoff semantics. Permit a
    # slightly lower confidence floor only because execution applies a small
    # dedicated capital sleeve and subtracts the modelled basis reserve.
    if low_basis and score >= 0.72 and completeness >= 0.75:
        return "HIGH_CONFIDENCE", score, reasons
    return "REVIEW", score, reasons

def _subject_block_key(subject: str | None) -> str | None:
    s = normalize_text(subject)
    if not s:
        return None
    # Normalize initials so "J.D. Vance" and "JD Vance" block together.
    parts = s.split()
    if len(parts) >= 2:
        compact = []
        i = 0
        while i < len(parts):
            if len(parts[i]) == 1 and i + 1 < len(parts) and len(parts[i + 1]) == 1:
                compact.append(parts[i] + parts[i + 1]); i += 2
            else:
                compact.append(parts[i]); i += 1
        s = " ".join(compact)
    return s


def _candidate_keys(sig: ContractSignature) -> list[tuple]:
    """Broad blocking keys; strict semantics are checked afterwards.

    V8.7 indexed on proposition+metric+threshold and therefore never compared
    many true pairs when one venue omitted a structured field. V8.8 blocks on
    entity/event identity first, then lets evaluate_equivalence do the safety
    work.
    """
    keys: list[tuple] = []
    subj = _subject_block_key(sig.subject)
    yr = sig.year or "*"
    if subj:
        # For location-sensitive contracts, geography belongs in the blocking
        # key itself. This prevents generic entities such as "democrats" from
        # producing huge cross-state candidate pools.
        if _location_sensitive(sig) and sig.jurisdiction_region:
            keys.append(("subject_region_year", subj, sig.jurisdiction_region, yr))
            if sig.jurisdiction_district:
                keys.append(("subject_district_year", subj, sig.jurisdiction_district, yr))
        else:
            keys.append(("subject_year", subj, yr))
            keys.append(("subject_any", subj))
        if sig.competition:
            keys.append(("subject_comp", subj, sig.competition))
    if sig.competition:
        keys.append(("comp_year", sig.competition, yr))
    # V16 high-recall structured retrieval. These keys recover candidates when
    # entity wording differs but payoff-defining structured fields agree. Hard
    # entity/proposition/threshold checks still run before acceptance.
    thr = None if sig.threshold_low is None else round(float(sig.threshold_low), 6)
    if sig.proposition and (sig.metric or sig.office_scope or sig.competition):
        keys.append(("structured", sig.domain, sig.proposition, sig.metric or "*", thr, sig.jurisdiction_region or "*", sig.office_scope or "*", yr))
    if sig.event_identity:
        toks = sorted(_tokens(sig.event_identity))
        for tok in toks[:3]:
            if len(tok) >= 4:
                keys.append(("event_token", sig.domain, yr, tok))
    core = tuple(sorted(_tokens(sig.event_identity or sig.context))[:6])
    if core:
        keys.append(("event_core", sig.domain, yr, core))
    return keys

def _build_match(ticker: str, row, pm: dict, ksig: ContractSignature, psig: ContractSignature, score: float, reasons: list[str], source: str, *, strict_rules: bool = False):
    rule_status, rule_reasons = _resolution_rule_comparison(ksig, psig, force_strict=strict_rules)
    basis_assessment = _basis_risk_assessment(ksig, psig, force_strict=strict_rules)
    basis_reserve = float(basis_assessment.reserve_per_contract) if rule_status == "LOW_BASIS" else 0.0
    resolution_lane = "LOW_BASIS" if rule_status == "LOW_BASIS" else "STRICT_ARB"
    latest_candidates = [x for x in (ksig.latest_settlement_ts, psig.latest_settlement_ts) if x is not None]
    latest_cross_settlement_ts = max(latest_candidates) if latest_candidates else None
    certificate = {
        "proposition": ksig.proposition,
        "subject": ksig.subject,
        "metric": ksig.metric,
        "threshold_op": ksig.threshold_op,
        "threshold_low": ksig.threshold_low,
        "threshold_high": ksig.threshold_high,
        "rank_semantics": ksig.rank_semantics,
        "period_scope": ksig.period_scope,
        "gender_scope": ksig.gender_scope,
        "competition": ksig.competition,
        "year": ksig.year,
        "stage": ksig.stage,
        "event_identity": ksig.event_identity,
        "jurisdiction_country": ksig.jurisdiction_country,
        "jurisdiction_region": ksig.jurisdiction_region,
        "jurisdiction_district": ksig.jurisdiction_district,
        "office_scope": ksig.office_scope,
        "structured_complete": bool(ksig.structured_complete and psig.structured_complete),
        "resolution_rule_status": rule_status,
        "resolution_rule_reasons": "; ".join(rule_reasons),
        "resolution_lane": resolution_lane,
        "basis_risk_reserve_per_contract": basis_reserve,
        "basis_risk_reasons": "; ".join(basis_assessment.reasons) if resolution_lane == "LOW_BASIS" else "",
        "kalshi_resolution_basis": ksig.resolution_basis,
        "polymarket_resolution_basis": psig.resolution_basis,
        "kalshi_resolution_flags": list(ksig.resolution_flags or ()),
        "polymarket_resolution_flags": list(psig.resolution_flags or ()),
        "kalshi_rule_text_present": bool(ksig.rule_text_present),
        "polymarket_rule_text_present": bool(psig.rule_text_present),
        "kalshi_latest_settlement_ts": ksig.latest_settlement_ts,
        "polymarket_latest_settlement_ts": psig.latest_settlement_ts,
        "latest_cross_settlement_ts": latest_cross_settlement_ts,
        "match_verdict": source.split(":", 1)[0] if ":" in source else None,
    }
    return {
        "kalshi_ticker": ticker,
        "kalshi_title": row.get("title"),
        "kalshi_yes_sub_title": row.get("yes_sub_title"),
        "signature": {
            "proposition": ksig.proposition,
            "topic": ksig.competition or ksig.event_identity or ksig.context or ksig.domain,
            "year": ksig.year, "subject": ksig.subject, "threshold": ksig.threshold_low,
            "domain": ksig.domain, "competition": ksig.competition, "stage": ksig.stage,
            "metric": ksig.metric, "rank_semantics": ksig.rank_semantics,
            "jurisdiction_country": ksig.jurisdiction_country, "jurisdiction_region": ksig.jurisdiction_region,
            "jurisdiction_district": ksig.jurisdiction_district, "office_scope": ksig.office_scope,
        },
        "polymarket_question": pm.get("question"), "polymarket_market": pm,
        "match_source": source, "equivalence_score": round(score, 6),
        "equivalence_reasons": "; ".join(reasons), "equivalence_certificate": certificate,
        "kalshi_signature": asdict(ksig), "polymarket_signature": asdict(psig),
    }


class BoundedAuditList(list):
    """Small in-memory audit sample plus exact aggregate counters.

    V8.9 deliberately does not retain every rejected candidate. On the expanded
    universe that can be millions of rows and previously caused a ~9.7 GB Arrow
    allocation. Aggregate counts remain exact; only representative rows are kept.
    """
    def __init__(self, max_rows: int = 25000):
        super().__init__()
        self.max_rows = max_rows
        self.verdict_counts = Counter()
        self.reason_counts = Counter()
        self.total_seen = 0

    def record(self, audit: MatchAudit):
        self.total_seen += 1
        self.verdict_counts[audit.verdict] += 1
        for reason in audit.reasons or ["unknown"]:
            self.reason_counts[reason] += 1
        # Always retain accepted rows; sample REVIEW/REJECT up to the cap.
        if audit.verdict in {"EXACT", "HIGH_CONFIDENCE"} or len(self) < self.max_rows:
            self.append(audit)


def _needs_resolution_rule_hydration(verdict: str, reasons: list[str], left: ContractSignature, right: ContractSignature) -> bool:
    """True when a structurally plausible pair failed only because rule evidence is thin.

    Hydration is intentionally lazy: the full 100k×178k universe is never
    detail-fetched. Only pairs that survive the payoff-semantic gates can cause
    a market-detail request.
    """
    # Even if V13 can eventually grade sparse evidence into LOW_BASIS, prefer
    # fetching public detail first whenever either side lacks rule text. This
    # preserves the stronger V11 behavior and avoids paying a basis reserve for
    # information that is actually available from the venue.
    if not left.rule_text_present or not right.rule_text_present:
        return True
    if verdict != "REVIEW":
        return False
    text = " ".join(str(x) for x in (reasons or [])).lower()
    return any(token in text for token in (
        "resolution rules missing",
        "resolution basis missing",
        "rule evidence",
        "insufficient material rule coverage",
    ))


def _unwrap_market_detail(value):
    if not isinstance(value, dict):
        return {}
    market = value.get("market")
    return market if isinstance(market, dict) else value


def find_universal_matches(
    kalshi_markets,
    polymarket_markets: list[dict],
    kalshi_market_metadata: dict[str, dict] | None = None,
    *,
    include_legacy: bool = True,
    kalshi_detail_fetcher=None,
    polymarket_detail_fetcher=None,
):
    metadata = kalshi_market_metadata or {}
    audits = BoundedAuditList(max_rows=25000)
    matches: list[dict] = []

    # V11 lazily enriches only rule-sensitive pairs that have already survived
    # all cheaper payoff checks. This preserves the ~full market universe while
    # avoiding tens of thousands of unnecessary detail API calls.
    hydrated_kmeta: dict[str, dict] = {}
    hydrated_ksig: dict[str, ContractSignature] = {}
    hydrated_pmarket: dict[str, dict] = {}
    hydrated_psig: dict[str, ContractSignature] = {}
    hydration_counts = Counter()
    # The production engine supplies both public detail fetchers. In that mode
    # every accepted cross-venue pair must prove material rule completeness.
    # Direct matcher unit tests without fetchers retain legacy default behavior.
    strict_rule_mode = bool(kalshi_detail_fetcher is not None and polymarket_detail_fetcher is not None)

    poly_sigs: dict[str, ContractSignature] = {}
    poly_index: dict[tuple, list[tuple[dict, ContractSignature]]] = {}
    for pm in polymarket_markets:
        sig = polymarket_signature(pm)
        if sig is None:
            continue
        pid = str(pm.get("id"))
        poly_sigs[pid] = sig
        for key in _candidate_keys(sig):
            poly_index.setdefault(key, []).append((pm, sig))

    row_by_ticker = {}
    ksig_by_ticker = {}
    seen_candidates: set[tuple[str, str]] = set()
    for _, row in kalshi_markets.iterrows():
        ticker = str(row.get("ticker") or "")
        row_by_ticker[ticker] = row
        ksig = kalshi_signature(row, metadata.get(ticker))
        if ksig is None:
            continue
        ksig_by_ticker[ticker] = ksig
        pool: dict[str, tuple[dict, ContractSignature]] = {}
        for key in _candidate_keys(ksig):
            for pm, psig in poly_index.get(key, []):
                pool[str(pm.get("id"))] = (pm, psig)
        for pid, (pm, psig0) in pool.items():
            ckey = (ticker, pid)
            if ckey in seen_candidates:
                continue
            seen_candidates.add(ckey)

            current_ksig = hydrated_ksig.get(ticker, ksig)
            current_pm = hydrated_pmarket.get(pid, pm)
            current_psig = hydrated_psig.get(pid, psig0)
            verdict, score, reasons = evaluate_equivalence(current_ksig, current_psig, require_rule_completeness=strict_rule_mode)

            if _needs_resolution_rule_hydration(verdict, reasons, current_ksig, current_psig):
                changed = False
                if not current_ksig.rule_text_present and kalshi_detail_fetcher is not None:
                    if ticker not in hydrated_kmeta:
                        try:
                            detail = _unwrap_market_detail(kalshi_detail_fetcher(ticker))
                            base = dict(metadata.get(ticker) or {})
                            event_ctx = base.get("_event")
                            merged = {**base, **detail}
                            if event_ctx is not None:
                                merged["_event"] = event_ctx
                            hydrated_kmeta[ticker] = merged
                            hydration_counts["kalshi_success"] += 1
                        except Exception:
                            hydrated_kmeta[ticker] = dict(metadata.get(ticker) or {})
                            hydration_counts["kalshi_failure"] += 1
                    candidate_sig = kalshi_signature(row, hydrated_kmeta[ticker])
                    if candidate_sig is not None:
                        current_ksig = candidate_sig
                        hydrated_ksig[ticker] = candidate_sig
                        ksig_by_ticker[ticker] = candidate_sig
                        changed = True

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
                    candidate_sig = polymarket_signature(current_pm)
                    if candidate_sig is not None:
                        current_psig = candidate_sig
                        hydrated_psig[pid] = candidate_sig
                        poly_sigs[pid] = candidate_sig
                        changed = True

                if changed:
                    verdict, score, reasons = evaluate_equivalence(current_ksig, current_psig, require_rule_completeness=strict_rule_mode)

            audits.record(MatchAudit(verdict, score, reasons, ticker, str(row.get("title") or ""), str(current_pm.get("question") or ""), asdict(current_ksig), asdict(current_psig)))
            if verdict in {"EXACT", "HIGH_CONFIDENCE"}:
                matches.append(_build_match(ticker, row, current_pm, current_ksig, current_psig, score, reasons, f"{verdict}:v16.0_hybrid_matcher", strict_rules=strict_rule_mode))

    # Legacy candidates are discovery hints only; they still must pass V8.8.
    if include_legacy:
        try:
            legacy = find_legacy_matches(kalshi_markets, polymarket_markets)
        except Exception:
            legacy = []
        for m in legacy:
            ticker = str(m.get("kalshi_ticker") or "")
            pm = m.get("polymarket_market") or {}
            pid = str(pm.get("id") or "")
            ckey = (ticker, pid)
            if ckey in seen_candidates:
                continue
            ksig = hydrated_ksig.get(ticker) or ksig_by_ticker.get(ticker)
            current_pm = hydrated_pmarket.get(pid, pm)
            psig = hydrated_psig.get(pid) or poly_sigs.get(pid) or polymarket_signature(current_pm)
            row = row_by_ticker.get(ticker)
            if ksig is None or psig is None or row is None:
                continue
            verdict, score, reasons = evaluate_equivalence(ksig, psig, require_rule_completeness=strict_rule_mode)
            # Legacy is only a discovery hint. If it reaches the same missing-rule
            # review state, apply the identical lazy detail hydration before it can
            # ever become an accepted match.
            if _needs_resolution_rule_hydration(verdict, reasons, ksig, psig):
                if not ksig.rule_text_present and kalshi_detail_fetcher is not None:
                    try:
                        if ticker not in hydrated_kmeta:
                            detail = _unwrap_market_detail(kalshi_detail_fetcher(ticker))
                            base = dict(metadata.get(ticker) or {})
                            event_ctx = base.get("_event")
                            merged = {**base, **detail}
                            if event_ctx is not None:
                                merged["_event"] = event_ctx
                            hydrated_kmeta[ticker] = merged
                        ksig = kalshi_signature(row, hydrated_kmeta[ticker]) or ksig
                        hydrated_ksig[ticker] = ksig
                    except Exception:
                        hydration_counts["kalshi_failure"] += 1
                if not psig.rule_text_present and polymarket_detail_fetcher is not None:
                    try:
                        if pid not in hydrated_pmarket:
                            detail = _unwrap_market_detail(polymarket_detail_fetcher(pid))
                            hydrated_pmarket[pid] = {**current_pm, **detail}
                        current_pm = hydrated_pmarket[pid]
                        psig = polymarket_signature(current_pm) or psig
                        hydrated_psig[pid] = psig
                    except Exception:
                        hydration_counts["polymarket_failure"] += 1
                verdict, score, reasons = evaluate_equivalence(ksig, psig, require_rule_completeness=strict_rule_mode)
            audits.record(MatchAudit(verdict, score, ["legacy candidate revalidated", *reasons], ticker, str(row.get("title") or ""), str(current_pm.get("question") or ""), asdict(ksig), asdict(psig)))
            if verdict in {"EXACT", "HIGH_CONFIDENCE"}:
                matches.append(_build_match(ticker, row, current_pm, ksig, psig, score, ["legacy candidate revalidated", *reasons], f"{verdict}:legacy_v16.0", strict_rules=strict_rule_mode))

    dedup: dict[tuple[str, str], dict] = {}
    for m in matches:
        key = (str(m["kalshi_ticker"]), str(m["polymarket_market"].get("id")))
        current = dedup.get(key)
        if current is None or float(m.get("equivalence_score", 0)) > float(current.get("equivalence_score", 0)):
            dedup[key] = m
    audits.hydration_counts = dict(hydration_counts)
    return list(dedup.values()), audits

