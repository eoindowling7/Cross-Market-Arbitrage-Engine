"""Conservative resolution-rule comparison for cross-venue prediction markets.

This module is deliberately rule-first.  Headline-equivalent markets are not
considered safe cross-venue arbitrage if their settlement contracts can pay
out differently in edge cases (for example, election result vs taking office,
or a venue-specific hard fallback deadline).

It extracts a small set of payoff-defining rule semantics from the public
market-rule text and provides a venue-agnostic compatibility verdict.  The
parser is intentionally conservative: missing material rule evidence produces
REVIEW rather than a profitable paper trade.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


def _norm(value) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass(frozen=True)
class ResolutionProfile:
    basis: str | None
    flags: tuple[str, ...]
    source_family: str | None
    explicit_deadline_ts: float | None
    text_present: bool
    material_coverage: float


# These flags can change which side receives $1, rather than merely when the
# venue administratively posts settlement.
MATERIAL_FLAGS = {
    "requires_take_office",
    "official_declaration_required",
    "certification_required",
    "final_court_outcome",
    "fallback_other",
    "fallback_market_price",
    "postponement_stays_open",
    "special_elections_included",
    "special_elections_excluded",
    "by_elections_included",
    "by_elections_excluded",
    "runoff_included",
    "runoff_excluded",
    "independent_caucus_counts",
    "party_at_election_controls",
    "party_at_resolution_controls",
    "death_after_win_counts",
    "death_after_win_excluded",
    "vacancy_after_election_counts_original_party",
    "overtime_included",
    "overtime_excluded",
    "shootout_included",
    "shootout_excluded",
    "cancelled_event_stays_open",
    "cancelled_event_voids",
    "cancelled_event_half_payout",
    "stat_corrections_included",
    "stat_corrections_excluded",
}


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(p in text for p in phrases)


def _extract_basis(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None

    if _contains_any(t, ("subsequently sworn in", "sworn in", "takes office", "take office", "taking office", "inauguration date", "assume office")):
        return "office_assumption"
    if _contains_any(t, ("final certified outcome", "official certification", "certified result", "certified outcome")):
        return "certified_result"
    if _contains_any(t, ("officially declared elected", "officially declared winner", "declared elected", "declared the winner")):
        return "declared_result"
    if (
        _contains_any(t, ("result of the election", "results of this election", "wins the election",
                          "winner of the election", "who wins the election", "number of seats",
                          "seats won", "seat count", "election winner"))
        or re.search(r"\bnumber of (?:[a-z0-9.-]+\s+){0,5}seats\b", t)
        or ("election" in t and " seats " in f" {t} " and _contains_any(t, ("resolves to", "resolve based on", "determined by")))
    ):
        return "event_result"
    if _contains_any(t, ("official data", "reported value", "published value", "closing price", "settlement value")):
        return "objective_measurement"
    return None


def _extract_flags(text: str) -> tuple[str, ...]:
    t = _norm(text)
    if not t:
        return ()
    flags: set[str] = set()

    if _contains_any(t, ("subsequently sworn in", "sworn in", "takes office", "take office", "inauguration date", "assume office")):
        flags.add("requires_take_office")
    if _contains_any(t, ("officially declared", "official declaration", "declared elected", "declared the winner")):
        flags.add("official_declaration_required")
    if _contains_any(t, ("official certification", "certified outcome", "certified result", "final certified outcome")):
        flags.add("certification_required")
    if _contains_any(t, ("highest court", "final court", "court with jurisdiction", "legal challenges have", "challenges have been resolved")):
        flags.add("final_court_outcome")

    # Venue-specific fallback mechanics are material because the two legs can
    # stop being complements even when the headline event is the same.
    if re.search(r"resolve(?:s|d)?\s+(?:to\s+)?['\"]?other['\"]?", t) or "resolve to other" in t:
        flags.add("fallback_other")
    if _contains_any(t, ("last fair market price", "market price", "fair market price")) and "resolve" in t:
        flags.add("fallback_market_price")
    if "postpon" in t and _contains_any(t, ("remain open", "stays open", "contract remains open")):
        flags.add("postponement_stays_open")

    if _contains_any(t, ("special elections are not included", "special elections to fill a vacancy")) and _contains_any(t, ("not included", "excluded")):
        flags.add("special_elections_excluded")
    if "special election" in t and _contains_any(t, ("included", "are included")) and "not included" not in t:
        flags.add("special_elections_included")
    if "by-election" in t or "by election" in t:
        if _contains_any(t, ("not included", "excluded")):
            flags.add("by_elections_excluded")
        elif "included" in t:
            flags.add("by_elections_included")

    if "runoff" in t or "second round" in t:
        if _contains_any(t, ("not included", "excluded")):
            flags.add("runoff_excluded")
        elif _contains_any(t, ("included", "will determine", "includes any potential second round")):
            flags.add("runoff_included")

    if "independent" in t and "caucus" in t:
        flags.add("independent_caucus_counts")
    if _contains_any(t, ("party under which the member was elected", "party under whose banner", "affiliation at the time of the election")):
        flags.add("party_at_election_controls")
    if _contains_any(t, ("party affiliation at resolution", "affiliation when the market resolves")):
        flags.add("party_at_resolution_controls")
    if "die" in t or "death" in t or "incapacitated" in t:
        if _contains_any(t, ("will resolve for that candidate", "still count", "counts")):
            flags.add("death_after_win_counts")
        elif _contains_any(t, ("will not count", "does not count", "excluded")):
            flags.add("death_after_win_excluded")
    if "vacates their seat" in t and _contains_any(t, ("attributed to the party under which", "original party")):
        flags.add("vacancy_after_election_counts_original_party")

    # Cross-venue sports/event basis risk: whether extra periods or a cancelled
    # event count can change which leg pays $1.
    if "overtime" in t or "extra time" in t:
        if _contains_any(t, ("included", "will count", "counts")) and not _contains_any(t, ("not included", "excluded")):
            flags.add("overtime_included")
        if _contains_any(t, ("not included", "excluded", "regulation only")):
            flags.add("overtime_excluded")
    if "shootout" in t or "penalty shootout" in t:
        if _contains_any(t, ("included", "will count", "counts")) and not _contains_any(t, ("not included", "excluded")):
            flags.add("shootout_included")
        if _contains_any(t, ("not included", "excluded")):
            flags.add("shootout_excluded")
    if _contains_any(t, ("cancelled", "canceled", "abandoned")):
        if _contains_any(t, ("remain open", "stays open", "rescheduled", "postponed")):
            flags.add("cancelled_event_stays_open")
        if _contains_any(t, ("void", "voided", "cancel the market", "market will be cancelled")):
            flags.add("cancelled_event_voids")
        if _contains_any(t, ("50-50", "50/50", "0.5", "half payout", "half-payout")):
            flags.add("cancelled_event_half_payout")
    if _contains_any(t, ("stat correction", "stat corrections", "official correction", "data correction")):
        if _contains_any(t, ("will be included", "included", "will count")) and not _contains_any(t, ("not included", "ignored")):
            flags.add("stat_corrections_included")
        if _contains_any(t, ("not included", "ignored", "will not count")):
            flags.add("stat_corrections_excluded")

    return tuple(sorted(flags))


def _source_family(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    families = []
    if _contains_any(t, ("independent national electoral commission", "inec")):
        families.append("nigeria_inec")
    if _contains_any(t, (
        "federal election commission", "fec.gov", "united states congress", "congress.gov",
        "state electoral", "state election", "secretary of state", "election authorities",
        "state authorities", "official state", "official election results",
    )):
        families.append("us_election_official")
    if _contains_any(t, ("associated press", " ap ", "reuters", "credible reporting", "consensus of credible reporting")):
        families.append("credible_reporting")
    if _contains_any(t, ("official league statistics", "official league", "official match report", "official tournament")):
        families.append("official_sport_body")
    if _contains_any(t, ("national weather service", "noaa")):
        families.append("us_weather_official")
    # Price/oracle families are deliberately distinct. Two 15-minute BTC
    # contracts are not guaranteed complements if one settles from Chainlink
    # and the other from CF Benchmarks/Coinbase.
    if "cf benchmarks" in t:
        families.append("cf_benchmarks")
    if "chainlink" in t:
        families.append("chainlink")
    if "pyth" in t:
        families.append("pyth")
    if "coinbase" in t:
        families.append("coinbase")
    if "binance" in t:
        families.append("binance")
    if "kraken" in t:
        families.append("kraken")
    if "coingecko" in t:
        families.append("coingecko")
    if "google trends" in t:
        families.append("google_trends")
    if "billboard" in t:
        families.append("billboard")
    if "spotify" in t:
        families.append("spotify")
    if not families:
        return None
    return "+".join(sorted(set(families)))


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


def _parse_explicit_deadline(text: str, market_year: str | None = None) -> float | None:
    """Extract a conservative explicit rule deadline when text states one.

    We only parse clear English month-day-year clauses plus the common
    "following calendar year" form.  Unknown timezone abbreviations are treated
    as UTC; this changes the horizon by hours, not months, and is conservative
    enough for ranking.  Venue metadata remains the primary max-payout clock.
    """
    t = _norm(text)
    if not t:
        return None

    candidates: list[float] = []
    month_pat = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
    # e.g. October 31, 2027 / Jan 4th 2027
    pat = rf"\b({month_pat})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(20\d{{2}})\b"
    for m in re.finditer(pat, t):
        try:
            dt = datetime(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)), 23, 59, 59, tzinfo=timezone.utc)
            candidates.append(dt.timestamp())
        except Exception:
            pass

    # e.g. January 4th ... of the following calendar year
    if market_year and "following calendar year" in t:
        pat2 = rf"\b({month_pat})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
        for m in re.finditer(pat2, t):
            try:
                dt = datetime(int(market_year) + 1, _MONTHS[m.group(1)], int(m.group(2)), 23, 59, 59, tzinfo=timezone.utc)
                candidates.append(dt.timestamp())
            except Exception:
                pass

    return max(candidates) if candidates else None


def build_resolution_profile(text: str, *, market_year: str | None = None) -> ResolutionProfile:
    t = _norm(text)
    flags = _extract_flags(t)
    basis = _extract_basis(t)
    source = _source_family(t)
    deadline = _parse_explicit_deadline(t, market_year)
    text_present = bool(t)

    # Coverage is intentionally coarse. It describes how much rule evidence we
    # have, not whether the markets are compatible.
    covered = int(basis is not None) + int(source is not None) + int(bool(flags)) + int(deadline is not None)
    material_coverage = covered / 4.0 if text_present else 0.0
    return ResolutionProfile(basis, flags, source, deadline, text_present, material_coverage)


def _conflicting_flag_sets(a: set[str], b: set[str]) -> str | None:
    conflicts = [
        ("requires_take_office", None),  # handled specially: one-sided requirement is material
        ("fallback_other", None),
        ("fallback_market_price", None),
        ("special_elections_included", "special_elections_excluded"),
        ("by_elections_included", "by_elections_excluded"),
        ("runoff_included", "runoff_excluded"),
        ("party_at_election_controls", "party_at_resolution_controls"),
        ("death_after_win_counts", "death_after_win_excluded"),
        ("overtime_included", "overtime_excluded"),
        ("shootout_included", "shootout_excluded"),
        ("cancelled_event_stays_open", "cancelled_event_voids"),
        ("stat_corrections_included", "stat_corrections_excluded"),
    ]
    for left, right in conflicts:
        if right is None:
            if (left in a) != (left in b):
                return f"one-sided material rule ({left})"
        elif (left in a and right in b) or (right in a and left in b):
            return f"contradictory material rules ({left} vs {right})"
    return None


def compare_resolution_profiles(
    left: ResolutionProfile,
    right: ResolutionProfile,
    *,
    rule_sensitive: bool,
) -> tuple[str, list[str]]:
    """Return EXACT, COMPATIBLE or REVIEW plus human-readable reasons."""
    reasons: list[str] = []

    if rule_sensitive and (not left.text_present or not right.text_present):
        return "REVIEW", ["material resolution rules missing on one venue"]

    # A sentence of boilerplate is not enough evidence for political/bespoke
    # contracts. Require at least two independently extracted rule dimensions
    # (for example basis+source or basis+material clauses) on each venue.
    if rule_sensitive and min(left.material_coverage, right.material_coverage) < 0.50:
        return "REVIEW", [
            f"insufficient material rule coverage ({left.material_coverage:.2f} vs {right.material_coverage:.2f})"
        ]

    if left.basis and right.basis:
        # office_assumption is materially different from an election/event result.
        if left.basis != right.basis:
            return "REVIEW", [f"resolution basis differs ({left.basis} vs {right.basis})"]
        reasons.append(f"resolution basis exact ({left.basis})")
    elif rule_sensitive and (left.basis or right.basis):
        return "REVIEW", ["resolution basis missing on one venue"]

    A, B = set(left.flags), set(right.flags)
    conflict = _conflicting_flag_sets(A, B)
    if conflict:
        return "REVIEW", [conflict]

    # Any one-sided material clause is a basis-risk warning for rule-sensitive
    # contracts. This is what blocks cases such as Polymarket's fallback-to-
    # Other deadline or a Kalshi take-office requirement.
    one_sided = sorted((A ^ B) & MATERIAL_FLAGS)
    if rule_sensitive and one_sided:
        return "REVIEW", [f"material rule present on one venue only ({', '.join(one_sided[:4])})"]

    if left.source_family and right.source_family:
        ls = set(left.source_family.split("+")); rs = set(right.source_family.split("+"))
        if not (ls & rs):
            # Different sources can still agree on an objective event, but this
            # is not guaranteed arbitrage evidence.
            return "REVIEW", [f"resolution source families differ ({left.source_family} vs {right.source_family})"]
        reasons.append("resolution source family overlaps")

    # An explicit hard deadline on only one venue is material for rule-sensitive
    # markets because it can create a non-complementary fallback outcome.
    if rule_sensitive and ((left.explicit_deadline_ts is None) != (right.explicit_deadline_ts is None)):
        # Only force REVIEW when the side with a deadline also has a payout-changing fallback.
        deadline_side = left if left.explicit_deadline_ts is not None else right
        if set(deadline_side.flags) & {"fallback_other", "fallback_market_price"}:
            return "REVIEW", ["one venue has a payout-changing hard resolution deadline"]

    exact = (
        left.basis == right.basis
        and set(left.flags) == set(right.flags)
        and left.source_family == right.source_family
        and left.text_present and right.text_present
    )
    if exact:
        return "EXACT", reasons + ["material resolution-rule fingerprint exact"]
    return "COMPATIBLE", reasons + ["no material resolution-rule conflict detected"]


def rule_sensitive_family(*, domain: str | None, proposition: str | None, metric: str | None, office_scope: str | None) -> bool:
    """Families that require rule completeness even outside final strict mode."""
    if domain == "politics":
        return True
    if proposition in {"winner", "primary_winner", "exact_count", "range", "above_threshold", "below_threshold"} and office_scope:
        return True
    if metric in {"seats", "votes", "search_rank"}:
        return True
    return False


# ---------------------------------------------------------------------------
# V13.0 graded basis-risk layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasisRiskAssessment:
    """Conservative assessment for non-identical but closely related rules.

    ``LOW_BASIS`` is *not* guaranteed arbitrage.  It is a separately capped
    research lane whose quoted edge is reduced by ``reserve_per_contract``.
    ``REJECT`` never reaches execution.
    """
    status: str
    reserve_per_contract: float
    reasons: tuple[str, ...]


# Clauses in this set can directly create opposite payouts in ordinary,
# foreseeable states of the world. They are never softened into the V12 basis
# lane.
_FATAL_BASIS_FLAGS = {
    "requires_take_office",
    "fallback_other",
    "fallback_market_price",
    "cancelled_event_voids",
    "cancelled_event_half_payout",
    "overtime_included",
    "overtime_excluded",
    "shootout_included",
    "shootout_excluded",
}

# These clauses can create basis risk, but generally only in narrower edge
# cases. V12 may research them with small capital *after* subtracting a fixed
# per-contract reserve from the executable spread. They are never reported as
# guaranteed-arbitrage profit.
_LOW_BASIS_FLAG_RESERVE = {
    "official_declaration_required": 0.005,
    "certification_required": 0.005,
    "final_court_outcome": 0.010,
    "postponement_stays_open": 0.008,
    "special_elections_included": 0.020,
    "special_elections_excluded": 0.020,
    "by_elections_included": 0.020,
    "by_elections_excluded": 0.020,
    "runoff_included": 0.020,
    "runoff_excluded": 0.020,
    "independent_caucus_counts": 0.025,
    "party_at_election_controls": 0.020,
    "party_at_resolution_controls": 0.020,
    "death_after_win_counts": 0.020,
    "death_after_win_excluded": 0.020,
    "vacancy_after_election_counts_original_party": 0.020,
    "cancelled_event_stays_open": 0.015,
    "stat_corrections_included": 0.010,
    "stat_corrections_excluded": 0.010,
}

_ORACLE_SOURCE_FAMILIES = {
    "cf_benchmarks", "chainlink", "pyth", "coinbase", "binance", "kraken",
    "coingecko", "google_trends", "billboard",
}


def assess_low_basis_risk(
    left: ResolutionProfile,
    right: ResolutionProfile,
    *,
    rule_sensitive: bool,
) -> BasisRiskAssessment:
    """Decide whether a strict REVIEW can enter the small V13 basis lane.

    This intentionally does not rescue missing-rule pairs or fundamentally
    different settlement contracts. It only grades narrow, explicit rule
    differences after both venues supplied material rule text.
    """
    strict_status, strict_reasons = compare_resolution_profiles(
        left, right, rule_sensitive=rule_sensitive
    )
    if strict_status in {"EXACT", "COMPATIBLE"}:
        return BasisRiskAssessment("STRICT_SAFE", 0.0, tuple(strict_reasons))

    # V13 presentation lane: missing/sparse rule text is not silently treated as
    # guaranteed arbitrage. It may enter LOW_BASIS only when the observable
    # payoff identity is already complete upstream, the extracted resolution
    # bases do not conflict, and we reserve extra edge for unknown rule risk.
    sparse_rule_reserve = 0.0
    sparse_rule_reasons: list[str] = []
    if not left.text_present or not right.text_present:
        sparse_rule_reserve += 0.020
        sparse_rule_reasons.append("rule text missing on one venue")
    min_coverage = min(left.material_coverage, right.material_coverage)
    if min_coverage < 0.50:
        # Strict comparison requires >=50% coverage for rule-sensitive families.
        # Earlier versions only reserved for <25%, leaving exactly-25% evidence
        # in a dead zone: too sparse for STRICT_ARB but impossible to grade into
        # LOW_BASIS. V15 assigns a conservative reserve instead of silently
        # discarding otherwise payoff-complete pairs.
        sparse_rule_reserve += 0.015 if min_coverage >= 0.25 else 0.025
        sparse_rule_reasons.append(f"material rule coverage sparse ({min_coverage:.2f})")

    # A genuinely different payout basis (election result vs taking office,
    # certified result vs objective measurement, etc.) is never a small basis
    # discrepancy.
    if left.basis and right.basis and left.basis != right.basis:
        return BasisRiskAssessment(
            "REJECT", 0.0, (f"fundamental resolution basis differs ({left.basis} vs {right.basis})",)
        )
    if rule_sensitive and (left.basis is None or right.basis is None):
        sparse_rule_reserve += 0.020
        sparse_rule_reasons.append("resolution basis omitted on one venue")

    A, B = set(left.flags), set(right.flags)
    conflict = _conflicting_flag_sets(A, B)
    # Conflicts involving fatal sports/cancellation/take-office mechanics are
    # hard rejects. Other explicit election edge-case conflicts may be studied
    # only with a large reserve and small bankroll sleeve.
    diff = (A ^ B) & MATERIAL_FLAGS
    if diff & _FATAL_BASIS_FLAGS:
        return BasisRiskAssessment("REJECT", 0.0, (conflict or "fatal payout-changing rule difference",))

    unknown_material = diff - set(_LOW_BASIS_FLAG_RESERVE)
    if unknown_material:
        return BasisRiskAssessment(
            "REJECT", 0.0, (f"unmodelled material rule difference ({', '.join(sorted(unknown_material))})",)
        )

    reserve = sparse_rule_reserve + sum(_LOW_BASIS_FLAG_RESERVE[x] for x in diff)
    reasons = list(sparse_rule_reasons)
    if diff:
        reasons.append(f"explicit low-basis rule difference ({', '.join(sorted(diff))})")

    if left.source_family and right.source_family:
        ls = set(left.source_family.split("+")); rs = set(right.source_family.split("+"))
        if not (ls & rs):
            # Oracle/source mismatches for numeric or ranked contracts can
            # produce ordinary divergent outcomes, so they remain hard rejects.
            if (ls | rs) & _ORACLE_SOURCE_FAMILIES:
                return BasisRiskAssessment(
                    "REJECT", 0.0,
                    (f"non-interchangeable resolution sources ({left.source_family} vs {right.source_family})",),
                )
            reserve += 0.015
            reasons.append(f"non-overlapping non-oracle source families ({left.source_family} vs {right.source_family})")
    elif rule_sensitive and (left.source_family or right.source_family):
        reserve += 0.010
        reasons.append("resolution source family specified on one venue only")

    # One-sided payout-changing hard deadlines should already have been
    # rejected by compare_resolution_profiles when paired with a fallback.
    # A pure administrative timing mismatch is handled by the engine using the
    # latest known cross-venue settlement timestamp and needs no extra payout
    # reserve.
    if not reasons:
        return BasisRiskAssessment("REJECT", 0.0, ("strict review has no safely modelled low-basis explanation",))

    # Cap the modelled reserve. If the identified basis risk consumes more than
    # six cents of a $1 contract, the opportunity is too dependent on subjective
    # probabilities to be useful for this project.
    if reserve > 0.06 + 1e-12:
        return BasisRiskAssessment("REJECT", reserve, tuple(reasons + ["basis reserve exceeds 6c hard cap"]))
    return BasisRiskAssessment("LOW_BASIS", reserve, tuple(reasons))
