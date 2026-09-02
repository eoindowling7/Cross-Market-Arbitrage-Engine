"""V8.1 precision guard for Kalshi/Polymarket semantic pairs.

Run after DeBERTa + production_veto_v7 + production_veto_v8_precision.

V8.1 does NOT add broad new hard rejections.  It downgrades a few logically
under-specified V8 PASS families to REVIEW so they cannot auto-trade without
raw-rule confirmation.

Additional guardrails found by adversarial audit of all 1,434 V8 PASS rows:
  1) office-holder-after-election vs generic next-office-holder scope;
  2) "next to leave" vs "first to leave" sequence wording;
  3) individual Emmy award where one venue keys the nominee to a specific work
     while the other side identifies only the person.

REVIEW must be treated as NO TRADE by the precision lane.
"""
from __future__ import annotations

import re
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_v8_path = Path(__file__).with_name("production_veto_v8_precision.py")
_spec = spec_from_file_location("v8_base", str(_v8_path))
v8 = module_from_spec(_spec)
_spec.loader.exec_module(v8)

norm = v8.norm


def precision_review_guard(row, k, p, certificate=None):
    """Return a V8.1 REVIEW reason, or None if no new ambiguity is present."""
    kt = norm(row.get("kalshi_title", ""))
    pt = norm(row.get("poly_question") or row.get("polymarket_question", ""))
    ke = norm(k.get("event_identity"))
    pe = norm(p.get("event_identity"))
    kall = " ".join([kt, ke])
    pall = " ".join([pt, pe])

    # 1) "become PM following election" is not logically identical, from titles
    # alone, to "be the next PM".  Raw rules may prove that the latter is also
    # explicitly conditioned on the election, but until then this is REVIEW.
    post_k = bool(re.search(r"\b(?:after|following) (?:the )?(?:next )?.*election\b", kt))
    post_p = bool(re.search(r"\b(?:after|following) (?:the )?(?:next )?.*election\b", pt))
    next_office_k = bool(re.search(
        r"\bnext (?:prime minister|chief minister|president|press secretary|majority leader|leader)\b",
        kt,
    ))
    next_office_p = bool(re.search(
        r"\bnext (?:prime minister|chief minister|president|press secretary|majority leader|leader)\b",
        pt,
    ))
    if (post_k and next_office_p and not post_p) or (post_p and next_office_k and not post_k):
        return "V8_1_REVIEW_SUCCESSION_SCOPE"

    # 2) "next to leave" and "first to leave" are equivalent only given a
    # shared start-state.  Do not infer that state from wording alone.
    next_leave_k = bool(re.search(r"\bnext to leave\b", kt))
    next_leave_p = bool(re.search(r"\bnext to leave\b", pt))
    first_leave_k = bool(re.search(r"\bfirst to leave\b", kt))
    first_leave_p = bool(re.search(r"\bfirst to leave\b", pt))
    if (next_leave_k and first_leave_p) or (next_leave_p and first_leave_k):
        return "V8_1_REVIEW_FIRST_VS_NEXT"

    # 3) For individual Emmy performance awards, person-only versus person+work
    # is not automatically exact payoff identity.  A later raw-rule check can
    # promote it if both contracts bind to the same nominated work.
    if "emmy" in kall and "emmy" in pall:
        a = norm(k.get("subject"))
        b = norm(p.get("subject"))
        if a and b and a != b:
            short, long = (a, b) if len(a) < len(b) else (b, a)
            if short in long:
                extra = set(long.split()) - set(short.split())
                benign = {"jr", "sr", "ii", "iii", "iv"}
                if extra - benign:
                    if re.search(r"\b(actor|actress)\b", kall) and re.search(r"\b(actor|actress)\b", pall):
                        return "V8_1_REVIEW_AWARD_WORK_SCOPE"

    return None


def pair_decision(row, k, p, certificate=None):
    """Apply V8, then V8.1's narrow review-only guardrails."""
    result = v8.pair_decision(row, k, p, certificate)
    if result.get("decision") != "PASS":
        return result

    # V8's generic metric detector recognizes the token "WAR" as the baseball
    # Wins Above Replacement metric.  Do not let an unrelated phrase such as
    # "War Machine" create positive metric proof outside a baseball context.
    if result.get("reason") == "PROOF_SAME_METRIC":
        kt = norm(row.get("kalshi_title", ""))
        pt = norm(row.get("poly_question") or row.get("polymarket_question", ""))
        all_text = " ".join([kt, pt, norm(k.get("event_identity")), norm(p.get("event_identity"))])
        if re.search(r"\bwar\b", all_text):
            baseballish = bool(re.search(r"\b(?:baseball|mlb|wins above replacement)\b", all_text))
            if not baseballish:
                out = dict(result)
                out["decision"] = "REVIEW"
                out["reason"] = "V8_1_REVIEW_METRIC_TOKEN_COLLISION"
                return out

    guard = precision_review_guard(row, k, p, certificate)
    if guard:
        out = dict(result)
        out["decision"] = "REVIEW"
        out["reason"] = guard
        return out
    return result


def veto(row, k, p):
    """Compatibility hard-veto wrapper. REVIEW is intentionally not a veto here."""
    return v8.veto(row, k, p)
