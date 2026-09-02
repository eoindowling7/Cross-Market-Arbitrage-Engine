"""V28 execution-recovery helpers.

Paper-only.  The goal is to distinguish temporary/missing evidence from an
explicit payoff contradiction, so candidates remain observable without turning
unknowns into fictitious arbitrage.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests

from src.api.kalshi_client import get_market_details
from src.api.polymarket_client import get_market_by_id
from src.arbitrage.universal_matcher import kalshi_signature, polymarket_signature
from src.arbitrage.resolution_equivalence import (
    build_resolution_profile,
    compare_resolution_profiles,
    rule_sensitive_family,
)

TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


def classify_exception(exc: Exception) -> tuple[str, bool]:
    """Return stable reason + whether retrying later can plausibly help."""
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return f"auth_http_{status}", False
        if status == 404:
            return "market_not_found_404", False
        if status in TRANSIENT_HTTP:
            return f"transient_http_{status}", True
        return f"http_{status or 'unknown'}", False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return type(exc).__name__.lower(), True
    if isinstance(exc, ValueError) and "max()" in str(exc) and "empty" in str(exc):
        return "empty_iterable_guarded", True
    return f"{type(exc).__name__}: {str(exc)[:120]}", False


def _market_id(pm: dict) -> str:
    return str(pm.get("id") or pm.get("market_id") or pm.get("conditionId") or "")


def _extract_rule_text(obj: dict) -> str:
    if not isinstance(obj, dict):
        return ""
    fields = (
        "rules_primary", "rulesSecondary", "rules_secondary", "rules", "description",
        "resolutionSource", "resolution_source", "settlementSource", "settlement_source",
        "subtitle", "yes_sub_title", "no_sub_title",
    )
    parts = []
    for key in fields:
        val = obj.get(key)
        if val not in (None, "", [], {}):
            parts.append(str(val))
    return "\n".join(parts)


def _explicit_structural_contradictions(ks: dict, ps: dict) -> list[str]:
    """Only reject observed disagreements; one-sided missing data is unresolved."""
    checks = (
        ("office_scope", "office/chamber"),
        ("jurisdiction_region", "jurisdiction"),
        ("jurisdiction_district", "district"),
        ("year", "year"),
        ("stage", "stage/round"),
    )
    out = []
    for key, label in checks:
        a, b = ks.get(key), ps.get(key)
        if a not in (None, "") and b not in (None, "") and str(a) != str(b):
            out.append(f"{label} mismatch ({a} vs {b})")
    # Threshold/operator only matter when both are explicitly known.
    for key, label in (("threshold_low", "threshold"), ("threshold_high", "threshold-high"), ("threshold_op", "operator")):
        a, b = ks.get(key), ps.get(key)
        if a not in (None, "") and b not in (None, "") and str(a) != str(b):
            out.append(f"{label} mismatch ({a} vs {b})")
    return out


def late_hydrate_candidate(candidate: dict, pm: dict, cache: dict, *, retries: int = 2) -> tuple[str, str]:
    """Hydrate rules for an economically interesting candidate.

    Returns (SAFE|UNRESOLVED|CONTRADICTION, human reason).  Successful evidence
    mutates candidate signatures/certificate in place so later cycles reuse it.
    """
    ticker = str(candidate.get("ticker") or candidate.get("kalshi_ticker") or "")
    pid = _market_id(pm)
    key = (ticker, pid)
    if key in cache:
        cached = cache[key]
        state, reason, payload, cached_at = cached if len(cached) == 4 else (*cached, 0.0)
        # SAFE/CONTRADICTION are stable for the run. UNRESOLVED evidence is
        # retried after a short TTL so temporary venue/API gaps do not kill a
        # candidate for the whole watch window.
        if state != "UNRESOLVED" or (time.monotonic() - float(cached_at)) < 60.0:
            if payload:
                candidate.update(payload)
            return state, reason

    ks0 = dict(candidate.get("kalshi_signature") or {})
    ps0 = dict(candidate.get("polymarket_signature") or {})
    contradictions = _explicit_structural_contradictions(ks0, ps0)
    if contradictions:
        result = ("CONTRADICTION", "; ".join(contradictions), {})
        cache[key] = (*result, time.monotonic())
        return result[0], result[1]

    kdetail = pdetail = None
    errors = []
    for attempt in range(retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fk = pool.submit(get_market_details, ticker)
                fp = pool.submit(get_market_by_id, pid) if pid else None
                kdetail = fk.result()
                pdetail = fp.result() if fp is not None else dict(pm)
            break
        except Exception as exc:
            reason, transient = classify_exception(exc)
            errors.append(reason)
            if not transient or attempt >= retries:
                break
            time.sleep(0.25 * (2 ** attempt))

    if not isinstance(kdetail, dict) or not isinstance(pdetail, dict):
        result = ("UNRESOLVED", "rule hydration failed: " + ", ".join(errors[-3:]), {})
        cache[key] = (*result, time.monotonic())
        return result[0], result[1]

    # Rebuild enriched signatures from public market details.  Preserve original
    # structural extraction if a detail page omits a field.
    krow = {
        "ticker": ticker,
        "title": candidate.get("kalshi_title") or kdetail.get("title") or "",
        "yes_sub_title": kdetail.get("yes_sub_title") or "",
        "subtitle": kdetail.get("subtitle") or "",
    }
    try:
        ksig_obj = kalshi_signature(krow, kdetail)
        psig_obj = polymarket_signature({**pm, **pdetail})
        ks1 = asdict(ksig_obj) if ksig_obj is not None else ks0
        ps1 = asdict(psig_obj) if psig_obj is not None else ps0
    except Exception:
        ks1, ps1 = ks0, ps0

    # Keep richer old structural fields when the hydrated parser has gaps.
    for key2, val in ks0.items():
        if ks1.get(key2) in (None, "", [], {}) and val not in (None, "", [], {}):
            ks1[key2] = val
    for key2, val in ps0.items():
        if ps1.get(key2) in (None, "", [], {}) and val not in (None, "", [], {}):
            ps1[key2] = val

    contradictions = _explicit_structural_contradictions(ks1, ps1)
    if contradictions:
        result = ("CONTRADICTION", "; ".join(contradictions), {})
        cache[key] = (*result, time.monotonic())
        return result[0], result[1]

    ktext = _extract_rule_text(kdetail)
    ptext = _extract_rule_text({**pm, **pdetail})
    ky = str(ks1.get("year") or "") or None
    py = str(ps1.get("year") or "") or None
    kp = build_resolution_profile(ktext, market_year=ky)
    pp = build_resolution_profile(ptext, market_year=py)
    sensitive = rule_sensitive_family(
        domain=ks1.get("domain") or ps1.get("domain"),
        proposition=ks1.get("proposition") or ps1.get("proposition"),
        metric=ks1.get("metric") or ps1.get("metric"),
        office_scope=ks1.get("office_scope") or ps1.get("office_scope"),
    )
    status, reasons = compare_resolution_profiles(kp, pp, rule_sensitive=sensitive)

    cert = dict(candidate.get("equivalence_certificate") or {})
    payload = {"kalshi_signature": ks1, "polymarket_signature": ps1}
    if status in {"EXACT", "COMPATIBLE"}:
        cert["resolution_rule_status"] = status
        cert["v28_late_rule_hydrated"] = True
        cert["v28_rule_reason"] = "; ".join(reasons)
        # Preserve LOW_BASIS lane while allowing compatible public rules to pass.
        if str(cert.get("resolution_lane") or "").upper() == "LOW_BASIS":
            cert["resolution_rule_status"] = "LOW_BASIS"
        payload["equivalence_certificate"] = cert
        candidate.update(payload)
        result = ("SAFE", "; ".join(reasons) or "late rule hydration compatible", payload)
    else:
        # REVIEW is not an explicit contradiction. Keep it observable and retry
        # on later cycles if data changes or a prior request was incomplete.
        result = ("UNRESOLVED", "; ".join(reasons) or "insufficient rule evidence", payload)
        candidate.update(payload)
    cache[key] = (*result, time.monotonic())
    return result[0], result[1]
