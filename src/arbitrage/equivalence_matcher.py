"""Final V3 cross-venue equivalence classifier + deterministic safety gates.

This module is deliberately narrow: BGE/FAISS is used elsewhere for recall-first
candidate retrieval.  This class decides whether a retrieved Kalshi/Polymarket
candidate is payoff-equivalent enough to enter pricing.

Decision policy
---------------
1. DeBERTa-v3 pair classifier probability must meet the frozen calibration threshold.
2. Deterministic contradictions are hard vetoes.
3. Unsupported/insufficiently-specified contracts return UNRESOLVED, never EQUIVALENT.

Paper-only. No order-placement code lives here.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_THRESHOLD = 0.0029228702187538147


def _clean(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().split())


def _field(sig, name, default=None):
    if sig is None:
        return default
    if isinstance(sig, dict):
        return sig.get(name, default)
    return getattr(sig, name, default)


def _first(obj, *names):
    if obj is None:
        return None
    for name in names:
        try:
            value = obj.get(name)
        except Exception:
            value = None
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_jsonish_list(value):
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def _extract_exact_count(text: str):
    t = _clean(text)
    for pat in (
        r"\bexactly\s+(\d+)\b",
        r"\bhold\s+exactly\s+(\d+)\b",
        r"\bwin\s+exactly\s+(\d+)\b",
    ):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return None


def _extract_top_n(text: str):
    t = _clean(text)
    for pat in (
        r"\btop[\s\-]*(\d+)\b",
        r"\bfinish(?:es)?\s+(?:in\s+)?(?:the\s+)?top[\s\-]*(\d+)\b",
    ):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return None


def _extract_numbered_scope(text: str, word: str):
    t = _clean(text)
    m = re.search(rf"\b{re.escape(word)}[\s\-]*(\d+)\b", t)
    return int(m.group(1)) if m else None


def _has_fixture(text: str) -> bool:
    t = f" {_clean(text)} "
    return bool(re.search(r"\bvs\.?\b|\bversus\b", t))


def _is_standard_yes_no_polymarket(pm: dict | None) -> bool:
    """Return True when outcome identity is explicit standard Yes/No.

    Non-Yes/No binary outcomes (common in esports winner markets) need outcome-aware
    serialization before they can safely enter the arbitrage leg mapper.
    """
    if not isinstance(pm, dict):
        return True  # caller may be evaluating saved text without raw market metadata
    outcomes = _extract_jsonish_list(pm.get("outcomes"))
    if not outcomes:
        return True
    norm = {_clean(x) for x in outcomes}
    return norm == {"yes", "no"}


def serialize_contract(*, venue: str, title: str, sig) -> str:
    """Serialize production markets in the same field-oriented style used in V3."""
    domain = _field(sig, "domain") or ""
    proposition = _field(sig, "proposition") or ""
    subject = _field(sig, "subject") or ""
    context = _field(sig, "context") or ""
    event = _field(sig, "event_identity") or _field(sig, "competition") or ""
    return (
        f"COL venue VAL {venue} "
        f"COL title VAL {title or ''} "
        f"COL domain VAL {domain} "
        f"COL proposition VAL {proposition} "
        f"COL subject VAL {subject} "
        f"COL context VAL {context} "
        f"COL event VAL {event}"
    )


@dataclass(frozen=True)
class GateResult:
    status: str  # PASS, REJECT, UNRESOLVED
    reason: str


@dataclass(frozen=True)
class MatchDecision:
    status: str  # EQUIVALENT, NON_EQUIVALENT, UNRESOLVED
    probability: float
    threshold: float
    gate_reason: str | None = None


def deterministic_gate(*, kalshi_title: str, polymarket_question: str, ksig, psig, polymarket_market=None) -> GateResult:
    """Hard-veto only facts that cannot both be true for payoff-equivalent contracts."""
    kt = _clean(kalshi_title)
    pt = _clean(polymarket_question)

    # Outcome identity must be explicit for the pricing layer. This prevents the
    # known esports failure mode where a question names two teams but token 0/1
    # are team outcomes rather than Yes/No.
    if not _is_standard_yes_no_polymarket(polymarket_market):
        return GateResult("UNRESOLVED", "non_yes_no_outcome_identity_requires_outcome_aware_serializer")

    # Exact counts.
    kc, pc = _extract_exact_count(kt), _extract_exact_count(pt)
    if kc is not None and pc is not None and kc != pc:
        return GateResult("REJECT", f"exact_count_mismatch:{kc}!={pc}")

    # Top-N/rank cutoffs.
    kn, pn = _extract_top_n(kt), _extract_top_n(pt)
    if kn is not None and pn is not None and kn != pn:
        return GateResult("REJECT", f"top_n_mismatch:{kn}!={pn}")

    # Numbered scopes that are payoff-defining.
    for scope in ("round", "map", "game"):
        a, b = _extract_numbered_scope(kt, scope), _extract_numbered_scope(pt, scope)
        if a is not None and b is not None and a != b:
            return GateResult("REJECT", f"{scope}_mismatch:{a}!={b}")

    # Explicit parser thresholds, only when both sides are populated.
    kop, pop = _field(ksig, "threshold_op"), _field(psig, "threshold_op")
    klo, plo = _field(ksig, "threshold_low"), _field(psig, "threshold_low")
    khi, phi = _field(ksig, "threshold_high"), _field(psig, "threshold_high")
    if kop and pop and kop != pop:
        return GateResult("REJECT", f"threshold_operator_mismatch:{kop}!={pop}")
    if klo is not None and plo is not None and float(klo) != float(plo):
        return GateResult("REJECT", f"threshold_low_mismatch:{klo}!={plo}")
    if khi is not None and phi is not None and float(khi) != float(phi):
        return GateResult("REJECT", f"threshold_high_mismatch:{khi}!={phi}")

    # Specific fixture vs season/competition winner. Restrict to winner-like
    # propositions and require the fixture marker to occur on exactly one side.
    kp, pp = _clean(_field(ksig, "proposition")), _clean(_field(psig, "proposition"))
    if kp == "winner" and pp == "winner" and _has_fixture(kt) != _has_fixture(pt):
        return GateResult("REJECT", "fixture_vs_competition_scope_mismatch")

    # Explicit year/jurisdiction conflicts are safe hard vetoes when both sides
    # contain the field. Missing values remain UNKNOWN rather than contradictions.
    ky, py = _field(ksig, "year"), _field(psig, "year")
    if ky and py and str(ky) != str(py):
        return GateResult("REJECT", f"year_mismatch:{ky}!={py}")

    for field in ("jurisdiction_country", "jurisdiction_region", "jurisdiction_district", "office_scope"):
        a, b = _clean(_field(ksig, field)), _clean(_field(psig, field))
        if a and b and a != b:
            return GateResult("REJECT", f"{field}_mismatch:{a}!={b}")

    return GateResult("PASS", "pass")


class EquivalenceMatcher:
    def __init__(self, model_path: str | Path, *, threshold: float = DEFAULT_THRESHOLD, batch_size: int = 64, device: str | None = None):
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.batch_size = max(1, int(batch_size))
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Final equivalence matcher dependencies are missing. Install transformers, torch, and sentencepiece."
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_env(cls):
        model_path = os.getenv("EQUIV_MODEL_PATH", "models/equivalence_v3/checkpoint-588")
        threshold = float(os.getenv("EQUIV_THRESHOLD", str(DEFAULT_THRESHOLD)))
        threshold_path = os.getenv("EQUIV_THRESHOLD_PATH")
        if threshold_path and Path(threshold_path).exists():
            try:
                threshold = float(json.loads(Path(threshold_path).read_text(encoding="utf-8"))["threshold"])
            except Exception as exc:
                raise RuntimeError(f"Could not read EQUIV_THRESHOLD_PATH={threshold_path}") from exc
        batch_size = int(os.getenv("EQUIV_BATCH_SIZE", "64"))
        device = os.getenv("EQUIV_DEVICE") or None
        return cls(model_path, threshold=threshold, batch_size=batch_size, device=device)

    def predict_probabilities(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float32)
        torch = self._torch
        out = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            left = [x[0] for x in batch]
            right = [x[1] for x in batch]
            enc = self.tokenizer(
                left,
                right,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
            out.extend(probs.detach().cpu().numpy().astype(np.float32).tolist())
        return np.asarray(out, dtype=np.float32)

    def decide(self, *, probability: float, kalshi_title: str, polymarket_question: str, ksig, psig, polymarket_market=None) -> MatchDecision:
        p = float(probability)
        if p < self.threshold:
            return MatchDecision("NON_EQUIVALENT", p, self.threshold, "classifier_below_threshold")
        gate = deterministic_gate(
            kalshi_title=kalshi_title,
            polymarket_question=polymarket_question,
            ksig=ksig,
            psig=psig,
            polymarket_market=polymarket_market,
        )
        if gate.status == "UNRESOLVED":
            return MatchDecision("UNRESOLVED", p, self.threshold, gate.reason)
        if gate.status == "REJECT":
            return MatchDecision("NON_EQUIVALENT", p, self.threshold, gate.reason)
        return MatchDecision("EQUIVALENT", p, self.threshold, None)
