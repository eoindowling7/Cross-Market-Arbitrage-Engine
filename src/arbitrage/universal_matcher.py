"""V27 semantic-hybrid cross-platform contract matcher.

The matcher deliberately separates *candidate discovery* from *payoff safety*:

1. BAAI/bge-small-en-v1.5 embeds Kalshi and Polymarket contract text.
2. FAISS HNSW retrieves a very small nearest-neighbour set per Kalshi market.
3. V8.4-style deterministic keys independently rescue obvious structural pairs.
4. The trained DeBERTa-v3 equivalence classifier scores only the small candidate union.
5. The frozen calibration threshold is applied exactly as selected before test.
6. Deterministic contradiction gates are hard vetoes; unsupported outcome mappings
   return UNRESOLVED and never enter pricing.
7. Legacy signature/resolution code is retained only for metadata/certification, not
   as the final equivalence decision.

The two ML models are local/open-source.  They download once through
SentenceTransformers/Hugging Face and are then cached by that library.  V27 also
keeps its own incremental embedding caches in data/cache/v27_semantic so later
runs encode only new/changed market texts.

Paper-only: this module contains no order-placement code.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.arbitrage import universal_matcher_rules as rules
from src.arbitrage.equivalence_matcher import EquivalenceMatcher, serialize_contract

# Re-export names used by tests/other modules.
ContractSignature = rules.ContractSignature
MatchAudit = rules.MatchAudit
normalize_text = rules.normalize_text
kalshi_signature = rules.kalshi_signature
polymarket_signature = rules.polymarket_signature
evaluate_equivalence = rules.evaluate_equivalence

EMBED_MODEL = os.getenv("V27_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE_DIR = Path(os.getenv("V27_SEMANTIC_CACHE", "data/cache/v27_semantic"))

# Conservative CPU defaults. They can be overridden without editing code.
EMBED_BATCH_SIZE = int(os.getenv("V27_EMBED_BATCH_SIZE", "128"))
SEMANTIC_TOP_K = int(os.getenv("V27_SEMANTIC_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("V27_RERANK_TOP_K", "4"))
GLOBAL_RESCUE_K = int(os.getenv("V27_GLOBAL_RESCUE_K", "2"))
MIN_EMBED_SIM = float(os.getenv("V27_MIN_EMBED_SIM", "0.66"))
# This is a sigmoid-transformed MS-MARCO logit, used as evidence rather than
# a calibrated probability of payoff equivalence.


def _require_semantic_deps():
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host env
        raise RuntimeError(
            "Semantic retrieval/final equivalence dependencies are missing. Run: "
            "python -m pip install -r requirements.txt\n"
            f"Underlying import error: {type(exc).__name__}: {exc}"
        ) from exc


def _compact(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _get_first(obj, names: Iterable[str]):
    for name in names:
        try:
            value = obj.get(name)
        except Exception:
            value = None
        if value not in (None, "", [], {}):
            return value
    return None


def _kalshi_semantic_text(row, metadata: dict | None) -> str:
    md = metadata or {}
    event = md.get("_event") or {}
    title = _get_first(row, ("title", "question")) or _get_first(md, ("title", "question"))
    subtitle = _get_first(row, ("yes_sub_title", "subtitle")) or _get_first(md, ("yes_sub_title", "subtitle"))
    rules_text = _get_first(md, ("rules_primary", "rules_secondary", "rules", "description"))
    event_title = _get_first(event, ("title", "name"))
    close = _get_first(row, ("close_time", "expiration_time")) or _get_first(md, ("close_time", "expiration_time"))
    # Retrieval text stays short: long rule boilerplate harms embeddings and
    # wastes transformer tokens. Detailed rule text is reserved for reranking.
    parts = [
        f"market: {_compact(title, 300)}",
        f"yes condition: {_compact(subtitle, 220)}" if subtitle else "",
        f"event: {_compact(event_title, 220)}" if event_title else "",
        f"resolves: {_compact(rules_text, 360)}" if rules_text else "",
        f"date: {_compact(close, 60)}" if close else "",
    ]
    return " | ".join(x for x in parts if x)


def _poly_semantic_text(market: dict) -> str:
    title = _get_first(market, ("question", "title"))
    rules_text = _get_first(market, ("description", "rules", "resolutionSource"))
    group = _get_first(market, ("groupItemTitle", "eventTitle", "category"))
    end = _get_first(market, ("endDate", "end_date", "endDateIso"))
    parts = [
        f"market: {_compact(title, 300)}",
        f"event: {_compact(group, 220)}" if group else "",
        f"resolves: {_compact(rules_text, 360)}" if rules_text else "",
        f"date: {_compact(end, 60)}" if end else "",
    ]
    return " | ".join(x for x in parts if x)


def _rerank_text(short_text: str, sig: ContractSignature) -> str:
    # Explicit structured hints help a general relevance model notice small
    # payoff-defining differences without trusting parser output as ground truth.
    fields = []
    for key in (
        "domain", "proposition", "subject", "metric", "unit", "year",
        "competition", "stage", "period_scope", "rank_semantics",
        "jurisdiction_country", "jurisdiction_region", "jurisdiction_district",
        "office_scope",
    ):
        value = getattr(sig, key, None)
        if value not in (None, "", [], {}):
            fields.append(f"{key}={value}")
    op, lo, hi = rules._normalized_threshold(sig)
    if any(x is not None for x in (op, lo, hi)):
        fields.append(f"threshold={op}:{lo}:{hi}")
    return _compact(short_text + " | structured: " + "; ".join(fields), 1450)


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=12).hexdigest()


def _load_embedding_cache(path: Path):
    if not path.exists():
        return {}
    try:
        z = np.load(path, allow_pickle=False)
        ids = z["ids"].astype(str)
        hashes = z["hashes"].astype(str)
        emb = z["embeddings"].astype(np.float32, copy=False)
        if len(ids) != len(hashes) or len(ids) != len(emb):
            return {}
        return {str(i): (str(h), emb[n]) for n, (i, h) in enumerate(zip(ids, hashes))}
    except Exception:
        return {}


def _save_embedding_cache(path: Path, ids: list[str], hashes: list[str], emb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Opening explicitly avoids numpy silently appending '.npz' to the temp name.
    with tmp.open("wb") as fh:
        np.savez(fh, ids=np.asarray(ids), hashes=np.asarray(hashes), embeddings=np.asarray(emb, dtype=np.float32))
    tmp.replace(path)


def _encode_incremental(model, *, ids: list[str], texts: list[str], cache_name: str) -> np.ndarray:
    cache_path = CACHE_DIR / f"{cache_name}.npz"
    cached = _load_embedding_cache(cache_path)
    hashes = [_hash_text(t) for t in texts]
    dim = int(model.get_sentence_embedding_dimension())
    out = np.empty((len(ids), dim), dtype=np.float32)
    missing_idx = []
    for n, (mid, th) in enumerate(zip(ids, hashes)):
        item = cached.get(mid)
        if item is not None and item[0] == th and int(item[1].shape[0]) == dim:
            out[n] = item[1]
        else:
            missing_idx.append(n)

    print(
        f"V27 embedding cache {cache_name} | total={len(ids)} reused={len(ids)-len(missing_idx)} "
        f"encode={len(missing_idx)}",
        flush=True,
    )
    if missing_idx:
        new_texts = [texts[i] for i in missing_idx]
        vecs = model.encode(
            new_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        for j, idx in enumerate(missing_idx):
            out[idx] = vecs[j]
        _save_embedding_cache(cache_path, ids, hashes, out)
    return out


@dataclass(frozen=True)
class _SemanticPoly:
    market: dict
    sig: ContractSignature
    text: str
    rerank_text: str
    domain: str


@dataclass
class _HybridCandidate:
    poly_idx: int
    embedding_score: float = -1.0
    retrieval_sources: set[str] | None = None
    deterministic_strength: float = 0.0
    equivalence_probability: float | None = None

    def __post_init__(self):
        if self.retrieval_sources is None:
            self.retrieval_sources = set()


def _sig_subject_key(sig):
    try:
        return rules._subject_key(sig.subject)
    except Exception:
        return normalize_text(sig.subject) or None


def _v84_key(sig: ContractSignature):
    """V8.4's useful broad blocking idea, never an auto-accept rule."""
    subj = _sig_subject_key(sig)
    if subj:
        return ("subject", sig.proposition, subj, sig.year)
    op, lo, hi = rules._normalized_threshold(sig)
    if any(x is not None for x in (op, lo, hi)):
        return ("threshold", sig.proposition, op, lo, hi, sig.unit, sig.year)
    # Context fallback is intentionally much narrower than old V8.4 because
    # generic context keys caused fan-out; use it only when 3+ informative tokens exist.
    toks = tuple(sorted(rules._doc_tokens(sig.context or sig.event_identity or ""))[:6])
    if len(toks) >= 3:
        return ("context", sig.proposition, toks, sig.year)
    return None


def _build_v84_index(polys: list[_SemanticPoly]):
    index = defaultdict(list)
    for i, p in enumerate(polys):
        key = _v84_key(p.sig)
        if key is not None:
            index[key].append(i)
    return index


def _round_number(text: str):
    t = normalize_text(text)
    m = re.search(r"\b(?:round|rd)\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1))
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    for word, n in words.items():
        if re.search(rf"\b{word}\s+round\b", t):
            return n
    return None


def _textual_hard_contradiction(ktext: str, ptext: str) -> str | None:
    # Parser-independent protection against one of the V25 audit's major errors.
    kr, pr = _round_number(ktext), _round_number(ptext)
    if kr is not None and pr is not None and kr != pr:
        return f"round differs ({kr} vs {pr})"
    return None


def _hard_contradiction(ksig, psig, ktext, ptext):
    c = rules._v20_hard_contradiction(ksig, psig)
    if c:
        return c
    return _textual_hard_contradiction(ktext, ptext)


def _sigmoid(x: float) -> float:
    # Stable transform for readable diagnostics only; MS-MARCO logits are not
    # calibrated probabilities of contract equivalence.
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _same_domain(a, b) -> bool:
    ad, bd = str(a.domain or "other"), str(b.domain or "other")
    return ad == bd or "other" in {ad, bd}


def _faiss_hnsw(vectors: np.ndarray):
    import faiss
    dim = int(vectors.shape[1])
    # HNSW avoids an O(N*M) exhaustive semantic scan.  Inner product on unit
    # vectors equals cosine similarity. M=32/efSearch=64 is a good recall/speed
    # tradeoff for ~50k Polymarket vectors on a CPU desktop.
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def _candidate_from(mapping: dict[int, _HybridCandidate], idx: int, *, source: str, embed_score: float = -1.0):
    cand = mapping.get(idx)
    if cand is None:
        cand = _HybridCandidate(poly_idx=idx)
        mapping[idx] = cand
    cand.retrieval_sources.add(source)
    cand.embedding_score = max(cand.embedding_score, float(embed_score))
    if source.startswith("v84_"):
        cand.deterministic_strength = max(cand.deterministic_strength, 1.0)
    return cand


def find_universal_matches(
    kalshi_markets,
    polymarket_markets: list[dict],
    kalshi_market_metadata: dict[str, dict] | None = None,
    *,
    include_legacy: bool = True,
    kalshi_detail_fetcher=None,
    polymarket_detail_fetcher=None,
):
    """BGE high-recall retrieval + frozen V3 DeBERTa classifier + hard gates."""
    _require_semantic_deps()
    import faiss
    from sentence_transformers import SentenceTransformer

    metadata = kalshi_market_metadata or {}
    audits = rules.BoundedAuditList(max_rows=25000)
    matches: list[dict] = []

    print(f"Final matcher | retrieval={EMBED_MODEL} | classifier=DeBERTa V3", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    equivalence_model = EquivalenceMatcher.from_env()
    print(f"Final matcher threshold={equivalence_model.threshold:.16f} | device={equivalence_model.device}", flush=True)

    # ---------------- Polymarket preprocessing / embeddings ----------------
    polys: list[_SemanticPoly] = []
    poly_ids: list[str] = []
    for pm in polymarket_markets:
        psig = polymarket_signature(pm)
        pid = str(pm.get("id") or pm.get("conditionId") or "")
        if psig is None or not pid:
            continue
        text = _poly_semantic_text(pm)
        polys.append(_SemanticPoly(pm, psig, text, _rerank_text(text, psig), str(psig.domain or "other")))
        poly_ids.append(pid)
    audits.bump("polymarket_signatures", len(polys))
    if not polys:
        return matches, audits

    poly_emb = _encode_incremental(
        embedder, ids=poly_ids, texts=[p.text for p in polys], cache_name="polymarket_bge_small_v15"
    )
    global_index = _faiss_hnsw(poly_emb)

    # Domain-local indexes improve both precision and speed, while a tiny global
    # rescue route protects category/parser asymmetry.
    domain_rows: dict[str, np.ndarray] = {}
    domain_indexes = {}
    by_domain = defaultdict(list)
    for i, p in enumerate(polys):
        by_domain[p.domain].append(i)
    for domain, rows in by_domain.items():
        arr = np.asarray(rows, dtype=np.int64)
        domain_rows[domain] = arr
        if len(arr) >= 2:
            domain_indexes[domain] = _faiss_hnsw(poly_emb[arr])
    v84_index = _build_v84_index(polys)

    # ---------------- Kalshi preprocessing / embeddings ----------------
    kitems = []
    k_ids = []
    k_texts = []
    for _, row in kalshi_markets.iterrows():
        ticker = str(row.get("ticker") or "")
        md = metadata.get(ticker) or {}
        ksig = kalshi_signature(row, md)
        if ksig is None or not ticker:
            continue
        text = _kalshi_semantic_text(row, md)
        kitems.append((ticker, row, md, ksig, text, _rerank_text(text, ksig)))
        k_ids.append(ticker)
        k_texts.append(text)
    audits.bump("kalshi_signatures", len(kitems))
    if not kitems:
        return matches, audits

    k_emb = _encode_incremental(embedder, ids=k_ids, texts=k_texts, cache_name="kalshi_bge_small_v15")

    # Process in batches so reranker work is vectorized and memory is bounded.
    block = 2000
    seen_pairs = set()
    for start in range(0, len(kitems), block):
        stop = min(start + block, len(kitems))
        block_maps: list[dict[int, _HybridCandidate]] = [dict() for _ in range(stop - start)]

        # Domain-local semantic retrieval.
        grouped = defaultdict(list)
        for local, (_, _, _, sig, _, _) in enumerate(kitems[start:stop]):
            grouped[str(sig.domain or "other")].append(local)
        for domain, locals_ in grouped.items():
            idx = domain_indexes.get(domain)
            rows = domain_rows.get(domain)
            if idx is None or rows is None:
                continue
            q = np.ascontiguousarray(k_emb[[start + x for x in locals_]], dtype=np.float32)
            scores, neigh = idx.search(q, min(SEMANTIC_TOP_K, len(rows)))
            for qi, local in enumerate(locals_):
                kept = 0
                for score, rel in zip(scores[qi], neigh[qi]):
                    if rel < 0:
                        continue
                    pi = int(rows[int(rel)])
                    if float(score) < MIN_EMBED_SIM - 0.08:
                        continue
                    _candidate_from(block_maps[local], pi, source="bge_domain", embed_score=float(score))
                    kept += 1
                    if kept >= SEMANTIC_TOP_K:
                        break

        # Tiny global rescue; it is cheap and protects bad category metadata.
        if GLOBAL_RESCUE_K > 0:
            q = np.ascontiguousarray(k_emb[start:stop], dtype=np.float32)
            scores, neigh = global_index.search(q, GLOBAL_RESCUE_K)
            for local in range(stop - start):
                for score, pi in zip(scores[local], neigh[local]):
                    if pi >= 0 and float(score) >= MIN_EMBED_SIM:
                        _candidate_from(block_maps[local], int(pi), source="bge_global", embed_score=float(score))

        # Independent V8.4 deterministic rescue. Cap pathological fan-out; a
        # key matching hundreds of markets is not sufficiently discriminative.
        for local, (_, _, _, ksig, _, _) in enumerate(kitems[start:stop]):
            key = _v84_key(ksig)
            if key is None:
                continue
            hits = v84_index.get(key, [])
            if len(hits) > 20:
                audits.bump("v84_ambiguous_key_skipped")
                continue
            source = f"v84_{key[0]}"
            for pi in hits:
                _candidate_from(block_maps[local], pi, source=source)

        # Keep only the strongest BGE candidates before the expensive final
        # classifier, while preserving deterministic retrieval rescues.
        pair_refs = []
        classifier_pairs = []
        for local, cmap in enumerate(block_maps):
            ranked = sorted(cmap.values(), key=lambda c: c.embedding_score, reverse=True)
            semantic = [c for c in ranked if any(src.startswith("bge_") for src in c.retrieval_sources)]
            selected_ids = {c.poly_idx for c in semantic[:RERANK_TOP_K]}
            selected_ids |= {c.poly_idx for c in ranked if any(src.startswith("v84_") for src in c.retrieval_sources)}
            for c in ranked:
                if c.poly_idx not in selected_ids:
                    continue
                kidx = start + local
                _, row, _, ksig, _, _ = kitems[kidx]
                poly = polys[c.poly_idx]
                k_title = str(row.get("title") or row.get("question") or "")
                p_title = str(poly.market.get("question") or poly.market.get("title") or "")
                pair_refs.append((local, c))
                classifier_pairs.append((
                    serialize_contract(venue="kalshi", title=k_title, sig=ksig),
                    serialize_contract(venue="polymarket", title=p_title, sig=poly.sig),
                ))

        audits.bump("retrieved_topk_pairs", len(pair_refs))
        if classifier_pairs:
            probs = equivalence_model.predict_probabilities(classifier_pairs)
            for (local, c), prob in zip(pair_refs, probs):
                c.equivalence_probability = float(prob)

        # Verify/rank within each Kalshi market. Strict verifier passes are
        # unlimited; exploratory semantic LOW_BASIS is capped at 2 per market.
        for local, cmap in enumerate(block_maps):
            kidx = start + local
            ticker, row, md, ksig, ktext, _ = kitems[kidx]
            if cmap:
                audits.bump("kalshi_with_candidates")
            ordered = sorted(
                [c for c in cmap.values() if c.equivalence_probability is not None],
                key=lambda c: (float(c.equivalence_probability or 0), c.embedding_score, c.deterministic_strength),
                reverse=True,
            )
            for cand in ordered:
                p = polys[cand.poly_idx]
                pid = poly_ids[cand.poly_idx]
                key = (ticker, pid)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                audits.bump("unique_pairs_verified")

                probability = float(cand.equivalence_probability or 0.0)
                decision = equivalence_model.decide(
                    probability=probability,
                    kalshi_title=str(row.get("title") or row.get("question") or ""),
                    polymarket_question=str(p.market.get("question") or p.market.get("title") or ""),
                    ksig=ksig,
                    psig=p.sig,
                    polymarket_market=p.market,
                )
                if decision.status == "EQUIVALENT":
                    verdict, score = "HIGH_CONFIDENCE", probability
                    reasons = [
                        "Final V3 DeBERTa classifier above frozen threshold",
                        "deterministic contradiction gates passed",
                    ]
                elif decision.status == "UNRESOLVED":
                    verdict, score = "REVIEW", probability
                    reasons = [f"UNRESOLVED: {decision.gate_reason}"]
                    audits.bump("unresolved_pairs")
                else:
                    verdict, score = "REJECT", probability
                    reasons = [str(decision.gate_reason or "classifier_reject")]

                reasons += [
                    f"BGE embedding_cosine={cand.embedding_score:.4f}",
                    f"DeBERTa equivalence_probability={probability:.8f}",
                    f"frozen_threshold={equivalence_model.threshold:.16f}",
                ]
                audit = MatchAudit(
                    verdict, score, reasons, ticker, str(row.get("title") or ""),
                    str(p.market.get("question") or ""), asdict(ksig), asdict(p.sig)
                )
                audits.record(audit)

                if verdict not in {"EXACT", "HIGH_CONFIDENCE"}:
                    continue

                # Hydrate accepted pairs only to enrich settlement metadata. Hydration
                # is not allowed to reverse the frozen classifier+gate decision.
                current_ksig, current_psig, current_pm = ksig, p.sig, p.market
                if kalshi_detail_fetcher is not None or polymarket_detail_fetcher is not None:
                    if not current_ksig.rule_text_present and kalshi_detail_fetcher is not None:
                        try:
                            detail = rules._unwrap_market_detail(kalshi_detail_fetcher(ticker))
                            merged = {**md, **detail}
                            if md.get("_event") is not None:
                                merged["_event"] = md["_event"]
                            ns = kalshi_signature(row, merged)
                            if ns is not None:
                                current_ksig = ns
                                audits.bump("hydration_kalshi_success")
                        except Exception:
                            audits.bump("hydration_kalshi_failure")
                    if not current_psig.rule_text_present and polymarket_detail_fetcher is not None:
                        try:
                            detail = rules._unwrap_market_detail(polymarket_detail_fetcher(pid))
                            current_pm = {**p.market, **detail}
                            ns = polymarket_signature(current_pm)
                            if ns is not None:
                                current_psig = ns
                                audits.bump("hydration_polymarket_success")
                        except Exception:
                            audits.bump("hydration_polymarket_failure")

                match = rules._build_match(
                    ticker, row, current_pm, current_ksig, current_psig, score, reasons,
                    "EQUIVALENT:deberta_v3", strict_rules=False,
                )
                cert = match.setdefault("equivalence_certificate", {})
                cert["retrieval_embedding_cosine"] = round(float(cand.embedding_score), 6)
                cert["equivalence_probability"] = round(probability, 8)
                cert["equivalence_threshold"] = equivalence_model.threshold
                cert["retrieval_sources"] = sorted(cand.retrieval_sources)
                cert["matcher_version"] = "deberta_v3_plus_deterministic_gates"
                cert["match_verdict"] = "EQUIVALENT"
                matches.append(match)
                audits.bump("accepted_pairs")

        done = stop
        print(
            f"Final matcher progress {done}/{len(kitems)} | classified={int(audits.stage_counts.get('retrieved_topk_pairs',0))} "
            f"| verified={int(audits.stage_counts.get('unique_pairs_verified',0))} "
            f"| accepted={int(audits.stage_counts.get('accepted_pairs',0))}",
            flush=True,
        )

    # De-duplicate by exact venue pair (strict beats relaxed if a path somehow duplicates).
    dedup = {}
    for m in matches:
        pid = str((m.get("polymarket_market") or {}).get("id") or (m.get("polymarket_market") or {}).get("conditionId") or "")
        key = (str(m.get("kalshi_ticker") or ""), pid)
        old = dedup.get(key)
        if old is None:
            dedup[key] = m
        else:
            old_low = str((old.get("equivalence_certificate") or {}).get("resolution_lane") or "") == "LOW_BASIS"
            new_low = str((m.get("equivalence_certificate") or {}).get("resolution_lane") or "") == "LOW_BASIS"
            if old_low and not new_low:
                dedup[key] = m

    return list(dedup.values()), audits
