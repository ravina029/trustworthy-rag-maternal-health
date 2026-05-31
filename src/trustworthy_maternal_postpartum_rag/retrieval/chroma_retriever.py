# src/trustworthy_maternal_postpartum_rag/retrieval/chroma_retriever.py

import os
import time
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

from trustworthy_maternal_postpartum_rag.utils.config import get_config


logger = logging.getLogger(__name__)

CFG = get_config("configs/pipeline_config.yaml")


# =============================================================================
# Runtime defaults
# =============================================================================

def _get_defaults() -> Dict[str, Any]:
    return {
        "chroma_path": os.getenv("TMPRAG_CHROMA_PATH", "data/chroma_db"),
        "collection": os.getenv(
            "TMPRAG_COLLECTION_NAME",
            CFG.get("indexing", {}).get("collection_name", "maternal_postpartum_chunks"),
        ),
        "embed_model": os.getenv(
            "TMPRAG_EMBED_MODEL",
            CFG.get("embedding", {}).get("model", "all-MiniLM-L6-v2"),
        ),
        "device": os.getenv(
            "TMPRAG_EMBED_DEVICE",
            CFG.get("embedding", {}).get("device", "cpu"),
        ),
        "disable_cache": os.getenv("TMPRAG_DISABLE_CHROMA_CACHE", "false").lower()
        in {"1", "true", "yes"},
        "debug": os.getenv("TMPRAG_RETRIEVER_DEBUG", "false").lower()
        in {"1", "true", "yes"},
    }


# =============================================================================
# Cache
# =============================================================================

_client_cache: Dict[str, chromadb.PersistentClient] = {}
_collection_cache: Dict[Tuple[str, str, str, str], Any] = {}


def reset_cache() -> None:
    _client_cache.clear()
    _collection_cache.clear()
    logger.info("Chroma retriever cache cleared.")


# =============================================================================
# Metadata normalization
# =============================================================================

_CANONICAL_META_DEFAULTS: Dict[str, Any] = {
    "chunk_id": "",
    "doc_id": "",
    "doc_title": "",
    "source_file": "unknown",
    "source_path": "",
    "page_number": -1,
    "publisher": "UNKNOWN",
    "source_tier": "unknown",
    "document_style": "unknown",
    "lifecycle_stage": "unknown",
    "topic_scope": "unknown",
    "country_scope": "unknown",
    "priority_score": 1,
    "target": "unknown",
    "language": "en",
    "metadata_source": "unknown",
    "doc_version": "",
    "preprocessing_version": "",
    "chunk_version": "",
    "inferred_lifecycle": "general",
    "medical_type": "general",
    "quality_score": 0.0,
    "chunk_word_count": 0,
    "topic_hint": "",
}


def _norm_str(value: Any, default: str) -> str:
    if value is None:
        return default

    out = str(value).strip()
    return out if out else default


def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize metadata retrieved from Chroma.

    Uses the new canonical metadata schema only.
    """

    meta_in = dict(meta or {})
    out = dict(_CANONICAL_META_DEFAULTS)

    for key, value in meta_in.items():
        out[key] = value

    out["chunk_id"] = _norm_str(out.get("chunk_id"), "")
    out["doc_id"] = _norm_str(out.get("doc_id"), "")
    out["doc_title"] = _norm_str(out.get("doc_title"), "")
    out["source_file"] = _norm_str(out.get("source_file"), "unknown")
    out["source_path"] = _norm_str(out.get("source_path"), "")
    out["publisher"] = _norm_str(out.get("publisher"), "UNKNOWN")
    out["source_tier"] = _norm_str(out.get("source_tier"), "unknown")
    out["document_style"] = _norm_str(out.get("document_style"), "unknown")
    out["lifecycle_stage"] = _norm_str(out.get("lifecycle_stage"), "unknown")
    out["topic_scope"] = _norm_str(out.get("topic_scope"), "unknown")
    out["country_scope"] = _norm_str(out.get("country_scope"), "unknown")
    out["target"] = _norm_str(out.get("target"), "unknown")
    out["language"] = _norm_str(out.get("language"), "en")
    out["metadata_source"] = _norm_str(out.get("metadata_source"), "unknown")
    out["doc_version"] = _norm_str(out.get("doc_version"), "")
    out["preprocessing_version"] = _norm_str(out.get("preprocessing_version"), "")
    out["chunk_version"] = _norm_str(out.get("chunk_version"), "")
    out["inferred_lifecycle"] = _norm_str(out.get("inferred_lifecycle"), "general")
    out["medical_type"] = _norm_str(out.get("medical_type"), "general")
    out["topic_hint"] = _norm_str(out.get("topic_hint"), "")

    try:
        out["page_number"] = int(out.get("page_number", -1))
    except Exception:
        out["page_number"] = -1

    try:
        out["priority_score"] = int(out.get("priority_score", 1))
    except Exception:
        out["priority_score"] = 1

    try:
        out["quality_score"] = float(out.get("quality_score", 0.0))
    except Exception:
        out["quality_score"] = 0.0

    try:
        out["chunk_word_count"] = int(out.get("chunk_word_count", 0))
    except Exception:
        out["chunk_word_count"] = 0


    # -------------------------------------------------------------------------
    # Legacy aliases for old downstream code.
    # Keep canonical fields, but also expose old keys expected by local_qa.py.
    # -------------------------------------------------------------------------
    lifecycle_stage = _norm_str(out.get("lifecycle_stage"), "unknown")
    topic_scope = _norm_str(out.get("topic_scope"), "unknown")
    country_scope = _norm_str(out.get("country_scope"), "unknown")
    source_tier = _norm_str(out.get("source_tier"), "unknown")
    document_style = _norm_str(out.get("document_style"), "unknown")

    combined_lifecycle_text = f"{lifecycle_stage} {topic_scope}".lower()

    # Preserve multi-stage canonical metadata for older downstream code.
    # Do not collapse "pregnancy_postpartum" into only "postpartum".
    if "pregnancy_postpartum" in combined_lifecycle_text:
        legacy_stage = "pregnancy_postpartum"
    elif "pregnancy_childbirth_postpartum_newborn" in combined_lifecycle_text:
        legacy_stage = "pregnancy_childbirth_postpartum_newborn"
    elif "postpartum_newborn" in combined_lifecycle_text:
        legacy_stage = "postpartum_newborn"
    elif "pregnancy" in combined_lifecycle_text or "antenatal" in combined_lifecycle_text:
        legacy_stage = "pregnancy"
    elif "postpartum" in combined_lifecycle_text or "postnatal" in combined_lifecycle_text:
        legacy_stage = "postpartum"
    elif any(x in combined_lifecycle_text for x in ["newborn", "infant", "baby", "child"]):
        legacy_stage = "newborn"
    else:
        legacy_stage = "general"

    out["stage"] = legacy_stage
    out["lifecycle"] = legacy_stage
    out["country"] = country_scope
    out["source_type"] = source_tier if source_tier != "unknown" else document_style

    if out.get("target") in [None, "", "unknown"]:
        if legacy_stage == "newborn":
            out["target"] = "baby"
        elif legacy_stage in {"pregnancy", "postpartum"}:
            out["target"] = "mother"
        else:
            out["target"] = "mother+baby"

    return out


def _validate_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if where is None:
        return None

    if not isinstance(where, dict):
        raise TypeError(f"where must be dict or None, got {type(where)}")

    return where or None


# =============================================================================
# Query classification
# =============================================================================

def classify_query_lifecycle(query: str) -> str:
    q = query.lower()

    if any(x in q for x in [
        "newborn", "baby", "infant", "toddler", "child",
        "feeding baby", "baby fever", "baby not feeding",
    ]):
        return "baby"

    if any(x in q for x in [
        "postpartum", "postnatal", "after delivery", "after birth",
        "after childbirth", "lochia", "perineal", "c-section",
        "c section", "cesarean", "breast pain", "mastitis",
    ]):
        return "postpartum"

    if any(x in q for x in [
        "pregnant", "pregnancy", "antenatal", "trimester",
        "gestation", "fetus", "foetal", "prenatal",
    ]):
        return "pregnancy"

    if any(x in q for x in [
        "breastfeeding", "lactation", "milk supply", "nipple",
        "expressing milk",
    ]):
        return "breastfeeding"

    return "general"


def classify_query_risk(query: str) -> str:
    q = query.lower()

    high_risk_terms = [
        "heavy bleeding",
        "severe bleeding",
        "fits",
        "convulsion",
        "seizure",
        "severe headache",
        "blurred vision",
        "chest pain",
        "difficulty breathing",
        "shortness of breath",
        "suicidal",
        "harm myself",
        "baby not feeding",
        "newborn fever",
        "fever in newborn",
        "not breathing",
        "unconscious",
        "severe abdominal pain",
    ]

    if any(term in q for term in high_risk_terms):
        return "high_risk"

    return "general"


def expand_query(query: str) -> str:
    """
    Controlled query expansion for common maternal/postpartum intents.
    """

    q = query.lower()
    boost_terms: List[str] = []

    if any(x in q for x in ["c section", "c-section", "cesarean"]):
        boost_terms += [
            "postpartum recovery",
            "wound healing",
            "pain",
            "bleeding",
            "infection",
        ]

    if "breastfeeding" in q:
        boost_terms += [
            "lactation",
            "milk supply",
            "breast pain",
            "sore nipple",
        ]

    if any(x in q for x in ["danger sign", "warning sign", "urgent", "emergency"]):
        boost_terms += [
            "seek care immediately",
            "go to hospital",
            "danger signs",
        ]

    if not boost_terms:
        return query.strip()

    expanded = query.strip() + " " + " ".join(boost_terms)
    return expanded[:1000]


# =============================================================================
# Collection getter
# =============================================================================

def _get_collection(
    chroma_path: Path,
    collection_name: str,
    embed_model: str,
    device: str,
    *,
    disable_cache: bool = False,
):
    key = (str(chroma_path), collection_name, embed_model, device)

    if not disable_cache and key in _collection_cache:
        return _collection_cache[key]

    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client_key = str(chroma_path)
    client = None if disable_cache else _client_cache.get(client_key)

    if client is None:
        client = chromadb.PersistentClient(path=str(chroma_path))

        if not disable_cache:
            _client_cache[client_key] = client

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model,
        device=device,
    )

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to open Chroma collection '{collection_name}': {e}"
        ) from e

    if not disable_cache:
        _collection_cache[key] = collection

    logger.info(
        "Chroma ready | path=%s collection=%s model=%s device=%s",
        chroma_path,
        collection_name,
        embed_model,
        device,
    )

    return collection


# =============================================================================
# Reranking
# =============================================================================

def _lexical_overlap_score(query: str, text: str) -> float:
    query_words = {
        w.strip(".,;:!?()[]{}").lower()
        for w in query.split()
        if len(w.strip(".,;:!?()[]{}")) > 2
    }

    text_words = {
        w.strip(".,;:!?()[]{}").lower()
        for w in text.split()
        if len(w.strip(".,;:!?()[]{}")) > 2
    }

    if not query_words or not text_words:
        return 0.0

    overlap = len(query_words.intersection(text_words))
    return min(overlap * 0.03, 0.30)


def _source_score(meta: Dict[str, Any], risk_level: str) -> float:
    score = 0.0

    source_tier = meta.get("source_tier", "unknown")
    publisher = meta.get("publisher", "UNKNOWN")
    priority_score = int(meta.get("priority_score", 1))

    score += min(priority_score, 4) * 0.025

    if source_tier == "core_authoritative":
        score += 0.12

    if source_tier == "secondary_patient_friendly":
        score -= 0.04

    if publisher in {"WHO", "NHS", "ACOG", "Government of India / PMSMA"}:
        score += 0.04

    if risk_level == "high_risk" and source_tier != "core_authoritative":
        score -= 0.25

    return score


def _lifecycle_score(meta: Dict[str, Any], query_lifecycle: str) -> float:
    lifecycle_stage = meta.get("lifecycle_stage", "unknown")
    topic_scope = meta.get("topic_scope", "unknown")
    inferred_lifecycle = meta.get("inferred_lifecycle", "general")

    if query_lifecycle == "general":
        return 0.0

    score = 0.0

    if query_lifecycle == "pregnancy":
        if "pregnancy" in lifecycle_stage:
            score += 0.12
        if "pregnancy" in topic_scope or "antenatal" in topic_scope:
            score += 0.08
        if inferred_lifecycle == "pregnancy":
            score += 0.04

    elif query_lifecycle == "postpartum":
        if "postpartum" in lifecycle_stage or "postnatal" in topic_scope:
            score += 0.14
        if "newborn" in lifecycle_stage and "postpartum" not in lifecycle_stage:
            score -= 0.06
        if inferred_lifecycle == "postpartum":
            score += 0.04

    elif query_lifecycle == "baby":
        if any(x in lifecycle_stage for x in ["newborn", "infant", "child"]):
            score += 0.14
        if any(x in topic_scope for x in ["newborn", "baby", "child"]):
            score += 0.08
        if inferred_lifecycle in {"newborn", "infant", "toddler"}:
            score += 0.04

    elif query_lifecycle == "breastfeeding":
        if any(x in lifecycle_stage for x in ["postpartum", "newborn", "infant"]):
            score += 0.10
        if inferred_lifecycle == "breastfeeding":
            score += 0.08

    return score


def _quality_score(meta: Dict[str, Any]) -> float:
    q = float(meta.get("quality_score", 0.0))
    return min(max(q, 0.0), 1.0) * 0.04


def _dedup_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for hit in hits:
        text = hit.get("text", "")
        key = hashlib.md5(text[:300].strip().lower().encode("utf-8")).hexdigest()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(hit)

    return deduped


def _balance_by_publisher(
    hits: List[Dict[str, Any]],
    k: int,
    max_per_pub: Optional[int],
) -> List[Dict[str, Any]]:
    if max_per_pub is None or max_per_pub <= 0:
        return hits[:k]

    balanced = []
    pub_counts = Counter()

    for hit in hits:
        publisher = hit["metadata"].get("publisher", "UNKNOWN")

        if pub_counts[publisher] >= max_per_pub:
            continue

        balanced.append(hit)
        pub_counts[publisher] += 1

        if len(balanced) >= k:
            break

    if len(balanced) < k:
        seen_ids = {h["id"] for h in balanced}

        for hit in hits:
            if hit["id"] in seen_ids:
                continue

            balanced.append(hit)
            seen_ids.add(hit["id"])

            if len(balanced) >= k:
                break

    return balanced[:k]


# =============================================================================
# Public retrieve()
# =============================================================================

def retrieve(
    query: str,
    k: int = 8,
    *,
    chroma_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
    embed_model: Optional[str] = None,
    device: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return retrieval hits:

    [
        {
            "id": str,
            "text": str,
            "metadata": dict,
            "distance": float,
            "similarity": float,
            "score": float
        }
    ]
    """

    start = time.time()

    if not query or not query.strip():
        return []

    k = int(k)

    if k <= 0:
        return []

    defaults = _get_defaults()

    chroma_path = Path(chroma_path or defaults["chroma_path"])
    collection_name = collection_name or defaults["collection"]
    embed_model = embed_model or defaults["embed_model"]
    device = device or defaults["device"]
    disable_cache = bool(defaults["disable_cache"])
    debug = bool(defaults["debug"])

    retrieval_cfg = CFG.get("retrieval", {})
    max_distance = retrieval_cfg.get("max_distance", None)
    max_per_pub = retrieval_cfg.get("max_hits_per_publisher", 3)
    candidate_multiplier = retrieval_cfg.get("candidate_multiplier", 8)
    min_candidate_pool = retrieval_cfg.get("min_candidate_pool", 32)

    where = _validate_where(where)

    original_query = " ".join(query.strip().split())
    expanded_query = expand_query(original_query)

    query_lifecycle = classify_query_lifecycle(original_query)
    risk_level = classify_query_risk(original_query)

    collection = _get_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embed_model=embed_model,
        device=device,
        disable_cache=disable_cache,
    )

    n_results = max(k * candidate_multiplier, min_candidate_pool)

    try:
        result = collection.query(
            query_texts=[expanded_query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("Retriever query failed")
        raise RuntimeError("Retriever query failed") from None

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    limit = min(len(ids), len(docs), len(metas), len(dists))

    raw_hits: List[Dict[str, Any]] = []

    dropped_empty = 0
    dropped_distance = 0

    for i in range(limit):
        text = (docs[i] or "").strip()

        if not text:
            dropped_empty += 1
            continue

        meta = _normalize_metadata(metas[i] or {})
        distance = float(dists[i]) if dists[i] is not None else 999.0

        if max_distance is not None and distance > float(max_distance):
            dropped_distance += 1
            continue

        similarity = 1.0 - distance

        lexical = _lexical_overlap_score(original_query, text)
        source = _source_score(meta, risk_level)
        lifecycle = _lifecycle_score(meta, query_lifecycle)
        quality = _quality_score(meta)

        final_score = similarity + lexical + source + lifecycle + quality

        raw_hits.append({
            "id": ids[i],
            "text": text,
            "metadata": meta,
            "distance": distance,
            "similarity": similarity,
            "score": final_score,
            "score_components": {
                "similarity": similarity,
                "lexical": lexical,
                "source": source,
                "lifecycle": lifecycle,
                "quality": quality,
            },

            # ---------------------------------------------------------------------
            # Top-level compatibility fields for older downstream code in local_qa.py
            # ---------------------------------------------------------------------
            "chunk_id": meta.get("chunk_id") or ids[i],
            "publisher": meta.get("publisher"),
            "source_file": meta.get("source_file"),
            "page_number": meta.get("page_number"),
            "doc_id": meta.get("doc_id"),
            "stage": meta.get("stage"),
            "lifecycle": meta.get("lifecycle"),
            "source_type": meta.get("source_type"),
            "country": meta.get("country"),
            "target": meta.get("target"),

            # Canonical fields also exposed at top level
            "source_tier": meta.get("source_tier"),
            "priority_score": meta.get("priority_score"),
            "lifecycle_stage": meta.get("lifecycle_stage"),
            "topic_scope": meta.get("topic_scope"),
            "country_scope": meta.get("country_scope"),
            "document_style": meta.get("document_style"),
            "quality_score": meta.get("quality_score"),
        })

    raw_hits.sort(key=lambda hit: hit.get("score", -999), reverse=True)

    deduped_hits = _dedup_hits(raw_hits)

    hits = _balance_by_publisher(
        hits=deduped_hits,
        k=k,
        max_per_pub=max_per_pub,
    )

    if not hits and limit > 0:
        logger.warning("All hits filtered; returning top raw Chroma result.")

        fallback_meta = _normalize_metadata(metas[0] or {})
        fallback_distance = float(dists[0]) if dists[0] is not None else 999.0

        return [{
            "id": ids[0],
            "text": (docs[0] or "").strip(),
            "metadata": fallback_meta,
            "distance": fallback_distance,
            "similarity": 1.0 - fallback_distance,
            "score": 1.0 - fallback_distance,
            "score_components": {
                "similarity": 1.0 - fallback_distance,
                "lexical": 0.0,
                "source": 0.0,
                "lifecycle": 0.0,
                "quality": 0.0,
            },

            "chunk_id": fallback_meta.get("chunk_id") or ids[0],
            "publisher": fallback_meta.get("publisher"),
            "source_file": fallback_meta.get("source_file"),
            "page_number": fallback_meta.get("page_number"),
            "doc_id": fallback_meta.get("doc_id"),
            "stage": fallback_meta.get("stage"),
            "lifecycle": fallback_meta.get("lifecycle"),
            "source_type": fallback_meta.get("source_type"),
            "country": fallback_meta.get("country"),
            "target": fallback_meta.get("target"),
            "source_tier": fallback_meta.get("source_tier"),
            "priority_score": fallback_meta.get("priority_score"),
            "lifecycle_stage": fallback_meta.get("lifecycle_stage"),
            "topic_scope": fallback_meta.get("topic_scope"),
            "country_scope": fallback_meta.get("country_scope"),
            "document_style": fallback_meta.get("document_style"),
            "quality_score": fallback_meta.get("quality_score"),
        }]

    if debug:
        pub_counts = Counter(hit["metadata"].get("publisher", "UNKNOWN") for hit in hits)
        tier_counts = Counter(hit["metadata"].get("source_tier", "unknown") for hit in hits)
        lifecycle_counts = Counter(hit["metadata"].get("lifecycle_stage", "unknown") for hit in hits)

        logger.info(
            "Retriever debug | query=%r expanded=%r k=%d where=%s hits=%d query_lifecycle=%s risk=%s",
            original_query,
            expanded_query,
            k,
            where,
            len(hits),
            query_lifecycle,
            risk_level,
        )

        logger.info(
            "Retriever debug counts | publishers=%s tiers=%s lifecycle=%s",
            dict(pub_counts),
            dict(tier_counts),
            dict(lifecycle_counts),
        )

        logger.info(
            "Retriever drops | empty=%d distance=%d",
            dropped_empty,
            dropped_distance,
        )

        if hits:
            top = hits[0]
            meta = top["metadata"]

            logger.info(
                "Top hit | score=%.4f distance=%.4f publisher=%s tier=%s lifecycle=%s topic=%s file=%s page=%s snip=%s",
                top.get("score", 0.0),
                top.get("distance", 0.0),
                meta.get("publisher"),
                meta.get("source_tier"),
                meta.get("lifecycle_stage"),
                meta.get("topic_scope"),
                meta.get("source_file"),
                meta.get("page_number"),
                top.get("text", "")[:180].replace("\n", " "),
            )

    logger.info(
        "Retrieval done | query_lifecycle=%s risk=%s hits=%d distances=%s latency=%.3fs",
        query_lifecycle,
        risk_level,
        len(hits),
        [round(hit["distance"], 4) for hit in hits],
        time.time() - start,
    )

    return hits