# src/trustworthy_maternal_postpartum_rag/retrieval/chroma_retriever.py
import os
import logging
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from warnings import filters
import chromadb
from chromadb.utils import embedding_functions
from trustworthy_maternal_postpartum_rag.utils.config import get_config
import time

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults (read from env, but re-read lazily when needed)
# -----------------------------------------------------------------------------

def _get_defaults() -> Dict[str, str]:
    """
    Lazy env reads so unit tests / notebooks can override env vars without
    needing a fresh Python process.
    """
    return {
        "chroma_path": os.getenv("TMPRAG_CHROMA_PATH", "data/chroma_db"),
        "collection": os.getenv("TMPRAG_COLLECTION_NAME", "maternal_postpartum_chunks"),
        "embed_model": os.getenv("TMPRAG_EMBED_MODEL", "all-MiniLM-L6-v2"),
        "device": os.getenv("TMPRAG_EMBED_DEVICE", "cpu"),
        # If true, always rebuild collection objects (useful for debugging)
        "disable_cache": os.getenv("TMPRAG_DISABLE_CHROMA_CACHE", "false").lower() in {"1", "true", "yes"},
        # Extra logs for retrieval
        "debug": os.getenv("TMPRAG_RETRIEVER_DEBUG", "false").lower() in {"1", "true", "yes"},
    }


# -----------------------------------------------------------------------------
# Cache: keyed by (path, collection, model, device)
# -----------------------------------------------------------------------------

_client_cache: Dict[str, chromadb.PersistentClient] = {}
_collection_cache: Dict[Tuple[str, str, str, str], Any] = {}


def reset_cache() -> None:
    """
    Useful for tests (pytest) or experiments where you want a clean slate.
    """
    _client_cache.clear()
    _collection_cache.clear()
    logger.info("Chroma retriever cache cleared.")


# -----------------------------------------------------------------------------
# Metadata normalization
# -----------------------------------------------------------------------------

_REQUIRED_META_DEFAULTS: Dict[str, Any] = {
    "publisher": "UNKNOWN",
    "source_file": "unknown",
    "page_number": -1,
    "lifecycle": "general",
    "stage": "",
    "country": "",
    "source_type": "",
    "topic_hint": "general",
    "language": "en",
}

def _norm_str(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure critical keys exist so downstream logic (publisher balancing,
    stage filtering, audits) behaves deterministically.

    Notes:
    - We do not aggressively canonicalize case for publisher because you already
      have meaningful casing ("Cleveland Clinic"). We only strip/blank-check.
    """
    meta_in = dict(meta or {})
    out = dict(_REQUIRED_META_DEFAULTS)

    # overlay stored keys
    for k, v in meta_in.items():
        out[k] = v

    out["publisher"] = _norm_str(out.get("publisher"), "UNKNOWN")
    out["source_file"] = _norm_str(out.get("source_file"), "unknown")
    out["source_type"] = _norm_str(out.get("source_type"), "")
    out["stage"] = _norm_str(out.get("stage"), "")
    out["lifecycle"] = _norm_str(out.get("lifecycle"), "general")
    out["country"] = _norm_str(out.get("country"), "")
    out["topic_hint"] = _norm_str(out.get("topic_hint"), "general")
    out["language"] = _norm_str(out.get("language"), "en")

    # Page number should be int if possible
    pn = out.get("page_number", -1)
    try:
        out["page_number"] = int(pn)
    except Exception:
        out["page_number"] = -1

    # Keep doc_id if present (handy for debugging)
    if "doc_id" in out and out["doc_id"] is None:
        out["doc_id"] = ""

    return out

# -----------------------------------------------------------------------------
# Where validation (lightweight)
# -----------------------------------------------------------------------------

def _validate_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma expects a dict with simple comparable values or operator dicts.
    We keep it permissive but avoid the most common footguns.
    """
    if where is None:
        return None
    if not isinstance(where, dict):
        raise TypeError(f"where must be a dict or None, got {type(where)}")
    # Avoid empty dict: behaves like no filter
    if not where:
        return None
    return where


# -----------------------------------------------------------------------------
# Collection getter (cached)
# -----------------------------------------------------------------------------

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

    # Cache client by path
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
        chroma_path, collection_name, embed_model, device
    )
    return collection


# -----------------------------------------------------------------------------
# Public retrieve()
# -----------------------------------------------------------------------------
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
    Returns a list of hits:
      [{"id": str, "text": str, "metadata": dict, "distance": float}, ...]

    Notes:
    - metadata is normalized to include keys your local_qa expects.
    - where is passed directly to Chroma (after light validation).
    """
    start = time.time()
    CFG = get_config()
    retr_cfg = CFG.get("retrieval", {})
    max_distance = retr_cfg.get("max_distance", None)
    max_per_pub = retr_cfg.get("max_hits_per_publisher", 2)
    collection_name = collection_name or (CFG.get("indexing", {}).get(
            "collection_name",
            "maternal_postpartum_chunks")
        )
    if not query:
        return []

    query = " ".join(query.strip().split())   # trims + removes extra spaces/newlines
    query_lower = query.lower()

    want_postpartum = any(x in query_lower for x in [
    "postpartum", "after delivery", "after c section",
    "after cesarean", "after birth", "c section"
])

    want_baby = any(x in query_lower for x in [
        "baby", "infant", "newborn", "toddler"
    ])

    # Intent-aware expansion
    boost_terms = []

    if any(x in query_lower for x in ["c section", "c-section", "cesarean"]):
        boost_terms += [
            "postpartum recovery",
            "surgery healing",
            "wound healing",
            "iron rich foods",
            "protein foods",
            "constipation relief",
            "hydration"
        ]

    if "protein" in query_lower:
            boost_terms += [
                "eggs paneer lentils greek yogurt milk tofu beans chickpeas"
            ]

    if "breastfeeding" in query_lower:
        boost_terms += [
            "milk supply calcium hydration oats"
        ]

    if boost_terms:
        query += " " + " ".join(boost_terms)
        if len(query) > 1000:
            logger.warning("Query truncated from %d chars", len(query))
        query = query[:1000]

    if not query:
        return []

    k = int(k)
    if k <= 0:
        return []
    defaults = _get_defaults()

    chroma_path = Path(chroma_path or defaults["chroma_path"])
    embed_model = embed_model or defaults["embed_model"]
    device = device or defaults["device"]
    disable_cache = bool(defaults["disable_cache"])
    debug = bool(defaults["debug"])

    where = _validate_where(where)

    # build internal filters safely
    filters = dict(where)

    col = _get_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embed_model=embed_model,
        device=device,
        disable_cache=disable_cache,
    )

    if any(x in query_lower for x in ["postpartum", "after delivery", "c section", "c-section", "cesarean"]):
        filters["lifecycle"] = "postpartum"

    if any(x in query_lower for x in ["baby", "infant", "newborn", "toddler"]):
        filters["target"] = "baby"

    where = filters or None

    try:
        res = col.query(
            query_texts=[query],
            n_results=k * 4,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("Retriever query failed")
        raise RuntimeError("Retriever query failed") from None

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    limit = min(len(ids), len(docs), len(metas), len(dists))
    if max_distance is not None and dists:
        closest = float(dists[0]) if dists[0] is not None else 999
        if closest > max_distance:
            logger.warning(
                "All raw hits exceed threshold | query=%r threshold=%.2f closest=%.4f",
                query, max_distance, closest
            )
    dropped_empty = 0
    dropped_distance = 0
    dropped_diversity = 0
    
    for i in range(limit):
        text = (docs[i] or "").strip()
        if not text:
            dropped_empty += 1
            continue

        meta = _normalize_metadata(metas[i] or {})
        dist_val = float(dists[i]) if dists[i] is not None else 999

        if max_distance is not None and dist_val > max_distance:
            dropped_distance += 1
            continue
        
        similarity = 1 - dist_val

        hits.append({
            "id": ids[i],
            "text": text,
            "metadata": meta,
            "distance": dist_val,
            "similarity": similarity,
        })
    h["score"] = h["similarity"] + overlap * 0.03

    # Safety: ensure non-decreasing distance ordering if anything upstream reorders
    hits.sort(key=lambda h: h.get("score", -999), reverse=True)
    seen = set()
    dedup_hits = []
    # lexical rerank
    query_words = set(query_lower.split())

    for h in hits:
        text_lower = h["text"].lower()
        text_words = set(text_lower.split())
        overlap = sum(1 for w in query_words if w in text_words)
        h["score"] = h["similarity"] + overlap * 0.03

    hits.sort(key=lambda x: x["score"], reverse=True)

    for h in hits:
        key = hashlib.md5(h["text"][:250].strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        dedup_hits.append(h)

    hits = dedup_hits

    balanced_hits = []
    pub_counts = {}
    if not hits:
        logger.warning("No retrieval hits passed threshold for query=%r", query)

    for h in hits:
        pub = h["metadata"].get("publisher", "UNKNOWN")

        if pub_counts.get(pub, 0) >= max_per_pub:
            dropped_diversity += 1
            continue

        balanced_hits.append(h)
        pub_counts[pub] = pub_counts.get(pub, 0) + 1

    hits = balanced_hits[:k]

    if debug:
        debug_pub_counts: Dict[str, int] = {}
        for h in hits:
            p = (h.get("metadata") or {}).get("publisher", "UNKNOWN")
            debug_pub_counts[p] = debug_pub_counts.get(p, 0) + 1
        logger.info(
            "Drops | empty=%d threshold=%d diversity=%d",
            dropped_empty,
            dropped_distance,
            dropped_diversity
        )

        logger.info(
            "Retriever debug | query=%r k=%d where=%s hits=%d pubs=%s",
            query, k, where, len(hits), debug_pub_counts
        )
        if hits:
            h0 = hits[0]
            m0 = h0["metadata"]
            logger.info(
                "Retriever top hit | dist=%.4f pub=%s file=%s page=%s stage=%s lc=%s snip=%s",
                float(h0.get("distance", 0.0)),
                m0.get("publisher"),
                m0.get("source_file"),
                m0.get("page_number"),
                m0.get("stage"),
                m0.get("lifecycle"),
                (h0.get("text", "")[:160].replace("\n", " ")),
            )
        
    if not hits and limit > 0:
            logger.warning("All hits filtered; returning top raw candidate")
            raw_meta = _normalize_metadata(metas[0] or {})
            return [{
                "id": ids[0],
                "text": (docs[0] or "").strip(),
                "metadata": raw_meta,
                "distance": float(dists[0]) if dists[0] is not None else 999,
                "similarity": 0.0,
            }]
    logger.info("Final distances: %s", [round(h["distance"],4) for h in hits])
    logger.info("Retrieval latency %.3fs", time.time() - start)
    return hits

