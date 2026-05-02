# src/trustworthy_maternal_postpartum_rag/ingestion/chroma_index.py

import json
from pathlib import Path
import logging
import uuid
import os

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from trustworthy_maternal_postpartum_rag.utils.config import get_config
import random
import numpy as np

# ============================================================
# Config + Seeds
# ============================================================

CFG = get_config("configs/pipeline_config.yaml")

random.seed(CFG["run"]["seed"])
np.random.seed(CFG["run"]["seed"])

# ============================================================
# Pipeline RUN_ID
# ============================================================

RUN_ID_ENV = "TMPRAG_RUN_ID"

def get_run_id() -> str:
    rid = os.environ.get(RUN_ID_ENV)
    if rid:
        return rid
    rid = str(uuid.uuid4())
    os.environ[RUN_ID_ENV] = rid
    return rid

RUN_ID = get_run_id()

class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID
        return True

# ============================================================
# Logging
# ============================================================

LOG_FILE = Path("logs/data_logs/indexing.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.indexing")
LOG_LEVEL = getattr(logging, CFG["logging"]["level"].upper(), logging.INFO)
logger.setLevel(LOG_LEVEL)

fmt = "%(asctime)s - %(levelname)s - run_id=%(run_id)s - %(message)s"

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE)
    for h in logger.handlers
):
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(fmt))
    fh.addFilter(RunIdFilter())
    logger.addHandler(fh)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(logging.Formatter(fmt))
    sh.addFilter(RunIdFilter())
    logger.addHandler(sh)

# ============================================================
# CONFIG (path-driven)
# ============================================================

CHUNKS_DIR = Path("data/chunks")
CHROMA_PATH = Path("data/chroma_db")

COLLECTION_NAME = f"{CFG['indexing']['collection_prefix']}_{CFG['run']['version']}"
BATCH_SIZE = CFG["indexing"]["batch_size"]
RECURSIVE_CHUNK_SCAN = False

# ============================================================
# Chunk Iterator
# ============================================================

def iter_chunks():
    pattern = "**/*_chunks.jsonl" if RECURSIVE_CHUNK_SCAN else "*_chunks.jsonl"
    files = sorted(CHUNKS_DIR.glob(pattern), key=lambda p: str(p).lower())

    if not files:
        logger.warning("[Index] No chunk files found in %s", CHUNKS_DIR)

    logger.info(
        "[Index] Scan | dir=%s files=%d recursive=%s",
        CHUNKS_DIR, len(files), RECURSIVE_CHUNK_SCAN
    )

    for f in files:
        logger.info("[Index] Read | file=%s", f.name)

        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue

                try:
                    rec = json.loads(line)
                except Exception as e:
                    logger.warning("[Index] Bad JSON skipped | file=%s error=%s", f.name, e)
                    continue

                text = (rec.get("text") or "").strip()
                chunk_id = rec.get("chunk_id")

                if not chunk_id or not text:
                    continue

                yield rec

# ============================================================
# Main Indexing Logic
# ============================================================

def main():
    logger.info(
        "[Index] Start | chroma_path=%s collection=%s model=%s",
        CHROMA_PATH, COLLECTION_NAME, CFG["embedding"]["model"]
    )

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=CFG["embedding"]["model"],
        device=CFG["embedding"]["device"],
    )

    # Safer deletion
    try:
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            if CFG["indexing"]["rebuild_collection"]:
                client.delete_collection(COLLECTION_NAME)
                logger.info("[Index] Old collection deleted")
    except Exception as e:
        logger.warning("[Index] Collection delete skipped | error=%s", e)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    total_upserted = 0
    seen_ids_in_run = set()

    for rec in tqdm(iter_chunks(), desc="Indexing chunks"):
        chunk_id = rec.get("chunk_id")
        text = (rec.get("text") or "").strip()

        if not chunk_id or not text:
            continue

        if chunk_id in seen_ids_in_run:
            logger.warning("[Index] Duplicate chunk_id skipped | id=%s", chunk_id)
            continue        
        seen_ids_in_run.add(chunk_id)
        ids.append(chunk_id)
        documents.append(text)

        # Clean metadata (avoid non-serializable types)
        meta = {
            k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in {
                "chunk_id": chunk_id,
                "source_file": rec.get("source_file"),
                "page_number": rec.get("page_number"),
                "doc_id": rec.get("doc_id"),
                "country": rec.get("country"),
                "stage": rec.get("stage"),
                "inferred_lifecycle": rec.get("inferred_lifecycle"),
                "target": rec.get("target"),
                "source_type": rec.get("source_type"),
                "publisher": rec.get("publisher"),
                "topic_hint": rec.get("topic_hint"),
                "version": rec.get("version"),
                "language": rec.get("language"),
                "medical_type": rec.get("medical_type"),
                "quality_score": float(rec["quality_score"]) if rec.get("quality_score") is not None else 0.0,
            }.items()
            if v is not None
        }

        metadatas.append(meta)

        if len(ids) >= BATCH_SIZE:
            try:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_upserted += len(ids)
                logger.info("[Index] Upserted | total=%d", total_upserted)
            except Exception as e:
                logger.error("[Index] Upsert failed | error=%s", e)

            ids, documents, metadatas = [], [], []

    # Final batch (FIXED: now protected)
    if ids:
        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_upserted += len(ids)
            logger.info("[IndexStats] total_chunks=%d", total_upserted)
        except Exception as e:
            logger.error("[Index] Final upsert failed | error=%s", e)

    logger.info("[IndexStats] collection_count=%d", collection.count())
    logger.info("[Index] Done")

    if total_upserted == 0:
        raise RuntimeError("No data indexed — check chunking output.")
# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    logger.info("Indexing started.")
    main()
    logger.info("Indexing finished.")