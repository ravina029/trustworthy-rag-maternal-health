# src/trustworthy_maternal_postpartum_rag/ingestion/chunking.py

import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import numpy as np

from trustworthy_maternal_postpartum_rag.utils.config import get_config
from trustworthy_maternal_postpartum_rag.ingestion.chunk_utils import (
    MAX_WORDS,
    OVERLAP_WORDS,
    est_words,
    infer_lifecycle,
    detect_medical_type,
    estimate_chunk_quality,
    split_on_headings,
    semantic_split,
    split_block_by_length,
    is_table_page,
    split_table_rows,
    is_emergency_card,
    split_emergency_card,
    chunk_fingerprint,
    stable_chunk_id,
)


CFG = get_config("configs/pipeline_config.yaml")

random.seed(CFG.get("run", {}).get("seed", 42))
np.random.seed(CFG.get("run", {}).get("seed", 42))


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

LOG_FILE = Path("logs/data_logs/chunking.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.chunking")
LOG_LEVEL = getattr(logging, CFG.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
logger.setLevel(LOG_LEVEL)

fmt = "%(asctime)s - %(levelname)s - run_id=%(run_id)s - %(message)s"
formatter = logging.Formatter(fmt)

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE)
    for h in logger.handlers
):
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    fh.addFilter(RunIdFilter())
    logger.addHandler(fh)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    sh.addFilter(RunIdFilter())
    logger.addHandler(sh)

for handler in logger.handlers:
    if not any(isinstance(f, RunIdFilter) for f in handler.filters):
        handler.addFilter(RunIdFilter())


# ============================================================
# Path-driven config
# ============================================================

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
PATTERN = "*_preprocessed.jsonl"

ENABLE_CHUNK_DEDUP = CFG.get("chunking", {}).get("enable_chunk_dedup", True)
DEDUP_MODE = CFG.get("chunking", {}).get("dedup_mode", "page")


# ============================================================
# Chunk record builder
# ============================================================

def make_chunk(
    text: str,
    page_metadata: Dict[str, Any],
    topic_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:

    text = (text or "").strip()

    if not text:
        return None

    doc_id = page_metadata.get("doc_id")
    page_number = page_metadata.get("page_number")
    source_file = page_metadata.get("source_file")

    if not doc_id:
        logger.warning("[Chunking] Missing doc_id in metadata | source_file=%s", source_file)
        doc_id = "unknown_doc"

    if page_number is None:
        logger.warning("[Chunking] Missing page_number in metadata | source_file=%s", source_file)
        page_number = -1

    chunk_id = stable_chunk_id(doc_id, page_number, text)

    chunk_metadata = {
        **page_metadata,

        "chunk_id": chunk_id,
        "chunk_run_id": RUN_ID,
        "topic_hint": topic_hint,
        "inferred_lifecycle": infer_lifecycle(text),
        "medical_type": detect_medical_type(text),
        "quality_score": estimate_chunk_quality(text),
        "chunk_word_count": est_words(text),
        "chunk_version": CFG.get("run", {}).get("version", "v1"),
    }

    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": chunk_metadata,
    }


# ============================================================
# Page chunking
# ============================================================

def chunk_page(
    record: Dict[str, Any],
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> List[Dict[str, Any]]:

    text = (record.get("text") or "").strip()

    if record.get("skipped") or not text or len(text.split()) < 5:
        return []

    page_metadata = record.get("metadata", {}) or {}

    source_file = page_metadata.get("source_file", "unknown_source")
    page_number = page_metadata.get("page_number", "unknown_page")

    if not page_metadata:
        logger.warning(
            "[Chunking] Missing canonical metadata | source_file=%s page=%s",
            source_file,
            page_number,
        )

    chunks: List[Dict[str, Any]] = []

    # Emergency-card splitting
    if is_emergency_card(text):
        for body in split_emergency_card(text):
            for block in split_block_by_length(body, max_words, overlap_words):
                chunk = make_chunk(
                    text=block,
                    page_metadata=page_metadata,
                    topic_hint="emergency",
                )

                if chunk is not None:
                    chunks.append(chunk)

        return chunks

    # Table-aware splitting
    if is_table_page(text):
        rows = split_table_rows(text)

        if rows:
            for row in rows:
                for block in split_block_by_length(row, max_words, overlap_words):
                    chunk = make_chunk(
                        text=block,
                        page_metadata=page_metadata,
                        topic_hint="table",
                    )

                    if chunk is not None:
                        chunks.append(chunk)

            if chunks:
                return chunks

    # Normal heading + semantic + length splitting
    blocks = split_on_headings(text)

    for block in blocks:
        if est_words(block) > 250 and len(re.split(r"[.!?]", block)) > 5:
            sub_blocks = semantic_split(block)
        else:
            sub_blocks = [block]

        for sub_block in sub_blocks:
            for body in split_block_by_length(sub_block, max_words, overlap_words):
                chunk = make_chunk(
                    text=body,
                    page_metadata=page_metadata,
                    topic_hint=None,
                )

                if chunk is not None:
                    chunks.append(chunk)

    # Fallback
    if not chunks:
        logger.warning(
            "[Chunking] Fallback triggered | file=%s page=%s text_len=%d",
            source_file,
            page_number,
            len(text),
        )

        for body in split_block_by_length(text, max_words, overlap_words):
            chunk = make_chunk(
                text=body,
                page_metadata=page_metadata,
                topic_hint="forced_fallback",
            )

            if chunk is not None:
                chunks.append(chunk)

    return chunks


# ============================================================
# Batch chunking
# ============================================================

def chunk_preprocessed_files(
    processed_dir: Path = PROCESSED_DIR,
    pattern: str = PATTERN,
    chunks_dir: Path = CHUNKS_DIR,
) -> None:

    chunks_dir.mkdir(parents=True, exist_ok=True)

    pre_files = sorted(
        processed_dir.glob(pattern),
        key=lambda path: path.name.lower(),
    )

    logger.info(
        "[Chunking] Batch start | processed_dir=%s files=%d dedup_enabled=%s dedup_mode=%s",
        processed_dir,
        len(pre_files),
        ENABLE_CHUNK_DEDUP,
        DEDUP_MODE,
    )

    total_pages_in = 0
    total_pages_skipped = 0
    total_chunks_out = 0
    total_chunks_deduped = 0
    total_words_batch = 0

    for pre_file in pre_files:
        out_file = chunks_dir / pre_file.name.replace("_preprocessed", "_chunks")

        logger.info(
            "[Chunking] File start | in=%s out=%s",
            pre_file.name,
            out_file.name,
        )

        seen_global = set()
        seen_by_page: Dict[Any, set] = {}

        pages_in = 0
        pages_skipped = 0
        chunks_out = 0
        chunks_deduped = 0
        file_words = 0

        with open(pre_file, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                pages_in += 1

                page_text = (record.get("text") or "").strip()

                if record.get("skipped") or not page_text:
                    pages_skipped += 1
                    continue

                page_chunks = chunk_page(record)

                if not page_chunks:
                    page_metadata = record.get("metadata", {}) or {}

                    logger.warning(
                        "[Chunking] Empty chunk_page output | file=%s page_number=%s",
                        pre_file.name,
                        page_metadata.get("page_number"),
                    )
                    continue

                for chunk in page_chunks:
                    chunk_text = chunk.get("text", "")
                    chunk_metadata = chunk.get("metadata", {}) or {}

                    if ENABLE_CHUNK_DEDUP and DEDUP_MODE != "off":
                        fp = chunk_fingerprint(chunk_text)

                        if DEDUP_MODE == "page":
                            page_number = chunk_metadata.get("page_number", -1)
                            bucket = seen_by_page.setdefault(page_number, set())

                            if fp in bucket:
                                chunks_deduped += 1
                                continue

                            bucket.add(fp)

                        elif DEDUP_MODE == "global":
                            if fp in seen_global:
                                chunks_deduped += 1
                                continue

                            seen_global.add(fp)

                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                    chunks_out += 1
                    words = est_words(chunk_text)
                    file_words += words
                    total_words_batch += words

        avg_chunk_len_file = file_words / max(chunks_out, 1)

        logger.info(
            "[ChunkStats] file=%s chunks=%d avg_len=%.2f deduped=%d",
            pre_file.name,
            chunks_out,
            avg_chunk_len_file,
            chunks_deduped,
        )

        logger.info(
            "[Chunking] File done | file=%s pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d dedup_enabled=%s dedup_mode=%s",
            pre_file.name,
            pages_in,
            pages_skipped,
            chunks_out,
            chunks_deduped,
            ENABLE_CHUNK_DEDUP,
            DEDUP_MODE,
        )

        total_pages_in += pages_in
        total_pages_skipped += pages_skipped
        total_chunks_out += chunks_out
        total_chunks_deduped += chunks_deduped

    avg_chunk_len_batch = total_words_batch / max(total_chunks_out, 1)

    logger.info(
        "[Chunking] Batch done | pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d avg_chunk_len=%.2f",
        total_pages_in,
        total_pages_skipped,
        total_chunks_out,
        total_chunks_deduped,
        avg_chunk_len_batch,
    )


if __name__ == "__main__":
    chunk_preprocessed_files()