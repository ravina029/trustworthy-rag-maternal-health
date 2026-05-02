# src/trustworthy_maternal_postpartum_rag/ingestion/chunking.py
import json
import re
import uuid
import hashlib
import logging
from pathlib import Path
import os
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import numpy as np
import random


from trustworthy_maternal_postpartum_rag.utils.config import get_config
CFG = get_config("configs/pipeline_config.yaml")

random.seed(CFG["run"]["seed"])
np.random.seed(CFG["run"]["seed"])

_chunk_model = None

def get_chunk_model():
    global _chunk_model
    if _chunk_model is None:
        _chunk_model = SentenceTransformer(CFG["embedding"]["model"])
    return _chunk_model

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

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
PATTERN = "*_preprocessed.jsonl"

MAX_WORDS = CFG["chunking"]["max_words"]
OVERLAP_WORDS = CFG["chunking"]["overlap_words"]

ENABLE_CHUNK_DEDUP = CFG["chunking"]["enable_chunk_dedup"]
DEDUP_MODE = CFG["chunking"]["dedup_mode"]

DEDUP_FINGERPRINT_CHARS = int(os.getenv("TMPRAG_DEDUP_CHARS", "8000"))
STRIP_BOILERPLATE_FOR_DEDUP = os.getenv("TMPRAG_STRIP_BOILERPLATE", "true").lower() == "true"

# Emergency/table heuristics can be tuned by env if needed
TABLE_BULLET_THRESHOLD = int(os.getenv("TMPRAG_TABLE_BULLET_THRESHOLD", "12"))
TABLE_COLONLINE_THRESHOLD = int(os.getenv("TMPRAG_TABLE_COLONLINE_THRESHOLD", "12"))

# ============================================================
# Token estimation
# ============================================================

def est_words(text: str) -> int:
    return len(text.split())

# ============================================================
# Lifecycle inference (text-side)
# ============================================================

def infer_lifecycle(text: str) -> str:
    t = (text or "").lower()

    # Priority-based classification (more specific → less specific)

    if any(w in t for w in [
        "pregnant", "pregnancy", "antenatal",
        "trimester", "gestation"
    ]):
        return "pregnancy"

    if any(w in t for w in [
        "postpartum", "after delivery", "after childbirth",
        "lochia", "perineal", "c-section", "cesarean",
        "maternal recovery"
    ]):
        return "postpartum"

    # Breastfeeding is a sub-phase → still useful to label separately
    if any(w in t for w in [
        "breastfeeding", "lactation", "milk supply",
        "nipple", "expressing milk"
    ]):
        return "breastfeeding"

    if any(w in t for w in [
        "newborn", "neonate", "first 28 days",
        "umbilical cord", "meconium"
    ]):
        return "newborn"

    if any(w in t for w in [
        "infant", "weaning", "complementary feeding"
    ]):
        return "infant"

    if any(w in t for w in [
        "toddler", "1 year", "2 years"
    ]):
        return "toddler"

    return "general"

# ============================================================
# Splitting helpers
# ============================================================

def split_on_headings(text: str) -> List[str]:
    """
    Split text using heading heuristics while preserving context.
    Falls back to full text if nothing valid is detected.
    """
    if not text or not text.strip():
        return []

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    blocks: List[str] = []
    current_lines: List[str] = []

    heading_pattern = re.compile(
    r"""^(
        (\d+(\.\d+)*\s+[A-Z][^\n]{5,})        # numbered headings
        |([A-Z][A-Z0-9 ,\-]{8,}\b[A-Z])       # ALLCAPS but must end properly
    )$""",
    re.X,
    )

    boilerplate_exact = {
        "RETURN TO TABLE OF CONTENTS",
        "TABLE OF CONTENTS",
    }

    for line in lines:
        up = line.upper()

        if up in boilerplate_exact:
            continue

        if re.match(r"^(CLINICAL PRACTICE GUIDELINES|NATIONAL INTEGRATED MATERNAL)", up):
            continue

        if heading_pattern.match(line):
            # ❌ Reject noisy OCR headings
            if len(line.split()) > 12 or (
                sum(c.isalpha() for c in line) / max(len(line), 1) < 0.6
            ):
                current_lines.append(line)
                continue

            # ✅ REAL SPLIT happens here
            if current_lines:
                block = "\n".join(current_lines).strip()
                if len(block.split()) >= 5:
                    blocks.append(block)
                current_lines = []

            current_lines.append(line)

        else:
            current_lines.append(line)

    if current_lines:
        block = "\n".join(current_lines).strip()
        if len(block.split()) >= 5:
            blocks.append(block)

    return blocks if blocks else [text.strip()]
def semantic_split(block_text: str) -> List[str]:
    if not CFG["chunking"]["semantic_split"]:
        return [block_text]

    if not block_text or not block_text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', block_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < CFG["chunking"]["min_sentences_for_semantic"]:
        return [block_text]

    emb_model = get_chunk_model()
    try:
        embs = emb_model.encode(sentences, normalize_embeddings=True)
    except Exception as e:
        logger.error("Embedding failed, fallback to no semantic split: %s", e)
        return [block_text]

    threshold = CFG["chunking"]["semantic_threshold"]

    chunks: List[str] = []
    current: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i], embs[i - 1]))
        sim = max(min(sim, 1.0), -1.0)

        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks
def split_block_by_length(
    block_text: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS
) -> List[str]:

    if not block_text or not block_text.strip():
        return []

    overlap_words = min(overlap_words, max_words // 2)  # safety clamp

    raw_parts = re.split(r"(?<=[\.\!\?])\s+|\n\n+", block_text)
    parts = [p.strip() for p in raw_parts if p.strip()]

    if not parts:
        return []

    chunks: List[str] = []
    current_words: List[str] = []

    for part in parts:
        words = part.split()

        # Force split very large parts
        if len(words) > max_words * 2:
            start = 0
            while start < len(words):
                end = min(len(words), start + max_words)
                chunk = words[start:end]

                if chunk:
                    chunks.append(" ".join(chunk))

                if overlap_words > 0:
                    start = max(start + 1, end - overlap_words)
                else:
                    start = end
            continue

        if len(current_words) + len(words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))

            current_words = current_words[-overlap_words:] if overlap_words > 0 else []

        current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

# ============================================================
# Table/emergency heuristics
# ============================================================

def is_table_page(text: str) -> bool:
    if not text:
        return False

    low = text.lower()

    # Better table detection
    if re.search(r"\btable\s+\d+|\btable\s*:", low):
        return True

    bullet_like = sum(
        1 for l in text.splitlines()
        if l.strip().startswith(("•", "-", "*"))
    )
    if bullet_like >= TABLE_BULLET_THRESHOLD:
        return True

    colon_lines = sum(
        1 for l in text.splitlines()
        if ":" in l and len(l.strip()) < 80 and len(l.split()) < 15
    )
    if colon_lines >= TABLE_COLONLINE_THRESHOLD:
        return True

    return False

def split_table_rows(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return []

    # Skip obvious headers
    title = lines[0] if len(lines[0].split()) < 10 else ""

    rows = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}|•| - ", line)
        parts = [p.strip() for p in parts if p.strip()]

        for p in parts:
            if len(p.split()) >= 3:
                if title:
                    rows.append(f"{title}: {p}")
                else:
                    rows.append(p)

    return rows


def is_emergency_card(text: str) -> bool:
    t = text or ""
    return (
        "EMERGENCY TREATMENTS FOR THE WOMAN" in t
        or (t.count("If the woman") >= 2 and t.count("Give ") >= 2)
    )


def split_emergency_card(text: str) -> List[str]:
    if not text:
        return []

    markers = [
        "Give oxytocin",
        "Give magnesium sulphate",
        "Refer the woman urgently",
    ]

    t = text  # DO NOT lowercase content

    for m in markers:
        t = re.sub(re.escape(m), f"\n\n### {m}\n", t, flags=re.IGNORECASE)

    parts = re.split(r"\n{2,}###\s+", t)

    return [p.strip() for p in parts if p.strip()]
# ============================================================
# Dedup helpers (safe defaults)
# ============================================================

def _strip_boilerplate_lines(text: str) -> str:
    """
    Conservative removal of common repeated header/footer patterns that dominate PDF extraction.
    Only removes lines that are likely boilerplate (short or numeric).
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    kept: List[str] = []

    for l in lines:
        low = l.lower()

        # Page numbers (standalone)
        if re.fullmatch(r"\d{1,4}", l):
            continue

        # Very common guide headers; remove only if short-ish
        if any(p in low for p in [
            "return to table of contents",
            "table of contents",
            "cleveland clinic",
            "women's health - pregnancy",
            "your guide to a healthy pregnancy",
            "healthy pregnancy",
            "edition",
        ]):
            if len(l) <= 80:
                continue

        kept.append(l)

    return "\n".join(kept).strip() if kept else (text or "").strip()


def _normalize_for_dedup(text: str) -> str:
    t = text or ""

    if STRIP_BOILERPLATE_FOR_DEDUP:
        t2 = _strip_boilerplate_lines(t)
        if len(t2.strip()) >= 50:
            t = t2

    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()

    return t

def _chunk_fingerprint(text: str) -> str:
    """
    Head+tail hashing prevents a shared intro boilerplate from collapsing everything.
    """
    norm = _normalize_for_dedup(text)

    n = DEDUP_FINGERPRINT_CHARS
    if len(norm) <= n:
        material = norm
    else:
        half = n // 2
        material = norm[:half] + " || " + norm[-half:]

    return hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()


# ============================================================
# Chunk builder
# ============================================================
def stable_chunk_id(doc_id, page_number, text):
    norm = re.sub(r"\s+", " ", text.strip())
    raw = f"{doc_id}_{page_number}_{norm[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def estimate_chunk_quality(text: str) -> float:
    words = text.split()
    length_score = min(len(words) / 200, 1.0)

    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)
    avg_sentence_len = len(words) / sentence_count

    structure_score = min(avg_sentence_len / 20, 1.0)

    noise_penalty = 0.0
    if re.search(r"[^\w\s\.,;:%()-]", text):
        noise_penalty = 0.2

    score = 0.5 * length_score + 0.5 * structure_score - noise_penalty
    return round(max(score, 0.0), 3)

def make_chunk(
    doc_id: str,
    source_file: str,
    page_number: int,
    text: str,
    doc_meta: Dict[str, Any],
    topic_hint: str = None
) -> Dict[str, Any]:

    text = (text or "").strip()
    if not text:
        return None

    metadata_stage = doc_meta.get("stage")  # trusted source
    inferred_lifecycle = infer_lifecycle(text)  # soft signal

    def detect_medical_type(t: str) -> str:
        t = t.lower()

        if re.search(r"\b(treatment|administer|therapy)\b", t):
            return "treatment"
        if re.search(r"\b(symptom|sign)\b", t):
            return "symptom"
        if re.search(r"\b(risk|complication)\b", t):
            return "risk"
        if re.search(r"\b(avoid|contraindicated|warning)\b", t):
            return "warning"

        return "general"

    return {
        "chunk_id": stable_chunk_id(doc_id, page_number, text),
        "doc_id": doc_id,
        "source_file": source_file,
        "page_number": page_number,
        "text": text,
        "language": "en",
        "version": CFG["run"]["version"],
        "country": doc_meta.get("country"),
        "target": doc_meta.get("target"),
        "source_type": doc_meta.get("source_type"),
        "publisher": doc_meta.get("publisher"),
        "topic_hint": topic_hint,
        "stage": metadata_stage,
        "inferred_lifecycle": inferred_lifecycle,   # heuristic from text
        "medical_type": detect_medical_type(text),
        "quality_score": estimate_chunk_quality(text),
    }
def chunk_page(
    record: Dict[str, Any],
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS
) -> List[Dict[str, Any]]:

    text = (record.get("text") or "").strip()

    if record.get("skipped") or not text or len(text.split()) < 5:
        return []
    doc_meta = record.get("doc_metadata", {}) or {}
    source_file = record.get("source_file")
    page_number = record.get("page_number")
    doc_id = record.get("doc_id")

    chunks: List[Dict[str, Any]] = []
    if not doc_meta:
        logger.warning("Missing doc_metadata | file=%s", source_file)
    # Emergency
    if is_emergency_card(text):
        for body in split_emergency_card(text):
            for b in split_block_by_length(body, max_words, overlap_words):
                if b:
                    ch = make_chunk(doc_id, source_file, page_number, b, doc_meta, "emergency")
                    if ch is not None:
                        chunks.append(ch)
        return chunks

    # Table
    if is_table_page(text):
        rows = split_table_rows(text)
        if rows:
            for row in rows:
                for b in split_block_by_length(row, max_words, overlap_words):
                    if b:
                        ch = make_chunk(doc_id, source_file, page_number, b, doc_meta, "table")
                        if ch is not None:
                            chunks.append(ch)
            if chunks:
                return chunks

    # Normal
    blocks = split_on_headings(text)

    for block in blocks:
    # ✅ Only apply semantic split for large blocks
        if est_words(block) > 250 and len(re.split(r"[.!?]", block)) > 5:
            sub_blocks = semantic_split(block)
        else:
            sub_blocks = [block]

        for sb in sub_blocks:
            for body in split_block_by_length(sb, max_words, overlap_words):
                if body:
                    ch = make_chunk(doc_id, source_file, page_number, body, doc_meta)
                    if ch is not None:
                        chunks.append(ch)

    # Fallback
    if not chunks:
        logger.warning(
            "[Chunking] Fallback triggered | file=%s page=%s text_len=%d",
            source_file, page_number, len(text)
        )

        for body in split_block_by_length(text, max_words, overlap_words):
            ch = make_chunk(doc_id, source_file, page_number, body, doc_meta, "forced_fallback")
            if ch is not None:
                chunks.append(ch)

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

    pre_files = sorted(processed_dir.glob(pattern), key=lambda p: p.name.lower())
    logger.info(
        "[Chunking] Batch start | processed_dir=%s files=%d dedup_enabled=%s dedup_mode=%s",
        processed_dir, len(pre_files), ENABLE_CHUNK_DEDUP, DEDUP_MODE
    )

    total_pages_in = 0
    total_pages_skipped = 0
    total_chunks_out = 0
    total_chunks_deduped = 0
    total_words_batch = 0


    for pre_file in pre_files:
        out_file = chunks_dir / pre_file.name.replace("_preprocessed", "_chunks")
        logger.info("[Chunking] File start | in=%s out=%s", pre_file.name, out_file.name)

        # Dedup state
        seen_global = set()
        seen_by_page: Dict[Any, set] = {}

        pages_in = 0
        pages_skipped = 0
        chunks_out = 0
        chunks_deduped = 0
        file_words = 0

        with open(pre_file, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                pages_in += 1

                page_text = (rec.get("text") or "").strip()
                if rec.get("skipped") or not page_text:
                    pages_skipped += 1
                    continue

                page_chunks = chunk_page(rec)
                if not page_chunks:
                    logger.warning(
                        "[Chunking] Empty chunk_page output | file=%s page_number=%s",
                        pre_file.name, rec.get("page_number")
                    )
                    continue

                for ch in page_chunks:
                    if ENABLE_CHUNK_DEDUP and DEDUP_MODE != "off":
                        fp = _chunk_fingerprint(ch.get("text", ""))

                        if DEDUP_MODE == "page":
                            pn = ch.get("page_number", -1)
                            bucket = seen_by_page.setdefault(pn, set())
                            if fp in bucket:
                                chunks_deduped += 1
                                continue
                            bucket.add(fp)

                        elif DEDUP_MODE == "global":
                            if fp in seen_global:
                                chunks_deduped += 1
                                continue
                            seen_global.add(fp)

                    fout.write(json.dumps(ch, ensure_ascii=False) + "\n")

                    chunks_out += 1
                    words = est_words(ch.get("text", ""))
                    file_words += words
                    total_words_batch += words

        # ✅ Correct file-level average
        avg_chunk_len_file = file_words / max(chunks_out, 1)

        logger.info(
            "[ChunkStats] file=%s chunks=%d avg_len=%.2f deduped=%d",
            pre_file.name,
            chunks_out,
            avg_chunk_len_file,
            chunks_deduped
        )

        logger.info(
            "[Chunking] File done | file=%s pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d dedup_enabled=%s dedup_mode=%s",
            pre_file.name, pages_in, pages_skipped, chunks_out, chunks_deduped, ENABLE_CHUNK_DEDUP, DEDUP_MODE
        )

        total_pages_in += pages_in
        total_pages_skipped += pages_skipped
        total_chunks_out += chunks_out
        total_chunks_deduped += chunks_deduped

    # ✅ Correct batch-level average
    avg_chunk_len_batch = total_words_batch / max(total_chunks_out, 1)

    logger.info(
        "[Chunking] Batch done | pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d avg_chunk_len=%.2f",
        total_pages_in,
        total_pages_skipped,
        total_chunks_out,
        total_chunks_deduped,
        avg_chunk_len_batch
    )
if __name__ == "__main__":
    chunk_preprocessed_files()

