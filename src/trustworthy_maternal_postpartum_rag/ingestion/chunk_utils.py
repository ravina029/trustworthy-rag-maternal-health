# src/trustworthy_maternal_postpartum_rag/ingestion/chunk_utils.py

import os
import re
import hashlib
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from trustworthy_maternal_postpartum_rag.utils.config import get_config


CFG = get_config("configs/pipeline_config.yaml")

logger = logging.getLogger("tmprag.ingestion.chunk_utils")

_chunk_model = None


def get_chunk_model():
    global _chunk_model
    if _chunk_model is None:
        _chunk_model = SentenceTransformer(CFG["embedding"]["model"])
    return _chunk_model


# ============================================================
# Config values
# ============================================================

MAX_WORDS = CFG.get("chunking", {}).get("max_words", 220)
OVERLAP_WORDS = CFG.get("chunking", {}).get("overlap_words", 40)

DEDUP_FINGERPRINT_CHARS = int(os.getenv("TMPRAG_DEDUP_CHARS", "8000"))
STRIP_BOILERPLATE_FOR_DEDUP = (
    os.getenv("TMPRAG_STRIP_BOILERPLATE", "true").lower() == "true"
)

TABLE_BULLET_THRESHOLD = int(os.getenv("TMPRAG_TABLE_BULLET_THRESHOLD", "12"))
TABLE_COLONLINE_THRESHOLD = int(os.getenv("TMPRAG_TABLE_COLONLINE_THRESHOLD", "12"))


# ============================================================
# Basic helpers
# ============================================================

def est_words(text: str) -> int:
    return len((text or "").split())


def infer_lifecycle(text: str) -> str:
    t = (text or "").lower()

    if any(w in t for w in [
        "pregnant", "pregnancy", "antenatal", "trimester", "gestation"
    ]):
        return "pregnancy"

    if any(w in t for w in [
        "postpartum", "after delivery", "after childbirth",
        "lochia", "perineal", "c-section", "cesarean",
        "maternal recovery"
    ]):
        return "postpartum"

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


def detect_medical_type(text: str) -> str:
    t = (text or "").lower()

    if re.search(r"\b(treatment|administer|therapy)\b", t):
        return "treatment"

    if re.search(r"\b(symptom|sign)\b", t):
        return "symptom"

    if re.search(r"\b(risk|complication)\b", t):
        return "risk"

    if re.search(r"\b(avoid|contraindicated|warning)\b", t):
        return "warning"

    return "general"


def estimate_chunk_quality(text: str) -> float:
    words = (text or "").split()

    if not words:
        return 0.0

    length_score = min(len(words) / 200, 1.0)

    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)
    avg_sentence_len = len(words) / sentence_count
    structure_score = min(avg_sentence_len / 20, 1.0)

    noise_penalty = 0.0
    if re.search(r"[^\w\s\.,;:%()\-]", text):
        noise_penalty = 0.2

    score = 0.5 * length_score + 0.5 * structure_score - noise_penalty

    return round(max(score, 0.0), 3)


# ============================================================
# Splitting helpers
# ============================================================

def split_on_headings(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    blocks: List[str] = []
    current_lines: List[str] = []

    heading_pattern = re.compile(
        r"""^(
            (\d+(\.\d+)*\s+[A-Z][^\n]{5,})
            |([A-Z][A-Z0-9 ,\-]{8,}\b[A-Z])
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
            if len(line.split()) > 12 or (
                sum(c.isalpha() for c in line) / max(len(line), 1) < 0.6
            ):
                current_lines.append(line)
                continue

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
    if not CFG.get("chunking", {}).get("semantic_split", False):
        return [block_text]

    if not block_text or not block_text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", block_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    min_sentences = CFG.get("chunking", {}).get("min_sentences_for_semantic", 6)
    if len(sentences) < min_sentences:
        return [block_text]

    emb_model = get_chunk_model()

    try:
        embs = emb_model.encode(sentences, normalize_embeddings=True)
    except Exception as e:
        logger.error("Embedding failed; falling back to no semantic split: %s", e)
        return [block_text]

    threshold = CFG.get("chunking", {}).get("semantic_threshold", 0.55)

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
    overlap_words: int = OVERLAP_WORDS,
) -> List[str]:

    if not block_text or not block_text.strip():
        return []

    overlap_words = min(overlap_words, max_words // 2)

    raw_parts = re.split(r"(?<=[.!?])\s+|\n\n+", block_text)
    parts = [p.strip() for p in raw_parts if p.strip()]

    if not parts:
        return []

    chunks: List[str] = []
    current_words: List[str] = []

    for part in parts:
        words = part.split()

        if len(words) > max_words:
            start = 0

            while start < len(words):
                end = min(len(words), start + max_words)
                chunk_words = words[start:end]

                if chunk_words:
                    chunks.append(" ".join(chunk_words))

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
# Table and emergency heuristics
# ============================================================

def is_table_page(text: str) -> bool:
    if not text:
        return False

    low = text.lower()

    if re.search(r"\btable\s+\d+|\btable\s*:", low):
        return True

    bullet_like = sum(
        1 for line in text.splitlines()
        if line.strip().startswith(("•", "-", "*"))
    )

    if bullet_like >= TABLE_BULLET_THRESHOLD:
        return True

    colon_lines = sum(
        1 for line in text.splitlines()
        if ":" in line and len(line.strip()) < 80 and len(line.split()) < 15
    )

    if colon_lines >= TABLE_COLONLINE_THRESHOLD:
        return True

    return False


def split_table_rows(text: str) -> List[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) < 2:
        return []

    title = lines[0] if len(lines[0].split()) < 10 else ""

    rows = []

    for line in lines[1:]:
        parts = re.split(r"\s{2,}|•| - ", line)
        parts = [p.strip() for p in parts if p.strip()]

        for part in parts:
            if len(part.split()) >= 3:
                if title:
                    rows.append(f"{title}: {part}")
                else:
                    rows.append(part)

    return rows


def is_emergency_card(text: str) -> bool:
    text = text or ""

    return (
        "EMERGENCY TREATMENTS FOR THE WOMAN" in text
        or (text.count("If the woman") >= 2 and text.count("Give ") >= 2)
    )


def split_emergency_card(text: str) -> List[str]:
    if not text:
        return []

    markers = [
        "Give oxytocin",
        "Give magnesium sulphate",
        "Refer the woman urgently",
    ]

    out = text

    for marker in markers:
        out = re.sub(
            re.escape(marker),
            f"\n\n### {marker}\n",
            out,
            flags=re.IGNORECASE,
        )

    parts = re.split(r"\n{2,}###\s+", out)

    return [p.strip() for p in parts if p.strip()]


# ============================================================
# Deduplication helpers
# ============================================================

def strip_boilerplate_lines(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    kept: List[str] = []

    for line in lines:
        low = line.lower()

        if re.fullmatch(r"\d{1,4}", line):
            continue

        if any(pattern in low for pattern in [
            "return to table of contents",
            "table of contents",
            "cleveland clinic",
            "women's health - pregnancy",
            "your guide to a healthy pregnancy",
            "healthy pregnancy",
            "edition",
        ]):
            if len(line) <= 80:
                continue

        kept.append(line)

    return "\n".join(kept).strip() if kept else (text or "").strip()


def normalize_for_dedup(text: str) -> str:
    out = text or ""

    if STRIP_BOILERPLATE_FOR_DEDUP:
        stripped = strip_boilerplate_lines(out)
        if len(stripped.strip()) >= 50:
            out = stripped

    out = out.lower()
    out = re.sub(r"\s+", " ", out).strip()

    return out


def chunk_fingerprint(text: str) -> str:
    norm = normalize_for_dedup(text)

    n = DEDUP_FINGERPRINT_CHARS

    if len(norm) <= n:
        material = norm
    else:
        half = n // 2
        material = norm[:half] + " || " + norm[-half:]

    return hashlib.sha256(
        material.encode("utf-8", errors="ignore")
    ).hexdigest()


def stable_chunk_id(doc_id: str, page_number: int, text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip())
    raw = f"{doc_id}_{page_number}_{norm[:200]}"

    return hashlib.sha256(
        raw.encode("utf-8", errors="ignore")
    ).hexdigest()