# src/trustworthy_maternal_postpartum_rag/ingestion/preprocessing.py

import logging
import time
import fitz  # PyMuPDF
import re
import uuid
from pathlib import Path
import unicodedata
import json
from collections import Counter
import hashlib
import os

from trustworthy_maternal_postpartum_rag.utils.config import get_config
from trustworthy_maternal_postpartum_rag.ingestion.document_registry import build_doc_metadata


CFG = get_config("configs/pipeline_config.yaml")

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

LOG_FILE = Path("logs/data_logs/preprocessing.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.preprocessing")
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
# Manual Per-PDF Skip Logic
# ============================================================

MANUAL_SKIP_MAP = {
    "who_antenatal care.pdf": {
        *range(0, 0),
        *range(9, 11),
    },
    "who_pcpnc_third_edition.pdf": {
        *range(1, 31),
        *range(206, 244),
    },
    "who_postnatal_positive_experience.pdf": {
        *range(1, 21),
        *range(170, 186),
    },
    "baby 411 clear answers and smart advice for your babys first year (brown, ari, fields, denise).pdf": {
        *range(1, 39),
        *range(946, 1102),
    },
}

MANUAL_SKIP_MAP = {k.lower(): v for k, v in MANUAL_SKIP_MAP.items()}

# ============================================================
# Text Cleaning Helpers
# ============================================================
def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\xa0": " ",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "•": "-",
        "–": "-",
        "—": "-",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    cleaned_chars = []
    for c in text:
        if c in {"\n", "\t"}:
            cleaned_chars.append(c)
        elif unicodedata.category(c)[0] != "C":
            cleaned_chars.append(c)

    return "".join(cleaned_chars)

def dehyphenate(text: str) -> str:
    return re.sub(r"(\w+)-\n([a-z])", r"\1\2", text)

def clean_headers_footers(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()

        if re.fullmatch(r"[A-Z ,\-]{15,}", s):
            continue
        if re.fullmatch(r"[A-Z]\d{1,3}", s):
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.match(r"^Page \d+ of \d+$", s, re.I):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def detect_repeated_headers_footers(pages_raw_text):
    line_counter = Counter()

    for text in pages_raw_text:
        for line in text.splitlines():
            c = line.strip()
            if c:
                line_counter[c] += 1

    threshold = max(5, len(pages_raw_text) * 0.2)
    return {line for line, count in line_counter.items() if count > threshold}

def remove_noise_lines(text: str, repeated_noise=None) -> str:
    lines = text.splitlines()
    out = []

    for line in lines:
        s = line.strip()

        if not s:
            continue
        if re.match(r"^[-=*_.]{3,}$", s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if repeated_noise and s in repeated_noise:
            continue

        out.append(s)

    return "\n".join(out)


def merge_paragraph_lines(text: str) -> str:
    lines = text.splitlines()
    merged = []
    buffer = []

    for line in lines:
        s = line.strip()

        if not s:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            continue

        if re.match(
            r"^(chapter|recommendation|recommendations|remarks|background|summary|introduction)\b",
            s.lower(),
        ):
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append(s)
            continue

        buffer.append(s)

    if buffer:
        merged.append(" ".join(buffer))

    return "\n\n".join(merged)


def should_remove_page(text: str) -> bool:
    lower = text.lower()

    if not text.strip():
        return True

    if any(term in lower for term in [
        "circumcision",
        "foreskin",
        "penis",
        "hiv acquisition",
        "sexually transmitted infections in men",
        "balanitis",
        "paraphimosis",
        "hpv infection in men",
    ]):
        return True

    if "table of contents" in lower or lower.startswith("contents"):
        if len(re.findall(r"\s\d{1,4}$", text, re.M)) >= 5:
            return True

    if lower.startswith("references"):
        return True

    if re.search(r"\b(appendix|annex|annexure)\b", lower):
        return True

    num_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    def is_table_like(page_text: str) -> bool:
        return page_text.count(":") > 10 or page_text.count("- ") > 10

    threshold = CFG.get("preprocessing", {}).get("remove_numeric_threshold", 0.35)

    if num_ratio > threshold and not is_table_like(text) and len(text) < 500:
        return True

    boilerplate_hit = re.search(
        r"(isbn|©|copyright|all rights reserved|produced by|available from:|tel:|fax:|email:|contact us|privacy statement|terms and conditions)",
        lower,
    )

    if boilerplate_hit:
        alpha_chars = sum(ch.isalpha() for ch in text)
        total_chars = max(len(text), 1)
        alpha_ratio = alpha_chars / total_chars

        if len(text) < 700 or alpha_ratio < 0.35:
            return True

    return False


def manual_page_skip(pdf_name: str, page_num: int) -> bool:
    pdf_name = pdf_name.lower()

    if pdf_name in MANUAL_SKIP_MAP:
        return page_num in MANUAL_SKIP_MAP[pdf_name]

    return False


# ============================================================
# Page Record Helper
# ============================================================

def build_page_record(
    text: str,
    doc_metadata: dict,
    page_number: int,
    skipped: bool,
    skip_reason: str | None,
):
    page_id = str(uuid.uuid4())

    metadata = {
        **doc_metadata,
        "page_id": page_id,
        "page_number": int(page_number),
        "language": "en",
        "preprocessing_version": CFG.get("run", {}).get("version", "v1"),
        "run_id": RUN_ID,
    }

    return {
        "page_id": page_id,
        "text": text,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "metadata": metadata,
    }


# ============================================================
# Deduplication Helpers
# ============================================================

def _normalize_for_dedup(text: str) -> str:
    if not text:
        return ""

    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()

    return t[:2000]


def _page_fingerprint(text: str) -> str:
    norm = _normalize_for_dedup(text)

    return hashlib.sha256(
        norm.encode("utf-8", errors="ignore")
    ).hexdigest()


# ============================================================
# Main PDF Preprocessing
# ============================================================

def preprocess_pdf_to_pages(pdf_path: Path, doc_id=None):
    pdf_name = pdf_path.name.lower()

    doc_id, doc_metadata = build_doc_metadata(pdf_path, doc_id=doc_id)

    t0 = time.time()

    logger.info(
        "[Preprocess] Start | file=%s | doc_id=%s | publisher=%s | source_tier=%s | priority=%s",
        pdf_path.name,
        doc_id,
        doc_metadata.get("publisher"),
        doc_metadata.get("source_tier"),
        doc_metadata.get("priority_score"),
    )

    with fitz.open(pdf_path) as doc:
        raw_pages = [page.get_text("text") for page in doc]

    normalized_raw_pages = [normalize_unicode(raw) for raw in raw_pages]
    repeated_noise = detect_repeated_headers_footers(normalized_raw_pages)

    preprocessed_pages = []
    seen_page_fps = set()

    manual_skipped = 0
    auto_filtered = 0
    deduped = 0
    kept = 0

    for i, raw in enumerate(normalized_raw_pages, start=1):
        if manual_page_skip(pdf_name, i):
            manual_skipped += 1

            preprocessed_pages.append(
                build_page_record(
                    text="",
                    doc_metadata=doc_metadata,
                    page_number=i,
                    skipped=True,
                    skip_reason="manual_skip",
                )
            )
            continue

        text = clean_headers_footers(raw)
        text = dehyphenate(text)
        text = remove_noise_lines(text, repeated_noise)
        text = merge_paragraph_lines(text)

        if CFG.get("chunking", {}).get("enable_chunk_dedup", True):
            fp = _page_fingerprint(text)

            if fp in seen_page_fps:
                deduped += 1

                preprocessed_pages.append(
                    build_page_record(
                        text="",
                        doc_metadata=doc_metadata,
                        page_number=i,
                        skipped=True,
                        skip_reason="deduped_duplicate_page",
                    )
                )
                continue

            seen_page_fps.add(fp)

        skipped = False
        skip_reason = None
        output_text = text

        if CFG.get("preprocessing", {}).get("enable_auto_filter", True) and should_remove_page(text):
            skipped = True
            skip_reason = "auto_filter"
            output_text = ""
            auto_filtered += 1
        else:
            kept += 1

        preprocessed_pages.append(
            build_page_record(
                text=output_text,
                doc_metadata=doc_metadata,
                page_number=i,
                skipped=skipped,
                skip_reason=skip_reason,
            )
        )

    dt = time.time() - t0

    logger.info(
        "[Preprocess] Done | file=%s pages=%d kept=%d manual=%d auto=%d deduped=%d seconds=%.2f",
        pdf_path.name,
        len(raw_pages),
        kept,
        manual_skipped,
        auto_filtered,
        deduped,
        dt,
    )

    logger.info(
        "[PreprocessStats] file=%s kept=%d skipped=%d deduped=%d ratio_kept=%.2f",
        pdf_path.name,
        kept,
        manual_skipped + auto_filtered + deduped,
        deduped,
        kept / max(len(raw_pages), 1),
    )

    return preprocessed_pages


# ============================================================
# Batch CLI
# ============================================================

if __name__ == "__main__":
    raw_root = Path("data/raw")
    pdf_paths = sorted(raw_root.rglob("*.pdf"))

    logger.info(
        "[Preprocess] Batch start | root=%s pdfs=%d",
        raw_root,
        len(pdf_paths),
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("*_preprocessed.jsonl"):
        old.unlink()

    for pdf in pdf_paths:
        logger.info(
            "[Preprocess] Running | rel=%s",
            pdf.relative_to(raw_root),
        )

        pages = preprocess_pdf_to_pages(pdf)

        rel_stem = "_".join(pdf.relative_to(raw_root).with_suffix("").parts)
        out_file = out_dir / f"{rel_stem}_preprocessed.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")

        kept = sum(1 for p in pages if not p.get("skipped"))
        skipped = sum(1 for p in pages if p.get("skipped"))
        deduped = sum(
            1 for p in pages
            if p.get("skip_reason") == "deduped_duplicate_page"
        )

        logger.info(
            "[Preprocess] Saved | out=%s pages=%d kept=%d skipped=%d deduped=%d",
            out_file.name,
            len(pages),
            kept,
            skipped,
            deduped,
        )

    logger.info("[Preprocess] Batch completed")
    print("Preprocessing completed.")