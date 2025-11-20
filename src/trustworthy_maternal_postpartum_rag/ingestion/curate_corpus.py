import os
import json
import logging
from pathlib import Path
import fitz  # PyMuPDF
from section_chunker import section_chunk_text  # your enhanced section chunker


# Paths
# -------------------------------
# ROOT = project root (one level above src/)
ROOT = Path(__file__).resolve().parents[3]

RAW_PDF_DIR = ROOT / "data" / "raw" / "pdfs"
PROCESSED_JSONL = ROOT / "data" / "processed" / "chunks.jsonl"

# logs directory is at project root, not inside src
LOG_FILE = ROOT / "logs" / "data_logs" / "ingestion.log"


# -------------------------------
# Logging setup
# -------------------------------
os.makedirs(LOG_FILE.parent, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting data curation pipeline...")

# -------------------------------
# Process PDFs
# -------------------------------
all_chunks = []
pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
logging.info(f"Found {len(pdf_files)} PDFs in {RAW_PDF_DIR}")

for pdf_path in pdf_files:
    logging.info(f"Loading PDF: {pdf_path.name}")
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF {pdf_path.name}: {e}")
        continue

    pdf_text = ""
    for page_num, page in enumerate(pdf):
        try:
            pdf_text += page.get_text()
        except Exception as e:
            logging.warning(f"Page {page_num} extraction failed in {pdf_path.name}: {e}")
            continue

    if not pdf_text.strip():
        logging.warning(f"No text extracted from {pdf_path.name}")
        continue

    # Section-based chunking
    chunks = section_chunk_text(pdf_text, pdf_path.stem)
    logging.info(f"Processed {pdf_path.stem} into {len(chunks)} chunks")
    all_chunks.extend(chunks)

# -------------------------------
# Save to JSONL
# -------------------------------
if all_chunks:
    os.makedirs(PROCESSED_JSONL.parent, exist_ok=True)
    with open(PROCESSED_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logging.info(f"Saved {len(all_chunks)} chunks to {PROCESSED_JSONL}")
else:
    logging.warning("No chunks generated!")

logging.info("Data curation pipeline completed.")
print("✅ Data curation pipeline completed. Check logs for details.")
