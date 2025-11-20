import os
from pathlib import Path
import logging
from typing import Dict
import fitz  # PyMuPDF

# Logging setup
LOG_FILE = Path("logs/data_logs/ingestion.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class PDFLoader:
    """
    Load all PDFs from a directory and extract plain text.
    Returns a dict with filename -> text.
    """

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory does not exist: {self.pdf_dir}")

    def load_pdfs(self) -> Dict[str, str]:
        pdf_texts = {}
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            try:
                text = self._extract_text(pdf_file)
                pdf_texts[pdf_file.stem] = text
                logging.info(f"Loaded PDF: {pdf_file.name}")
            except Exception as e:
                logging.error(f"Failed to load {pdf_file.name}: {e}")
        return pdf_texts

    def _extract_text(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            text = page.get_text()
            if text:
                full_text.append(text)
        doc.close()
        return "\n".join(full_text)
