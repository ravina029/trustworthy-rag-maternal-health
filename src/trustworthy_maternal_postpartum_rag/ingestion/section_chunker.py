import re
import logging

# Use the central logger (configured in curate_corpus.py)
logger = logging.getLogger(__name__)


def section_chunk_text(text, source_name, chunk_size=500, overlap=50):
    """
    Splits the text into chunks based on section headings and approximate token size.

    Args:
        text (str): Full text from PDF.
        source_name (str): Name of the PDF (for logging).
        chunk_size (int): Number of words per chunk.
        overlap (int): Overlap between chunks.

    Returns:
        List[dict]: List of chunk dictionaries with 'source', 'section', 'text'.
    """

    # Regex to detect common section headings
    section_pattern = re.compile(
        r'^\s*(Chapter|Section|Part|Postpartum|Prenatal|Antenatal|Newborn|Care|Guidelines)[\s\S]{0,100}?$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(section_pattern.finditer(text))
    chunks = []

    if not matches:
        logger.warning(f"No section headings found in {source_name}, doing simple chunking")

        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size]).strip()

            if chunk_text:
                chunks.append({
                    "source": source_name,
                    "section": "full_text",
                    "text": chunk_text
                })
            else:
                logger.info(f"Skipped empty chunk in {source_name} at position {i}")

        return chunks

    # Section-based splitting
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        section_name = match.group().strip()

        if not section_text:
            logger.info(f"Skipped empty section '{section_name}' in {source_name}")
            continue

        words = section_text.split()

        for j in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[j:j + chunk_size]).strip()

            if chunk_text:
                chunks.append({
                    "source": source_name,
                    "section": section_name,
                    "text": chunk_text
                })
            else:
                logger.info(
                    f"Skipped empty chunk in section '{section_name}' of {source_name} at position {j}"
                )

    return chunks
