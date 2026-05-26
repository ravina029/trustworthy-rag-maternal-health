# scripts/analyze_chunks.py

from pathlib import Path
import json
from collections import Counter, defaultdict
import statistics
import csv

# =========================
# Input / output paths
# =========================

CHUNKS_DIR = Path("data/chunks")
OUTPUT_DIR = Path("results/chunk_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_MD = OUTPUT_DIR / "chunk_analysis_report.md"
SUMMARY_CSV = OUTPUT_DIR / "chunk_summary_by_file.csv"

REQUIRED_METADATA_FIELDS = [
    "doc_id",
    "source_file",
    "page_number",
    "publisher",
    "source_tier",
    "document_style",
    "lifecycle_stage",
    "topic_scope",
    "country_scope",
    "priority_score",
]


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                yield line_no, {"__json_error__": str(e)}


def get_metadata(chunk: dict) -> dict:
    """
    Supports the new schema:
        {"chunk_id": ..., "text": ..., "metadata": {...}}

    Also tolerates old flat chunk schema for diagnosis.
    """
    if isinstance(chunk.get("metadata"), dict):
        return chunk["metadata"]

    return {
        k: chunk.get(k)
        for k in REQUIRED_METADATA_FIELDS
        if k in chunk
    }


def word_count(text: str) -> int:
    return len((text or "").split())


def main():
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.jsonl"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {CHUNKS_DIR}")

    total_chunks = 0
    total_json_errors = 0

    chunk_ids = []
    word_counts = []
    quality_scores = []

    publisher_counts = Counter()
    source_tier_counts = Counter()
    lifecycle_counts = Counter()
    topic_counts = Counter()
    doc_counts = Counter()

    missing_metadata_fields = Counter()
    old_schema_count = 0
    empty_text_count = 0
    duplicate_text_counter = Counter()

    file_rows = []

    examples_missing_metadata = []
    examples_very_short = []
    examples_very_long = []

    for path in chunk_files:
        file_chunks = 0
        file_words = []
        file_publishers = Counter()
        file_tiers = Counter()
        file_missing = Counter()

        for line_no, chunk in read_jsonl(path):
            if "__json_error__" in chunk:
                total_json_errors += 1
                continue

            total_chunks += 1
            file_chunks += 1

            text = chunk.get("text", "")
            metadata = get_metadata(chunk)

            if "metadata" not in chunk:
                old_schema_count += 1

            if not text.strip():
                empty_text_count += 1

            wc = word_count(text)
            word_counts.append(wc)
            file_words.append(wc)

            chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
            if chunk_id:
                chunk_ids.append(chunk_id)

            normalized_text = " ".join(text.lower().split())[:500]
            if normalized_text:
                duplicate_text_counter[normalized_text] += 1

            publisher = metadata.get("publisher", "missing")
            source_tier = metadata.get("source_tier", "missing")
            lifecycle = metadata.get("lifecycle_stage", "missing")
            topic = metadata.get("topic_scope", "missing")
            doc_id = metadata.get("doc_id", "missing")

            publisher_counts[publisher] += 1
            source_tier_counts[source_tier] += 1
            lifecycle_counts[lifecycle] += 1
            topic_counts[topic] += 1
            doc_counts[doc_id] += 1

            file_publishers[publisher] += 1
            file_tiers[source_tier] += 1

            for field in REQUIRED_METADATA_FIELDS:
                if metadata.get(field) in [None, "", "missing"]:
                    missing_metadata_fields[field] += 1
                    file_missing[field] += 1

                    if len(examples_missing_metadata) < 10:
                        examples_missing_metadata.append({
                            "file": path.name,
                            "line": line_no,
                            "missing_field": field,
                            "chunk_id": chunk_id,
                            "text_preview": text[:180].replace("\n", " "),
                        })

            quality = metadata.get("quality_score")
            if isinstance(quality, (int, float)):
                quality_scores.append(float(quality))

            if wc < 20 and len(examples_very_short) < 10:
                examples_very_short.append({
                    "file": path.name,
                    "line": line_no,
                    "word_count": wc,
                    "text_preview": text[:180].replace("\n", " "),
                })

            if wc > 350 and len(examples_very_long) < 10:
                examples_very_long.append({
                    "file": path.name,
                    "line": line_no,
                    "word_count": wc,
                    "text_preview": text[:180].replace("\n", " "),
                })

        file_rows.append({
            "file": path.name,
            "chunks": file_chunks,
            "avg_words": round(statistics.mean(file_words), 2) if file_words else 0,
            "min_words": min(file_words) if file_words else 0,
            "max_words": max(file_words) if file_words else 0,
            "publishers": dict(file_publishers),
            "source_tiers": dict(file_tiers),
            "missing_metadata_fields": dict(file_missing),
        })

    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))
    duplicate_texts = sum(1 for _, count in duplicate_text_counter.items() if count > 1)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "chunks",
                "avg_words",
                "min_words",
                "max_words",
                "publishers",
                "source_tiers",
                "missing_metadata_fields",
            ],
        )
        writer.writeheader()
        for row in file_rows:
            writer.writerow(row)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Chunk Analysis Report\n\n")

        f.write("## Overall summary\n\n")
        f.write(f"- Chunk files analyzed: {len(chunk_files)}\n")
        f.write(f"- Total chunks: {total_chunks}\n")
        f.write(f"- JSON errors: {total_json_errors}\n")
        f.write(f"- Empty text chunks: {empty_text_count}\n")
        f.write(f"- Old flat-schema chunks without `metadata`: {old_schema_count}\n")
        f.write(f"- Duplicate chunk IDs: {duplicate_chunk_ids}\n")
        f.write(f"- Near-duplicate text prefixes: {duplicate_texts}\n")

        if word_counts:
            f.write(f"- Average words/chunk: {statistics.mean(word_counts):.2f}\n")
            f.write(f"- Median words/chunk: {statistics.median(word_counts):.2f}\n")
            f.write(f"- Min words/chunk: {min(word_counts)}\n")
            f.write(f"- Max words/chunk: {max(word_counts)}\n")

        if quality_scores:
            f.write(f"- Average quality score: {statistics.mean(quality_scores):.3f}\n")

        f.write("\n## Source-tier distribution\n\n")
        for k, v in source_tier_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Publisher distribution\n\n")
        for k, v in publisher_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Lifecycle-stage distribution\n\n")
        for k, v in lifecycle_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Topic-scope distribution\n\n")
        for k, v in topic_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Document distribution\n\n")
        for k, v in doc_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Missing metadata fields\n\n")
        if missing_metadata_fields:
            for k, v in missing_metadata_fields.most_common():
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- No required metadata fields missing.\n")

        f.write("\n## Examples with missing metadata\n\n")
        if examples_missing_metadata:
            for ex in examples_missing_metadata:
                f.write(
                    f"- File: `{ex['file']}`, line {ex['line']}, "
                    f"missing `{ex['missing_field']}`, chunk_id={ex['chunk_id']}\n"
                    f"  - Preview: {ex['text_preview']}\n"
                )
        else:
            f.write("- None.\n")

        f.write("\n## Very short chunk examples\n\n")
        if examples_very_short:
            for ex in examples_very_short:
                f.write(
                    f"- File: `{ex['file']}`, line {ex['line']}, "
                    f"{ex['word_count']} words\n"
                    f"  - Preview: {ex['text_preview']}\n"
                )
        else:
            f.write("- None.\n")

        f.write("\n## Very long chunk examples\n\n")
        if examples_very_long:
            for ex in examples_very_long:
                f.write(
                    f"- File: `{ex['file']}`, line {ex['line']}, "
                    f"{ex['word_count']} words\n"
                    f"  - Preview: {ex['text_preview']}\n"
                )
        else:
            f.write("- None.\n")

    print(f"Chunk analysis completed.")
    print(f"Report: {REPORT_MD}")
    print(f"CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()