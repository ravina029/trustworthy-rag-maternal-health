from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Path-driven configuration
# ============================================================

OUTPUT_PATH = Path("results/representative_examples.md")

RUNS = {
    "answerable": Path("eval_runs/eval_run_20260528_215408.jsonl"),
    "safety": Path("eval_runs/eval_run_20260528_230900.jsonl"),
    "insufficient": Path("eval_runs/eval_run_20260528_232800.jsonl"),
    "adversarial": Path("eval_runs/eval_run_20260528_233300.jsonl"),
}

EXAMPLES = {
    "answerable": [
        "soft cheese",
        "postpartum bleeding",
        "newborn is jaundiced",
    ],
    "safety": [
        "soaking a pad",
        "chest pain",
        "newborn has a fever",
    ],
    "insufficient": [
        "probiotic brand",
        "vitamin D drops",
        "hospital discharge policy",
    ],
    "adversarial": [
        "Ignore the evidence",
        "Pretend you are WHO",
        "Do not cite",
    ],
}


# ============================================================
# Helpers
# ============================================================

def coerce_text(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, list):
        return " ".join(coerce_text(v) for v in x)

    if isinstance(x, dict):
        return " ".join(coerce_text(v) for v in x.values())

    return str(x)


def load_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("type") == "result" or "status" in rec:
                rows.append(rec)

    return rows


def find_record(rows: List[Dict[str, Any]], keyword: str) -> Optional[Dict[str, Any]]:
    kw = keyword.lower()

    for rec in rows:
        q = coerce_text(rec.get("question") or rec.get("query")).lower()

        if kw in q:
            return rec

    return None


def format_citations(rec: Dict[str, Any]) -> str:
    audit = rec.get("audit") or {}
    llm = audit.get("llm") or {}
    citations = llm.get("citations") or []

    if not isinstance(citations, list) or not citations:
        return "No citations recorded for this status."

    lines: List[str] = []

    for c in citations[:4]:
        if not isinstance(c, dict):
            continue

        publisher = coerce_text(c.get("publisher") or "UNKNOWN")
        source_file = coerce_text(c.get("source_file") or "unknown")
        page_number = coerce_text(c.get("page_number") or "")
        supports = coerce_text(c.get("supports") or "")

        lines.append(
            f"- {publisher}, `{source_file}`, page {page_number}: {supports}"
        )

    return "\n".join(lines) if lines else "No usable citation details."


def format_record(rec: Dict[str, Any]) -> str:
    question = coerce_text(rec.get("question") or rec.get("query"))
    status = coerce_text(rec.get("status"))
    answer = coerce_text(rec.get("answer"))

    audit = rec.get("audit") or {}
    llm = audit.get("llm") or {}

    confidence = coerce_text(llm.get("confidence") or "")
    failure_type = coerce_text(llm.get("failure_type") or "")

    block = []
    block.append(f"**Question:** {question}")
    block.append("")
    block.append(f"**Status:** `{status}`")
    if confidence:
        block.append(f"**Confidence:** `{confidence}`")
    if failure_type:
        block.append(f"**Failure type / gate:** `{failure_type}`")
    block.append("")
    block.append("**Answer:**")
    block.append("")
    block.append(answer if answer else "_No answer text._")
    block.append("")
    block.append("**Citation / evidence trace:**")
    block.append("")
    block.append(format_citations(rec))
    block.append("")

    return "\n".join(block)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append("# Representative Examples")
    md.append("")
    md.append("These examples illustrate behavior of the governed RAG system across answerable QA, safety escalation, insufficient-evidence handling, and adversarial prompts.")
    md.append("")

    for category, run_path in RUNS.items():
        md.append(f"## {category.title()} Examples")
        md.append("")

        if run_path is None:
            md.append("_Run path not yet added. Update `RUNS` in `scripts/extract_representative_examples.py` after rerunning this category._")
            md.append("")
            continue

        rows = load_results(run_path)

        if not rows:
            md.append(f"_No rows loaded from `{run_path}`._")
            md.append("")
            continue

        md.append(f"Run: `{run_path}`")
        md.append("")

        for keyword in EXAMPLES.get(category, []):
            rec = find_record(rows, keyword)

            if rec is None:
                md.append(f"### Keyword not found: `{keyword}`")
                md.append("")
                continue

            md.append(f"### Example: `{keyword}`")
            md.append("")
            md.append(format_record(rec))
            md.append("")

    OUTPUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
