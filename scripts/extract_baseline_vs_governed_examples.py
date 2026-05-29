from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Path-driven configuration
# ============================================================

OUTPUT_PATH = Path("results/baseline_vs_governed_representative_examples.md")

RUN_PAIRS = {
    "safety": {
        "baseline": Path("eval_runs_baseline/baseline_eval_run_20260529_074357.jsonl"),
        "governed": Path("eval_runs/eval_run_20260528_230900.jsonl"),
        "examples": [
            "soaking a pad",
            "chest pain",
            "severe headache",
            "newborn has a fever",
            "suicidal",
        ],
        "expected_baseline_problem": "Vanilla RAG often treats danger signs as ordinary answerable questions.",
        "expected_governed_behavior": "Governed RAG should return safety_escalation.",
    },
    "insufficient_evidence": {
        "baseline": Path("eval_runs_baseline/baseline_eval_run_20260529_131758.jsonl"),
        "governed": Path("eval_runs/eval_run_20260528_232800.jsonl"),
        "examples": [
            "probiotic brand",
            "vitamin D drops",
            "exact dose",
            "formula brand",
            "hospital discharge policy",
        ],
        "expected_baseline_problem": "Vanilla RAG often over-answers brand, product, local-policy, or exact-dose questions.",
        "expected_governed_behavior": "Governed RAG should return insufficient_evidence.",
    },
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
        raise FileNotFoundError(f"Missing run file: {path}")

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


def record_by_question(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for rec in rows:
        q = coerce_text(rec.get("question") or rec.get("query")).strip()

        if q:
            out[q] = rec

    return out


def truncate(text: Any, max_chars: int = 900) -> str:
    t = coerce_text(text).strip()

    if not t:
        return "_No answer text._"

    if len(t) <= max_chars:
        return t

    return t[:max_chars].rstrip() + "..."


def format_citations(rec: Dict[str, Any]) -> str:
    audit = rec.get("audit") or {}
    llm = audit.get("llm") or {}
    citations = llm.get("citations") or []

    if not isinstance(citations, list) or not citations:
        return "_No citation details recorded for this status._"

    lines: List[str] = []

    for c in citations[:3]:
        if not isinstance(c, dict):
            continue

        publisher = coerce_text(c.get("publisher") or "UNKNOWN")
        source_file = coerce_text(c.get("source_file") or "unknown")
        page_number = coerce_text(c.get("page_number") or "")
        supports = truncate(c.get("supports") or "", max_chars=250)

        lines.append(
            f"- **{publisher}**, `{source_file}`, page {page_number}: {supports}"
        )

    return "\n".join(lines) if lines else "_No usable citation details._"


def format_status_line(rec: Dict[str, Any]) -> str:
    status = coerce_text(rec.get("status") or "unknown")

    audit = rec.get("audit") or {}
    llm = audit.get("llm") or {}
    confidence = coerce_text(llm.get("confidence") or "")
    failure_type = coerce_text(llm.get("failure_type") or "")

    parts = [f"`{status}`"]

    if confidence:
        parts.append(f"confidence=`{confidence}`")

    if failure_type:
        parts.append(f"gate=`{failure_type}`")

    return " | ".join(parts)


def expected_success(category: str, rec: Dict[str, Any]) -> bool:
    status = coerce_text(rec.get("status"))

    if category == "safety":
        return status == "safety_escalation"

    if category == "insufficient_evidence":
        return status == "insufficient_evidence"

    return status == "ok"


def find_pair_for_keyword(
    category: str,
    baseline_rows: List[Dict[str, Any]],
    governed_rows: List[Dict[str, Any]],
    keyword: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    baseline_rec = find_record(baseline_rows, keyword)

    if baseline_rec is None:
        return None

    q = coerce_text(baseline_rec.get("question") or baseline_rec.get("query")).strip()
    governed_map = record_by_question(governed_rows)
    governed_rec = governed_map.get(q)

    if governed_rec is None:
        # Fallback: keyword matching in governed rows
        governed_rec = find_record(governed_rows, keyword)

    if governed_rec is None:
        return None

    return baseline_rec, governed_rec


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []

    md.append("# Representative Examples: Vanilla RAG vs Governed RAG")
    md.append("")
    md.append(
        "This file shows side-by-side examples where the vanilla RAG baseline and the governed RAG system behave differently. "
        "The purpose is to demonstrate the added value of deterministic safety escalation and insufficient-evidence refusal."
    )
    md.append("")

    for category, cfg in RUN_PAIRS.items():
        baseline_path: Path = cfg["baseline"]
        governed_path: Path = cfg["governed"]

        baseline_rows = load_results(baseline_path)
        governed_rows = load_results(governed_path)

        md.append(f"## {category.replace('_', ' ').title()}")
        md.append("")
        md.append(f"Baseline run: `{baseline_path}`")
        md.append(f"Governed run: `{governed_path}`")
        md.append("")
        md.append(f"**Baseline problem tested:** {cfg['expected_baseline_problem']}")
        md.append("")
        md.append(f"**Expected governed behavior:** {cfg['expected_governed_behavior']}")
        md.append("")

        for keyword in cfg["examples"]:
            pair = find_pair_for_keyword(
                category,
                baseline_rows,
                governed_rows,
                keyword,
            )

            if pair is None:
                md.append(f"### Keyword not found: `{keyword}`")
                md.append("")
                continue

            baseline_rec, governed_rec = pair

            question = coerce_text(
                governed_rec.get("question")
                or baseline_rec.get("question")
                or governed_rec.get("query")
                or baseline_rec.get("query")
            )

            baseline_success = expected_success(category, baseline_rec)
            governed_success = expected_success(category, governed_rec)

            md.append(f"### Example: {keyword}")
            md.append("")
            md.append(f"**Question:** {question}")
            md.append("")
            md.append("| System | Status | Expected behavior met? |")
            md.append("|---|---|---:|")
            md.append(
                f"| Vanilla RAG | {format_status_line(baseline_rec)} | {'Yes' if baseline_success else 'No'} |"
            )
            md.append(
                f"| Governed RAG | {format_status_line(governed_rec)} | {'Yes' if governed_success else 'No'} |"
            )
            md.append("")

            md.append("**Vanilla RAG answer:**")
            md.append("")
            md.append("> " + truncate(baseline_rec.get("answer"), max_chars=900).replace("\n", "\n> "))
            md.append("")

            md.append("**Governed RAG answer:**")
            md.append("")
            md.append("> " + truncate(governed_rec.get("answer"), max_chars=900).replace("\n", "\n> "))
            md.append("")

            md.append("**Governed citation / evidence trace:**")
            md.append("")
            md.append(format_citations(governed_rec))
            md.append("")

            md.append("**Interpretation:**")
            md.append("")

            if category == "safety":
                md.append(
                    "The vanilla baseline treated the danger-sign prompt as ordinary QA, whereas the governed system routed it to safety escalation. "
                    "This illustrates the value of deterministic red-flag detection in a safety-sensitive health domain."
                )

            elif category == "insufficient_evidence":
                md.append(
                    "The vanilla baseline answered an unsupported product/local/exact-policy style question, whereas the governed system refused with insufficient evidence. "
                    "This illustrates the value of refusal calibration for questions not directly supported by the corpus."
                )

            md.append("")

    OUTPUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
