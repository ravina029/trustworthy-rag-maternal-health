from __future__ import annotations

import csv
import glob
import json
import re
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Path-driven input
# ============================================================

# You can override this with:
# TMPRAG_EVAL_RUN_PATH=eval_runs/eval_run_xxx.jsonl python scripts/eval_metadata_metrics.py
INPUT_LOG_PATH = os.getenv("TMPRAG_EVAL_RUN_PATH", "").strip()

OUTPUT_JSON_PATH = Path("results/metadata_metrics_summary.json")
OUTPUT_CSV_PATH = Path("results/metadata_metrics_details.csv")


CORE_AUTHORITATIVE_PUBLISHERS = {
    "WHO",
    "NHS",
    "ACOG",
    "Government of India",
    "Government of India / PMSMA",
}


# ============================================================
# Helpers
# ============================================================

def latest_eval_run() -> Path:
    files = sorted(glob.glob("eval_runs/eval_run_*.jsonl"))

    if not files:
        raise FileNotFoundError("No eval_runs/eval_run_*.jsonl files found.")

    return Path(files[-1])


def load_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

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


def lower_text(x: Any) -> str:
    return coerce_text(x).lower()


def infer_expected_stage(question: str) -> str:
    """
    Simple deterministic question-stage classifier.
    This is only for evaluation, not retrieval.
    """

    q = lower_text(question)

    breastfeeding_terms = [
        "breastfeeding",
        "breastfeed",
        "breast milk",
        "milk supply",
        "nipple",
        "mastitis",
        "clogged duct",
        "ducts",
        "lactation",
    ]

    newborn_terms = [
        "newborn",
        "baby",
        "infant",
        "6-month",
        "6 month",
        "wet diaper",
        "wet diapers",
        "jaundice",
        "jaundiced",
        "dehydration",
        "solid foods",
        "safe sleeping",
        "sleeping advice",
        "toddler",
        "rash and fever",
        "not feeding",
        "fever",
        "fast breathing",
        "convulsions",
        "yellow palms",
        "yellow soles",
    ]

    postpartum_terms = [
        "postpartum",
        "after delivery",
        "after birth",
        "postnatal",
        "lochia",
        "c-section",
        "c section",
        "cesarean",
        "bleeding",
        "infection after birth",
        "exercise postpartum",
        "resume exercise",
        "severe headache",
        "vision changes postpartum",
        "chest pain",
        "shortness of breath",
        "suicidal",
    ]

    pregnancy_terms = [
        "pregnancy",
        "pregnant",
        "trimester",
        "soft cheese",
        "raw eggs",
        "undercooked eggs",
        "foods should be avoided",
        "round ligament",
        "ibuprofen in the third trimester",
        "antenatal",
    ]

    if any(t in q for t in breastfeeding_terms):
        return "breastfeeding"

    if any(t in q for t in newborn_terms):
        return "newborn_infant"

    if any(t in q for t in postpartum_terms):
        return "postpartum"

    if any(t in q for t in pregnancy_terms):
        return "pregnancy"

    return "general"


def evidence_stage_text(ev: Dict[str, Any]) -> str:
    meta = ev.get("metadata") or {}

    fields = [
        ev.get("stage"),
        ev.get("lifecycle"),
        ev.get("lifecycle_stage"),
        ev.get("topic_hint"),
        ev.get("topic_scope"),
        ev.get("inferred_lifecycle"),
        meta.get("stage"),
        meta.get("lifecycle"),
        meta.get("lifecycle_stage"),
        meta.get("topic_hint"),
        meta.get("topic_scope"),
        meta.get("inferred_lifecycle"),
    ]

    return " ".join(coerce_text(x) for x in fields if x is not None).lower()


def stage_aligns(expected_stage: str, ev: Dict[str, Any]) -> bool:
    text = evidence_stage_text(ev)

    if expected_stage == "general":
        return True

    if expected_stage == "pregnancy":
        return "pregnancy" in text or "antenatal" in text

    if expected_stage == "postpartum":
        return (
            "postpartum" in text
            or "postnatal" in text
            or "pregnancy_postpartum" in text
            or "postpartum_newborn" in text
            or "pregnancy_childbirth_postpartum_newborn" in text
        )

    if expected_stage == "breastfeeding":
        return (
            "breast" in text
            or "lactation" in text
            or "postpartum" in text
            or "postnatal" in text
            or "newborn" in text
            or "infant" in text
            or "pregnancy_childbirth_postpartum_newborn" in text
        )

    if expected_stage == "newborn_infant":
        return (
            "newborn" in text
            or "infant" in text
            or "baby" in text
            or "child" in text
            or "toddler" in text
            or "postpartum_newborn" in text
            or "pregnancy_childbirth_postpartum_newborn" in text
        )

    return True


def get_publisher_from_obj(x: Dict[str, Any]) -> str:
    meta = x.get("metadata") or {}

    publisher = (
        x.get("publisher")
        or meta.get("publisher")
        or x.get("source_publisher")
        or "UNKNOWN"
    )

    return coerce_text(publisher).strip() or "UNKNOWN"


def get_source_tier_from_obj(x: Dict[str, Any]) -> str:
    meta = x.get("metadata") or {}

    tier = (
        x.get("source_tier")
        or meta.get("source_tier")
        or x.get("source_type")
        or meta.get("source_type")
        or ""
    )

    return coerce_text(tier).strip()


def evidence_id(ev: Dict[str, Any], idx: int) -> str:
    return coerce_text(ev.get("chunk_id") or f"E{idx + 1}").strip()


def get_used_evidence(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prefer audit.llm.citations[].chunk_id if available.
    Fall back to all evidence in the record.
    """

    evidence = rec.get("evidence") or []

    if not isinstance(evidence, list):
        return []

    evidence_clean = [e for e in evidence if isinstance(e, dict)]

    by_id: Dict[str, Dict[str, Any]] = {}

    for idx, ev in enumerate(evidence_clean):
        by_id[evidence_id(ev, idx)] = ev
        by_id[f"E{idx + 1}"] = ev

    audit = rec.get("audit") or {}
    llm = audit.get("llm") or {}
    citations = llm.get("citations") or []

    used: List[Dict[str, Any]] = []

    if isinstance(citations, list):
        for c in citations:
            if not isinstance(c, dict):
                continue

            cid = coerce_text(c.get("chunk_id")).strip()

            if cid in by_id:
                ev = dict(by_id[cid])

                # Citation publisher can be cleaner than evidence publisher.
                if c.get("publisher"):
                    ev["publisher"] = c.get("publisher")

                used.append(ev)

    if used:
        return used

    return evidence_clean


def result_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    question = coerce_text(rec.get("question") or rec.get("query") or "")
    status = coerce_text(rec.get("status") or "unknown")
    expected_stage = infer_expected_stage(question)

    used_evidence = get_used_evidence(rec)

    publishers = []
    core_hits = 0
    aligned_hits = 0

    for ev in used_evidence:
        pub = get_publisher_from_obj(ev)
        tier = get_source_tier_from_obj(ev)

        if pub and pub != "UNKNOWN":
            publishers.append(pub)

        if pub in CORE_AUTHORITATIVE_PUBLISHERS or tier == "core_authoritative":
            core_hits += 1

        if stage_aligns(expected_stage, ev):
            aligned_hits += 1

    distinct_publishers = sorted(set(publishers))
    n_used = len(used_evidence)

    stage_alignment_fraction: Optional[float]

    if n_used == 0:
        stage_alignment_fraction = None
    else:
        stage_alignment_fraction = aligned_hits / n_used

    publisher_diverse = len(distinct_publishers) >= 2
    has_core_authoritative = core_hits > 0

    return {
        "idx": rec.get("idx"),
        "question": question,
        "status": status,
        "expected_stage": expected_stage,
        "used_evidence_count": n_used,
        "distinct_publishers": len(distinct_publishers),
        "publishers": "; ".join(distinct_publishers),
        "publisher_diverse": publisher_diverse,
        "stage_alignment_fraction": stage_alignment_fraction,
        "stage_aligned": (
            None if stage_alignment_fraction is None else stage_alignment_fraction >= 0.5
        ),
        "has_core_authoritative_source": has_core_authoritative,
    }


def safe_mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]

    if not values:
        return None

    return mean(values)


def fraction_true(values: List[bool]) -> Optional[float]:
    if not values:
        return None

    return sum(1 for v in values if v) / len(values)


def main() -> None:
    log_path = Path(INPUT_LOG_PATH) if INPUT_LOG_PATH else latest_eval_run()

    records = load_results(log_path)
    details = [result_metrics(rec) for rec in records]

    ok_details = [d for d in details if d["status"] == "ok"]
    evidence_bearing = [d for d in details if d["used_evidence_count"] > 0]
    ok_evidence_bearing = [d for d in ok_details if d["used_evidence_count"] > 0]

    summary = {
        "input_log_path": str(log_path),
        "n_results": len(details),
        "n_ok": len(ok_details),

        "mean_distinct_publishers_all": safe_mean(
            [float(d["distinct_publishers"]) for d in evidence_bearing]
        ),
        "mean_distinct_publishers_ok": safe_mean(
            [float(d["distinct_publishers"]) for d in ok_evidence_bearing]
        ),

        "publisher_diversity_rate_all": fraction_true(
            [bool(d["publisher_diverse"]) for d in evidence_bearing]
        ),
        "publisher_diversity_rate_ok": fraction_true(
            [bool(d["publisher_diverse"]) for d in ok_evidence_bearing]
        ),

        "mean_stage_alignment_all": safe_mean(
            [
                float(d["stage_alignment_fraction"])
                for d in evidence_bearing
                if d["stage_alignment_fraction"] is not None
            ]
        ),
        "mean_stage_alignment_ok": safe_mean(
            [
                float(d["stage_alignment_fraction"])
                for d in ok_evidence_bearing
                if d["stage_alignment_fraction"] is not None
            ]
        ),

        "stage_alignment_pass_rate_all": fraction_true(
            [
                bool(d["stage_aligned"])
                for d in evidence_bearing
                if d["stage_aligned"] is not None
            ]
        ),
        "stage_alignment_pass_rate_ok": fraction_true(
            [
                bool(d["stage_aligned"])
                for d in ok_evidence_bearing
                if d["stage_aligned"] is not None
            ]
        ),

        "core_authoritative_source_rate_all": fraction_true(
            [bool(d["has_core_authoritative_source"]) for d in evidence_bearing]
        ),
        "core_authoritative_source_rate_ok": fraction_true(
            [bool(d["has_core_authoritative_source"]) for d in ok_evidence_bearing]
        ),
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "details": details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "idx",
            "status",
            "expected_stage",
            "used_evidence_count",
            "distinct_publishers",
            "publisher_diverse",
            "stage_alignment_fraction",
            "stage_aligned",
            "has_core_authoritative_source",
            "publishers",
            "question",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in details:
            writer.writerow({k: d.get(k) for k in fieldnames})

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary → {OUTPUT_JSON_PATH}")
    print(f"Saved details → {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
