# src/trustworthy_maternal_postpartum_rag/eval/scorecard.py

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

EVAL_RUNS_DIR = Path("eval_runs")

EVAL_SCRIPTS = [
    "eval_explainability.py",
    "eval_trustworthiness.py",
    "eval_robustness.py",
]

EVAL_DIR = Path("src/trustworthy_maternal_postpartum_rag/eval")
HARNESS = Path("src/trustworthy_maternal_postpartum_rag/rag/eval_harness.py")

DEFAULT_FAST_QUESTIONS_PATH = "eval_questions_fast_adversarial.txt"
DEFAULT_FULL_QUESTIONS_PATH = "eval_questions_full_mixed.txt"

EXPL_REPORT = EVAL_RUNS_DIR / "explainability_report.json"
TRUST_REPORT = EVAL_RUNS_DIR / "trustworthiness_report.json"
ROBUST_REPORT = EVAL_RUNS_DIR / "robustness_report.json"


# -------------------------
# Helpers
# -------------------------

def _latest_eval_run_log() -> Optional[Path]:
    if not EVAL_RUNS_DIR.exists():
        return None
    candidates = sorted(EVAL_RUNS_DIR.glob("eval_run_*.jsonl"))
    return candidates[-1] if candidates else None


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_results(path: Path):
    objs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                objs.append(json.loads(line))

    last_meta_idx = -1
    for i, o in enumerate(objs):
        if o.get("type") == "run_meta":
            last_meta_idx = i

    if last_meta_idx == -1:
        return [o for o in objs if o.get("type") == "result"]

    return [o for o in objs[last_meta_idx + 1:] if o.get("type") == "result"]


def _run(cmd, env):
    return subprocess.run(cmd, env=env).returncode


# -------------------------
# Main
# -------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")

    parser.add_argument("--min-grounded-ok", type=float, default=0.95)
    parser.add_argument("--min-ok", type=int, default=10)

    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")

    # -------------------------
    # Question config
    # -------------------------
    if "TMPRAG_QUESTIONS_PATH" not in env:
        if args.fast:
            env["TMPRAG_QUESTIONS_PATH"] = DEFAULT_FAST_QUESTIONS_PATH
        elif args.full:
            env["TMPRAG_QUESTIONS_PATH"] = DEFAULT_FULL_QUESTIONS_PATH

    if args.fast:
        env.setdefault("TMPRAG_MAX_QUESTIONS", "5")
    elif args.full:
        env.setdefault("TMPRAG_MAX_QUESTIONS", "None")

    # -------------------------
    # 1. Run harness
    # -------------------------
    rc = _run(["python", str(HARNESS)], env)
    if rc != 0:
        print("Harness failed", file=sys.stderr)
        return rc

    latest_log = _latest_eval_run_log()
    if not latest_log:
        print("No eval log found", file=sys.stderr)
        return 2

    # -------------------------
    # 2. Run eval scripts
    # -------------------------
    for script in EVAL_SCRIPTS:
        script_path = EVAL_DIR / script
        if script_path.exists():
            rc = _run(["python", str(script_path)], env)
            if rc != 0:
                print(f"{script} failed", file=sys.stderr)
                return rc

    # -------------------------
    # 3. Load reports
    # -------------------------
    expl = _read_json(EXPL_REPORT)
    trust = _read_json(TRUST_REPORT)
    robust = _read_json(ROBUST_REPORT)

    # Sanity check
    if not trust or "mean_grounded_sentence_rate" not in trust:
        print("Invalid trust report", file=sys.stderr)
        return 2

    rows = _read_jsonl_results(latest_log)

    # -------------------------
    # 4. Metrics
    # -------------------------
    status_counts = {}
    for r in rows:
        st = r.get("status", "")
        status_counts[st] = status_counts.get(st, 0) + 1

    n_results = len(rows)

    if n_results == 0:
        print("No evaluation results found", file=sys.stderr)
        return 2

    n_ok = status_counts.get("ok", 0)
    n_insuff = status_counts.get("insufficient_evidence", 0)
    n_safety = status_counts.get("safety_escalation", 0)
    n_exception = status_counts.get("exception", 0)
    n_gen_error = status_counts.get("generation_error", 0)

    failures: List[str] = []

    # -------------------------
    # 🎯 CORE GATES (FINAL)
    # -------------------------

    # Traceability
    if expl.get("trace_complete_rate") != 1.0:
        failures.append("traceability failed")

    # Groundedness (OK-only)
    grounded_ok = trust.get("mean_grounded_sentence_rate_ok") or 0.0
    if grounded_ok < args.min_grounded_ok:
        failures.append(f"groundedness below threshold ({grounded_ok:.2f})")

    # Groundedness (overall — prevents selection bias)
    overall_grounded = trust.get("mean_grounded_sentence_rate") or 0.0
    if overall_grounded < 0.7:
        failures.append(f"overall groundedness too low ({overall_grounded:.2f})")

    # Coverage
    min_ok_required = int(0.6 * n_results)
    if n_ok < min_ok_required:
        failures.append("too many refusals")

    # Absolute minimum OK samples
    if n_ok < args.min_ok:
        failures.append(f"not enough valid samples (n_ok={n_ok} < {args.min_ok})")

    # Insufficient evidence rate (CRITICAL FIX)
    insuff_rate = n_insuff / n_results
    if insuff_rate > 0.3:
        failures.append(f"too many insufficient_evidence cases ({insuff_rate:.2f})")

    # Exceptions
    if n_exception > 0:
        failures.append(f"{n_exception} exceptions occurred")

    # Generation errors (NEW — requires pipeline support)
    if n_gen_error > 0:
        failures.append(f"{n_gen_error} generation errors")

    # Robustness
    if robust.get("external_link_leak_rate") != 0.0:
        failures.append("link leakage detected")

    # -------------------------
    # 5. Summary
    # -------------------------
    print("\n=== SCORECARD ===")
    print(
        f"Total={n_results} | OK={n_ok} | Insufficient={n_insuff} | "
        f"Safety={n_safety} | Exceptions={n_exception} | GenErr={n_gen_error}"
    )
    print(f"Traceability={expl.get('trace_complete_rate')}")
    print(f"Groundedness_OK={grounded_ok}")
    print(f"Groundedness_ALL={overall_grounded}")
    print(f"LinkLeak={robust.get('external_link_leak_rate')}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("-", f)
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())