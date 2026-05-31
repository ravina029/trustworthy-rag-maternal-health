# src/trustworthy_maternal_postpartum_rag/rag/eval_harness.py

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trustworthy_maternal_postpartum_rag.app.final_answer_generation import answer_question_final
from trustworthy_maternal_postpartum_rag.utils import call_ollama


OUTPUT_DIR = Path("eval_runs")


def now():
    return datetime.now().isoformat(timespec="seconds")


def load_config():
    config_path = os.getenv("TMPRAG_CONFIG_PATH", "configs/pipeline_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_questions(config: Dict[str, Any]) -> List[str]:
    # Priority: env override > config file
    path = os.getenv("TMPRAG_QUESTIONS_PATH")

    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

    return config.get("queries", [])


def ensure_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir()

    config = load_config()

    questions = load_questions(config)

    max_q = os.getenv("TMPRAG_MAX_QUESTIONS")
    if max_q and max_q.lower() != "none":
        questions = questions[: int(max_q)]

    k = config["retrieval"]["top_k"]
    debug = config.get("runtime", {}).get("debug", False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"eval_run_{run_id}.jsonl"

    with out_file.open("w", encoding="utf-8") as f:

        # metadata
        f.write(json.dumps({
            "type": "run_meta",
            "run_id": run_id,
            "ts": now(),
            "config_used": os.getenv("TMPRAG_CONFIG_PATH", "configs/pipeline_config.yaml"),
            "k": k,
            "n_questions": len(questions)
        }) + "\n")

        for i, q in enumerate(questions, 1):

            try:
                result = answer_question_final(
                    query=q,
                    k=k,
                    llm_fn=call_ollama,
                    debug=debug
                )
            except Exception as e:
                result = {
                    "status": "exception",
                    "answer": "",
                    "audit": {"error": str(e)},
                    "evidence": []
                }

            f.write(json.dumps({
                "type": "result",
                "idx": i,
                "question": q,
                "status": result.get("status"),
                "answer": result.get("answer", ""),
                "audit": result.get("audit", {}),
                "evidence": result.get("evidence", []),
                "ts": now()
            }) + "\n")

            print(f"[{i}] {result.get('status')}")

    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    sys.exit(main())