# scripts/run_pipeline.py

import yaml
import os
import sys
import json
from datetime import datetime
import numpy as np
import random
import subprocess

random.seed(CFG["run"]["seed"])
np.random.seed(CFG["run"]["seed"])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath("src"))

from trustworthy_maternal_postpartum_rag.app.final_answer_generation import (
    answer_question_final,
    call_ollama,
)


def run_pipeline(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    queries = config.get("queries", [])

    if "retrieval" not in config or "top_k" not in config["retrieval"]:
        raise ValueError("Config must contain retrieval.top_k")

    k = config["retrieval"]["top_k"]
    debug = config.get("runtime", {}).get("debug", False)

    if config.get("runtime", {}).get("rebuild_index", False):
    subprocess.run(["python", "scripts/run_ingestion.py"])

    

    run_meta = {
        "timestamp": str(datetime.now()),
        "config": open("config/pipeline.yaml").read()
    }

    with open("eval_runs/run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    results = []

    for q in queries:
        print(f"\n🔍 Query: {q}")

        output = answer_question_final(
            query=q,
            k=k,
            llm_fn=call_ollama,
            debug=debug,
        )

        results.append(output)

        print(f"✅ Status: {output['status']}")
        print(f"💬 Answer: {output['answer'][:200]}...\n")

    return results


if __name__ == "__main__":
    config_path = os.getenv("TMPRAG_CONFIG_PATH", "configs/default.yaml")
    run_pipeline(config_path)