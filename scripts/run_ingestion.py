import subprocess
import json
from datetime import datetime

def run():
    print("STEP 1: Preprocessing")
    subprocess.run(["python", "-m", "trustworthy_maternal_postpartum_rag.ingestion.preprocessing"], check=True)

    print("STEP 2: Chunking")
    subprocess.run(["python", "-m", "trustworthy_maternal_postpartum_rag.ingestion.chunk_and_merge"], check=True)

    print("STEP 3: Indexing")
    subprocess.run(["python", "-m", "trustworthy_maternal_postpartum_rag.ingestion.chroma_index"], check=True)

    # Save run metadata (reproducibility)
    run_meta = {
        "timestamp": str(datetime.now())
    }

    with open("eval_runs/ingestion_run.json", "w") as f:
        json.dump(run_meta, f, indent=2)

if __name__ == "__main__":
    run()