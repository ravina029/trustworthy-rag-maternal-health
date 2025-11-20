import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------
# New Parent Package Name
# ----------------------------------------------------------
PACKAGE = "src/trustworthy_maternal_postpartum_rag"

# ----------------------------------------------------------
# Folder structure — new clean package + your old modules kept
# ----------------------------------------------------------
FOLDERS = [

    # Main package
    f"{PACKAGE}",
    f"{PACKAGE}/ingestion",
    f"{PACKAGE}/preprocessing",
    f"{PACKAGE}/retrieval",
    f"{PACKAGE}/pipeline",
    f"{PACKAGE}/rag_app",

    # Existing module folders kept (no deletion)
    "src/retriever",
    "src/generator",
    "src/safety",
    "src/explainability",
    "src/evaluation",
    "src/ui",

    # Data folders
    "data/raw/pdfs",
    "data/processed",

    # Logs folders
    "logs/data_logs",
    "logs/pipeline_logs",

    # Other project folders
    "notebooks",
    "models/local_llm",
    "models/sentence_transformer",
    "scripts",
]

# ----------------------------------------------------------
# Initial files — only created if NEW (never overwrite)
# ----------------------------------------------------------
FILES = {

    # Package init
    f"{PACKAGE}/__init__.py": "",
    f"{PACKAGE}/utils.py": "# utility functions will go here\n",

    # New submodules
    f"{PACKAGE}/ingestion/__init__.py": "",
    f"{PACKAGE}/ingestion/pdf_loader.py": "",
    f"{PACKAGE}/ingestion/text_cleaning.py": "",

    f"{PACKAGE}/preprocessing/__init__.py": "",
    f"{PACKAGE}/preprocessing/splitter.py": "",

    f"{PACKAGE}/retrieval/__init__.py": "",
    f"{PACKAGE}/retrieval/vectorstore.py": "",
    f"{PACKAGE}/retrieval/embeddings.py": "",

    f"{PACKAGE}/pipeline/__init__.py": "",
    f"{PACKAGE}/pipeline/build_index.py": "",

    f"{PACKAGE}/rag_app/__init__.py": "",
    f"{PACKAGE}/rag_app/rag_engine.py": "",

    # Old files kept exactly as before
    "data/sample_corpus.txt": "",
    "src/retriever/embedder.py": "",
    "src/retriever/retriever.py": "",
    "src/retriever/indexing.py": "",

    "src/generator/rag_model.py": "",

    "src/safety/hallucination_check.py": "",
    "src/safety/safety_classifier.py": "",
    "src/safety/rule_based_filters.py": "",

    "src/explainability/lime_explainer.py": "",
    "src/explainability/shap_explainer.py": "",

    "src/evaluation/faithfulness_eval.py": "",
    "src/evaluation/safety_eval.py": "",

    "src/ui/streamlit_app.py": "",

    "scripts/prepare_corpus.py": "",
    "scripts/build_index.py": "",
    "scripts/run_inference.py": "",
}

# ----------------------------------------------------------
# Executor
# ----------------------------------------------------------
def create_folders_and_files():
    print("\n🚀 Updating project structure safely...\n")

    # Create folders safely
    for folder in FOLDERS:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Folder ready: {path}")

    # Create missing files safely
    for file_path, content in FILES.items():
        path = ROOT / file_path
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 File created: {path}")
        else:
            print(f"⚠️ File exists, skipped: {path}")

    print("\n✨ Folder structure updated without losing any existing work.\n")


if __name__ == "__main__":
    create_folders_and_files()
