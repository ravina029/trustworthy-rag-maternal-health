from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Path-driven configuration
# ==========================================================
QUESTIONS_PATH = Path(os.getenv("TMPRAG_QUESTIONS_PATH", "configs/eval_sets/eval_questions_answerable_core.txt"))
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "maternal_postpartum_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OUTPUT_DIR = Path("eval_runs_baseline")
TOP_K = int(os.getenv("TMPRAG_TOP_K", "4"))
OLLAMA_MODEL = os.getenv("TMPRAG_OLLAMA_MODEL", "llama3")


# ============================================================
# Helpers
# ============================================================

def coerce_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return " ".join(coerce_text(v) for v in value)

    if isinstance(value, dict):
        return " ".join(coerce_text(v) for v in value.values())

    return str(value)


def strip_links(text: Any) -> str:
    t = coerce_text(text)
    t = re.sub(r"(https?://\S+|www\.\S+)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def read_questions(path: Path) -> List[str]:
    questions: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()

            if not q:
                continue

            if q.startswith("#"):
                continue

            questions.append(q)

    return questions


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))

        return coerce_text(data.get("response")).strip()

    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )

        return result.stdout.strip()


def extract_json_obj(raw: Any) -> Optional[Dict[str, Any]]:
    s = coerce_text(raw).strip()

    if not s:
        return None

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start, end = s.find("{"), s.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        obj = json.loads(s[start: end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def format_evidence_for_prompt(evidence: List[Dict[str, Any]]) -> str:
    blocks = []

    for ev in evidence:
        cid = ev.get("chunk_id", "E?")
        pub = ev.get("publisher", "UNKNOWN")
        src = ev.get("source_file", "unknown")
        page = ev.get("page_number", -1)
        text = coerce_text(ev.get("text", "")).strip()

        if len(text) > 900:
            text = text[:900].rstrip() + "..."

        blocks.append(
            f"[{cid}] publisher={pub} | file={src} | page={page}\n\"\"\"\n{text}\n\"\"\""
        )

    return "\n\n".join(blocks)


def build_baseline_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    evidence_text = format_evidence_for_prompt(evidence)

    return f"""
You are a maternal/postpartum/newborn health assistant.

Use the evidence below to answer the question.
Do not use external links.
Return valid JSON only.

QUESTION:
{question}

EVIDENCE:
{evidence_text}

Return JSON with exactly these keys:
{{
  "status": "ok" | "insufficient_evidence" | "safety_escalation",
  "answer": "string",
  "evidence_used": ["E1"],
  "citations": [
    {{
      "chunk_id": "E1",
      "publisher": "publisher name",
      "source_file": "file name",
      "page_number": 1,
      "supports": "short support statement"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "safety_notes": [],
  "follow_up_questions": []
}}

Important:
- This is a baseline system.
- Do not apply any deterministic safety gate.
- Do not apply any deterministic insufficient-evidence gate.
- Answer from retrieved evidence only.
""".strip()


def retrieve_baseline(
    collection,
    embedder: SentenceTransformer,
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_embedding = embedder.encode(question).tolist()

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    evidence: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        meta = metas[i] or {}
        dist = distances[i] if i < len(distances) else None

        ev = {
            "chunk_id": f"E{i + 1}",
            "text": coerce_text(doc),
            "metadata": meta,
            "distance": dist,
            "publisher": meta.get("publisher", "UNKNOWN"),
            "source_file": meta.get("source_file", "unknown"),
            "page_number": meta.get("page_number", -1),
            "stage": meta.get("stage") or meta.get("lifecycle_stage") or "",
            "lifecycle": meta.get("lifecycle") or meta.get("lifecycle_stage") or "",
            "source_tier": meta.get("source_tier") or meta.get("source_type") or "",
        }

        evidence.append(ev)

    return evidence


def normalize_baseline_output(
    parsed: Optional[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if parsed is None:
        return {
            "status": "generation_error",
            "answer": "",
            "evidence_used": [],
            "citations": [],
            "confidence": "low",
            "safety_notes": [],
            "follow_up_questions": [],
        }

    valid_ids = {ev["chunk_id"] for ev in evidence}
    evidence_by_id = {ev["chunk_id"]: ev for ev in evidence}

    status = parsed.get("status", "ok")

    if status not in {"ok", "insufficient_evidence", "safety_escalation"}:
        status = "ok"

    answer = strip_links(parsed.get("answer", ""))

    evidence_used_raw = parsed.get("evidence_used") or []
    evidence_used = [
        x for x in evidence_used_raw
        if isinstance(x, str) and x in valid_ids
    ]

    citations_raw = parsed.get("citations") or []
    citations: List[Dict[str, Any]] = []

    if isinstance(citations_raw, list):
        for c in citations_raw:
            if not isinstance(c, dict):
                continue

            cid = c.get("chunk_id")

            if cid not in valid_ids:
                continue

            ev = evidence_by_id[cid]

            citations.append(
                {
                    "chunk_id": cid,
                    "publisher": ev.get("publisher", "UNKNOWN"),
                    "source_file": ev.get("source_file", "unknown"),
                    "page_number": ev.get("page_number", -1),
                    "supports": strip_links(c.get("supports", "")),
                }
            )

    if status == "ok" and not evidence_used and citations:
        evidence_used = [c["chunk_id"] for c in citations]

    confidence = parsed.get("confidence", "low")

    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "status": status,
        "answer": answer,
        "evidence_used": evidence_used,
        "citations": citations,
        "confidence": confidence,
        "safety_notes": parsed.get("safety_notes", []) if isinstance(parsed.get("safety_notes", []), list) else [],
        "follow_up_questions": parsed.get("follow_up_questions", []) if isinstance(parsed.get("follow_up_questions", []), list) else [],
    }


def main() -> None:
    questions = read_questions(QUESTIONS_PATH)

    if not questions:
        raise ValueError(f"No questions found in {QUESTIONS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("baseline_eval_run_%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{run_id}.jsonl"

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Questions: {len(questions)}")
    print(f"Output: {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for idx, question in enumerate(questions, start=1):
            try:
                evidence = retrieve_baseline(collection, embedder, question, TOP_K)
                prompt = build_baseline_prompt(question, evidence)
                raw = call_ollama(prompt)
                parsed = extract_json_obj(raw)
                normalized = normalize_baseline_output(parsed, evidence)

                rec = {
                    "type": "result",
                    "idx": idx,
                    "question": question,
                    "status": normalized["status"],
                    "answer": normalized["answer"],
                    "prompt": prompt,
                    "audit": {
                        "mode": "baseline_vanilla_rag",
                        "retrieval": {
                            "top_k": TOP_K,
                            "collection": COLLECTION_NAME,
                            "embedding_model": EMBEDDING_MODEL,
                        },
                        "llm": {
                            "failure_type": None if normalized["status"] != "generation_error" else "json_parse_failure",
                            "confidence": normalized["confidence"],
                            "evidence_used": normalized["evidence_used"],
                            "citations": normalized["citations"],
                            "safety_notes": normalized["safety_notes"],
                            "follow_up_questions": normalized["follow_up_questions"],
                        },
                    },
                    "evidence": evidence,
                }

            except Exception as e:
                rec = {
                    "type": "result",
                    "idx": idx,
                    "question": question,
                    "status": "exception",
                    "answer": "",
                    "audit": {
                        "mode": "baseline_vanilla_rag",
                        "error": str(e),
                    },
                    "evidence": [],
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx}] {rec['status']}")

    print(f"\nSaved baseline run → {output_path}")


if __name__ == "__main__":
    main()

