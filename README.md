# Trustworthy RAG for Maternal and Postpartum Health

A privacy-preserving, fully local retrieval-augmented generation (RAG) system for maternal, postpartum, newborn, and infant health information. The project studies how deterministic governance layers can improve RAG behavior in a safety-sensitive health-information domain.

> **Research claim:** Vanilla RAG can answer many ordinary health-information questions, but it may also treat urgent danger signs and unsupported product/local-policy questions as ordinary answerable prompts. This project adds a metadata-governed reliability layer with lifecycle-aware retrieval, citation governance, deterministic safety escalation, and insufficient-evidence refusal.

---

## Results at a Glance

The governed RAG system was compared with a vanilla RAG baseline under controlled conditions. Both systems used the same source PDFs, Chroma vector database, embedding model, local LLM, and question sets.

The percentages below are **expected-behavior pass rates on curated evaluation sets**, not estimates of general clinical accuracy.

| Evaluation regime | Expected behavior | Vanilla RAG | Governed RAG | Observation |
|---|---|---:|---:|---|
| Answerable QA | Evidence-grounded `ok` answer | 100.0% | 96.7% | Both systems answered ordinary questions well; governed RAG conservatively escalated one potentially risky baby rash + fever question. |
| Safety stress test | `safety_escalation` | 0.0% | 100.0% | On the curated danger-sign set, governed RAG routed all safety-critical prompts to escalation, while vanilla RAG treated them as ordinary `ok` prompts. |
| Insufficient-evidence stress test | `insufficient_evidence` | 0.0% | 100.0% | On the curated unsupported-question set, governed RAG refused all product/local-policy/exact-dose traps, while vanilla RAG answered them as `ok`. |
| Adversarial prompts | Evidence-bound / no fake links / no crash | 100.0% | 100.0% | Both systems remained link-safe in this setup; governed RAG added one conservative refusal. |

### Reliability Dashboard

| Reliability outcome | Final observed result |
|---|---:|
| Exceptions | 0 |
| Generation errors | 0 |
| External-link leakage | 0.0 |
| Curated safety stress-test pass rate | 0.0% → 100.0% |
| Curated insufficient-evidence stress-test pass rate | 0.0% → 100.0% |
| Governed answerable groundedness (`Groundedness_OK`) | 0.978 |
| Governed traceability | 1.0 across evaluated sets |

### Key Research Observations

| Observation | Interpretation |
|---|---|
| Vanilla RAG performed well on ordinary answerable questions. | Retrieval plus generation can be sufficient for many low-risk informational questions. |
| Vanilla RAG showed observed safety and refusal failures on curated stress tests. | A generic RAG pipeline can over-answer urgent danger signs and unsupported brand/local-policy questions. |
| Governed RAG eliminated the observed safety under-escalation and unsupported over-answering on the curated stress tests. | Deterministic gates improved behavioral reliability in the evaluated safety-sensitive settings. |
| Governance did not mainly improve ordinary answerability. | The strongest contribution is reliability under high-risk and out-of-scope conditions. |

### Result Artifacts

| Artifact | Path |
|---|---|
| Final comparison summary | `results/final_baseline_vs_governed_summary.md` |
| Evaluation manifest | `results/evaluation_run_manifest.md` |
| Representative examples | `results/baseline_vs_governed_representative_examples.md` |
| Answerable comparison | `results/baseline_vs_final_answerable.md` |
| Safety comparison | `results/baseline_vs_final_safety.md` |
| Insufficient-evidence comparison | `results/baseline_vs_final_insufficient.md` |
| Adversarial comparison | `results/baseline_vs_final_adversarial.md` |

---

## Motivation

RAG systems are often evaluated mainly by whether they produce plausible answers. In health-information settings, plausibility is not enough. A reliable system should also know when to escalate, when to refuse, how to cite evidence, and how to avoid unsupported claims.

This project focuses on maternal and newborn health because the domain has several reliability challenges:

| Challenge | Why it matters |
|---|---|
| Lifecycle specificity | Pregnancy, postpartum, breastfeeding, newborn, infant, and child-care evidence should not be mixed carelessly. |
| Safety-critical symptoms | Emergency symptoms should not be handled as ordinary informational prompts. |
| Unsupported local/product questions | Brand, product, local hospital policy, pharmacy, and exact-dose questions may not be answerable from the corpus. |
| Citation traceability | Answers should be linked to source documents and page-level evidence. |
| Privacy | User questions can be sensitive; the system runs locally without external LLM APIs. |

---

## Research Contributions

| Contribution | Implementation in this project |
|---|---|
| Lifecycle-aware metadata governance | Chunks preserve lifecycle stage, publisher, source tier, topic scope, source file, and page number. Retrieval and evaluation use this metadata to reduce cross-stage evidence leakage. |
| Publisher/source-tier evidence handling | Sources such as WHO, NHS, ACOG, government guidance, and patient-friendly secondary sources are tracked separately for source-reliability analysis. |
| Citation-grounded generation | Answers include evidence traces with publisher, source file, page number, and support text. Citation hygiene and traceability are evaluated explicitly. |
| Deterministic safety escalation | Red-flag prompts involving severe postpartum bleeding, chest pain, shortness of breath, severe headache with vision changes, suicidal ideation, newborn fever, poor feeding, convulsions, fast breathing, and severe jaundice signs are escalated. |
| Insufficient-evidence refusal | Product recommendations, exact local policies, commercial brands, pharmacy availability, and unsupported exact-dose questions are refused when the corpus does not support them. |
| Controlled baseline comparison | Vanilla RAG and governed RAG are compared using the same corpus, Chroma index, embedding model, local LLM, and question sets. |
| Category-aware evaluation | Evaluation is separated into answerable QA, safety escalation, insufficient evidence, and adversarial prompts. |

---

## System Architecture

```text
Source PDFs
   |
   v
PDF preprocessing and page extraction
   |
   v
Canonical metadata registry
   |
   v
Chunking and quality checks
   |
   v
Chroma vector index
   |
   v
Metadata-aware retrieval
   |
   v
Governed answer generation
   |      |        |
   |      |        +--> Insufficient-evidence refusal
   |      +-----------> Safety escalation gate
   +------------------> Citation and evidence validation
   |
   v
Final answer + evidence trace + evaluation audit
```

### Pipeline Components

| Component | Role |
|---|---|
| PDF preprocessing | Extracts page-level text and preserves document metadata. |
| Chunking | Produces retrieval-ready evidence chunks. |
| Chroma indexing | Stores local vector embeddings and metadata. |
| Retriever | Retrieves semantically relevant and metadata-aware evidence. |
| Governed generator | Produces evidence-bounded answers with citation traces. |
| Safety gate | Escalates danger-sign questions deterministically. |
| Insufficient-evidence gate | Refuses unsupported product/local/exact-policy questions. |
| Scorecard | Evaluates traceability, groundedness, safety, refusal, and robustness. |
| Metadata metrics | Evaluates publisher diversity, stage alignment, and source authority. |
| Baseline comparison | Compares vanilla RAG with governed RAG. |

---

## Corpus and Data Policy

The system was evaluated on a local corpus of nine public-health PDF documents covering pregnancy, postpartum, breastfeeding, newborn, infant, and child-care topics.

The raw PDFs are **not committed to this repository**. They should be downloaded from official public sources and placed locally under the configured raw-data directory.

A source-link file should be maintained at:

```text
docs/source_links.md
```

Recommended source-link format:

| Publisher | Document | Source URL | Local filename |
|---|---|---|---|
| WHO | ... | ... | `who_antenatal care.pdf` |
| NHS | ... | ... | `NHS_pregnancy_postpartum_guide.pdf` |
| ACOG | ... | ... | `ACOG_Pregnancy Guide.pdf` |

### Indexed Corpus Summary

| Source tier | Indexed chunks |
|---|---:|
| Core authoritative | 2,474 |
| Secondary patient-friendly | 1,725 |
| **Total** | **4,199** |

| Publisher | Indexed chunks |
|---|---:|
| Baby 411 | 1,376 |
| WHO | 1,317 |
| NHS | 928 |
| Cleveland Clinic | 349 |
| ACOG | 205 |
| Government of India / PMSMA | 24 |

---

## Models and Runtime

| Component | Model / Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | Chroma |
| Local LLM | Ollama `llama3` |
| Runtime | Local Python environment |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ravina029/trustworthy-rag-maternal-health.git
cd trustworthy-rag-maternal-health
```

Create and activate a virtual environment:

```bash
python -m venv ragvenv
source ragvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama separately, then pull the local model:

```bash
ollama pull llama3
```

> The project uses Chroma for the vector database. Ensure `chromadb>=0.5.0` is included in `requirements.txt`.

### Recommended `requirements.txt` additions

| Dependency | Purpose |
|---|---|
| `chromadb>=0.5.0` | Local vector database. |
| `sentence-transformers>=2.2.2` | Embedding generation. |
| `torch>=2.2.0` | Transformer backend. |
| `pymupdf>=1.23.0` | PDF parsing. |
| `pyyaml>=6.0.0` | Config loading. |
| `streamlit>=1.27.0` | Planned optional UI. |

---

## Running the Pipeline

A typical local workflow is:

```bash
PYTHONPATH=src python scripts/run_pipeline.py
```

The main configuration file is:

```text
configs/pipeline_config.yaml
```

Important local folders:

| Path | Purpose | Committed? |
|---|---|---|
| `data/raw/` | Local source PDFs | No |
| `data/processed/` | Processed page-level outputs | No |
| `data/chunks/` | Chunk files | No |
| `data/chroma_db/` | Chroma vector database | No |
| `eval_runs/` | Governed evaluation logs | Usually no, except selected summaries |
| `eval_runs_baseline/` | Vanilla baseline logs | Usually no, except selected summaries |
| `results/` | Selected summaries and reports | Yes, selected markdown summaries only |

---

## Evaluation Design

The evaluation is divided into four behavioral regimes. This category-aware design avoids judging safety-only or refusal-only files using ordinary OK-answer rate.

| Evaluation set | Purpose | Expected behavior |
|---|---|---|
| Answerable core | Normal maternal/newborn health-information questions | Evidence-grounded `ok` answer |
| Safety curated | Danger-sign prompts | `safety_escalation` |
| Insufficient evidence curated | Product, brand, exact local policy, or unsupported exact-dose questions | `insufficient_evidence` |
| Adversarial curated | Prompt-injection and evidence-override attempts | Evidence-bound answer or refusal; no fake links |

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| OK rate | Whether answerable questions receive normal answers. |
| Safety-escalation rate | Whether danger-sign prompts are escalated. |
| Insufficient-evidence rate | Whether unsupported questions are refused. |
| Traceability | Whether outputs retain evidence/citation audit trails. |
| Groundedness | Whether answer sentences are supported by retrieved evidence. |
| Hallucination rate | Unsupported sentence fraction. |
| Citation hygiene | Whether citations include source metadata and support text. |
| External-link leakage | Whether the model invents or emits external links. |
| Stage-alignment rate | Whether evidence matches the expected lifecycle stage. |
| Core-authoritative-source rate | Whether evidence includes authoritative sources. |
| Exception / generation-error rate | Software robustness. |

### Metric Interpretation Note

The percentages reported in this repository are **not broad clinical accuracy claims**. They are pass rates on curated evaluation sets designed to stress specific behaviors: normal answerability, safety escalation, insufficient-evidence refusal, and adversarial robustness. The safety and insufficient-evidence sets intentionally contain prompts that should not receive ordinary `ok` answers.

---

## Final Governed RAG Results

| Evaluation set | Total | OK | Safety | Insufficient | Exceptions | GenErr | Traceability | Groundedness_OK | LinkLeak | Observation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Answerable core | 30 | 29 | 1 | 0 | 0 | 0 | 1.0 | 0.978 | 0.0 | Passed; one conservative escalation. |
| Safety curated | 15 | 0 | 15 | 0 | 0 | 0 | 1.0 | N/A | 0.0 | Passed on curated set; 15/15 safety escalations. |
| Insufficient evidence | 12 | 0 | 0 | 12 | 0 | 0 | 1.0 | N/A | 0.0 | Passed on curated set; 12/12 insufficient-evidence refusals. |
| Adversarial curated | 15 | 14 | 0 | 1 | 0 | 0 | 1.0 | 1.0 | 0.0 | Passed; no link leakage, no exceptions. |

---

## Metadata-Governance Results

| Evaluation set | Mean distinct publishers | Publisher diversity rate | Mean stage alignment | Stage-alignment pass rate | Core-authoritative-source rate | Observation |
|---|---:|---:|---:|---:|---:|---|
| Answerable core | 1.63 | 0.600 | 0.821 | 0.967 | 0.933 | Strong stage alignment and source authority. |
| Safety curated | 2.13 | 0.733 | 0.917 | 1.000 | 1.000 | Strong metadata governance during safety escalation. |
| Insufficient evidence | 2.00 | 0.833 | 0.906 | 1.000 | 1.000 | Strong metadata governance despite refusal behavior. |
| Adversarial curated | 1.00 | 0.000 | 1.000 | 1.000 | 0.933 | Stage alignment strong; publisher diversity low under attack prompts. |

---

## Baseline vs Governed RAG

| Set | Vanilla RAG behavior | Governed RAG behavior | Curated expected-behavior pass rate | Main finding |
|---|---|---|---:|---|
| Answerable | 30/30 OK | 29/30 OK + 1 safety escalation | 100.0% → 96.7% | Governed system remains competitive on ordinary QA. |
| Safety stress test | 15/15 OK | 15/15 safety escalation | 0.0% → 100.0% | Governance eliminated observed safety under-escalation on this curated set. |
| Insufficient-evidence stress test | 12/12 OK | 12/12 insufficient evidence | 0.0% → 100.0% | Governance eliminated observed unsupported over-answering on this curated set. |
| Adversarial | 15/15 OK | 14 OK + 1 insufficient evidence | 100.0% → 100.0% | Both remained link-safe; governed added refusal behavior. |

### Comparison Artifacts

| Comparison | File |
|---|---|
| Answerable baseline vs governed | `results/baseline_vs_final_answerable.md` |
| Safety baseline vs governed | `results/baseline_vs_final_safety.md` |
| Insufficient-evidence baseline vs governed | `results/baseline_vs_final_insufficient.md` |
| Adversarial baseline vs governed | `results/baseline_vs_final_adversarial.md` |

---

## Representative Examples

Side-by-side qualitative examples are available in:

```text
results/baseline_vs_governed_representative_examples.md
```

| Example type | Vanilla RAG behavior | Governed RAG behavior |
|---|---|---|
| Safety-danger prompts | Treats urgent symptoms as ordinary answerable questions. | Routes prompts to safety escalation. |
| Unsupported product/brand/local-policy prompts | Gives normal answers despite insufficient corpus support. | Refuses with insufficient evidence. |

---

## Streamlit Demo Status

A Streamlit demo is planned but not yet included in the final evaluation snapshot. The current project should be treated as a research-grade CLI/local pipeline first.

| Planned UI feature | Status |
|---|---|
| User question input | Planned |
| Retrieved evidence display | Planned |
| Final answer display | Planned |
| Citation trace | Planned |
| Safety/refusal status | Planned |
| Metadata audit | Planned |

Recommended future branch:

```bash
git checkout -b feature/streamlit-demo
```

---

## Reproducibility Status

The final governed-vs-vanilla evaluation snapshot is committed and pushed. A clean-clone reproducibility test is still pending.

Recommended clean-clone test:

```bash
cd ..
git clone https://github.com/ravina029/trustworthy-rag-maternal-health.git tmprag_clean_test
cd tmprag_clean_test
python -m venv ragvenv
source ragvenv/bin/activate
pip install -r requirements.txt
ollama pull llama3
```

Then add source PDFs locally, configure `configs/pipeline_config.yaml`, and run the pipeline/evaluation commands.

---

## Limitations

| Limitation | Current status / mitigation |
|---|---|
| Not a clinical decision system | Research prototype for trustworthy health-information retrieval. |
| Corpus coverage dependence | Outputs are limited by the local PDF corpus. |
| Raw PDFs not committed | Source links should be added under `docs/source_links.md`. |
| Curated-test-set scope | Reported pass rates are stress-test results, not general medical accuracy estimates. |
| Scorecard not fully category-aware | Safety-only and refusal-only files may trigger generic warnings. |
| Adversarial baseline not fully raw | Baseline still uses structured JSON-style generation and link normalization. |
| Streamlit demo not implemented | Planned as a future branch. |
| Clean-clone test pending | To be completed before broad sharing. |

---

## Roadmap

| Priority | Task |
|---:|---|
| 1 | Add `docs/source_links.md` with official PDF source links. |
| 2 | Perform and document a clean-clone reproducibility test. |
| 3 | Make the scorecard category-aware. |
| 4 | Fix robustness evaluator handling of `json_ok_rate` and `injection_resistance_rate`. |
| 5 | Add ablation variants: vanilla RAG, +metadata retrieval, +citation governance, +safety gate, +insufficient-evidence gate, final governed system. |
| 6 | Build Streamlit demo. |
| 7 | Add README screenshots or demo GIF after the UI is stable. |

---

## License

This project is released under the MIT License. See `LICENSE`.
