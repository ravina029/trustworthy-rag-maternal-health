# Trustworthy RAG for Maternal & Postpartum Health  
### An Evaluation-First Retrieval-Augmented Generation Framework for Safety-Sensitive Question Answering

---

## Overview

This project develops a **trustworthy Retrieval-Augmented Generation (RAG) system** for maternal and postpartum health information, a domain where hallucinated or weakly grounded responses may create real-world harm.

Rather than optimizing only for fluent text generation, the system prioritizes:

- **Evidence-grounded answers**
- **Claim-level verification**
- **Traceable reasoning outputs**
- **Robust behavior under adversarial or low-evidence inputs**
- **Safety-aware refusal when evidence is insufficient**

The project is designed as a **research-oriented prototype** demonstrating how LLM-based question answering systems can be constrained, audited, and evaluated for higher reliability in sensitive domains.

---

## Why This Problem Matters

Maternal and postpartum users frequently seek guidance regarding:

- Recovery after childbirth  
- Breastfeeding and nutrition  
- Postpartum symptoms  
- Medication-related concerns  
- Escalation to professional medical care  

General-purpose LLMs may produce fluent but unsupported advice. In healthcare-adjacent settings, this creates a strong need for systems that explicitly manage uncertainty and provide verifiable outputs.

---

## Research Motivation

Modern RAG pipelines improve factuality by retrieving external documents, yet many systems still fail in critical ways:

- Unsupported claims remain in generated answers  
- Retrieved evidence is not explicitly linked to claims  
- Systems answer confidently under weak evidence  
- Evaluation often focuses on fluency rather than trustworthiness  

This project investigates an alternative direction:

> **Can RAG systems be designed and evaluated primarily around reliability rather than generation quality?**

---

## Core Contributions

### 1. Domain-Focused Trustworthy RAG Pipeline

A modular RAG system for maternal and postpartum health with explicit retrieval, grounding, verification, and safety stages.

### 2. Claim-Level Evidence Auditing

Generated answers are decomposed into claims and checked against retrieved evidence before final output.

### 3. Safety-Aware Refusal Logic

The system can decline to answer or escalate uncertainty when evidence quality is weak or absent.

### 4. Multi-Dimensional Evaluation Framework

Evaluation goes beyond standard QA accuracy and measures:

- Groundedness  
- Traceability  
- Robustness  
- Refusal behavior  

### 5. Reproducible Research Workflow

Config-driven experiments, structured outputs, and modular evaluation scripts support systematic experimentation.

---

## System Architecture

```text
User Query
   ↓
Retriever (Vector Search / Chroma)
   ↓
Draft Answer Generation
   ↓
Claim Segmentation
   ↓
Claim-Evidence Verification
   ↓
Safety / Refusal Policy
   ↓
Final Audited Response
```

---

## Repository Structure

```text
trustworthy-rag-maternal-health/

configs/        # experiment and pipeline settings
docs/           # documentation and notes
scripts/        # runnable entrypoints
src/
└── trustworthy_maternal_postpartum_rag/
    ├── app/          # QA orchestration
    ├── retrieval/    # retriever logic
    ├── grounding/    # evidence validation
    ├── ingestion/    # corpus processing
    ├── safety/       # refusal / policy logic
    ├── eval/         # metrics and scorecards
    └── rag/          # experiment harness
```

---

## Example Research Questions

- Does claim verification reduce unsupported outputs?  
- How sensitive is groundedness to retrieval quality?  
- When should systems refuse rather than answer?  
- How robust is the pipeline against adversarial prompts?  
- Which trust metrics correlate most strongly with failure cases?  

---

## Evaluation Framework

The system is evaluated across multiple dimensions.

### 1. Groundedness

Measures whether answer claims are supported by retrieved evidence.

**Example Metrics**

- Grounded sentence rate  
- Unsupported claim rate  

### 2. Traceability

Measures whether claims can be linked back to specific retrieved passages.

**Example Metrics**

- Trace completeness  
- Citation coverage  

### 3. Robustness

Measures behavior under prompt attacks, irrelevant instructions, or noisy inputs.

**Example Metrics**

- External link leakage  
- Unsafe instruction following  

### 4. Safety Behavior

Measures whether the system appropriately refuses when evidence is insufficient.

**Example Metrics**

- Refusal precision  
- Refusal recall  

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Main Pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Evaluation

```bash
python scripts/run_pipeline.py --config configs/experiment_config.yaml
```

---

## Example Intended Output Behavior

### Query

> Can I resume exercise two weeks after a cesarean section?

### Desired System Behavior

- Retrieve postpartum recovery evidence  
- Avoid overconfident medical advice  
- Mention variability and need for clinician guidance  
- Cite supporting evidence  
- Refuse certainty if corpus support is insufficient  

---

## Current Limitations

- Retrieval quality still constrains downstream performance  
- Some grounding metrics are heuristic rather than clinically validated  
- Corpus scale is limited compared with production medical systems  
- This project is a research prototype, not a medical device  

---

## Future Research Directions

- Retrieval reranking with cross-encoders  
- Uncertainty estimation for answer calibration  
- Human evaluation with domain experts  
- Multilingual maternal-health support  
- Benchmarking against standard RAG baselines  
- Explainable citation confidence scoring  

---

## Relevance to NLP / Trustworthy AI Research

This project aligns with current research themes in:

- Trustworthy NLP  
- Retrieval-Augmented Generation  
- Explainable AI  
- Factuality and hallucination reduction  
- Safety-aware LLM systems  
- Evaluation methodology for generative models  

It is particularly relevant for PhD groups working on:

- LLM reliability  
- Robust NLP  
- Explainable AI  
- Healthcare NLP  
- Safe generative systems  

---

## Author

**Ravina**  
Independent research project focused on trustworthy NLP systems, RAG evaluation, and reliable language model behavior.

---

## License

MIT License
