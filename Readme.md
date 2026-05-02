# Trustworthy RAG for Maternal & Postpartum Health

## 🚨 Problem

Large Language Models (LLMs) are unreliable in **safety-critical domains** due to hallucinations and lack of verifiable grounding.

This project builds a **trustworthy RAG system** that enforces:

* evidence-grounded generation
* claim-level verification
* safety-aware response behavior

The focus is **not on generation quality**, but on **verifiability and reliability**.

---

## 🎯 Contribution

This work introduces a **trustworthiness-focused RAG pipeline** with:

* **Claim-level grounding** of generated answers
* **Explicit evidence traceability**
* **Robustness against adversarial inputs**
* **Safety-aware refusal mechanisms**

---

## 🧠 System Pipeline

```text
User Query  
→ Retrieval  
→ Answer Generation  
→ Claim Verification  
→ Safety Filtering  
```

---

## 📂 Project Structure

```text
src/trustworthy_maternal_postpartum_rag/
    app/
    retrieval/
    grounding/
    pipeline/
    eval/
    safety/
    ingestion/

scripts/
    run_pipeline.py

configs/
    default.yaml
```

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/default.yaml
```

---

## 📊 Evaluation

The system is evaluated along **three core dimensions**:

### 1. Explainability (Traceability)

* Metric: `trace_complete_rate`
* Measures whether each answer is fully supported by retrieved evidence

---

### 2. Trustworthiness (Groundedness)

* Metric: `mean_grounded_sentence_rate_ok`
* Measures factual alignment between generated claims and retrieved documents

---

### 3. Robustness

* Metric: `external_link_leak_rate`
* Evaluates resistance to prompt injection and unsafe external references

---

### Additional Behavior Checks

* Refusal handling (`insufficient_evidence`)
* Safety escalation behavior

---

## 📈 Key Insight

Claim-level verification significantly improves reliability by:

* reducing unsupported claims
* enforcing strict grounding
* preventing hallucinated outputs

---

## 🧪 Research Contributions

* Evidence-grounded RAG architecture
* Deterministic evaluation of groundedness
* Integrated safety + verification pipeline
* End-to-end trustworthiness evaluation

---

## ⚠️ Limitations

* Dependent on retrieval quality
* Limited domain corpus
* Heuristic grounding metrics

---

## 🚀 Future Work

* Retrieval reranking
* Uncertainty estimation
* Larger clinical datasets
* Formal evaluation benchmarks

---

## 📚 Citation

```text
@misc{trustworthy_rag_2025,
    title={Trustworthy RAG for Maternal & Postpartum Health},
    author={Ravina Verma},
    year={2025}
}
```

---

## 📝 License

MIT License
