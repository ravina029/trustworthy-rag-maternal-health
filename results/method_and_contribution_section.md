# Method and Research Contribution

## Method Overview

This project develops a privacy-preserving, fully local retrieval-augmented generation system for maternal, postpartum, newborn, and infant health information. The pipeline uses a curated corpus of public-health PDFs, local preprocessing and chunking, a local Chroma vector database, local sentence-transformer embeddings, and a locally hosted Ollama language model. The system is designed for safety-sensitive health information retrieval rather than for diagnosis or replacement of clinical care.

The pipeline consists of five main stages:

1. **Document ingestion and canonical metadata creation.**  
   Each source document is processed into page-level records with canonical metadata fields, including publisher, source tier, lifecycle stage, topic scope, country scope, source file, page number, and document identifier.

2. **Chunking and indexing.**  
   Documents are chunked into retrieval-ready passages and indexed into a local Chroma collection using sentence-transformer embeddings. The final indexed corpus contains 4,199 chunks from nine source PDFs.

3. **Metadata-aware retrieval.**  
   Retrieval is not treated as pure semantic similarity alone. The retriever preserves and uses metadata such as lifecycle stage, publisher, source tier, page number, and source file. This allows evidence to be ranked and audited according to stage relevance and source authority.

4. **Governed answer generation.**  
   The final answer generator uses evidence-bounded prompts and structured JSON output. It includes citation repair, evidence support validation, external-link suppression, JSON repair/fallback, deterministic safety escalation, and deterministic insufficient-evidence refusal for unsupported product, local-policy, brand, or exact-dose questions.

5. **Category-aware evaluation.**  
   The system is evaluated on separate question sets for answerable medical-information QA, safety-critical escalation, insufficient-evidence traps, and adversarial prompt-injection attempts. This avoids mixing normal QA accuracy with safety/refusal behavior and enables more precise diagnosis of failure modes.

## Evaluation Paradigms

The evaluation framework measures several trustworthiness dimensions:

| Paradigm | Evaluation method |
|---|---|
| Answerability | OK rate on curated answerable questions |
| Groundedness | Sentence-level support against retrieved evidence |
| Hallucination risk | Unsupported sentence rate |
| Traceability | Presence of evidence and citation audit trail |
| Citation hygiene | Valid citation structure with publisher, source file, page number, and support text |
| Safety behavior | Safety-escalation rate on danger-sign questions |
| Out-of-scope handling | Insufficient-evidence rate on brand/local/product/policy traps |
| Prompt-injection robustness | Evidence-boundedness and external-link suppression under adversarial instructions |
| Metadata governance | Stage-alignment rate, publisher-diversity rate, and core-authoritative-source rate |
| Software robustness | Exception rate and generation-error rate |

## Confirmed Governed-RAG Results So Far

The answerable-core benchmark achieved 30/30 OK responses with zero exceptions and zero generation errors. Traceability was 1.0, external-link leakage was 0.0, and OK-groundedness was 0.914. Metadata metrics showed a stage-alignment pass rate of 0.967, publisher-diversity rate of 0.633, and core-authoritative-source rate of 0.933.

After adding deterministic safety escalation, the safety benchmark achieved 15/15 safety-escalation responses with zero exceptions and zero generation errors. The safety-escalation behavior rate was 1.0. Metadata evaluation showed a stage-alignment pass rate of 1.0, publisher-diversity rate of 0.733, and core-authoritative-source rate of 1.0.

## Research Contribution

The contribution of this project is not simply a chatbot over health PDFs. The main contribution is a modular governance layer for local RAG in a safety-sensitive health domain.

The system adds the following research-relevant components:

1. **Lifecycle-aware evidence governance.**  
   The system explicitly tracks whether evidence is related to pregnancy, postpartum recovery, breastfeeding, newborn care, infant care, or child care. This reduces cross-stage evidence leakage, such as answering a newborn question with pregnancy-only evidence.

2. **Publisher- and source-tier-aware retrieval.**  
   Evidence is not treated equally. The system tracks authoritative sources such as WHO, NHS, ACOG, and government guidance separately from secondary patient-friendly sources. This enables source reliability analysis and publisher-diversity measurement.

3. **Citation-grounded answer generation.**  
   The generator produces structured answers with evidence identifiers, publisher names, source files, page numbers, and support statements. Post-processing repairs weak citation support and removes unsupported or externally linked content.

4. **Safety-first routing for danger signs.**  
   The system uses deterministic red-flag detection for urgent maternal and newborn symptoms, such as severe bleeding, chest pain, shortness of breath, severe headache with vision changes, suicidal ideation, newborn fever, poor feeding, convulsions, fast breathing, and severe jaundice signs.

5. **Refusal calibration for unsupported questions.**  
   The system detects questions that cannot be reliably answered from the corpus, such as brand recommendations, local pharmacy availability, exact local hospital policies, commercial product comparisons, unsupported herbal/essential-oil claims, and exact country-specific dosing.

6. **Structured evaluation across behavioral regimes.**  
   Instead of using a single mixed benchmark, the evaluation separates answerable QA, safety escalation, insufficient evidence, and adversarial attacks. This makes it possible to identify whether failures are due to retrieval, generation, safety routing, refusal calibration, or evaluator mismatch.

7. **Baseline-vs-governed comparison design.**  
   The project includes a controlled vanilla-RAG baseline using the same corpus, Chroma database, embedding model, LLM, and question sets. This allows the governance layer to be evaluated as an intervention rather than as an uncontrolled system change.

## Interview Explanation

A concise way to explain the project is:

> I built a privacy-preserving, fully local RAG system for maternal, postpartum, newborn, and infant health guidance. The research focus is not just retrieval or chatbot generation, but trustworthy RAG governance. I added lifecycle-aware metadata retrieval, publisher-aware source ranking, citation-grounded generation, deterministic safety escalation, insufficient-evidence refusal, JSON repair, and adversarial robustness checks. I evaluate the system separately on answerable questions, safety-critical questions, insufficient-evidence traps, and prompt-injection attacks, and I compare the governed system against a vanilla RAG baseline using the same corpus, vector database, embedding model, and local LLM.

A shorter version:

> My contribution is a metadata-governed, safety-aware local RAG framework for a high-risk health domain, with evaluation metrics for groundedness, hallucination, traceability, stage alignment, source authority, safety escalation, refusal accuracy, and adversarial robustness.

## Limitations and Next Steps

Current limitations include the need to rerun insufficient-evidence and adversarial benchmarks after the latest deterministic gates, to inspect the robustness evaluator’s `json_ok_rate` and `injection_resistance_rate`, and to complete the vanilla-RAG baseline comparison. The system is intended as a research prototype for trustworthy health-information retrieval and not as a clinical decision system.
