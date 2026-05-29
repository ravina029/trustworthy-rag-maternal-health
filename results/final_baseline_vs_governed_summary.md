# Final Baseline vs Governed RAG Summary

Governed RAG commit: `d69eb68`

## Main Comparison Table

| Evaluation set | Expected behavior | Vanilla RAG | Governed RAG | Main finding |
|---|---|---:|---:|---|
| Answerable core | Normal `ok` answers | 30/30 expected behavior, 100.0% | 29/30 expected behavior, 96.7% | Both perform well; governed RAG conservatively escalated one potentially risky baby rash + fever question |
| Safety | `safety_escalation` | 0/15, 0.0% | 15/15, 100.0% | Governed safety gate corrected complete baseline under-escalation |
| Insufficient evidence | `insufficient_evidence` | 0/12, 0.0% | 12/12, 100.0% | Governed refusal gate corrected complete baseline over-answering |
| Adversarial | Evidence-bound / no links / no crash | 15/15, 100.0% | 15/15, 100.0% | Both remained link-safe in this comparison; governed RAG added one conservative refusal |

## Detailed Status Counts

| Set | System | Total | OK | Safety | Insufficient | Exceptions | GenErr | Link leaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Answerable | Vanilla | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| Answerable | Governed | 30 | 29 | 1 | 0 | 0 | 0 | 0 |
| Safety | Vanilla | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| Safety | Governed | 15 | 0 | 15 | 0 | 0 | 0 | 0 |
| Insufficient | Vanilla | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| Insufficient | Governed | 12 | 0 | 0 | 12 | 0 | 0 | 0 |
| Adversarial | Vanilla | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| Adversarial | Governed | 15 | 14 | 0 | 1 | 0 | 0 | 0 |

## Interpretation

The vanilla RAG baseline uses the same document corpus, Chroma vector database, embedding model, local LLM, and question files, but removes the deterministic governance layer. It performs well on ordinary answerable questions, but it over-answers safety-critical and insufficient-evidence questions.

The governed RAG system adds lifecycle-aware metadata handling, citation governance, deterministic safety escalation, deterministic insufficient-evidence refusal, JSON repair/fallback, and external-link suppression. The largest reliability gains are seen in the safety and insufficient-evidence benchmarks.

## Main Research Finding

The governance layer does not mainly improve ordinary answerability; it improves **behavioral reliability** in safety-sensitive and out-of-scope scenarios. In particular, it prevents vanilla RAG from treating emergency symptoms and unsupported product/local-policy questions as ordinary answerable QA.

## Interview Summary

A concise interview explanation:

> I compared vanilla RAG and governed RAG under controlled conditions using the same PDFs, Chroma index, embedding model, local LLM, and question sets. Vanilla RAG performed well on ordinary answerable questions, but it failed safety and refusal behavior: it answered 15/15 safety-danger questions as normal OK responses and answered 12/12 insufficient-evidence traps as OK. The governed version corrected this using deterministic safety escalation and insufficient-evidence gates, achieving 15/15 safety escalation and 12/12 refusal accuracy, with zero exceptions and zero generation errors.

## Important Caveat

The current vanilla baseline is still somewhat conservative because the baseline script asks for JSON and strips links during normalization. Therefore, the adversarial link-leak comparison is not a fully raw LLM baseline. The safety and insufficient-evidence comparisons remain highly informative because the baseline intentionally lacks deterministic safety and refusal governance.
