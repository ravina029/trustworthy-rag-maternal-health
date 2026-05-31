# Baseline vs Governed RAG Comparison: adversarial

Baseline log: `eval_runs_baseline/baseline_eval_run_20260529_132822.jsonl`

Final log: `eval_runs/eval_run_20260528_233300.jsonl`

| Metric | Baseline vanilla RAG | Final governed RAG |
|---|---:|---:|
| Total questions | 15 | 15 |
| OK | 15 | 14 |
| Safety escalations | 0 | 0 |
| Insufficient evidence | 0 | 1 |
| Exceptions | 0 | 0 |
| Generation errors | 0 | 0 |
| External link leaks | 0 | 0 |
| Expected-behavior accuracy | 100.0% | 100.0% |

## Interpretation

The baseline uses the same corpus, embeddings, Chroma collection, and LLM, but removes deterministic metadata/safety/refusal governance. The final system adds lifecycle-aware metadata handling, citation governance, safety escalation, insufficient-evidence refusal, and link suppression.

The final system maintains or improves external-link suppression.
