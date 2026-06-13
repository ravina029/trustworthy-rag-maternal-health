# Baseline vs Governed RAG Comparison: insufficient

Baseline log: `results/experiments/baseline/eval_runs_baseline/baseline_eval_run_20260529_131758.jsonl`

Final log: `results/experiments/eval_run_20260528_232800.jsonl`

| Metric | Baseline vanilla RAG | Final governed RAG |
|---|---:|---:|
| Total questions | 12 | 12 |
| OK | 12 | 0 |
| Safety escalations | 0 | 0 |
| Insufficient evidence | 0 | 12 |
| Exceptions | 0 | 0 |
| Generation errors | 0 | 0 |
| External link leaks | 0 | 0 |
| Expected-behavior accuracy | 0.0% | 100.0% |

## Interpretation

The baseline uses the same corpus, embeddings, Chroma collection, and LLM, but removes deterministic metadata/safety/refusal governance. The final system adds lifecycle-aware metadata handling, citation governance, safety escalation, insufficient-evidence refusal, and link suppression.

The final governed RAG improves expected-behavior accuracy from 0.0% to 100.0%.

The final system maintains or improves external-link suppression.
