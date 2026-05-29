# Evaluation Run Manifest

Project: Trustworthy Maternal/Postpartum RAG  
Governed RAG commit: `d69eb68`  
Governed snapshot time: Thu May 28 21:30:09 IDT 2026  
Python/platform metadata: `results/run_metadata/system_info_governed.txt`

## Main Evaluation Runs

| System | Question file | Run path | Total | OK | Safety | Insufficient | Exceptions | GenErr | Traceability | Groundedness_OK | LinkLeak | Expected behavior | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Governed RAG | eval_questions_answerable_core.txt | eval_runs/eval_run_20260528_215408.jsonl | 30 | 29 | 1 | 0 | 0 | 0 | 1.0 | 0.978 | 0.0 | OK answers | Passed; one safety escalation for baby rash + fever is clinically conservative |
| Governed RAG | eval_questions_safety_curated.txt | eval_runs/eval_run_20260528_230900.jsonl | 15 | 0 | 15 | 0 | 0 | 0 | 1.0 | N/A | 0.0 | Safety escalation | Passed; 15/15 safety escalation |
| Governed RAG | eval_questions_insufficient_evidence_curated.txt | eval_runs/eval_run_20260528_232800.jsonl | 12 | 0 | 0 | 12 | 0 | 0 | 1.0 | N/A | 0.0 | Insufficient evidence | Passed; 12/12 refusal behavior |
| Governed RAG | eval_questions_adversarial_curated.txt | eval_runs/eval_run_20260528_233300.jsonl | 15 | 14 | 0 | 1 | 0 | 0 | 1.0 | 1.000 | 0.0 | Evidence-bound / no links | Passed; no link leakage, no exceptions, evidence-bounded |
| Vanilla RAG | eval_questions_answerable_core.txt | pending |  |  |  |  |  |  |  |  |  | OK answers | Pending |
| Vanilla RAG | eval_questions_safety_curated.txt | pending |  |  |  |  |  |  |  |  |  | Safety escalation | Pending |
| Vanilla RAG | eval_questions_insufficient_evidence_curated.txt | pending |  |  |  |  |  |  |  |  |  | Insufficient evidence | Pending |
| Vanilla RAG | eval_questions_adversarial_curated.txt | pending |  |  |  |  |  |  |  |  |  | Evidence-bound / no links | Pending |

## Metadata-Governance Metrics

| System | Question file | Run path | Mean distinct publishers | Publisher diversity rate | Mean stage alignment | Stage-alignment pass rate | Core-authoritative-source rate | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---|
| Governed RAG | eval_questions_answerable_core.txt | eval_runs/eval_run_20260528_215408.jsonl | 1.63 | 0.600 | 0.821 | 0.967 | 0.933 | Strong stage alignment and source authority |
| Governed RAG | eval_questions_safety_curated.txt | eval_runs/eval_run_20260528_230900.jsonl | 2.13 | 0.733 | 0.917 | 1.000 | 1.000 | Strong metadata governance during safety escalation |
| Governed RAG | eval_questions_insufficient_evidence_curated.txt | eval_runs/eval_run_20260528_232800.jsonl | 2.00 | 0.833 | 0.906 | 1.000 | 1.000 | Strong metadata governance despite refusal behavior |
| Governed RAG | eval_questions_adversarial_curated.txt | eval_runs/eval_run_20260528_233300.jsonl | 1.00 | 0.000 | 1.000 | 1.000 | 0.933 | Stage alignment strong; publisher diversity low under attack prompts |
| Vanilla RAG | eval_questions_answerable_core.txt | pending |  |  |  |  |  | Pending |
| Vanilla RAG | eval_questions_safety_curated.txt | pending |  |  |  |  |  | Pending |
| Vanilla RAG | eval_questions_insufficient_evidence_curated.txt | pending |  |  |  |  |  | Pending |
| Vanilla RAG | eval_questions_adversarial_curated.txt | pending |  |  |  |  |  | Pending |

## Category-Aware Interpretation Rules

| Question file | Expected dominant status | Main metric |
|---|---|---|
| eval_questions_answerable_core.txt | ok | OK rate, groundedness, traceability, citation hygiene |
| eval_questions_safety_curated.txt | safety_escalation | safety_escalation_behavior_rate |
| eval_questions_insufficient_evidence_curated.txt | insufficient_evidence | refusal accuracy |
| eval_questions_adversarial_curated.txt | ok or insufficient_evidence | no external links, evidence-boundedness, no unsupported claims |

## Notes

- Generic scorecard warnings such as `too many refusals` are not treated as failures for safety-only or insufficient-evidence-only benchmarks.
- Safety and insufficient-evidence templates are intentionally not normal evidence-grounded answers, so `Groundedness_ALL` may be low for those sets even when behavior is correct.
- `json_ok_rate` and `injection_resistance_rate` still appear unreliable because the evaluator checks log-level JSON compliance rather than the normalized final response.
- Baseline comparison must use the same corpus, Chroma index, embedding model, local LLM, and question files.

## Completed Vanilla Baseline Runs

| System | Question file | Run path | Total | OK | Safety | Insufficient | Exceptions | GenErr | Expected behavior accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla RAG | eval_questions_answerable_core.txt | eval_runs_baseline/baseline_eval_run_20260529_070419.jsonl | 30 | 30 | 0 | 0 | 0 | 0 | 100.0% |
| Vanilla RAG | eval_questions_safety_curated.txt | eval_runs_baseline/baseline_eval_run_20260529_074357.jsonl | 15 | 15 | 0 | 0 | 0 | 0 | 0.0% |
| Vanilla RAG | eval_questions_insufficient_evidence_curated.txt | eval_runs_baseline/baseline_eval_run_20260529_131758.jsonl | 12 | 12 | 0 | 0 | 0 | 0 | 0.0% |
| Vanilla RAG | eval_questions_adversarial_curated.txt | eval_runs_baseline/baseline_eval_run_20260529_132822.jsonl | 15 | 15 | 0 | 0 | 0 | 0 | 100.0% |

## Completed Comparison Files

| Comparison | File |
|---|---|
| Answerable baseline vs governed | results/baseline_vs_final_answerable.md |
| Safety baseline vs governed | results/baseline_vs_final_safety.md |
| Insufficient-evidence baseline vs governed | results/baseline_vs_final_insufficient.md |
| Adversarial baseline vs governed | results/baseline_vs_final_adversarial.md |
| Final summary | results/final_baseline_vs_governed_summary.md |

## Representative Baseline-vs-Governed Examples

| Example file | Purpose |
|---|---|
| results/baseline_vs_governed_representative_examples.md | Side-by-side vanilla vs governed examples for safety escalation and insufficient-evidence refusal |
