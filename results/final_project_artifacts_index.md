# Final Project Artifacts Index

Project: Trustworthy Maternal/Postpartum RAG  
Governed RAG commit: `d69eb68`

## Main Evaluation Reports

| Artifact | Path |
|---|---|
| Evaluation manifest | results/evaluation_run_manifest.md |
| Final baseline-vs-governed summary | results/final_baseline_vs_governed_summary.md |
| Representative baseline-vs-governed examples | results/baseline_vs_governed_representative_examples.md |
| Governed run paths | results/governed_final_run_paths.md |
| Method and contribution section | results/method_and_contribution_section.md |

## Governed RAG Runs

| Category | Run path |
|---|---|
| Answerable | eval_runs/eval_run_20260528_215408.jsonl |
| Safety | eval_runs/eval_run_20260528_230900.jsonl |
| Insufficient evidence | eval_runs/eval_run_20260528_232800.jsonl |
| Adversarial | eval_runs/eval_run_20260528_233300.jsonl |

## Vanilla Baseline Runs

| Category | Run path |
|---|---|
| Answerable | eval_runs_baseline/baseline_eval_run_20260529_070419.jsonl |
| Safety | eval_runs_baseline/baseline_eval_run_20260529_074357.jsonl |
| Insufficient evidence | eval_runs_baseline/baseline_eval_run_20260529_131758.jsonl |
| Adversarial | eval_runs_baseline/baseline_eval_run_20260529_132822.jsonl |

## Main Result

Vanilla RAG performs well on ordinary answerable questions, but fails safety and refusal behavior.  
Governed RAG improves:

- Safety expected-behavior accuracy: 0.0% → 100.0%
- Insufficient-evidence expected-behavior accuracy: 0.0% → 100.0%
- Exceptions: 0
- Generation errors: 0
- External link leakage: 0.0

## Interpretation

The governance layer mainly improves behavioral reliability, not ordinary answerability. It prevents the system from treating danger-sign prompts and unsupported product/local-policy questions as ordinary answerable QA.
