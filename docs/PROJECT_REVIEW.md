# Project Review

| Area | Assessment | Evidence and caveat |
|---|---:|---|
| Problem framing | 7/10 | Clear exploratory label-prediction task; clinical use must be excluded. |
| Modelling | 7/10 | Interpretable logistic path and statsmodels summary are implemented. |
| Evaluation | 5/10 | A rerun records 0.9010 accuracy and 0.90 weighted F1 on 101 holdout records (`results/model_summary.txt`), but uses one non-stratified seeded split with no repeated splits, calibration, or external validation. |
| Reproducibility | 6/10 | Data is included and paths/defaults are centralized; dependencies are not pinned. |
| Overall | 6.3/10 | Useful transparent prototype with high-stakes validation limits. |

## Priorities

1. Add stratified repeated validation and confidence intervals.
2. Report class prevalence, ROC/PR metrics, calibration, and subgroup checks.
3. Establish an ethics/privacy review before any use beyond exploratory work.
