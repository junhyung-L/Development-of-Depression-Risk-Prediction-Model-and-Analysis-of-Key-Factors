# Student Depression-Label Exploration: Choosing Interpretability over Complexity

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

This project explores associations between a student-survey depression label and
lifestyle or academic variables. It is not a diagnostic model. Logistic
regression was chosen so predictive output could be examined alongside
coefficients and p-values.

## Data and approach

The included survey has 502 rows. After encoding mixed responses and reviewing
exploratory associations, the maintained model uses dietary habits,
suicidal-thought history, academic pressure, and financial stress. The
workflow uses an 80/20 split with seed 42, then runs scikit-learn logistic
regression for metrics and statsmodels Logit for inference output.

## Results

The retained rerun records 0.9010 accuracy and 0.90 weighted F1 on 101 holdout
records, with confusion matrix `[[42, 6], [4, 49]]`. All four retained Logit
features have p<0.001 in the training summary. These are survey-sample
associations, not causal or clinical findings.

## What this demonstrates

The project keeps feature selection, holdout metrics, and coefficient review in
one path rather than presenting a single opaque score. It also generates
correlation, response-distribution, and learning-curve views for context.

## Limitations

There is no repeated stratified validation, calibration, subgroup analysis, or
external validation. The work is exploratory only and must not inform diagnosis
or automated intervention.

## Evidence

- [`results/model_summary.txt`](results/model_summary.txt)
- [`src/train.py`](src/train.py)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Why an interpretable model is the reference point

Survey variables are sensitive and the sample is only a few hundred rows, so complexity is not automatically a virtue. The project retains four predictors—dietary habits, suicidal-thought history, academic pressure, and financial stress—and runs scikit-learn metrics beside statsmodels coefficients and p-values. It therefore separates “how often did this split classify labels correctly?” from “which variable combination was associated with that result?”

The retained confusion matrix, `[[42, 6], [4, 49]]`, means that 49 of 53 positive labels and 42 of 48 negative labels were correct on this one split. Weighted F1 of 0.90 is a support-weighted average; it is not a clinical sensitivity, a fairness result, or an individual risk statement. Coefficients reported at p<0.001 are associations inside this sample and selected specification, not causal evidence.

Learning curves, response distributions, and correlation exploration are kept alongside the model so the conclusion does not rest on one score. Repeated stratified validation, external data, measurement-equivalence checks, and clinical review would be required before any use beyond educational or exploratory analysis.
