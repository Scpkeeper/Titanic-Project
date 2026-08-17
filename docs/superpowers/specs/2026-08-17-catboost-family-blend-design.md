# Leakage-Safe CatBoost/Family Blend Design

## Goal

Test whether a probability blend of the existing CatBoost baseline and the
cross-fitted family/ticket CatBoost model improves Kaggle Titanic performance
without using public-leaderboard feedback, external labels, or test outcomes
for selection.

## Evidence and boundary

- Uploaded baseline CatBoost score: `0.79186`.
- Uploaded family/ticket CatBoost score: `0.78229`.
- The family/ticket standalone submission is weaker on Kaggle and must not be
  promoted by itself.
- Only `source/train.csv`, `source/test.csv`, and the `Survived` labels in
  `source/train.csv` may influence model training, configuration selection, or
  blend weight selection.
- The two public scores are logged after the fact for audit; neither is used as
  an optimization objective.

## Design

### Candidate models and blend

For each training split, fit the existing raw-data CatBoost baseline and the
existing cross-fitted family/ticket CatBoost estimator with the same selected
family configuration. For every validation row, collect both probability
vectors and form:

`blend_probability = family_weight * family_probability + (1 - family_weight) * baseline_probability`

Evaluate family weights `0.00`, `0.25`, `0.50`, `0.75`, and `1.00` with a
decision threshold of `0.50`. A weight of `0.00` is the baseline control; a
weight of `1.00` is the family model control.

### Leakage-safe selection

The existing outer/inner stratified validation structure remains authoritative:

1. In each outer-training partition, fit both models only in each inner-training
   fold and gather inner validation probabilities.
2. Choose a blend weight by mean inner-fold accuracy, with deterministic ties
   resolved toward the smaller family weight, then smaller standard deviation.
3. Refit both models on the full outer-training partition, create the outer
   validation blend using only the selected weight, and record its out-of-fold
   probability and prediction.
4. Select the deployment blend weight using the same inner-fold procedure over
   all training rows.

This keeps a row's target separate from both its group-signal construction and
blend-weight selection.

### Outputs and promotion

The experiment writes:

- `outputs/submission_catboost_family_blend.csv` — the deployment blend.
- `reports/family_ticket/blend_configuration_results.csv` — inner and full
  training candidate metrics.
- blend probabilities, selected weights, metrics, and prediction differences in
  the run manifest and overview.

The promotion gate compares nested outer-fold blend accuracy with the same-split
baseline accuracy. It recommends the blend only on a strict improvement;
otherwise it retains `outputs/submission_recommended.csv` as the baseline.

## Validation

- Unit test candidate-weight selection and deterministic tie behavior.
- Quick integration test validates the blend submission contract, manifest
  fields, and recommendation artifact.
- Full experiment validates all outputs, hashes, serialization/reproducibility,
  and the existing pipeline test suite before publication.

## Non-goals

- No use of external Titanic data, labels, or public-leaderboard results for
  model tuning.
- No claim that a `0.83` Kaggle score has been achieved until Kaggle evaluates a
  newly uploaded submission.
