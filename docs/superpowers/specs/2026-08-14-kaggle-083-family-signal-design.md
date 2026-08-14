# Kaggle Titanic 0.83 Family/Ticket Signal Design

## Objective

Improve the current CatBoost Kaggle public score of `0.79186` toward the
project target of `0.83` using only the official Titanic competition
`train.csv`, `test.csv`, and the labels contained in `train.csv`.

The implementation must not use historical survivor lists, public answer
keys, manually discovered test labels, or PassengerId-specific corrections.
The `0.83` score is a target rather than a guaranteed outcome because the
hidden Kaggle labels are unavailable locally.

## Current Evidence

- Kaggle scores supplied by the project owner:
  - CatBoost: `0.79186`
  - ExtraTrees: `0.77990`
  - Logistic pipeline: `0.77511`
  - Random Forest: `0.77033`
- The uploaded `submission_catboost.csv` has the valid Kaggle contract:
  418 unique PassengerIds from 892 through 1309 and binary `Survived` values.
- The uploaded predictions match the local rank-boost CatBoost candidate for
  all 418 passengers.
- Existing CatBoost nested stratified OOF accuracy is approximately `0.8440`,
  while its Kaggle score is `0.79186`. This validation-to-leaderboard gap is
  evidence that ordinary model tuning alone is unlikely to be sufficient.
- Within the official data, 126 test passengers match a non-singleton
  surname/family-size key observed in training and 152 match a ticket observed
  in training. These are legitimate train-to-test relationships that are not
  explicitly converted into label-derived evidence by the current pipeline.

## Approved Scope

Implement Approach 1 only: CatBoost augmented with cross-fitted family and
ticket survival signals. Existing Logistic Regression, XGBoost, ExtraTrees,
and blend results remain comparison baselines; a new general-purpose ensemble
search is outside this change.

The canonical codebase is `Scpkeeper/Titanic-Project`. The local Member C
rank-boost delivery will be integrated into that repository without replacing
or weakening the existing Module B pipeline.

## Signal Definitions

### Stable group keys

- `FamilyKey`: normalized surname plus `FamilySize`, where
  `FamilySize = SibSp + Parch + 1`.
- Singleton families do not share family evidence, even when surnames match.
- `TicketKey`: whitespace-normalized, uppercase ticket value.
- Missing or unusable keys receive no group evidence and fall back to priors.

### Label-derived features

For both family and ticket groups, the transformer produces:

- observed training support;
- smoothed survival probability;
- distance from the relevant prior;
- confidence derived from support;
- an availability flag.

The smoothed probability is calculated only from the labels made available to
the group-map builder. Smoothing strength, minimum support, and confidence
gates are selected from a small predefined inner-validation grid. Priors use
training-only demographic context, with global survival as the final fallback.

### Cross-fitting contract

Training rows must never receive group survival features calculated from their
own labels. Each estimator fit creates internal stratified folds:

1. Build family and ticket maps from the internal training portion.
2. Transform only the held-out portion with those maps.
3. Concatenate all held-out results into one complete cross-fitted feature
   matrix.
4. Train CatBoost on that matrix.
5. Rebuild group maps from the complete estimator training set for inference
   on its validation fold or on Kaggle `test.csv`.

This contract applies independently inside every outer validation fold.

## Model Architecture

Create a focused group-signal module rather than adding target-aware behavior
to the existing target-independent feature transformer.

- The existing rank feature engineer continues to own raw-data validation,
  missing-value handling, and non-target-derived features.
- A group-map builder owns normalized keys, training-only aggregation,
  smoothing, and inference-time fallbacks.
- A cross-fitted CatBoost estimator owns internal fold generation, assembly of
  target-independent and group-signal features, model fitting, and probability
  prediction.
- A candidate policy module may apply only predefined, confidence-gated family
  or ticket corrections. Every correction must be derived from training labels
  through the same cross-fitting contract and must be evaluated as a separate
  candidate, never silently applied.

The saved model bundle contains the fitted base feature engineer, full-training
group maps, CatBoost model, decision threshold, signal configuration, source
hashes, package versions, and random seeds needed for deterministic inference.

## Validation and Selection

### Primary deployment-aligned evaluation

Use repeated stratified outer cross-validation. For every outer fold, all
target-derived group signals for validation rows come only from outer-training
labels. Hyperparameters and any confidence-gated correction policy are chosen
only with inner cross-validation.

Report fold-level and aggregate accuracy, F1, ROC-AUC, prediction rate, and
the number of rows receiving family or ticket evidence. Compare the new model
against CatBoost generated on the exact same splits.

### Secondary robustness evaluation

Retain family/ticket-disjoint grouped cross-validation as a robustness monitor.
It answers how the model behaves for entirely unseen groups, not how well it
can exploit legitimate train-to-test group overlap. Its score must be reported
separately and must not be presented as directly interchangeable with the
deployment-aligned score.

### Candidate selection rules

- Use a small, declared grid for smoothing, minimum support, confidence gates,
  CatBoost regularization, and the binary decision threshold.
- Select the configuration using inner-fold mean accuracy, then lower standard
  deviation, then fewer corrected rows as deterministic tie-breakers.
- Do not choose individual passenger changes from Kaggle feedback.
- Keep the current CatBoost submission as the immutable baseline candidate.

## Experiment Logging

Every formal run writes:

- a machine-readable run manifest with timestamp, source hashes, code commit,
  environment versions, seeds, and configuration;
- fold-level metrics and evidence coverage;
- aggregate candidate comparison;
- OOF probabilities and predictions;
- a prediction-difference audit against the uploaded CatBoost baseline;
- final submission hashes and Kaggle format checks;
- a concise Markdown experiment report explaining what changed and why.

The repository will include an append-only experiment index. A new run may add
an entry but must not rewrite previous measured results.

## Outputs

The formal run produces:

- `outputs/submission_family_ticket_catboost.csv` as the primary submission;
- `outputs/submission_catboost_baseline.csv` as the reproduced baseline;
- an optional conservative candidate only if it wins the predefined inner-CV
  selection policy;
- a model bundle under `models/`;
- detailed reports under `reports/family_ticket/`.

All submission files must contain exactly `PassengerId,Survived`, 418 unique
passengers, binary integer predictions, no missing values, and the official
test PassengerId set.

## Automated Tests

Tests must prove:

- family and ticket key normalization is deterministic;
- singleton surnames do not share family evidence;
- a row's own label cannot affect its cross-fitted features;
- validation labels cannot affect outer-fold features or predictions;
- unknown groups use the documented fallback;
- every training row receives exactly one cross-fitted feature row;
- repeated runs with the same seed reproduce probabilities and submissions;
- serialized inference reproduces the saved submission;
- experiment manifests contain the required audit fields;
- output submissions satisfy the Kaggle contract.

Implementation follows test-driven development: each new behavior begins with
a failing test, followed by the smallest implementation that passes it.

## GitHub Delivery

Development occurs on `kaggle-083-family-signal`. Only files belonging
to this improvement are staged. After fresh verification, the branch is pushed
to `Scpkeeper/Titanic-Project` and opened as a draft pull request describing
the method, evidence, limitations, generated submission, and validation run.

## Completion Criteria

The change is complete when:

- all existing and new tests pass in a clean environment;
- a full deterministic run completes from the official raw Kaggle files;
- the primary deployment-aligned OOF result is no worse than the same-split
  CatBoost baseline and the selected candidate is supported across folds;
- robustness metrics and the validation-to-leaderboard limitation are clearly
  documented;
- the final CSV and experiment reports are committed and available through a
  draft GitHub pull request.

Actual achievement of `0.83` can be confirmed only after the project owner
uploads the generated primary submission to Kaggle and records its score.
