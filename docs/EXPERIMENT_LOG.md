# Official-Kaggle-Data 0.83 Experiment Log

## Scope

- Data: only `source/train.csv`, `source/test.csv`, and the training labels in
  the official Kaggle Titanic files.
- Comparison submission: the uploaded CatBoost CSV with public score `0.79186`.
- Excluded: external labels, third-party Titanic datasets, and hand-entered test
  outcomes.
- Reproducibility: Python/package versions, source hashes, output hashes, seed,
  fold metrics, and selected configuration are recorded in
  `reports/family_ticket/run_manifest.json`.

## Work completed

1. Audited the existing preprocessing pipeline and model submissions.
2. Added deterministic surname-family and ticket group keys.
3. Added cross-fitted group survival evidence so a passenger's own target is
   never used to construct that passenger's training feature.
4. Wrapped CatBoost in a raw-data estimator that repeats feature engineering
   inside each validation fold.
5. Added nested threshold/policy selection and an independent group-disjoint
   robustness check.
6. Added a promotion gate: the experiment can replace the existing submission
   only when nested OOF accuracy is strictly better than the baseline.
7. Added fixed-weight inner-fold blend selection and evaluated the selected
   weight independently in each outer fold.
8. Generated and validated the model, family, baseline, blend, and recommended
   submissions, fold reports, configuration results, prediction-difference
   audit, manifest, and SHA-256 hashes.

## Full-run result

| Measure | Family/ticket signal | Same-split CatBoost baseline | Nested blend |
|---|---:|---:|---:|
| Nested OOF accuracy | 0.8328 | 0.8384 | 0.8406 |
| OOF F1 | 0.7639 | 0.7700 | 0.7753 |
| OOF ROC-AUC | 0.8904 | 0.8871 | 0.8884 |
| Group-disjoint accuracy | 0.8103 | 0.8283 | not evaluated |

The signal improves probability ranking slightly (ROC-AUC), but it does not
improve classification accuracy and is weaker under the group-disjoint test.
Its family-only promotion result is therefore `baseline_retained`. The nested
blend improves OOF accuracy over the same-split baseline and its promotion
result is `promoted` under the strict accuracy-only gate. Full-data inner
selection chose family weight `0.00`, so the deployment candidate uses the
freshly fitted same-split baseline probabilities at threshold `0.50`; it is not
a family-probability mixture. The recommended file is byte-equivalent to
`outputs/submission_catboost_family_blend.csv` and changes 15 of 418 predictions
from the uploaded baseline. The family-only file changes 16 predictions.

The family/ticket submission later received a public Kaggle score of `0.78229`,
which is `0.00957` below the uploaded CatBoost baseline score of `0.79186`.
This leaderboard score is a post-run audit observation only: it was not used to
tune features, thresholds, policies, or blend weight, and it does not alter the
strict nested-OOF promotion gate.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python run_family_ticket.py --train source/train.csv --test source/test.csv `
  --output . --baseline-submission path\to\submission_catboost.csv
python -m unittest discover -s tests -v
```

## What to do next

1. Upload the family/CatBoost probability blend as the next single controlled
   Kaggle candidate and record its public score separately from the family-only
   score of `0.78229`. Do not use either public score for model selection.
2. Keep `outputs/submission_recommended.csv` as the safe current recommendation
   unless the blend passes the strict nested-OOF promotion gate.
3. Stop model iteration if repeated CV does not show a stable gain. A guaranteed
   `0.83` from official data alone cannot be honestly promised from local CV.
