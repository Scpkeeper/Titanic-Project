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
7. Generated and validated the model, submissions, fold reports, configuration
   results, prediction-difference audit, manifest, and SHA-256 hashes.

## Full-run result

| Measure | Family/ticket signal | Same-split CatBoost baseline |
|---|---:|---:|
| Nested OOF accuracy | 0.8328 | 0.8384 |
| OOF F1 | 0.7639 | 0.7700 |
| OOF ROC-AUC | 0.8904 | 0.8871 |
| Group-disjoint accuracy | 0.8103 | 0.8283 |

The signal improves probability ranking slightly (ROC-AUC), but it does not
improve classification accuracy and is weaker under the group-disjoint test.
The promotion result is therefore `baseline_retained`. The recommended file is
byte-equivalent to the uploaded CatBoost baseline; the experimental signal file
changes 16 of 418 predictions and is retained for an optional Kaggle comparison.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python run_family_ticket.py --train source/train.csv --test source/test.csv `
  --output . --baseline-submission path\to\submission_catboost.csv
python -m unittest discover -s tests -v
```

## What to do next

1. Upload `outputs/submission_family_ticket_catboost.csv` to Kaggle as a single
   controlled experiment and record its public score. Do not call it a 0.83
   model before Kaggle verifies it.
2. Keep `outputs/submission_recommended.csv` as the safe current recommendation
   unless the experimental file beats `0.79186`.
3. If another iteration is warranted, test repeated nested CV and a compact
   probability blend between the baseline and signal model. Select blend weight
   entirely inside inner folds and keep the same promotion gate.
4. Stop model iteration if repeated CV does not show a stable gain. A guaranteed
   `0.83` from official data alone cannot be honestly promised from local CV.
