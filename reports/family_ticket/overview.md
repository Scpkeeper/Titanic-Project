# Family/Ticket CatBoost Experiment

This run uses only the official Kaggle Titanic train/test files and training labels.

- Same-split signal OOF accuracy: `0.8328`
- Same-split CatBoost baseline OOF accuracy: `0.8384`
- Group-disjoint signal accuracy: `0.8103`
- Group-disjoint baseline accuracy: `0.8283`
- Uploaded CatBoost Kaggle baseline: `0.79186`
- Test predictions changed from uploaded baseline: `16`
- Selected configuration: `{"decision_threshold": 0.5, "min_support": 2, "policy": "none", "smoothing": 2.0}`
- Promotion decision: `baseline_retained`
- Recommended Kaggle file: `outputs/submission_recommended.csv`

The 0.83 Kaggle target remains unverified until the recommended CSV is uploaded.
