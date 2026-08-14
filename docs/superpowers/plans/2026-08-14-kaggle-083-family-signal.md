# Kaggle Titanic Family/Ticket Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reproducible CatBoost submission that uses only official Kaggle Titanic data and leak-safe, cross-fitted family/ticket survival evidence.

**Architecture:** A target-aware group-signal module derives normalized family and ticket maps from training labels. A scikit-learn-compatible CatBoost estimator cross-fits those signals for its own training rows and uses full-training maps only for inference. A separate experiment runner performs nested selection, compares the new estimator with a same-split CatBoost baseline, writes audit reports, and creates the final Kaggle CSV.

**Tech Stack:** Python 3.12+, pandas 3.0.5, scikit-learn 1.9.0, CatBoost 1.2.10, joblib 1.5.3, unittest, Git/GitHub.

## Global Constraints

- Use only `source/train.csv`, `source/test.csv`, and `Survived` labels from `source/train.csv`.
- Do not use historical survivor lists, public answer keys, Kaggle-derived per-passenger corrections, or PassengerId as a predictive feature.
- Random seeds are explicit and default to `42`.
- Every target-derived feature for a training row is produced without that row's label.
- The existing Module B pipeline and tests remain intact.
- `0.83` is a Kaggle target, not a locally provable guarantee.

---

### Task 1: Group survival evidence

**Files:**
- Create: `src/family_ticket_signal.py`
- Create: `tests/test_family_ticket_signal.py`

**Interfaces:**
- Produces: `normalize_group_keys(frame: pd.DataFrame) -> pd.DataFrame`
- Produces: `GroupSignalMap(smoothing: float, min_support: int).fit(X, y) -> GroupSignalMap`
- Produces: `GroupSignalMap.transform(X) -> pd.DataFrame`
- Produces: `cross_fit_group_signals(X, y, n_splits, random_state, smoothing, min_support) -> pd.DataFrame`

- [ ] **Step 1: Write failing key and fallback tests**

Add literal fixtures proving that surnames and tickets normalize deterministically, singleton families receive unique non-sharing keys, known groups receive support/rate/confidence, and unknown groups receive prior probability with availability `0`.

- [ ] **Step 2: Run the signal tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_signal`

Expected: import failure because `src.family_ticket_signal` does not exist.

- [ ] **Step 3: Implement normalized keys and fitted group maps**

Use uppercase whitespace-normalized tickets and `SURNAME_FAMILYSIZE` only when family size exceeds one. Fit family and ticket count/sum maps from the provided labels. Compute smoothed probability as `(survivors + smoothing * prior) / (support + smoothing)` and return explicit numeric support, probability, delta, confidence, and availability columns.

- [ ] **Step 4: Run the signal tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_signal`

Expected: all key and map tests pass.

- [ ] **Step 5: Write the failing own-label isolation test**

Create a fixture with repeated family/ticket keys. Generate cross-fitted features twice after flipping one row's own label. Assert that row's features are byte-equivalent while at least one peer's training-derived feature changes.

- [ ] **Step 6: Run the isolation test and verify RED**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_signal.GroupSignalTests.test_cross_fit_excludes_own_label`

Expected: failure because `cross_fit_group_signals` is absent.

- [ ] **Step 7: Implement deterministic cross-fitting**

Use `StratifiedKFold(shuffle=True, random_state=random_state)`. Fit a fresh `GroupSignalMap` on each training fold, transform only its held-out fold, preserve original index/order, and assert complete one-time coverage.

- [ ] **Step 8: Run all signal tests**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_signal`

Expected: all signal tests pass.

- [ ] **Step 9: Commit Task 1**

Run:

```powershell
git add src/family_ticket_signal.py tests/test_family_ticket_signal.py
git commit -m "Add cross-fitted family-ticket signals"
```

### Task 2: Cross-fitted CatBoost estimator

**Files:**
- Create: `src/family_ticket_model.py`
- Create: `tests/test_family_ticket_model.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `cross_fit_group_signals` and `GroupSignalMap`
- Consumes: `TitanicFeatureEngineer` and `RAW_FEATURE_COLUMNS` from `titanic_b_pipeline.py`
- Produces: `FamilyTicketCatBoostClassifier(BaseEstimator, ClassifierMixin)` with `fit`, `predict_proba`, and `predict`
- Produces: `make_catboost_baseline(**params) -> FamilyTicketCatBoostClassifier` with group signals disabled

- [ ] **Step 1: Add CatBoost to the declared environment**

Add `catboost==1.2.10` to `requirements.txt` while preserving existing pins.

- [ ] **Step 2: Write failing estimator contract tests**

Test that the estimator fits raw Kaggle columns, returns finite two-column probabilities summing to one, emits binary predictions, never exposes PassengerId as a model feature, and is deterministic for a fixed seed. Use a stratified 180-row fixture to keep the test fast.

- [ ] **Step 3: Run estimator tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_model`

Expected: import failure because `src.family_ticket_model` does not exist.

- [ ] **Step 4: Implement the minimal estimator**

Fit `TitanicFeatureEngineer` on the estimator's complete training subset, build the target-independent clean matrix, append cross-fitted group signals, cast declared categorical columns to strings, and fit CatBoost with writing disabled and one thread. For inference, transform with the fitted feature engineer and a full-training `GroupSignalMap`. Add `use_group_signals=False` for the exact same-feature baseline.

- [ ] **Step 5: Run estimator tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_model`

Expected: all estimator tests pass.

- [ ] **Step 6: Add serialization reproduction test**

Fit on the fixture, save with joblib into a temporary directory, reload, and assert identical probabilities and predictions for held-out raw rows.

- [ ] **Step 7: Run estimator and legacy tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_model
.venv\Scripts\python.exe run_tests.py
```

Expected: new estimator tests and all 10 Module B tests pass.

- [ ] **Step 8: Commit Task 2**

Run:

```powershell
git add requirements.txt src/family_ticket_model.py tests/test_family_ticket_model.py
git commit -m "Add family-ticket CatBoost estimator"
```

### Task 3: Nested experiment selection and audit trail

**Files:**
- Create: `src/family_ticket_experiment.py`
- Create: `run_family_ticket.py`
- Create: `tests/test_family_ticket_experiment.py`

**Interfaces:**
- Consumes: `FamilyTicketCatBoostClassifier`
- Produces: `validate_submission(frame, expected_ids) -> None`
- Produces: `run_experiment(train_path, test_path, output_root, baseline_submission_path=None, quick=False) -> dict`
- CLI: `python run_family_ticket.py --train source/train.csv --test source/test.csv --output . --baseline-submission <path>`

- [ ] **Step 1: Write failing submission and manifest tests**

Test valid 418-row submissions, reject duplicate/missing IDs and non-binary predictions, and require manifest fields for source hashes, commit, versions, seeds, configuration, fold metrics, evidence coverage, output paths, and output hashes.

- [ ] **Step 2: Run experiment tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_experiment`

Expected: import failure because `src.family_ticket_experiment` does not exist.

- [ ] **Step 3: Implement validation, hashing, and reporting helpers**

Keep file/hash/report functions separate from fitting. JSON output must be UTF-8, stable-key ordered where practical, and contain only JSON-native values.

- [ ] **Step 4: Run helper tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_experiment`

Expected: helper tests pass while the quick integration test remains absent.

- [ ] **Step 5: Write failing quick-run integration test**

Run a 3-outer/2-inner quick experiment with reduced CatBoost iterations into a temporary directory. Assert complete OOF coverage, same-split baseline comparison, a saved loadable model, baseline difference audit, append-only index entry, and valid primary submission.

- [ ] **Step 6: Run quick integration test and verify RED**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_quick_run_writes_complete_audit`

Expected: failure because `run_experiment` has no orchestration behavior.

- [ ] **Step 7: Implement nested selection and formal outputs**

Evaluate a declared small grid of smoothing/minimum-support configurations inside each outer training fold. Compare the winning signal model with `use_group_signals=False` on identical outer folds. Select the deployment configuration on full-training inner CV using mean accuracy, standard deviation, and coverage tie-breakers. Fit the deployment model, create predictions, validate them, save the bundle, and write all reports from the approved design.

- [ ] **Step 8: Add the CLI entry point**

Expose `--quick`, `--train`, `--test`, `--output`, and optional `--baseline-submission`. Print the primary output path, same-split OOF comparison, selected configuration, number of changes from the uploaded baseline, and an explicit statement that Kaggle `0.83` is unverified until submission.

- [ ] **Step 9: Run all experiment tests**

Run: `.venv\Scripts\python.exe -m unittest -v tests.test_family_ticket_experiment`

Expected: all experiment tests pass.

- [ ] **Step 10: Commit Task 3**

Run:

```powershell
git add src/family_ticket_experiment.py run_family_ticket.py tests/test_family_ticket_experiment.py
git commit -m "Add audited family-ticket experiment"
```

### Task 4: Full run, documentation, and generated artifacts

**Files:**
- Modify: `README.md`
- Modify: `TASKS.md`
- Create: `reports/family_ticket/experiment_index.jsonl`
- Create: `reports/family_ticket/run_manifest.json`
- Create: `reports/family_ticket/model_comparison.csv`
- Create: `reports/family_ticket/outer_fold_metrics.csv`
- Create: `reports/family_ticket/oof_predictions.csv`
- Create: `reports/family_ticket/submission_difference.csv`
- Create: `reports/family_ticket/overview.md`
- Create: `outputs/submission_family_ticket_catboost.csv`
- Create: `outputs/submission_catboost_baseline.csv`
- Create: `models/family_ticket_catboost.joblib`

**Interfaces:**
- Consumes: the uploaded CatBoost submission as an audit baseline only
- Produces: the final user-facing submission and overview

- [ ] **Step 1: Run the full formal experiment**

Run:

```powershell
.venv\Scripts\python.exe run_family_ticket.py `
  --train source/train.csv `
  --test source/test.csv `
  --output . `
  --baseline-submission "C:\Users\swp\Documents\xwechat_files\hmd70925_3b51\msg\file\2026-08\submission_catboost.csv"
```

Expected: exit `0`, complete reports, serialized model, and two valid submissions.

- [ ] **Step 2: Inspect the measured result before documenting it**

Read the complete manifest and comparison table. Report measured values exactly; do not claim Kaggle `0.83` from local CV.

- [ ] **Step 3: Update README and TASKS**

Document the official-data-only method, exact reproduction command, current verified Kaggle baseline `0.79186`, new offline results, output locations, and the remaining action of uploading the primary CSV to Kaggle.

- [ ] **Step 4: Verify generated artifact contracts**

Run the submission validator against both output files and load the model bundle to reproduce the primary predictions exactly.

- [ ] **Step 5: Commit Task 4**

Stage only the documented source, tests, reports, model, and output files, then commit:

```powershell
git commit -m "Generate family-ticket CatBoost submission"
```

### Task 5: Final verification and GitHub publication

**Files:**
- Verify all files changed on `kaggle-083-family-signal`

- [ ] **Step 1: Run the complete verification suite**

Run:

```powershell
.venv\Scripts\python.exe run_tests.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe run_family_ticket.py --help
git diff origin/main...HEAD --check
git status --short
```

Expected: all tests pass, CLI help exits `0`, no whitespace errors, and only intended generated state remains.

- [ ] **Step 2: Audit requirements against the design**

Confirm official-data-only provenance, own-label isolation, outer-fold isolation, deterministic inference, complete logging, valid submissions, and accurate limitation language.

- [ ] **Step 3: Push the renamed branch**

Run: `git push -u origin kaggle-083-family-signal`

- [ ] **Step 4: Open a draft pull request**

Target `Scpkeeper/Titanic-Project:main`. The body must state what changed, why family/ticket evidence is legitimate, measured offline results, CatBoost Kaggle baseline `0.79186`, test evidence, final submission path, and that the `0.83` target requires a Kaggle upload to verify.
