# CatBoost Family Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe, nested-selected probability blend of the baseline CatBoost and family/ticket CatBoost models, then publish its audited results.

**Architecture:** The experiment module remains the single orchestration point. A small blend evaluator consumes out-of-fold baseline and family probabilities, deterministically selects the best family weight, and the existing experiment runner writes blend artifacts plus a promotion decision against the baseline. The Kaggle scores are documentation-only observations.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, CatBoost, joblib, unittest, Git/GitHub.

## Global Constraints

- Use only `source/train.csv`, `source/test.csv`, and the `Survived` label in `source/train.csv` for model or blend selection.
- Evaluate family weights exactly `0.00`, `0.25`, `0.50`, `0.75`, and `1.00` at threshold `0.50`.
- Select by inner-fold mean accuracy; deterministic ties prefer lower family weight, then lower standard deviation.
- Promote only on a strict outer-OOF accuracy improvement over the baseline.
- Log public Kaggle scores `0.79186` and `0.78229` as audit observations only.

---

### Task 1: Deterministic blend-weight evaluation

**Files:**
- Modify: `src/family_ticket_experiment.py`
- Modify: `tests/test_family_ticket_experiment.py`

**Interfaces:**
- Produces: `_blend_weights() -> tuple[float, ...]` returning `(0.0, 0.25, 0.5, 0.75, 1.0)`.
- Produces: `_choose_blend_weight(results: list[dict[str, Any]]) -> float` where each result has `family_weight`, `mean_accuracy`, and `std_accuracy`.
- Consumes: validation labels and independently produced baseline/family probabilities.

- [ ] **Step 1: Write the failing tests**

```python
from src.family_ticket_experiment import _blend_weights, _choose_blend_weight

def test_blend_weights_are_fixed_and_include_both_controls(self) -> None:
    self.assertEqual(_blend_weights(), (0.0, 0.25, 0.5, 0.75, 1.0))

def test_blend_selection_prefers_lower_family_weight_on_accuracy_tie(self) -> None:
    results = [
        {"family_weight": 0.50, "mean_accuracy": 0.82, "std_accuracy": 0.01},
        {"family_weight": 0.25, "mean_accuracy": 0.82, "std_accuracy": 0.03},
    ]
    self.assertEqual(_choose_blend_weight(results), 0.25)
```

The production mutations these tests catch are omitting a required control
weight or selecting a more family-heavy blend when accuracy is tied.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_blend_weights_are_fixed_and_include_both_controls tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_blend_selection_prefers_lower_family_weight_on_accuracy_tie -v`

Expected: FAIL because the two helper functions are not defined.

- [ ] **Step 3: Write the minimal implementation**

```python
def _blend_weights() -> tuple[float, ...]:
    return (0.0, 0.25, 0.5, 0.75, 1.0)

def _choose_blend_weight(results: list[dict[str, Any]]) -> float:
    ordered = sorted(
        results,
        key=lambda item: (
            -round(item["mean_accuracy"], 12),
            item["family_weight"],
            round(item["std_accuracy"], 12),
        ),
    )
    return float(ordered[0]["family_weight"])
```

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_blend_weights_are_fixed_and_include_both_controls tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_blend_selection_prefers_lower_family_weight_on_accuracy_tie -v`

Expected: PASS.

- [ ] **Step 5: Commit the test-covered helper**

```powershell
git add src/family_ticket_experiment.py tests/test_family_ticket_experiment.py
git commit -m "feat: add deterministic blend selection"
```

### Task 2: Nested blend orchestration and audited output

**Files:**
- Modify: `src/family_ticket_experiment.py`
- Modify: `tests/test_family_ticket_experiment.py`

**Interfaces:**
- Consumes: `_blend_weights`, `_choose_blend_weight`, `_build_model`, `_apply_policy`, `validate_submission`.
- Produces: `outputs/submission_catboost_family_blend.csv`.
- Produces: manifest keys `blend`, `blend_promotion_status`, and output paths/hashes for the blend submission and `blend_configuration_results.csv`.

- [ ] **Step 1: Write the failing integration assertions**

```python
blend = pd.read_csv(output / "outputs" / "submission_catboost_family_blend.csv")
validate_submission(blend, test["PassengerId"])
self.assertIn("blend", manifest)
self.assertIn("selected_family_weight", manifest["blend"])
self.assertIn("blend_configuration_results", manifest["output_paths"])
self.assertIn(manifest["blend_promotion_status"], {"promoted", "baseline_retained"})
```

The production mutations these assertions catch are failing to create the
submission, failing to make its selection auditable, or bypassing the
promotion gate.

- [ ] **Step 2: Run the quick integration test to verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_quick_run_writes_complete_audit -v`

Expected: FAIL because the blend CSV and manifest keys do not exist.

- [ ] **Step 3: Implement nested blend evaluation**

Within each outer fold, generate inner-fold baseline/family validation
probabilities, measure every `_blend_weights()` candidate, select with
`_choose_blend_weight`, then fit both models on the outer training split and
write the selected blend probability to outer OOF. Repeat the same inner-fold
procedure on all training rows for the deployment family weight. Write the
deployment blend CSV with `PassengerId,Survived`, a CSV of blend candidate
metrics, selected weights and OOF comparison to the manifest, and a strict
baseline-vs-blend promotion decision.

- [ ] **Step 4: Run the quick integration test to verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest tests.test_family_ticket_experiment.FamilyTicketExperimentTests.test_quick_run_writes_complete_audit -v`

Expected: PASS with schema-valid family, baseline, recommended, and blend CSVs.

- [ ] **Step 5: Run experiment-specific tests and commit**

Run: `.venv\Scripts\python.exe -m unittest tests.test_family_ticket_experiment -v`

Expected: PASS.

```powershell
git add src/family_ticket_experiment.py tests/test_family_ticket_experiment.py
git commit -m "feat: add nested CatBoost family blend"
```

### Task 3: Score logging, full execution, verification, and publication

**Files:**
- Modify: `docs/EXPERIMENT_LOG.md`
- Modify: `README.md`
- Modify: `reports/family_ticket/*`
- Modify: `outputs/*`
- Modify: `models/*` only if the runner produces a changed serialized model

**Interfaces:**
- Consumes: `run_family_ticket.py --baseline-submission <uploaded CatBoost CSV>`.
- Produces: a full manifest with blend artifacts and a documented `0.78229`
  family-model Kaggle score.

- [ ] **Step 1: Record the observed leaderboard score without using it for tuning**

Add `0.78229` as the family/ticket submission public score and state that it is
`0.00957` below the baseline. Amend the next-action section: the blend is the
next controlled candidate, and its Kaggle score must be recorded separately.

- [ ] **Step 2: Run the full experiment**

Run:

```powershell
.venv\Scripts\python.exe run_family_ticket.py --train source\train.csv --test source\test.csv --output . --baseline-submission C:\Users\swp\Documents\xwechat_files\hmd70925_3b51\msg\file\2026-08\submission_catboost.csv
```

Expected: a blend CSV, manifest, overview, configuration audit, and promotion
decision generated from the full five-outer-fold evaluation.

- [ ] **Step 3: Verify all behavior and artifacts**

Run:

```powershell
.venv\Scripts\python.exe run_tests.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff origin/main...HEAD --check
```

Reload the deployment bundle, reproduce the blend prediction probabilities,
and verify SHA-256 hashes for every manifest output path before committing.

- [ ] **Step 4: Commit generated audited outputs and documentation**

```powershell
git add README.md docs/EXPERIMENT_LOG.md outputs reports/family_ticket models
git commit -m "docs: record CatBoost family blend results"
```

- [ ] **Step 5: Push and update the existing draft PR**

Run: `git push -u origin kaggle-083-family-signal`

Update GitHub pull request `Scpkeeper/Titanic-Project#2` with blend method,
full nested OOF metrics, the `0.78229` observed family score, the current
recommendation, artifact paths, and fresh test evidence.
