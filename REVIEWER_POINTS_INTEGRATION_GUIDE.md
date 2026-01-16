# Complete Guide: Integrating Reviewer Points with ML Pipeline

## Overview

Your boss is correct: the reviewer point cells need to work **WITH** the ML pipeline, not independently. This guide shows how to modify each reviewer point cell.

## What Changed from Original Implementation

| Aspect | Original | Updated |
|--------|----------|---------|
| **Data source** | Loads CSV independently | Uses X_train, y_train from notebook |
| **Models** | Creates model_factory | Can use trained model from pipeline |
| **Task types** | Classification only | Both classification AND regression |
| **Integration** | Standalone Python modules | Direct notebook cells |
| **Model reuse** | Always retrains | Optional: base_model (reuse) or model_factory (retrain) |

## Key Architecture

All reviewer points now support:
1. **Pipeline mode:** Use `base_model=model` (trained model from earlier cells)
2. **Retrain mode:** Use `model_factory=callable` (create fresh models for each fold)
3. **Auto-detection:** `task_type='auto'` detects classification vs regression

---

## POINT 1: Outer-Fold Metrics, 95% CI, Calibration, Decision Curves

### File
`/Users/romirpatel/nature-mitmcg/REVIEWER_POINT_1_UPDATED.py`

### What It Does
- Explicit outer-fold CV (5 independent test folds)
- 95% confidence intervals using t-distribution
- Calibration plots with Wilson CI (classification)
- Decision curve analysis (classification)
- Pred-vs-actual plots (regression)
- Fold-wise distributions

### Usage in Notebook

**OPTION 1: Pipeline Mode (Use trained model)**
```python
# After you've trained your model in the ML cell
# Just run this:

reporter_point1 = NestedCVReporter(X_train.copy(), y_train.copy(), task_type='auto', n_outer=5)
reporter_point1.run_outer_cv(base_model=model, results_dir='nestedcv_results')
reporter_point1.generate_report(title="[POINT 1] Nested CV Performance - RP")
```

**OPTION 2: Retrain Mode (Fresh models for comparison)**
```python
from sklearn.linear_model import LogisticRegression

def model_factory():
    return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

reporter_point1 = NestedCVReporter(X_train.copy(), y_train.copy(), task_type='auto', n_outer=5)
reporter_point1.run_outer_cv(model_factory=model_factory, results_dir='nestedcv_results')
reporter_point1.generate_report(title="[POINT 1] Nested CV Performance - RP")
```

### Outputs
- `nestedcv_results/oof_predictions_classification.csv` - Per-fold predictions
- `nestedcv_results/fold_metrics_classification.csv` - Per-fold metrics
- Console table: metrics with 95% CI
- Console table: metrics at clinical thresholds (0.3, 0.5, 0.7)
- Plots: fold distributions, calibration curve, decision curve

### Task Type Handling
- **Classification:** AUC, sensitivity, specificity, PPV, NPV, accuracy, F1
- **Regression:** R², RMSE, MAE, explained variance

---

## POINT 2: Stratified Performance by Sex & SES

### What It Does
- Stratify analysis by sex (male/female)
- Stratify analysis by SES (low/mid/high from parent_income)
- Separate calibration plots per subgroup (classification)
- Separate pred-vs-actual per subgroup (regression)
- Compare model fairness across demographics

### Key Changes from Original
1. **Extract labels from sample DataFrame:**
   ```python
   sex_labels = sample_valid['sex'].map({1: 'Male', 2: 'Female'})
   ses_labels = pd.qcut(sample_valid['parent_income'], q=3, labels=['Low', 'Mid', 'High'])
   ```

2. **Support both task types:**
   ```python
   stratified_reporter = NestedCVReporterStratified(X_train, y_train, task_type='auto')
   ```

3. **Pipeline integration:**
   ```python
   stratified_reporter.run_outer_cv_stratified(
       base_model=model,  # or model_factory for retraining
       sex_labels=sex_labels,
       ses_labels=ses_labels
   )
   ```

### Modifications Needed
In your notebook cell that runs POINT 2:
- Extract `sex_labels` and `ses_labels` BEFORE calling the reporter
- Ensure labels are aligned with X_train (same row ordering)
- For SES, use tertile split of parent_income or similar

### Outputs
- Metrics table stratified by sex × SES (all combinations)
- Separate calibration curves for each subgroup
- Comparison of model fairness across demographics

---

## POINT 3: Robustness via Family/Site Grouping

### What It Does
- Test robustness via family-aware CV (siblings stay together)
- Test robustness via site-aware CV (site members stay together)
- Compare standard CV vs grouped CV (performance drop = robustness issue)
- Handles confounds: family relatedness, site effects

### Key Changes from Original
1. **Detect grouping columns:**
   ```python
   has_family = 'rel_family_id' in sample.columns
   has_site = 'L_site_id' in sample.columns
   ```

2. **Support both task types:**
   - Classification: Use StratifiedGroupKFold
   - Regression: Use GroupKFold

3. **Pipeline integration:**
   ```python
   robust_reporter = RobustnessReporter(X_train, y_train, task_type='auto')

   if has_family:
       family_ids = sample_valid.loc[valid_idx, 'rel_family_id']
       robust_reporter.run_group_cv(
           base_model=model,
           groups=family_ids,
           group_type='family'
       )

   if has_site:
       site_ids = sample_valid.loc[valid_idx, 'L_site_id']
       robust_reporter.run_group_cv(
           base_model=model,
           groups=site_ids,
           group_type='site'
       )
   ```

### Modifications Needed
- Extract family_id and site_id from sample DataFrame
- Ensure group IDs are aligned with X_train rows
- For family analysis: filter to subjects with valid family_id
- For site analysis: filter to subjects with valid site_id

### Outputs
- Comparison: standard CV metrics vs family-grouped CV metrics
- Comparison: standard CV metrics vs site-grouped CV metrics
- Performance drop table (if any, indicates confound issue)

---

## POINT 4: Permutation-Based Null Distributions

### What It Does
- Test model significance via label permutation
- Build null distribution of performance metrics
- Compute p-value: observed vs null (is model significant?)
- Addresses "predictive pleiotropy" concern from reviewers

### Key Changes from Original
1. **Support both task types:**
   - Classification: AUC as scoring metric
   - Regression: R² as scoring metric

2. **Pipeline integration:**
   ```python
   perm_reporter = PermutationTestReporter(
       X_train, y_train,
       task_type='auto',
       n_permutations=100,  # or 1000 for final
       cv_folds=5
   )

   result = perm_reporter.run_permutation_test(
       base_model=model,  # or model_factory
       scoring_metric='auto'  # auto-detects AUC or R²
   )
   ```

3. **Auto-detect scoring:**
   ```python
   if is_classification:
       scoring_metric = 'roc_auc'
   else:
       scoring_metric = 'r2'
   ```

### Modifications Needed
- Add `task_type` parameter to detect scoring automatically
- Support both `base_model` (reuse) and `model_factory` (retrain) modes
- For regression: use R² instead of AUC

### Outputs
- Histogram: null distribution vs observed performance
- P-value: proportion of null models ≥ observed (2-tailed)
- Conclusion: "Model is significant at p < 0.05" (if p < 0.05)

---

## POINT 5: Stacking Nested CV Validation

### What It Does
- Audit stacking methodology for data leakage
- Verify meta-learner trained on OOF predictions only
- Check base models don't train on data they predict for meta-learner
- Support both classification and regression meta-learners

### Key Changes from Original
1. **Support both task types:**
   - Classification: LogisticRegression meta-learner
   - Regression: LinearRegression meta-learner

2. **Pipeline integration (if model is stacking ensemble):**
   ```python
   stacking_auditor = StackingCVAuditor(X_train, y_train, task_type='auto')

   if hasattr(model, 'meta_learner'):  # Check if model is stacking
       validation_results = stacking_auditor.audit_existing_stacking(model)
   ```

3. **Or audit proposed stacking setup:**
   ```python
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.linear_model import LogisticRegression

   base_models = {
       'rf': RandomForestClassifier(n_estimators=100, random_state=42),
       'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
   }

   meta_learner = LogisticRegression(random_state=42)

   validation_results = stacking_auditor.validate_stacking_setup(
       base_models,
       meta_learner,
       n_inner=5
   )
   ```

### Modifications Needed
- Add `task_type` detection
- Auto-select meta-learner based on task type
- Support auditing existing stacking model OR validating proposed setup

### Outputs
- Report: "Meta-learner trained on OOF predictions" ✅
- Report: "No data leakage detected" ✅
- Table: per-base-model metrics in stacking
- Table: meta-learner weights/importances

---

## Implementation Checklist

### For Each Reviewer Point Cell

- [ ] **Add task_type detection:**
  ```python
  def detect_task_type(y):
      unique_vals = np.unique(y)
      return 'classification' if len(unique_vals) <= 10 else 'regression'
  ```

- [ ] **Support two input modes:**
  ```python
  # Mode 1: Pipeline (use trained model)
  reporter.run_outer_cv(base_model=model, ...)

  # Mode 2: Retrain (create fresh models)
  reporter.run_outer_cv(model_factory=model_factory, ...)
  ```

- [ ] **Handle classification metrics:**
  - AUC, sensitivity, specificity, PPV, NPV, accuracy, F1
  - Calibration curves, decision curves

- [ ] **Handle regression metrics:**
  - R², RMSE, MAE, explained variance
  - Pred-vs-actual plots, residual distributions

- [ ] **Use pipeline variables:**
  ```python
  # Instead of:
  # X, y = load_csv()

  # Do this:
  if 'X_train' in locals() and 'y_train' in locals():
      X, y = X_train.copy(), y_train.copy()
  ```

- [ ] **Extract additional labels from sample DataFrame:**
  ```python
  # For POINT 2 (sex/SES):
  sex_labels = sample_valid['sex'].map({1: 'Male', 2: 'Female'})
  ses_labels = pd.qcut(sample_valid['parent_income'], q=3)

  # For POINT 3 (family/site):
  family_ids = sample_valid['rel_family_id']
  site_ids = sample_valid['L_site_id']
  ```

- [ ] **Test both task types:**
  ```python
  # Classification test:
  target_options = "dep_onset_rci_1.96"  # Binary

  # Regression test:
  target_options = "depress_D_p"  # Continuous
  ```

---

## How to Modify Your Notebook

### Step 1: Replace Cell 55 ([POINT 1])
- Delete current cell content
- Copy entire content from `REVIEWER_POINT_1_UPDATED.py`
- Modify the execution section at bottom:
  ```python
  # Check if pipeline variables exist
  if 'X_train' in locals() and 'y_train' in locals() and 'model' in locals():
      reporter_point1 = NestedCVReporter(X_train.copy(), y_train.copy(), task_type='auto')
      reporter_point1.run_outer_cv(base_model=model, results_dir='nestedcv_results')
      reporter_point1.generate_report(title="[POINT 1] Nested CV Performance - RP")
  ```

### Step 2-5: Modify Cells 56-59 (POINTS 2-5)
- Similar approach: add task_type detection + support both modes
- Extract necessary labels from `sample` DataFrame
- Test with both classification and regression targets

### Step 3: Test Execution
Run in this order:
1. Data preparation cell (loads `sample`)
2. ML training cell (trains `model`, creates `X_train`, `y_train`, `X_test`, `y_test`)
3. POINT 1 cell
4. POINT 2 cell (with extracted `sex_labels`, `ses_labels`)
5. POINT 3 cell (with extracted `family_ids`, `site_ids`)
6. POINT 4 cell
7. POINT 5 cell

---

## Critical Code Snippets

### Auto-detect and handle both task types
```python
# At start of reporter class
task_type = 'auto' if task_type == 'auto' else task_type
if task_type == 'auto':
    unique_vals = np.unique(y)
    task_type = 'classification' if len(unique_vals) <= 10 else 'regression'

is_classification = (task_type == 'classification')
is_regression = (task_type == 'regression')

# When computing metrics
if is_classification:
    # Use AUC, sensitivity, specificity, PPV, NPV, F1
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
else:
    # Use R², RMSE, MAE
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

### Support both pipeline and retrain modes
```python
def run_outer_cv(self, model_factory=None, base_model=None, ...):
    if base_model is not None:
        # Pipeline mode: use trained model
        return self._analyze_pretrained_model(base_model, ...)
    elif model_factory is not None:
        # Retrain mode: train fresh models
        return self._run_explicit_outer_cv(model_factory, ...)
    else:
        raise ValueError("Either base_model or model_factory required")
```

### Extract stratification labels
```python
# In notebook, before calling stratified_reporter
sex_labels = sample_valid['sex'].map({1: 'Male', 2: 'Female'})

# For SES: use tertile split
ses_labels = pd.qcut(sample_valid['parent_income'],
                     q=3,
                     labels=['Low', 'Mid', 'High'],
                     duplicates='drop')  # Drop duplicates if income has ties

# OR use existing SES column if available
if 'socioeconomic_status' in sample_valid.columns:
    ses_labels = sample_valid['socioeconomic_status']
```

---

## Troubleshooting

### "X_train not found"
**Cause:** Reporter cell runs before ML training cell
**Fix:** Run cells in order: data prep → ML training → reviewer cells

### "Task type not detected correctly"
**Solution:** Override with explicit task_type parameter:
```python
reporter = NestedCVReporter(X_train, y_train, task_type='classification', n_outer=5)
# Or for regression:
reporter = NestedCVReporter(X_train, y_train, task_type='regression', n_outer=5)
```

### "Model has incompatible API"
**Cause:** Pretrained model doesn't have `predict_proba` or `predict`
**Fix:** Use model_factory (retrain mode) instead:
```python
def model_factory():
    return MyCompatibleModel(...)
reporter.run_outer_cv(model_factory=model_factory)
```

### "Performance very different from ML cell"
**Cause:** Different CV fold structure (reporter uses 5-fold, ML cell might use 3-fold or single train/test split)
**Solution:** This is expected - reporter does more rigorous evaluation. Report both values.

---

## Summary

**BEFORE (incorrect):**
- Independent CSV loading
- Classification only
- No pipeline integration
- Separate Python files

**AFTER (correct):**
- Uses X_train, y_train from notebook
- Both classification and regression
- Integrates with trained model
- Direct notebook cells
- Supports model reuse or retraining

All changes maintain backward compatibility while adding ML pipeline integration.
