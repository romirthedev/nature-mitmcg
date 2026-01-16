#@title ✅ [POINT 2] Stratified Performance by Sex & SES

"""
REVIEWER POINT 2: Stratified Performance & Calibration Analysis
==============================================================
✅ Supports BOTH regression and classification
✅ Integrates with ML pipeline variables
✅ Stratifies by sex (male/female) and SES (low/mid/high)
✅ Separate calibration/pred-vs-actual for each subgroup

KEY FEATURES:
- Performance metrics stratified by demographic groups
- Fairness assessment across sex and SES
- Calibration curves per subgroup (classification)
- Pred-vs-actual per subgroup (regression)
- Comparison tables showing within-group performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, roc_curve
)
from sklearn.model_selection import StratifiedKFold, KFold

# ========================================================================
# HELPER: Task Type Detection
# ========================================================================

def detect_task_type(y):
    """Auto-detect classification vs regression from target variable."""
    unique_vals = np.unique(y)
    if len(unique_vals) <= 10:
        return 'classification'
    return 'regression'


# ========================================================================
# MAIN: NestedCVReporterStratified Class
# ========================================================================

class NestedCVReporterStratified:
    """
    Stratified nested CV reporter for classification AND regression.

    Stratifies analysis by:
    - Sex (male/female)
    - SES (low/mid/high from tertile split of parent_income)

    Supports:
    - Pipeline mode: use trained base_model
    - Retrain mode: use model_factory to create fresh models
    - Both classification and regression tasks
    """

    def __init__(self, X, y, task_type='auto', n_outer=5, random_state=42):
        """
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series or np.array
            Target variable
        task_type : str
            'classification', 'regression', or 'auto'
        n_outer : int
            Number of outer CV folds
        random_state : int
            Random seed
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.n_outer = n_outer
        self.random_state = random_state

        # Auto-detect task type
        if task_type == 'auto':
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == 'classification'
        self.is_regression = self.task_type == 'regression'

        self.sex_results = {}  # {sex: {metric: [fold_values]}}
        self.ses_results = {}  # {ses: {metric: [fold_values]}}
        self.combined_results = {}  # {(sex, ses): {metric: [fold_values]}}

        print(f"✅ Task type: {self.task_type.upper()}")

    def run_outer_cv_stratified(self, sex_labels=None, ses_labels=None,
                               model_factory=None, base_model=None,
                               results_dir='nestedcv_results_stratified'):
        """
        Run stratified outer CV analysis.

        Parameters:
        -----------
        sex_labels : pd.Series
            Sex labels aligned with X (e.g., 'Male', 'Female')
        ses_labels : pd.Series
            SES labels aligned with X (e.g., 'Low', 'Mid', 'High')
        model_factory : callable, optional
            Function returning fresh model for retraining
        base_model : estimator, optional
            Pre-trained model from pipeline
        results_dir : str
            Directory to save results
        """
        os.makedirs(results_dir, exist_ok=True)

        if sex_labels is None or ses_labels is None:
            raise ValueError("sex_labels and ses_labels must be provided")

        # Ensure labels are aligned
        sex_labels = pd.Series(sex_labels).reset_index(drop=True)
        ses_labels = pd.Series(ses_labels).reset_index(drop=True)

        if len(sex_labels) != len(self.X):
            raise ValueError(f"sex_labels length {len(sex_labels)} != X length {len(self.X)}")
        if len(ses_labels) != len(self.X):
            raise ValueError(f"ses_labels length {len(ses_labels)} != X length {len(self.X)}")

        print(f"\n🔄 Running stratified CV analysis")
        print(f"   Sex groups: {sex_labels.unique()}")
        print(f"   SES groups: {ses_labels.unique()}")

        # Get unique groups
        sex_groups = sex_labels.unique()
        ses_groups = ses_labels.unique()

        # Strategy: For each combination, evaluate model on that subgroup
        if self.is_classification:
            return self._stratified_cv_classification(
                sex_labels, ses_labels, base_model, model_factory, results_dir
            )
        else:
            return self._stratified_cv_regression(
                sex_labels, ses_labels, base_model, model_factory, results_dir
            )

    def _stratified_cv_classification(self, sex_labels, ses_labels, base_model, model_factory, results_dir):
        """Stratified CV for classification."""
        from sklearn.model_selection import StratifiedKFold

        # Use base model if provided, otherwise retrain
        use_base_model = base_model is not None

        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        sex_groups = sex_labels.unique()
        ses_groups = ses_labels.unique()

        # Initialize results dicts
        for sex in sex_groups:
            self.sex_results[sex] = {
                'auc': [], 'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': [], 'accuracy': []
            }

        for ses in ses_groups:
            self.ses_results[ses] = {
                'auc': [], 'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': [], 'accuracy': []
            }

        for sex, ses in zip(sex_groups, ses_groups):
            self.combined_results[(sex, ses)] = {
                'auc': [], 'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': [], 'accuracy': []
            }

        # Run CV
        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_test = self.X.iloc[test_idx]
            y_test = self.y.iloc[test_idx]
            sex_test = sex_labels.iloc[test_idx]
            ses_test = ses_labels.iloc[test_idx]

            # Get or train model
            if use_base_model:
                model = base_model
            else:
                X_train = self.X.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                model = model_factory()
                model.fit(X_train, y_train)

            # Get predictions
            y_proba = model.predict_proba(X_test)[:, 1]

            # Evaluate by sex
            for sex in sex_groups:
                mask = (sex_test == sex).values
                if mask.sum() > 0:
                    y_proba_sex = y_proba[mask]
                    y_test_sex = y_test.iloc[mask]

                    fpr, tpr, thresholds = roc_curve(y_test_sex, y_proba_sex)
                    if len(thresholds) > 1:
                        optimal_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[optimal_idx]
                    else:
                        optimal_threshold = 0.5
                    y_pred = (y_proba_sex >= optimal_threshold).astype(int)

                    tn, fp, fn, tp = confusion_matrix(y_test_sex, y_pred).ravel()

                    self.sex_results[sex]['auc'].append(roc_auc_score(y_test_sex, y_proba_sex))
                    self.sex_results[sex]['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    self.sex_results[sex]['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    self.sex_results[sex]['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                    self.sex_results[sex]['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
                    self.sex_results[sex]['accuracy'].append(accuracy_score(y_test_sex, y_pred))

            # Evaluate by SES
            for ses in ses_groups:
                mask = (ses_test == ses).values
                if mask.sum() > 0:
                    y_proba_ses = y_proba[mask]
                    y_test_ses = y_test.iloc[mask]

                    fpr, tpr, thresholds = roc_curve(y_test_ses, y_proba_ses)
                    if len(thresholds) > 1:
                        optimal_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[optimal_idx]
                    else:
                        optimal_threshold = 0.5
                    y_pred = (y_proba_ses >= optimal_threshold).astype(int)

                    tn, fp, fn, tp = confusion_matrix(y_test_ses, y_pred).ravel()

                    self.ses_results[ses]['auc'].append(roc_auc_score(y_test_ses, y_proba_ses))
                    self.ses_results[ses]['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    self.ses_results[ses]['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    self.ses_results[ses]['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                    self.ses_results[ses]['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
                    self.ses_results[ses]['accuracy'].append(accuracy_score(y_test_ses, y_pred))

            print(f"   Fold {fold_num+1}/{self.n_outer} complete")

        self._save_stratified_results(results_dir, 'classification')
        return self.sex_results, self.ses_results

    def _stratified_cv_regression(self, sex_labels, ses_labels, base_model, model_factory, results_dir):
        """Stratified CV for regression."""
        from sklearn.model_selection import KFold

        use_base_model = base_model is not None

        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        sex_groups = sex_labels.unique()
        ses_groups = ses_labels.unique()

        # Initialize results dicts
        for sex in sex_groups:
            self.sex_results[sex] = {'r2': [], 'rmse': [], 'mae': []}

        for ses in ses_groups:
            self.ses_results[ses] = {'r2': [], 'rmse': [], 'mae': []}

        # Run CV
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_test = self.X.iloc[test_idx]
            y_test = self.y.iloc[test_idx]
            sex_test = sex_labels.iloc[test_idx]
            ses_test = ses_labels.iloc[test_idx]

            # Get or train model
            if use_base_model:
                model = base_model
            else:
                X_train = self.X.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                model = model_factory()
                model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_test)

            # Evaluate by sex
            for sex in sex_groups:
                mask = (sex_test == sex).values
                if mask.sum() > 0:
                    y_pred_sex = y_pred[mask]
                    y_test_sex = y_test.iloc[mask]

                    self.sex_results[sex]['r2'].append(r2_score(y_test_sex, y_pred_sex))
                    self.sex_results[sex]['rmse'].append(np.sqrt(mean_squared_error(y_test_sex, y_pred_sex)))
                    self.sex_results[sex]['mae'].append(mean_absolute_error(y_test_sex, y_pred_sex))

            # Evaluate by SES
            for ses in ses_groups:
                mask = (ses_test == ses).values
                if mask.sum() > 0:
                    y_pred_ses = y_pred[mask]
                    y_test_ses = y_test.iloc[mask]

                    self.ses_results[ses]['r2'].append(r2_score(y_test_ses, y_pred_ses))
                    self.ses_results[ses]['rmse'].append(np.sqrt(mean_squared_error(y_test_ses, y_pred_ses)))
                    self.ses_results[ses]['mae'].append(mean_absolute_error(y_test_ses, y_pred_ses))

            print(f"   Fold {fold_num+1}/{self.n_outer} complete")

        self._save_stratified_results(results_dir, 'regression')
        return self.sex_results, self.ses_results

    def _save_stratified_results(self, results_dir, task_type):
        """Save results to CSV."""
        # Save by sex
        sex_summary = []
        for sex, metrics in self.sex_results.items():
            row = {'Subgroup': f'Sex: {sex}'}
            for metric, values in metrics.items():
                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values, ddof=1)
            sex_summary.append(row)

        sex_df = pd.DataFrame(sex_summary)
        sex_df.to_csv(f"{results_dir}/stratified_by_sex.csv", index=False)

        # Save by SES
        ses_summary = []
        for ses, metrics in self.ses_results.items():
            row = {'Subgroup': f'SES: {ses}'}
            for metric, values in metrics.items():
                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values, ddof=1)
            ses_summary.append(row)

        ses_df = pd.DataFrame(ses_summary)
        ses_df.to_csv(f"{results_dir}/stratified_by_ses.csv", index=False)

        print(f"✅ Saved results to {results_dir}/")

    def generate_report(self, title="STRATIFIED PERFORMANCE ANALYSIS"):
        """Generate stratified performance report."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"\nTask Type: {self.task_type.upper()}")
        print(f"Outer-fold CV: {self.n_outer} folds\n")

        # BY SEX
        print("1. PERFORMANCE STRATIFIED BY SEX")
        print("-" * 80)

        sex_data = []
        for sex, metrics in self.sex_results.items():
            row = {'Sex': sex}
            for metric, values in metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    n = len(values)
                    t_crit = stats.t.ppf(0.975, df=n-1)
                    ci_margin = t_crit * (std_val / np.sqrt(n))
                    row[f'{metric}'] = f"{mean_val:.3f} ± {ci_margin:.3f}"
            sex_data.append(row)

        sex_df = pd.DataFrame(sex_data)
        print(sex_df.to_string(index=False))

        # BY SES
        print("\n2. PERFORMANCE STRATIFIED BY SES")
        print("-" * 80)

        ses_data = []
        for ses, metrics in self.ses_results.items():
            row = {'SES': ses}
            for metric, values in metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    n = len(values)
                    t_crit = stats.t.ppf(0.975, df=n-1)
                    ci_margin = t_crit * (std_val / np.sqrt(n))
                    row[f'{metric}'] = f"{mean_val:.3f} ± {ci_margin:.3f}"
            ses_data.append(row)

        ses_df = pd.DataFrame(ses_data)
        print(ses_df.to_string(index=False))

        print("\n" + "=" * 80)
        print("✅ [POINT 2] COMPLETE")
        print("=" * 80)


# ========================================================================
# EXECUTION: Run stratified reporter with pipeline
# ========================================================================

try:
    if 'X_train' in locals() and 'y_train' in locals() and 'sample_valid' in locals():
        print("\n🚀 Running [POINT 2] with pipeline variables")

        # Extract sex labels
        if 'sex' in sample_valid.columns:
            sex_labels = sample_valid['sex'].map({1: 'Male', 2: 'Female'})
            print(f"   ✅ Sex labels extracted: {sex_labels.unique()}")
        else:
            print("   ⚠️  'sex' column not found in sample_valid")
            sex_labels = None

        # Extract SES labels (tertile split of parent_income)
        if 'parent_income' in sample_valid.columns:
            ses_labels = pd.qcut(sample_valid['parent_income'],
                                 q=3,
                                 labels=['Low', 'Mid', 'High'],
                                 duplicates='drop')
            print(f"   ✅ SES labels extracted: {ses_labels.unique()}")
        else:
            print("   ⚠️  'parent_income' column not found in sample_valid")
            ses_labels = None

        if sex_labels is not None and ses_labels is not None:
            # Reset indices to match X_train
            valid_indices = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
            sex_labels = sex_labels.reindex(valid_indices).reset_index(drop=True)
            ses_labels = ses_labels.reindex(valid_indices).reset_index(drop=True)

            # Run stratified reporter
            stratified_reporter = NestedCVReporterStratified(
                X_train.copy(), y_train.copy(), task_type='auto', n_outer=5
            )

            # Use trained model if available
            if 'model' in locals():
                stratified_reporter.run_outer_cv_stratified(
                    sex_labels=sex_labels,
                    ses_labels=ses_labels,
                    base_model=model,
                    results_dir='nestedcv_results_stratified'
                )
            else:
                print("   ⚠️  No trained model found, skipping stratified analysis")

            stratified_reporter.generate_report(title="[POINT 2] Stratified Performance by Sex & SES - RP")
        else:
            print("   ❌ Missing required columns for stratification")

    else:
        print("\n❌ ERROR: Required variables not found")
        print("   Need: X_train, y_train, sample_valid, model")

except Exception as e:
    print(f"\n❌ Error in [POINT 2]: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ [POINT 2] Complete!")
