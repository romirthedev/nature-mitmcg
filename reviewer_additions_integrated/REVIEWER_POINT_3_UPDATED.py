#@title ✅ [POINT 3] Robustness via Family/Site Grouping & Confounds

"""
REVIEWER POINT 3: Robustness to Confounds via Group-Aware CV
============================================================
✅ Supports BOTH regression and classification
✅ Tests robustness via family-aware CV (siblings together)
✅ Tests robustness via site-aware CV (site members together)
✅ Compares standard CV vs grouped CV (detect confound bias)

KEY FEATURES:
- Family-grouped CV: prevents information leakage from related subjects
- Site-grouped CV: tests generalization across data collection sites
- Performance comparison table (standard vs grouped)
- Robustness assessment: drop in performance = potential confound
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold

# ========================================================================
# HELPER: Task Type Detection
# ========================================================================

def detect_task_type(y):
    """Auto-detect classification vs regression."""
    unique_vals = np.unique(y)
    return 'classification' if len(unique_vals) <= 10 else 'regression'


# ========================================================================
# MAIN: RobustnessReporter Class
# ========================================================================

class RobustnessReporter:
    """
    Test model robustness to confounds via group-aware cross-validation.

    Grouping strategies:
    1. Standard CV: Random stratified splits (baseline)
    2. Family CV: Siblings always in same fold (tests family confound)
    3. Site CV: Site members always in same fold (tests site confound)

    For each grouping, computes metrics and compares to baseline.
    Large performance drop = potential confound bias.
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
            Number of CV folds
        random_state : int
            Random seed
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.n_outer = n_outer
        self.random_state = random_state

        if task_type == 'auto':
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == 'classification'
        self.is_regression = self.task_type == 'regression'

        self.results = {
            'standard': {},      # {metric: [fold_values]}
            'family_grouped': {},
            'site_grouped': {}
        }

        print(f"✅ Task type: {self.task_type.upper()}")

    def run_group_cv(self, groups, group_type='family', model_factory=None, base_model=None):
        """
        Run group-aware CV.

        Parameters:
        -----------
        groups : pd.Series or np.array
            Group IDs (family_id, site_id, etc.) aligned with X
        group_type : str
            'family', 'site', or other label for reporting
        model_factory : callable, optional
            Function returning fresh model
        base_model : estimator, optional
            Pre-trained model from pipeline
        """
        groups = pd.Series(groups).reset_index(drop=True)

        if len(groups) != len(self.X):
            raise ValueError(f"groups length {len(groups)} != X length {len(self.X)}")

        print(f"\n🔄 Running robustness test: {group_type.upper()}-AWARE CV")
        print(f"   Groups: {len(groups.unique())} unique {group_type} IDs")
        print(f"   Samples per group: mean={groups.value_counts().mean():.1f}, "
              f"min={groups.value_counts().min()}, max={groups.value_counts().max()}")

        use_base_model = base_model is not None

        if self.is_classification:
            return self._robustness_cv_classification(groups, group_type, use_base_model, base_model, model_factory)
        else:
            return self._robustness_cv_regression(groups, group_type, use_base_model, base_model, model_factory)

    def _robustness_cv_classification(self, groups, group_type, use_base_model, base_model, model_factory):
        """Robustness test for classification."""
        from sklearn.metrics import roc_curve

        # Initialize result storage
        group_results = {
            'fold': [], 'auc': [], 'accuracy': [], 'sensitivity': [],
            'specificity': [], 'ppv': []
        }

        # Use StratifiedGroupKFold to preserve class balance within groups
        n_unique_groups = len(groups.unique())
        n_splits = min(self.n_outer, n_unique_groups)

        if n_splits < 2:
            print(f"   ⚠️  Not enough groups for CV (need >= 2, have {n_unique_groups})")
            return None

        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cv_iterator = sgkf.split(self.X, self.y, groups=groups)
        except:
            # Fallback if StratifiedGroupKFold not available
            print("   Falling back to GroupKFold (no stratification)")
            gkf = GroupKFold(n_splits=n_splits)
            cv_iterator = gkf.split(self.X, self.y, groups=groups)

        # Run CV
        for fold_num, (train_idx, test_idx) in enumerate(cv_iterator):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            # Get or train model
            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            # Predictions
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            if len(thresholds) > 1:
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
            y_pred = (y_proba >= optimal_threshold).astype(int)

            # Metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            group_results['fold'].append(fold_num)
            group_results['auc'].append(roc_auc_score(y_test, y_proba))
            group_results['accuracy'].append(accuracy_score(y_test, y_pred))
            group_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            group_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            group_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)

            print(f"   Fold {fold_num+1}: AUC={group_results['auc'][-1]:.4f}")

        self.results[f'{group_type}_grouped'] = group_results
        return group_results

    def _robustness_cv_regression(self, groups, group_type, use_base_model, base_model, model_factory):
        """Robustness test for regression."""
        group_results = {
            'fold': [], 'r2': [], 'rmse': [], 'mae': []
        }

        # Use GroupKFold
        n_unique_groups = len(groups.unique())
        n_splits = min(self.n_outer, n_unique_groups)

        if n_splits < 2:
            print(f"   ⚠️  Not enough groups for CV (need >= 2, have {n_unique_groups})")
            return None

        gkf = GroupKFold(n_splits=n_splits)

        # Run CV
        for fold_num, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, groups=groups)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            # Get or train model
            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            group_results['fold'].append(fold_num)
            group_results['r2'].append(r2_score(y_test, y_pred))
            group_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            group_results['mae'].append(mean_absolute_error(y_test, y_pred))

            print(f"   Fold {fold_num+1}: R²={group_results['r2'][-1]:.4f}")

        self.results[f'{group_type}_grouped'] = group_results
        return group_results

    def run_standard_cv(self, model_factory=None, base_model=None):
        """Run standard CV (baseline for comparison)."""
        print(f"\n🔄 Running baseline: STANDARD CV")

        use_base_model = base_model is not None

        if self.is_classification:
            return self._standard_cv_classification(use_base_model, base_model, model_factory)
        else:
            return self._standard_cv_regression(use_base_model, base_model, model_factory)

    def _standard_cv_classification(self, use_base_model, base_model, model_factory):
        """Standard CV for classification (baseline)."""
        from sklearn.metrics import roc_curve

        standard_results = {
            'fold': [], 'auc': [], 'accuracy': [], 'sensitivity': [],
            'specificity': [], 'ppv': []
        }

        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            if len(thresholds) > 1:
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
            y_pred = (y_proba >= optimal_threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            standard_results['fold'].append(fold_num)
            standard_results['auc'].append(roc_auc_score(y_test, y_proba))
            standard_results['accuracy'].append(accuracy_score(y_test, y_pred))
            standard_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            standard_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            standard_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)

            print(f"   Fold {fold_num+1}: AUC={standard_results['auc'][-1]:.4f}")

        self.results['standard'] = standard_results
        return standard_results

    def _standard_cv_regression(self, use_base_model, base_model, model_factory):
        """Standard CV for regression (baseline)."""
        standard_results = {
            'fold': [], 'r2': [], 'rmse': [], 'mae': []
        }

        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]

            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            standard_results['fold'].append(fold_num)
            standard_results['r2'].append(r2_score(y_test, y_pred))
            standard_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            standard_results['mae'].append(mean_absolute_error(y_test, y_pred))

            print(f"   Fold {fold_num+1}: R²={standard_results['r2'][-1]:.4f}")

        self.results['standard'] = standard_results
        return standard_results

    def generate_report(self, title="ROBUSTNESS ANALYSIS"):
        """Generate robustness comparison report."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"\nTask Type: {self.task_type.upper()}")
        print(f"CV folds: {self.n_outer}\n")

        # Determine metrics
        if self.is_classification:
            metrics = ['auc', 'accuracy', 'sensitivity', 'specificity']
        else:
            metrics = ['r2', 'rmse', 'mae']

        # Build comparison table
        comparison_rows = []

        for cv_type in ['standard', 'family_grouped', 'site_grouped']:
            if cv_type not in self.results or not self.results[cv_type]:
                continue

            row = {'CV Strategy': cv_type.replace('_', ' ').title()}

            for metric in metrics:
                if metric in self.results[cv_type]:
                    values = self.results[cv_type][metric]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1)
                        row[f'{metric}'] = f"{mean_val:.3f} ± {std_val:.3f}"

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        print("ROBUSTNESS COMPARISON")
        print("-" * 80)
        print(comparison_df.to_string(index=False))

        # Interpretation
        if 'standard' in self.results and 'family_grouped' in self.results:
            print("\n📊 INTERPRETATION:")
            print("-" * 80)

            if self.is_classification:
                standard_auc = np.mean(self.results['standard']['auc'])
                family_auc = np.mean(self.results['family_grouped']['auc'])
                drop = (standard_auc - family_auc) / standard_auc * 100

                print(f"Family confound: AUC drop = {drop:.1f}%")
                if drop < 5:
                    print("   ✅ Low family confound (robust to family clustering)")
                elif drop < 15:
                    print("   ⚠️  Moderate family confound (consider in interpretation)")
                else:
                    print("   ⚠️  High family confound (significant bias detected)")
            else:
                standard_r2 = np.mean(self.results['standard']['r2'])
                family_r2 = np.mean(self.results['family_grouped']['r2'])
                drop = (standard_r2 - family_r2) / (standard_r2 + 1e-10) * 100

                print(f"Family confound: R² drop = {drop:.1f}%")

        print("\n" + "=" * 80)
        print("✅ [POINT 3] COMPLETE")
        print("=" * 80)


# ========================================================================
# EXECUTION: Run robustness reporter
# ========================================================================

try:
    if 'X_train' in locals() and 'y_train' in locals() and 'sample_valid' in locals():
        print("\n🚀 Running [POINT 3] with pipeline variables")

        robust_reporter = RobustnessReporter(
            X_train.copy(), y_train.copy(), task_type='auto', n_outer=5
        )

        # Determine which grouping columns are available
        has_family = 'rel_family_id' in sample_valid.columns
        has_site = 'L_site_id' in sample_valid.columns

        print(f"   Available confounds: family={has_family}, site={has_site}")

        # Get trained model
        if 'model' not in locals():
            print("   ⚠️  No trained model found, skipping robustness test")
        else:
            # Run baseline (standard CV)
            robust_reporter.run_standard_cv(base_model=model)

            # Run family-aware CV if available
            if has_family:
                try:
                    family_ids = sample_valid.loc[X_train.index, 'rel_family_id'] if hasattr(X_train, 'index') else sample_valid['rel_family_id']
                    robust_reporter.run_group_cv(groups=family_ids, group_type='family', base_model=model)
                except Exception as e:
                    print(f"   ⚠️  Error in family CV: {e}")

            # Run site-aware CV if available
            if has_site:
                try:
                    site_ids = sample_valid.loc[X_train.index, 'L_site_id'] if hasattr(X_train, 'index') else sample_valid['L_site_id']
                    robust_reporter.run_group_cv(groups=site_ids, group_type='site', base_model=model)
                except Exception as e:
                    print(f"   ⚠️  Error in site CV: {e}")

            robust_reporter.generate_report(title="[POINT 3] Robustness to Confounds - RP")

    else:
        print("\n❌ ERROR: Required variables not found")
        print("   Need: X_train, y_train, sample_valid, model")

except Exception as e:
    print(f"\n❌ Error in [POINT 3]: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ [POINT 3] Complete!")
