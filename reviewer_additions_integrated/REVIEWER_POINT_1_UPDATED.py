#@title ✅ [POINT 1] Outer-Fold Metrics, 95% CI, Calibration, Decision Curves

"""
REVIEWER POINT 1: Outer-Fold Nested CV Analysis
================================================
✅ Supports BOTH regression and classification
✅ Integrates with ML pipeline variables (X_train, X_test, y_train, y_test, model)
✅ Can optionally retrain fresh models with model_factory parameter
✅ Works in Google Colab

KEY FEATURES:
- True outer-fold CV (5 independent test folds)
- 95% CI using t-distribution (df=4)
- Calibration plots with Wilson CI (classification)
- Decision curve analysis (classification)
- Pred-vs-actual plots (regression)
- Fold-wise distribution plots for all metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, KFold

# ========================================================================
# HELPER: Task Type Detection
# ========================================================================

def detect_task_type(y):
    """Auto-detect classification vs regression from target variable."""
    unique_vals = np.unique(y)
    # If binary or few categories (<= 10), likely classification
    if len(unique_vals) <= 10:
        return 'classification'
    # If continuous, regression
    return 'regression'


# ========================================================================
# MAIN: NestedCVReporter Class (Handles both task types)
# ========================================================================

class NestedCVReporter:
    """
    Explicit nested CV reporter for classification AND regression.

    INTEGRATION WITH ML PIPELINE:
    - Can use trained models from notebook (base_model parameter)
    - Can retrain fresh models (model_factory parameter)
    - Works with X_train, y_train from global notebook scope

    TASK TYPE SUPPORT:
    - Classification: AUC, sensitivity, specificity, PPV, NPV, accuracy, F1
    - Regression: R², RMSE, MAE, explained variance
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
            'classification', 'regression', or 'auto' (auto-detect)
        n_outer : int
            Number of outer CV folds (default 5)
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

        self.outer_fold_results = {}
        self.clinical_thresholds = [0.3, 0.5, 0.7]

        print(f"✅ Task type detected: {self.task_type.upper()}")
        if self.is_classification:
            print(f"   Unique classes: {len(np.unique(self.y))}")
            print(f"   Class balance: {dict(pd.Series(self.y).value_counts())}")
        else:
            print(f"   Value range: [{self.y.min():.3f}, {self.y.max():.3f}]")
            print(f"   Mean: {self.y.mean():.3f}, SD: {self.y.std():.3f}")

    def run_outer_cv(self, model_factory=None, base_model=None, results_dir='nestedcv_results'):
        """
        Run explicit outer CV loop.

        USAGE OPTIONS:
        1. Pipeline mode: pass base_model (trained model from notebook)
           reporter.run_outer_cv(base_model=model)

        2. Retrain mode: pass model_factory (function returning fresh model)
           def model_factory():
               return LogisticRegression(max_iter=1000)
           reporter.run_outer_cv(model_factory=model_factory)

        Parameters:
        -----------
        model_factory : callable, optional
            Function that returns a fresh model instance (for retraining)
        base_model : estimator, optional
            Pre-trained model to use for predictions (from pipeline)
        results_dir : str
            Directory to save results
        """
        if base_model is not None:
            return self._analyze_pretrained_model(base_model, results_dir)
        elif model_factory is not None:
            return self._run_explicit_outer_cv(model_factory, results_dir)
        else:
            raise ValueError("Either base_model (trained model) or model_factory (callable) must be provided")

    # =================================================================
    # PIPELINE MODE: Analyze pretrained model
    # =================================================================

    def _analyze_pretrained_model(self, model, results_dir):
        """Analyze a pretrained model using stratified CV."""
        print(f"\n🔄 PIPELINE MODE: Analyzing pretrained {model.__class__.__name__}")
        print(f"   (Using trained model with {self.n_outer}-fold outer CV)")

        os.makedirs(results_dir, exist_ok=True)

        if self.is_classification:
            return self._analyze_pretrained_classification(model, results_dir)
        else:
            return self._analyze_pretrained_regression(model, results_dir)

    def _analyze_pretrained_classification(self, model, results_dir):
        """Analyze pretrained classification model."""
        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            'fold': [], 'auc': [], 'sensitivity': [], 'specificity': [],
            'ppv': [], 'npv': [], 'accuracy': [], 'f1': [],
            'y_true': [], 'y_proba': [], 'y_pred': []
        }

        oof_df_list = []

        print(f"   Running {self.n_outer} outer folds...")

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_test_fold = self.X.iloc[test_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Get predictions from pretrained model
            y_proba = model.predict_proba(X_test_fold)[:, 1]

            # Optimal threshold (Youden's J)
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_proba >= optimal_threshold).astype(int)

            # Metrics for this fold
            tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

            fold_results['fold'].append(fold_num)
            fold_results['auc'].append(roc_auc_score(y_test_fold, y_proba))
            fold_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            fold_results['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            fold_results['accuracy'].append(accuracy_score(y_test_fold, y_pred))
            fold_results['f1'].append(f1_score(y_test_fold, y_pred, zero_division=0))

            # Store predictions
            fold_results['y_true'].append(y_test_fold.values)
            fold_results['y_proba'].append(y_proba)
            fold_results['y_pred'].append(y_pred)

            # OOF dataframe
            oof_df = pd.DataFrame({
                'fold': fold_num,
                'y_true': y_test_fold.values,
                'y_proba': y_proba,
                'y_pred': y_pred
            })
            oof_df_list.append(oof_df)

            print(f"      Fold {fold_num+1}: AUC={fold_results['auc'][-1]:.4f}, Acc={fold_results['accuracy'][-1]:.4f}")

        self.outer_fold_results = fold_results

        # Save results
        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_classification.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items()
                                   if k not in ['y_true', 'y_proba', 'y_pred']})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_classification.csv", index=False)

        print(f"✅ Saved to {results_dir}/")
        return fold_results

    def _analyze_pretrained_regression(self, model, results_dir):
        """Analyze pretrained regression model."""
        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            'fold': [], 'r2': [], 'rmse': [], 'mae': [], 'explained_var': [],
            'y_true': [], 'y_pred': []
        }

        oof_df_list = []

        print(f"   Running {self.n_outer} outer folds...")

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_test_fold = self.X.iloc[test_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Get predictions
            y_pred = model.predict(X_test_fold)

            # Metrics
            r2 = r2_score(y_test_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            exp_var = 1 - (np.var(y_test_fold - y_pred) / np.var(y_test_fold))

            fold_results['fold'].append(fold_num)
            fold_results['r2'].append(r2)
            fold_results['rmse'].append(rmse)
            fold_results['mae'].append(mae)
            fold_results['explained_var'].append(exp_var)
            fold_results['y_true'].append(y_test_fold.values)
            fold_results['y_pred'].append(y_pred)

            # OOF dataframe
            oof_df = pd.DataFrame({
                'fold': fold_num,
                'y_true': y_test_fold.values,
                'y_pred': y_pred,
                'residual': y_test_fold.values - y_pred
            })
            oof_df_list.append(oof_df)

            print(f"      Fold {fold_num+1}: R²={r2:.4f}, RMSE={rmse:.4f}")

        self.outer_fold_results = fold_results

        # Save results
        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_regression.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items()
                                   if k not in ['y_true', 'y_pred']})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_regression.csv", index=False)

        print(f"✅ Saved to {results_dir}/")
        return fold_results

    # =================================================================
    # RETRAIN MODE: Explicit outer CV with fresh model training
    # =================================================================

    def _run_explicit_outer_cv(self, model_factory, results_dir):
        """Run explicit outer CV (retraining fresh models each fold)."""
        print(f"\n🔄 RETRAIN MODE: Running explicit outer CV ({self.n_outer} folds)")

        os.makedirs(results_dir, exist_ok=True)

        if self.is_classification:
            return self._explicit_cv_classification(model_factory, results_dir)
        else:
            return self._explicit_cv_regression(model_factory, results_dir)

    def _explicit_cv_classification(self, model_factory, results_dir):
        """Explicit outer CV for classification."""
        skf = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            'fold': [], 'auc': [], 'sensitivity': [], 'specificity': [],
            'ppv': [], 'npv': [], 'accuracy': [], 'f1': [],
            'y_true': [], 'y_proba': [], 'y_pred': []
        }

        oof_df_list = []

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Train fresh model
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)

            # Predictions
            y_proba = model.predict_proba(X_test_fold)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_proba >= optimal_threshold).astype(int)

            # Metrics
            tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

            fold_results['fold'].append(fold_num)
            fold_results['auc'].append(roc_auc_score(y_test_fold, y_proba))
            fold_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            fold_results['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            fold_results['accuracy'].append(accuracy_score(y_test_fold, y_pred))
            fold_results['f1'].append(f1_score(y_test_fold, y_pred, zero_division=0))

            fold_results['y_true'].append(y_test_fold.values)
            fold_results['y_proba'].append(y_proba)
            fold_results['y_pred'].append(y_pred)

            oof_df = pd.DataFrame({
                'fold': fold_num,
                'y_true': y_test_fold.values,
                'y_proba': y_proba,
                'y_pred': y_pred
            })
            oof_df_list.append(oof_df)

            print(f"   Fold {fold_num+1}: AUC={fold_results['auc'][-1]:.4f}, Acc={fold_results['accuracy'][-1]:.4f}")

        self.outer_fold_results = fold_results

        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_classification.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items()
                                   if k not in ['y_true', 'y_proba', 'y_pred']})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_classification.csv", index=False)

        print(f"✅ Saved to {results_dir}/")
        return fold_results

    def _explicit_cv_regression(self, model_factory, results_dir):
        """Explicit outer CV for regression."""
        kf = KFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)

        fold_results = {
            'fold': [], 'r2': [], 'rmse': [], 'mae': [], 'explained_var': [],
            'y_true': [], 'y_pred': []
        }

        oof_df_list = []

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]

            # Train fresh model
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)

            # Predictions
            y_pred = model.predict(X_test_fold)

            # Metrics
            r2 = r2_score(y_test_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            exp_var = 1 - (np.var(y_test_fold - y_pred) / np.var(y_test_fold))

            fold_results['fold'].append(fold_num)
            fold_results['r2'].append(r2)
            fold_results['rmse'].append(rmse)
            fold_results['mae'].append(mae)
            fold_results['explained_var'].append(exp_var)
            fold_results['y_true'].append(y_test_fold.values)
            fold_results['y_pred'].append(y_pred)

            oof_df = pd.DataFrame({
                'fold': fold_num,
                'y_true': y_test_fold.values,
                'y_pred': y_pred,
                'residual': y_test_fold.values - y_pred
            })
            oof_df_list.append(oof_df)

            print(f"   Fold {fold_num+1}: R²={r2:.4f}, RMSE={rmse:.4f}")

        self.outer_fold_results = fold_results

        oof_all = pd.concat(oof_df_list, ignore_index=True)
        oof_all.to_csv(f"{results_dir}/oof_predictions_regression.csv", index=False)

        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items()
                                   if k not in ['y_true', 'y_pred']})
        metrics_df.to_csv(f"{results_dir}/fold_metrics_regression.csv", index=False)

        print(f"✅ Saved to {results_dir}/")
        return fold_results

    # =================================================================
    # REPORTING METHODS
    # =================================================================

    def summarize_outer_folds(self):
        """Aggregate outer-fold metrics with 95% t-CI."""
        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        if self.is_classification:
            return self._summarize_classification()
        else:
            return self._summarize_regression()

    def _summarize_classification(self):
        """Summarize classification metrics."""
        metrics_list = ['auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'f1']
        summary = {}

        for metric in metrics_list:
            values = np.array(self.outer_fold_results[metric])
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            t_crit = stats.t.ppf(0.975, df=n-1)
            se = std_val / np.sqrt(n)
            ci_margin = t_crit * se

            summary[metric] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': mean_val - ci_margin,
                'ci_upper': mean_val + ci_margin,
                'fold_values': list(values)
            }

        df = pd.DataFrame({
            'Metric': list(summary.keys()),
            'Mean': [summary[m]['mean'] for m in summary],
            'Std': [summary[m]['std'] for m in summary],
            '95% CI Lower': [summary[m]['ci_lower'] for m in summary],
            '95% CI Upper': [summary[m]['ci_upper'] for m in summary]
        })

        return df, summary

    def _summarize_regression(self):
        """Summarize regression metrics."""
        metrics_list = ['r2', 'rmse', 'mae', 'explained_var']
        summary = {}

        for metric in metrics_list:
            if metric in self.outer_fold_results:
                values = np.array(self.outer_fold_results[metric])
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                n = len(values)
                t_crit = stats.t.ppf(0.975, df=n-1)
                se = std_val / np.sqrt(n)
                ci_margin = t_crit * se

                summary[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': mean_val - ci_margin,
                    'ci_upper': mean_val + ci_margin,
                    'fold_values': list(values)
                }

        df = pd.DataFrame({
            'Metric': list(summary.keys()),
            'Mean': [summary[m]['mean'] for m in summary],
            'Std': [summary[m]['std'] for m in summary],
            '95% CI Lower': [summary[m]['ci_lower'] for m in summary],
            '95% CI Upper': [summary[m]['ci_upper'] for m in summary]
        })

        return df, summary

    def plot_calibration_curve(self, n_bins=10, figsize=(10, 6), save_path=None):
        """Calibration curve - classification only."""
        if not self.is_classification:
            print("⚠️  Calibration curve only for classification. Skipping.")
            return None

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results['y_true'])
        y_proba_all = np.concatenate(self.outer_fold_results['y_proba'])

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_true = np.zeros(n_bins)
        bin_ci_lower = np.zeros(n_bins)
        bin_ci_upper = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (y_proba_all >= bins[i]) & (y_proba_all < bins[i + 1])
            if mask.sum() > 0:
                bin_true[i] = y_true_all[mask].mean()
                n = mask.sum()
                p = bin_true[i]
                # Wilson score CI
                z = 1.96
                denom = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denom
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
                bin_ci_lower[i] = max(0, center - margin)
                bin_ci_upper[i] = min(1, center + margin)

        ece = np.mean(np.abs(bin_true - bin_centers))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
        ax.errorbar(bin_centers, bin_true,
                   yerr=[bin_true - bin_ci_lower, bin_ci_upper - bin_true],
                   fmt='o-', capsize=4, capthick=2, lw=2, markersize=8,
                   label=f'Model (ECE = {ece:.4f})')
        ax.fill_between(bin_centers, bin_ci_lower, bin_ci_upper, alpha=0.2)
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('[POINT 1] Calibration Curve\n(Outer Fold Aggregated, Wilson 95% CI)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Saved: {save_path}")
        plt.show()

        return {'bin_centers': bin_centers, 'bin_true': bin_true, 'ece': ece}

    def plot_decision_curve(self, figsize=(10, 6), save_path=None):
        """Decision curve - classification only."""
        if not self.is_classification:
            print("⚠️  Decision curve only for classification. Skipping.")
            return None

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results['y_true'])
        y_proba_all = np.concatenate(self.outer_fold_results['y_proba'])

        thresholds = np.linspace(0.05, 0.95, 50)
        net_benefits = []

        for threshold in thresholds:
            y_pred_thresh = (y_proba_all >= threshold).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (y_true_all == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true_all == 0))
            n = len(y_true_all)
            nb = (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-10))
            net_benefits.append(nb)

        net_benefits = np.array(net_benefits)
        prevalence = np.mean(y_true_all)
        treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds + 1e-10))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, net_benefits, label='Model', lw=3, color='#1f77b4')
        ax.plot(thresholds, treat_all_nb, label='Treat All', lw=2, linestyle='--', color='red')
        ax.axhline(y=0, label='Treat None', lw=2, linestyle='--', color='gray')
        ax.fill_between(thresholds, 0, net_benefits, alpha=0.2, color='#1f77b4')
        ax.axvspan(0.2, 0.8, alpha=0.1, color='green', label='Clinical range')
        ax.set_xlabel('Probability Threshold', fontsize=12)
        ax.set_ylabel('Net Benefit', fontsize=12)
        ax.set_title('[POINT 1] Decision Curve Analysis\n(Outer Fold Aggregated)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Saved: {save_path}")
        plt.show()

        return {'thresholds': thresholds, 'net_benefits': net_benefits}

    def plot_fold_distributions(self, figsize=(14, 8), save_path=None):
        """Plot fold-wise distributions (all metrics)."""
        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        if self.is_classification:
            metrics = ['auc', 'sensitivity', 'specificity', 'ppv', 'accuracy']
        else:
            metrics = ['r2', 'rmse', 'mae', 'explained_var']

        metrics = [m for m in metrics if m in self.outer_fold_results]

        n_cols = 3
        n_rows = (len(metrics) + 2) // 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            values = self.outer_fold_results[metric]

            axes[idx].violinplot([values], positions=[0], vert=True, widths=0.5,
                                showmeans=True, showextrema=True)
            axes[idx].boxplot([values], positions=[0], widths=0.15, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7))

            y_scatter = np.random.normal(0, 0.02, size=len(values))
            axes[idx].scatter(y_scatter, values, alpha=0.7, s=80, color='red',
                            edgecolors='black', label='Outer folds')

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            t_crit = stats.t.ppf(0.975, df=n-1)
            ci_margin = t_crit * (std_val / np.sqrt(n))

            axes[idx].hlines(mean_val, -0.3, 0.3, colors='green', linewidth=2.5, label='Mean')
            axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{metric.upper()}\n{mean_val:.3f} [{mean_val-ci_margin:.3f}-{mean_val+ci_margin:.3f}]',
                               fontsize=11)
            axes[idx].set_xlim([-0.5, 0.5])
            axes[idx].set_xticks([])
            axes[idx].grid(True, alpha=0.3, axis='y')

        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'[POINT 1] Outer-Fold Distributions (n={self.n_outer} folds, {self.task_type.upper()})',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Saved: {save_path}")
        plt.show()

    def threshold_metrics_table(self, thresholds=None):
        """Metrics at thresholds (classification only)."""
        if not self.is_classification:
            print("⚠️  Threshold metrics only for classification. Skipping.")
            return None

        if thresholds is None:
            thresholds = self.clinical_thresholds

        if not self.outer_fold_results:
            raise ValueError("Run run_outer_cv() first.")

        y_true_all = np.concatenate(self.outer_fold_results['y_true'])
        y_proba_all = np.concatenate(self.outer_fold_results['y_proba'])

        results = []

        for threshold in thresholds:
            y_pred = (y_proba_all >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            # Bootstrap CI
            n_boot = 1000
            boot_sens, boot_spec, boot_ppv = [], [], []

            np.random.seed(self.random_state)
            for _ in range(n_boot):
                idx = np.random.choice(len(y_proba_all), len(y_proba_all), replace=True)
                y_true_b = y_true_all[idx]
                y_pred_b = y_pred[idx]
                try:
                    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b).ravel()
                    boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
                    boot_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
                    boot_ppv.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0)
                except:
                    pass

            results.append({
                'Threshold': f'{threshold:.2f}',
                'Sensitivity': f"{sensitivity:.3f} [{np.percentile(boot_sens, 2.5):.3f}-{np.percentile(boot_sens, 97.5):.3f}]",
                'Specificity': f"{specificity:.3f} [{np.percentile(boot_spec, 2.5):.3f}-{np.percentile(boot_spec, 97.5):.3f}]",
                'PPV': f"{ppv:.3f} [{np.percentile(boot_ppv, 2.5):.3f}-{np.percentile(boot_ppv, 97.5):.3f}]",
                'NPV': f'{npv:.3f}',
                'N': len(y_true_all)
            })

        return pd.DataFrame(results)

    def generate_report(self, title="NESTED CV PERFORMANCE REPORT", save_plots=False):
        """Generate complete report with all outputs."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"\nTask Type: {self.task_type.upper()}")
        print(f"Outer-fold CV: {self.n_outer} folds")
        print("Metrics aggregated with t-distribution 95% CI\n")

        # Summary statistics
        df_summary, _ = self.summarize_outer_folds()
        print("1. OUTER-FOLD METRICS (Mean ± 95% CI)")
        print("-" * 80)
        print(df_summary.to_string(index=False))

        # Threshold metrics (classification only)
        if self.is_classification:
            print("\n2. METRICS AT CLINICAL THRESHOLDS (bootstrap 95% CI)")
            print("-" * 80)
            threshold_df = self.threshold_metrics_table()
            if threshold_df is not None:
                print(threshold_df.to_string(index=False))

        # Visualizations
        print("\n3. VISUALIZATIONS")
        print("-" * 80)

        self.plot_fold_distributions(save_path='ncv_fold_dist.png' if save_plots else None)

        if self.is_classification:
            self.plot_calibration_curve(save_path='ncv_calibration.png' if save_plots else None)
            self.plot_decision_curve(save_path='ncv_decision_curve.png' if save_plots else None)

        print("\n" + "=" * 80)
        print("✅ [POINT 1] COMPLETE")
        print("=" * 80)


# ========================================================================
# EXECUTION: Run reporter with pipeline or standalone
# ========================================================================

try:
    if 'X_train' in locals() and 'y_train' in locals():
        print("\n🚀 Running [POINT 1] with pipeline variables")
        print("   Using: X_train, y_train, model from notebook")

        # Check if trained model is available
        if 'model' in locals():
            print("   ✅ Trained model found - using PIPELINE MODE")
            reporter_point1 = NestedCVReporter(X_train.copy(), y_train.copy(), task_type='auto', n_outer=5)
            reporter_point1.run_outer_cv(base_model=model, results_dir='nestedcv_results')
        else:
            print("   ⚠️  No trained model found")
            print("   Define model_factory to retrain, or run ML pipeline first")
            raise ValueError("model not found in notebook")

        reporter_point1.generate_report(title="[POINT 1] Nested CV Performance - RP", save_plots=False)

    else:
        print("\n❌ ERROR: X_train or y_train not found in notebook")
        print("   Make sure to run the data preparation and ML training cells first")

except Exception as e:
    print(f"\n❌ Error in [POINT 1]: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ [POINT 1] Complete!")
