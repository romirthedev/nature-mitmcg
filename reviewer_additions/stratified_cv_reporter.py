#!/usr/bin/env python3
"""
STRATIFIED CV REPORTER
======================
Extended nested CV reporter that computes performance metrics and calibration
stratified by demographic subgroups (sex, SES, etc.).

Implements "cheap version" approach: train once on full sample, evaluate by subgroup.

Key Functions:
- create_ses_groups(): Convert parent_income to low/high SES groups
- get_subgroup_metrics(): Compute performance metrics for each subgroup
- NestedCVReporterStratified: Extended class with stratified analysis
- plot_calibration_by_subgroup(): Calibration curves by group
- plot_decision_curve_by_subgroup(): Decision curves by group
- fairness_summary_report(): Print demographic equity summary

Usage:
    from stratified_cv_reporter import NestedCVReporterStratified, create_ses_groups
    from nested_cv_reporter import NestedCVReporter, load_abcd_data
    
    # Prepare data
    df = load_abcd_data()
    X = df.drop(columns=['target', 'sex', 'parent_income'])
    y = df['target']
    sex_labels = df['sex'].map({1: 'Male', 2: 'Female'})
    ses_labels = create_ses_groups(df)['ses_group']
    
    # Run stratified analysis
    reporter = NestedCVReporterStratified(X, y, n_outer=5)
    reporter.run_outer_cv_stratified(model_factory, sex_labels, ses_labels)
    
    # Get results
    print(reporter.get_stratified_metrics_summary())
    reporter.plot_all_stratified_calibrations()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import base class from existing module
import sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from reviewer_additions.nested_cv_reporter import NestedCVReporter, get_results_dir
except ImportError:
    print("Warning: nested_cv_reporter not found. Some features may not work.")
    NestedCVReporter = None


# ============================================================================
# SUBGROUP CREATION HELPERS
# ============================================================================

def create_ses_groups(data, ses_column='parent_income', method='median'):
    """
    Create low/high SES groups from a continuous SES variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing SES variable
    ses_column : str
        Column name for SES indicator (default: 'parent_income')
    method : str
        'median' (default): Split at median
        'tertile': Split at tertiles (low, mid, high)
        'quartile': Split at quartiles
    
    Returns:
    --------
    pd.DataFrame
        Copy of data with 'ses_group' column added
    """
    data = data.copy()
    
    if ses_column not in data.columns:
        raise ValueError(f"Column '{ses_column}' not found in data. Available: {data.columns.tolist()}")
    
    # Remove NaN
    valid_idx = data[ses_column].notna()
    
    if method == 'median':
        threshold = data.loc[valid_idx, ses_column].median()
        data['ses_group'] = 'Low_SES'
        data.loc[data[ses_column] >= threshold, 'ses_group'] = 'High_SES'
    
    elif method == 'tertile':
        low_thresh = data.loc[valid_idx, ses_column].quantile(0.33)
        high_thresh = data.loc[valid_idx, ses_column].quantile(0.67)
        data['ses_group'] = 'Low_SES'
        data.loc[data[ses_column] >= low_thresh, 'ses_group'] = 'Mid_SES'
        data.loc[data[ses_column] >= high_thresh, 'ses_group'] = 'High_SES'
    
    elif method == 'quartile':
        q25 = data.loc[valid_idx, ses_column].quantile(0.25)
        q75 = data.loc[valid_idx, ses_column].quantile(0.75)
        data['ses_group'] = 'Low_SES'
        data.loc[data[ses_column] >= q75, 'ses_group'] = 'High_SES'
        data.loc[(data[ses_column] >= q25) & (data[ses_column] < q75), 'ses_group'] = 'Mid_SES'
    
    return data


def create_binary_subgroups(data, sex_column='sex', ses_column='parent_income'):
    """
    Create comprehensive subgroup labels (sex × SES combinations).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with sex and SES columns
    sex_column : str
        Column name for sex (values: 1=Male, 2=Female, or M/F strings)
    ses_column : str
        Column name for SES indicator
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: 'sex_group', 'ses_group', 'combined_group'
    """
    subgroups = pd.DataFrame(index=data.index)
    
    # Convert sex to readable labels
    if data[sex_column].dtype == 'object':
        subgroups['sex_group'] = data[sex_column]
    else:
        subgroups['sex_group'] = data[sex_column].map({1: 'Male', 2: 'Female'})
    
    # Create SES groups
    ses_data = create_ses_groups(data, ses_column=ses_column)
    subgroups['ses_group'] = ses_data['ses_group']
    
    # Combined group labels
    subgroups['combined_group'] = (
        subgroups['sex_group'] + '_' + subgroups['ses_group']
    )
    
    return subgroups


# ============================================================================
# SUBGROUP METRICS COMPUTATION
# ============================================================================

def get_subgroup_metrics(y_true, y_pred, y_proba, subgroup_labels, group_name=None):
    """
    Calculate performance metrics for each subgroup.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels (at threshold 0.5)
    y_proba : array-like
        Predicted probabilities
    subgroup_labels : pd.Series or array-like
        Group membership for each sample
    group_name : str, optional
        Name of grouping variable (for reporting)
    
    Returns:
    --------
    dict
        Metrics for each subgroup with structure:
        {
            'group_name': {
                'auc': float,
                'sensitivity': float,
                'specificity': float,
                'ppv': float,
                'npv': float,
                'accuracy': float,
                'n_samples': int,
                'n_positive': int,
                'prevalence': float
            },
            ...
        }
    """
    results = {}
    
    for group in np.unique(subgroup_labels):
        mask = np.array(subgroup_labels) == group
        
        if mask.sum() == 0:
            continue
        
        y_true_group = np.array(y_true)[mask]
        y_pred_group = np.array(y_pred)[mask]
        y_proba_group = np.array(y_proba)[mask]
        
        # Handle edge cases
        if len(np.unique(y_true_group)) < 2:
            results[str(group)] = {
                'auc': np.nan,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'ppv': np.nan,
                'npv': np.nan,
                'accuracy': np.nan,
                'n_samples': mask.sum(),
                'n_positive': y_true_group.sum(),
                'prevalence': y_true_group.mean(),
                'warning': 'Only one class in group'
            }
            continue
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        # Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        try:
            auc = roc_auc_score(y_true_group, y_proba_group)
        except:
            auc = np.nan
        
        results[str(group)] = {
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy,
            'n_samples': int(mask.sum()),
            'n_positive': int(y_true_group.sum()),
            'prevalence': float(y_true_group.mean())
        }
    
    return results


# ============================================================================
# EXTENDED NESTED CV REPORTER WITH STRATIFICATION
# ============================================================================

class NestedCVReporterStratified:
    """
    Extended nested CV reporter with demographic stratification.
    
    Computes performance metrics and calibration separately for:
    - Sex subgroups (Male/Female)
    - SES subgroups (Low/High)
    - Combined subgroups (e.g., Male_Low_SES, Female_High_SES)
    """
    
    def __init__(self, base_reporter_or_X, y=None, n_outer=5, random_state=42):
        """
        Initialize stratified reporter.
        
        Parameters:
        -----------
        base_reporter_or_X : NestedCVReporter or pd.DataFrame
            Either existing NestedCVReporter instance or feature matrix X
        y : pd.Series, optional
            Target variable (required if first arg is X)
        n_outer : int
            Number of outer CV folds
        random_state : int
            Random seed
        """
        if isinstance(base_reporter_or_X, NestedCVReporter):
            self.base_reporter = base_reporter_or_X
            self.X = base_reporter_or_X.X
            self.y = base_reporter_or_X.y
        else:
            # Create new base reporter
            self.X = base_reporter_or_X
            self.y = y
            if NestedCVReporter is None:
                raise ImportError("nested_cv_reporter module required")
            self.base_reporter = NestedCVReporter(self.X, self.y, n_outer=n_outer, random_state=random_state)
        
        self.n_outer = n_outer
        self.random_state = random_state
        
        # Storage for stratified results
        self.stratified_results = {}  # fold_id -> group -> metrics
        self.sex_results = {}
        self.ses_results = {}
        self.combined_results = {}
        
        # Store subgroup labels
        self.sex_labels = None
        self.ses_labels = None
        self.combined_labels = None
    
    def run_outer_cv_stratified(self, model_factory, sex_labels=None, ses_labels=None, results_dir=None):
        """
        Run outer CV with stratified evaluation.
        
        Parameters:
        -----------
        model_factory : callable
            Function returning fresh model instance
        sex_labels : pd.Series or array-like, optional
            Sex group labels (index-aligned to X)
        ses_labels : pd.Series or array-like, optional
            SES group labels (index-aligned to X)
        results_dir : str, optional
            Directory to save results
        """
        if results_dir is None:
            results_dir = get_results_dir()
        
        # Store labels
        if sex_labels is not None:
            self.sex_labels = pd.Series(sex_labels, index=self.X.index)
        
        if ses_labels is not None:
            self.ses_labels = pd.Series(ses_labels, index=self.X.index)
        
        if self.sex_labels is not None and self.ses_labels is not None:
            self.combined_labels = (
                self.sex_labels.astype(str) + '_' + self.ses_labels.astype(str)
            )
        
        # Run base outer CV
        print("\n[1] Running outer CV with stratified evaluation...")
        self.base_reporter.run_outer_cv(model_factory, results_dir=results_dir)
        
        # Compute stratified metrics for each fold
        print("[2] Computing metrics by demographic subgroup...")
        
        self.sex_results = {'all_folds': {}}
        self.ses_results = {'all_folds': {}}
        self.combined_results = {'all_folds': {}}
        
        # Aggregate across folds
        y_true_all = np.concatenate(self.base_reporter.outer_fold_results['y_true'])
        y_pred_all = np.concatenate(self.base_reporter.outer_fold_results['y_pred'])
        y_proba_all = np.concatenate(self.base_reporter.outer_fold_results['y_proba'])
        
        # Get indices for reconstruction
        idx_counter = 0
        for fold_idx in range(self.n_outer):
            n_fold = len(self.base_reporter.outer_fold_results['y_true'][fold_idx])
            fold_indices = range(idx_counter, idx_counter + n_fold)
            idx_counter += n_fold
            
            # Stratified metrics for this fold
            if self.sex_labels is not None:
                sex_labels_fold = self.sex_labels.iloc[fold_indices]
                self.sex_results[f'fold_{fold_idx}'] = get_subgroup_metrics(
                    self.base_reporter.outer_fold_results['y_true'][fold_idx],
                    self.base_reporter.outer_fold_results['y_pred'][fold_idx],
                    self.base_reporter.outer_fold_results['y_proba'][fold_idx],
                    sex_labels_fold
                )
            
            if self.ses_labels is not None:
                ses_labels_fold = self.ses_labels.iloc[fold_indices]
                self.ses_results[f'fold_{fold_idx}'] = get_subgroup_metrics(
                    self.base_reporter.outer_fold_results['y_true'][fold_idx],
                    self.base_reporter.outer_fold_results['y_pred'][fold_idx],
                    self.base_reporter.outer_fold_results['y_proba'][fold_idx],
                    ses_labels_fold
                )
            
            if self.combined_labels is not None:
                combined_labels_fold = self.combined_labels.iloc[fold_indices]
                self.combined_results[f'fold_{fold_idx}'] = get_subgroup_metrics(
                    self.base_reporter.outer_fold_results['y_true'][fold_idx],
                    self.base_reporter.outer_fold_results['y_pred'][fold_idx],
                    self.base_reporter.outer_fold_results['y_proba'][fold_idx],
                    combined_labels_fold
                )
        
        # Compute aggregated metrics (all folds combined)
        if self.sex_labels is not None:
            self.sex_results['all_folds'] = get_subgroup_metrics(
                y_true_all, y_pred_all, y_proba_all, self.sex_labels
            )
        
        if self.ses_labels is not None:
            self.ses_results['all_folds'] = get_subgroup_metrics(
                y_true_all, y_pred_all, y_proba_all, self.ses_labels
            )
        
        if self.combined_labels is not None:
            self.combined_results['all_folds'] = get_subgroup_metrics(
                y_true_all, y_pred_all, y_proba_all, self.combined_labels
            )
        
        # Save stratified results
        self._save_stratified_results(results_dir)
        print("[3] ✅ Stratified analysis complete!")
    
    def _save_stratified_results(self, results_dir):
        """Save stratified results to disk."""
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save as pickle
        with open(os.path.join(results_dir, 'stratified_results_sex.pkl'), 'wb') as f:
            pickle.dump(self.sex_results, f)
        
        with open(os.path.join(results_dir, 'stratified_results_ses.pkl'), 'wb') as f:
            pickle.dump(self.ses_results, f)
        
        with open(os.path.join(results_dir, 'stratified_results_combined.pkl'), 'wb') as f:
            pickle.dump(self.combined_results, f)
        
        # Save as CSV (aggregated)
        if self.sex_results:
            sex_df = pd.DataFrame(self.sex_results['all_folds']).T
            sex_df.to_csv(os.path.join(results_dir, 'metrics_by_sex.csv'))
        
        if self.ses_results:
            ses_df = pd.DataFrame(self.ses_results['all_folds']).T
            ses_df.to_csv(os.path.join(results_dir, 'metrics_by_ses.csv'))
        
        if self.combined_results:
            combined_df = pd.DataFrame(self.combined_results['all_folds']).T
            combined_df.to_csv(os.path.join(results_dir, 'metrics_by_combined.csv'))
    
    def get_stratified_metrics_summary(self, stratification='sex'):
        """
        Get formatted summary of stratified metrics.
        
        Parameters:
        -----------
        stratification : str
            'sex', 'ses', or 'combined'
        
        Returns:
        --------
        pd.DataFrame
            Metrics table with groups as rows
        """
        if stratification == 'sex':
            results = self.sex_results.get('all_folds', {})
        elif stratification == 'ses':
            results = self.ses_results.get('all_folds', {})
        elif stratification == 'combined':
            results = self.combined_results.get('all_folds', {})
        else:
            raise ValueError("stratification must be 'sex', 'ses', or 'combined'")
        
        if not results:
            print(f"No results for {stratification}")
            return None
        
        df = pd.DataFrame(results).T
        df = df.round(4)
        return df
    
    def print_stratified_summary(self):
        """Print formatted stratified analysis summary."""
        print("\n" + "="*80)
        print("STRATIFIED PERFORMANCE ANALYSIS")
        print("="*80)
        
        if self.sex_results and self.sex_results['all_folds']:
            print("\n📊 PERFORMANCE BY SEX:")
            print("-"*80)
            df_sex = self.get_stratified_metrics_summary('sex')
            print(df_sex[['auc', 'sensitivity', 'specificity', 'ppv', 'n_samples']].to_string())
        
        if self.ses_results and self.ses_results['all_folds']:
            print("\n📊 PERFORMANCE BY SES:")
            print("-"*80)
            df_ses = self.get_stratified_metrics_summary('ses')
            print(df_ses[['auc', 'sensitivity', 'specificity', 'ppv', 'n_samples']].to_string())
        
        if self.combined_results and self.combined_results['all_folds']:
            print("\n📊 PERFORMANCE BY SEX × SES:")
            print("-"*80)
            df_combined = self.get_stratified_metrics_summary('combined')
            print(df_combined[['auc', 'sensitivity', 'specificity', 'n_samples']].to_string())
        
        print("\n" + "="*80)


# ============================================================================
# STRATIFIED VISUALIZATION
# ============================================================================

def plot_calibration_by_subgroup(y_true, y_proba, subgroup_labels, group_name, 
                                 n_bins=10, figsize=None, save_path=None):
    """
    Create calibration curves for each subgroup.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    subgroup_labels : pd.Series or array-like
        Group membership for each sample
    group_name : str
        Name of grouping variable (e.g., 'Sex', 'SES')
    n_bins : int
        Number of calibration bins
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    groups = np.unique(subgroup_labels)
    n_groups = len(groups)
    
    if figsize is None:
        figsize = (6 * min(n_groups, 2), 5 * (n_groups // 2 + 1))
    
    if n_groups > 4:
        n_cols = 2
        n_rows = (n_groups + 1) // 2
    else:
        n_cols = min(2, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        mask = np.array(subgroup_labels) == group
        
        y_true_group = np.array(y_true)[mask]
        y_proba_group = np.array(y_proba)[mask]
        
        if len(np.unique(y_true_group)) < 2:
            ax.text(0.5, 0.5, f'Single class only\n(n={mask.sum()})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{group}")
            continue
        
        # Plot calibration
        frac_pos, mean_pred = calibration_curve(y_true_group, y_proba_group, n_bins=n_bins)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', lw=2, markersize=8, label='Model')
        ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.2)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f"{group} (n={mask.sum()}, prev={y_true_group.mean():.2f})", fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"[POINT 2] Calibration by {group_name}", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()


def plot_decision_curve_by_subgroup(y_true, y_proba, subgroup_labels, group_name,
                                    figsize=None, save_path=None):
    """
    Create decision curves (net benefit) for each subgroup.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    subgroup_labels : pd.Series or array-like
        Group membership for each sample
    group_name : str
        Name of grouping variable
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    groups = np.unique(subgroup_labels)
    n_groups = len(groups)
    
    if figsize is None:
        figsize = (6 * min(n_groups, 2), 5 * (n_groups // 2 + 1))
    
    if n_groups > 4:
        n_cols = 2
        n_rows = (n_groups + 1) // 2
    else:
        n_cols = min(2, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        mask = np.array(subgroup_labels) == group
        
        y_true_group = np.array(y_true)[mask]
        y_proba_group = np.array(y_proba)[mask]
        n_group = mask.sum()
        
        # Decision curve
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba_group >= threshold).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (y_true_group == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true_group == 0))
            
            net_benefit = (tp / n_group) - (fp / n_group) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        
        ax.plot(thresholds, net_benefits, lw=2.5, label='Model')
        ax.axhline(y=0, color='k', linestyle='--', lw=1, alpha=0.5)
        ax.fill_between(thresholds, 0, net_benefits, alpha=0.2)
        
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Net Benefit', fontsize=11)
        ax.set_title(f"{group} (n={n_group})", fontsize=12)
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"[POINT 2] Decision Curve by {group_name}", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict, metrics_to_plot=None, figsize=(12, 6)):
    """
    Create comparison bar plots for stratified metrics.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of group -> metrics mapping
    metrics_to_plot : list, optional
        Which metrics to plot (default: AUC, sensitivity, specificity, PPV)
    figsize : tuple
        Figure size
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['auc', 'sensitivity', 'specificity', 'ppv']
    
    df = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)
    
    for ax, metric in zip(axes, metrics_to_plot):
        df[metric].sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel(metric.capitalize(), fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def fairness_summary_report(stratified_reporter):
    """
    Print comprehensive fairness and equity summary.
    
    Parameters:
    -----------
    stratified_reporter : NestedCVReporterStratified
        Fitted stratified reporter
    """
    print("\n" + "="*80)
    print("FAIRNESS & DEMOGRAPHIC EQUITY ANALYSIS")
    print("="*80)
    
    # Sex analysis
    if stratified_reporter.sex_results and stratified_reporter.sex_results['all_folds']:
        print("\n🔍 SEX EQUITY ANALYSIS:")
        print("-"*80)
        sex_metrics = stratified_reporter.sex_results['all_folds']
        
        # Find performance gaps
        aucs = {g: v['auc'] for g, v in sex_metrics.items() if not np.isnan(v['auc'])}
        if len(aucs) > 1:
            auc_gap = max(aucs.values()) - min(aucs.values())
            print(f"  AUC Range: {min(aucs.values()):.4f} - {max(aucs.values()):.4f}")
            print(f"  AUC Gap: {auc_gap:.4f}")
            if auc_gap > 0.05:
                print(f"  ⚠️  Substantial gap detected (>0.05)")
            else:
                print(f"  ✅ Minimal gap (<0.05)")
        
        sens = {g: v['sensitivity'] for g, v in sex_metrics.items() if not np.isnan(v['sensitivity'])}
        if len(sens) > 1:
            sens_gap = max(sens.values()) - min(sens.values())
            print(f"  Sensitivity Range: {min(sens.values()):.4f} - {max(sens.values()):.4f}")
            print(f"  Sensitivity Gap: {sens_gap:.4f}")
    
    # SES analysis
    if stratified_reporter.ses_results and stratified_reporter.ses_results['all_folds']:
        print("\n🔍 SES EQUITY ANALYSIS:")
        print("-"*80)
        ses_metrics = stratified_reporter.ses_results['all_folds']
        
        aucs = {g: v['auc'] for g, v in ses_metrics.items() if not np.isnan(v['auc'])}
        if len(aucs) > 1:
            auc_gap = max(aucs.values()) - min(aucs.values())
            print(f"  AUC Range: {min(aucs.values()):.4f} - {max(aucs.values()):.4f}")
            print(f"  AUC Gap: {auc_gap:.4f}")
            if auc_gap > 0.05:
                print(f"  ⚠️  Substantial gap detected (>0.05)")
            else:
                print(f"  ✅ Minimal gap (<0.05)")
        
        # Check if low SES is disadvantaged
        sens = {g: v['sensitivity'] for g, v in ses_metrics.items() if not np.isnan(v['sensitivity'])}
        if 'Low_SES' in sens and 'High_SES' in sens:
            print(f"  Low-SES Sensitivity: {sens['Low_SES']:.4f}")
            print(f"  High-SES Sensitivity: {sens['High_SES']:.4f}")
            if sens['Low_SES'] < sens['High_SES'] - 0.05:
                print(f"  ⚠️  Model underperforms for low-SES group")
    
    print("\n" + "="*80)


# ============================================================================
# STANDALONE EXECUTION (when run as python script)
# ============================================================================

def run_standalone_demo():
    """
    Run complete stratified analysis on synthetic data.
    Called when script is run directly: python stratified_cv_reporter.py
    """
    print("="*80)
    print("STRATIFIED CV REPORTER - STANDALONE DEMO")
    print("="*80)
    
    # Load base reporter for synthetic data
    from reviewer_additions.nested_cv_reporter import NestedCVReporter, load_abcd_data, generate_synthetic_data
    from sklearn.linear_model import LogisticRegression
    
    print("\n[1] Loading data...")
    df = load_abcd_data()
    if df is None:
        print("    Using synthetic data...")
        X, y = generate_synthetic_data()
        df = pd.concat([X, pd.Series(y, name='target', index=X.index)], axis=1)
    
    print(f"    Shape: {df.shape}")
    
    print("\n[2] Preparing features and target...")
    # Use last column as target
    X = df.iloc[:, :-1]
    y_series = df.iloc[:, -1]
    
    # Ensure binary target
    if y_series.dtype == 'object':
        y = (y_series > y_series.median()).astype(int)
    else:
        y = y_series.astype(int)
    
    print(f"    X shape: {X.shape}, y shape: {y.shape}")
    print(f"    Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    print("\n[3] Creating demographic groups...")
    # Create synthetic sex groups (randomly assign)
    np.random.seed(42)
    sex_labels = pd.Series(['Male' if np.random.random() > 0.5 else 'Female' for _ in range(len(X))], index=X.index)
    
    # Create SES groups based on feature variation
    ses_base = X.iloc[:, 0]  # Use first feature as proxy
    ses_threshold = ses_base.median()
    ses_labels = pd.Series(['Low_SES' if v < ses_threshold else 'High_SES' for v in ses_base], index=X.index)
    
    print(f"    Sex: {sex_labels.value_counts().to_dict()}")
    print(f"    SES: {ses_labels.value_counts().to_dict()}")
    
    print("\n[4] Initializing stratified reporter...")
    reporter = NestedCVReporterStratified(X, y, n_outer=5, random_state=42)
    
    print("\n[5] Defining model factory...")
    def model_factory():
        return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    print("\n[6] Running stratified outer CV...")
    reporter.run_outer_cv_stratified(model_factory, sex_labels, ses_labels)
    
    print("\n[7] Generating stratified reports...")
    reporter.print_stratified_summary()
    
    print("\n[8] Creating visualizations...")
    y_true_all = np.concatenate(reporter.base_reporter.outer_fold_results['y_true'])
    y_proba_all = np.concatenate(reporter.base_reporter.outer_fold_results['y_proba'])
    
    print("\n    📊 Calibration by Sex...")
    plot_calibration_by_subgroup(y_true_all, y_proba_all, sex_labels, 'Sex', 
                                save_path='calibration_by_sex.png')
    
    print("\n    📊 Calibration by SES...")
    plot_calibration_by_subgroup(y_true_all, y_proba_all, ses_labels, 'SES',
                                save_path='calibration_by_ses.png')
    
    print("\n    📊 Decision Curves by Sex...")
    plot_decision_curve_by_subgroup(y_true_all, y_proba_all, sex_labels, 'Sex',
                                   save_path='decision_curve_by_sex.png')
    
    print("\n    📊 Decision Curves by SES...")
    plot_decision_curve_by_subgroup(y_true_all, y_proba_all, ses_labels, 'SES',
                                   save_path='decision_curve_by_ses.png')
    
    print("\n[9] Fairness Analysis...")
    fairness_summary_report(reporter)
    
    print("\n[10] Exporting results...")
    results_dir = get_results_dir()
    
    # Export metrics tables
    df_sex = reporter.get_stratified_metrics_summary('sex')
    df_sex.to_csv(f"{results_dir}/metrics_by_sex_final.csv")
    
    df_ses = reporter.get_stratified_metrics_summary('ses')
    df_ses.to_csv(f"{results_dir}/metrics_by_ses_final.csv")
    
    print(f"    ✅ Results saved to: {results_dir}/")
    
    print("\n" + "="*80)
    print("✅ STRATIFIED ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - calibration_by_sex.png")
    print(f"  - calibration_by_ses.png")
    print(f"  - decision_curve_by_sex.png")
    print(f"  - decision_curve_by_ses.png")
    print(f"  - metrics_by_sex_final.csv")
    print(f"  - metrics_by_ses_final.csv")
    print(f"\nTo use in Jupyter notebook:")
    print(f"  from stratified_cv_reporter import NestedCVReporterStratified")
    print(f"  reporter = NestedCVReporterStratified(X, y)")
    print(f"  reporter.run_outer_cv_stratified(model_factory, sex_labels, ses_labels)")


if __name__ == "__main__":
    run_standalone_demo()
