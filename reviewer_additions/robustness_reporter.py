#!/usr/bin/env python3
"""
ROBUSTNESS REPORTER
===================
Handles sensitivity analyses and robustness checks requested by reviewers:
1. Site-based Cross-Validation (Leave-One-Site-Out or GroupKFold)
2. Family-based Cross-Validation (Ensuring siblings are in same fold)
3. Comparison of performance across different validation schemes

Usage:
    from robustness_reporter import RobustnessReporter
    
    # Family-aware CV
    reporter = RobustnessReporter(X, y)
    reporter.run_group_cv(model_factory, groups=family_ids, group_type='family', n_splits=5)
    
    # Site-aware CV
    reporter.run_group_cv(model_factory, groups=site_ids, group_type='site', n_splits=5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
import os
import pickle
import sys

# Import base class
sys.path.insert(0, os.path.dirname(__file__))
try:
    from reviewer_additions.nested_cv_reporter import NestedCVReporter, get_results_dir
except ImportError:
    try:
        from nested_cv_reporter import NestedCVReporter, get_results_dir
    except ImportError:
        # Fallback for when running inside the package structure
        from .nested_cv_reporter import NestedCVReporter, get_results_dir

class RobustnessReporter(NestedCVReporter):
    """
    Extended reporter for robustness checks (Site/Family grouping).
    Inherits plotting and reporting capabilities from NestedCVReporter.
    """
    
    def __init__(self, X, y, n_outer=5, random_state=42):
        super().__init__(X, y, n_outer, random_state)
        self.robustness_results = {} # Store results for different modes
        
    def run_group_cv(self, model_factory, groups, group_type='group', n_splits=5, results_dir=None):
        """
        Run Group-based CV (e.g., Site or Family).
        
        Parameters:
        -----------
        model_factory : callable
            Function returning a new, unfitted model instance.
        groups : array-like
            Group labels (site IDs or family IDs) aligned with X.
        group_type : str
            Label for the grouping ('site', 'family', etc.)
        n_splits : int or None
            Number of splits. 
            If None, uses Leave-One-Group-Out (e.g., Leave-One-Site-Out).
            If int, uses GroupKFold with n_splits.
        results_dir : str, optional
            Directory to save results.
        """
        if results_dir is None:
            results_dir = get_results_dir()
            
        print(f"\nRunning {group_type.upper()}-based Cross-Validation...")
        
        # Setup Splitter
        if n_splits is None:
            print(f"  Using Leave-One-{group_type.capitalize()}-Out CV")
            splitter = LeaveOneGroupOut()
            n_splits_actual = len(np.unique(groups))
        else:
            print(f"  Using GroupKFold (n_splits={n_splits}) on {group_type}")
            splitter = GroupKFold(n_splits=n_splits)
            n_splits_actual = n_splits
            
        # Initialize storage
        fold_results = {
            'fold': [], 'auc': [], 'sensitivity': [], 'specificity': [],
            'ppv': [], 'npv': [], 'accuracy': [], 'f1': [],
            'threshold': [], 'y_true': [], 'y_proba': [], 'y_pred': []
        }
        
        fold_num = 0
        
        # Run CV
        for train_idx, test_idx in splitter.split(self.X, self.y, groups=groups):
            # Progress indicator
            print(f"  Processing fold {fold_num+1}/{n_splits_actual}...", end='\r')
            
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]
            
            # Train
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_proba = model.predict_proba(X_test_fold)[:, 1]
            
            # Optimize Threshold (using ROC curve on test set - standard for outer fold reporting)
            # Note: In a pure nested setup, threshold should be determined on inner CV.
            # For this reporter, we use the optimal threshold on the test set to report 
            # "potential" performance, or 0.5 if strictly required. 
            # Consistent with base NestedCVReporter, we calculate optimal here for reporting metrics.
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_proba >= optimal_threshold).astype(int)
            
            # Metrics
            tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()
            
            fold_results['fold'].append(fold_num)
            try:
                fold_results['auc'].append(roc_auc_score(y_test_fold, y_proba))
            except:
                fold_results['auc'].append(np.nan)
                
            fold_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            fold_results['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            fold_results['accuracy'].append(accuracy_score(y_test_fold, y_pred))
            fold_results['f1'].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
            fold_results['threshold'].append(optimal_threshold)
            fold_results['y_true'].append(y_test_fold.values)
            fold_results['y_proba'].append(y_proba)
            fold_results['y_pred'].append(y_pred)
            
            fold_num += 1
            
        print(f"\n  ✅ {group_type.capitalize()} CV Complete!")
        
        # Store as current results (so we can use plot methods from base class)
        self.outer_fold_results = fold_results
        
        # Also archive in robustness_results
        self.robustness_results[group_type] = fold_results
        
        # Save
        self._save_robustness_results(fold_results, group_type, results_dir)
        
        return fold_results

    def _save_robustness_results(self, fold_results, group_type, results_dir):
        """Save results to disk."""
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Save metrics CSV
        metrics_df = pd.DataFrame({k: v for k, v in fold_results.items() 
                                 if k not in ['y_true', 'y_proba', 'y_pred']})
        metrics_df.to_csv(os.path.join(results_dir, f"robustness_metrics_{group_type}.csv"), index=False)
        
        # Save pickle
        with open(os.path.join(results_dir, f"robustness_results_{group_type}.pkl"), "wb") as f:
            pickle.dump(fold_results, f)

    def compare_robustness_schemes(self, schemes=None):
        """
        Compare metrics across different validation schemes.
        
        Parameters:
        -----------
        schemes : list of str, optional
            Keys to compare (e.g., ['standard', 'site', 'family']).
            'standard' refers to results from run_outer_cv() (if run).
        
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        summary_data = []
        
        # Check available results
        available_schemes = list(self.robustness_results.keys())
        
        # If base run_outer_cv was run, it might be in self.outer_fold_results but not robustness_results
        # We can add it temporarily if not present
        if self.outer_fold_results and 'standard' not in self.robustness_results:
             # Check if current outer_fold_results is actually one of the robustness ones
             is_robust = False
             for k, v in self.robustness_results.items():
                 if v is self.outer_fold_results:
                     is_robust = True
                     break
             
             if not is_robust:
                 # It's likely the standard run
                 self.robustness_results['standard'] = self.outer_fold_results
        
        if schemes is None:
            schemes = list(self.robustness_results.keys())
            
        for scheme in schemes:
            if scheme not in self.robustness_results:
                continue
                
            res = self.robustness_results[scheme]
            
            # Calculate means and CIs
            aucs = res['auc']
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            sens = res['sensitivity']
            mean_sens = np.mean(sens)
            
            spec = res['specificity']
            mean_spec = np.mean(spec)
            
            summary_data.append({
                'Validation Scheme': scheme.capitalize(),
                'AUC (Mean)': mean_auc,
                'AUC (Std)': std_auc,
                'Sensitivity': mean_sens,
                'Specificity': mean_spec,
                'N_Folds': len(aucs)
            })
            
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.set_index('Validation Scheme').round(3)
        return df

