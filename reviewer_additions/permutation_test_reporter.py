"""
Permutation Test Reporter for Reviewer Point 4

Implements label-permutation significance testing to demonstrate that model
performance exceeds what would be expected from chance given the covariance
structure of the features.

Addresses reviewer concern:
"I would recommend that authors measure model significance against null 
distributions using label permutations because a massive predictive pleiotropy 
within the feature space could render even highly predictive models nonsignificant."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score, make_scorer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PermutationTestReporter:
    """
    Performs permutation-based significance testing for ML models.
    
    This addresses the concern that high performance might be due to 
    massive covariance in features rather than meaningful prediction.
    """
    
    def __init__(self, X, y, n_permutations=100, cv_folds=5, random_state=42):
        """
        Initialize the permutation test framework.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        n_permutations : int
            Number of permutation tests to run (default: 100)
        cv_folds : int
            Number of cross-validation folds (default: 5)
        random_state : int
            Random seed for reproducibility
        """
        self.X = X
        self.y = np.array(y)
        self.n_permutations = n_permutations
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        
        # Determine task type
        self.is_classification = len(np.unique(y)) <= 10
        
    def run_permutation_test(self, model_factory, test_name="Model", scoring=None):
        """
        Run permutation test for model significance.
        
        Parameters:
        -----------
        model_factory : callable
            Function that returns a fresh model instance
        test_name : str
            Name for this test (for results tracking)
        scoring : str or callable
            Scoring metric ('roc_auc', 'r2', or custom scorer)
            
        Returns:
        --------
        dict : Results containing observed score, null distribution, p-value
        """
        np.random.seed(self.random_state)
        
        # Auto-select scoring if not provided
        if scoring is None:
            scoring = 'roc_auc' if self.is_classification else 'r2'
        
        print(f"\n🔬 Running Permutation Test: {test_name}")
        print(f"   Metric: {scoring}")
        print(f"   Permutations: {self.n_permutations}")
        print(f"   CV Folds: {self.cv_folds}")
        
        # 1. Compute observed performance
        print(f"\n[1/3] Computing observed performance...")
        model_obs = model_factory()
        
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        obs_scores = cross_val_score(model_obs, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1)
        obs_mean = np.mean(obs_scores)
        obs_std = np.std(obs_scores)
        
        print(f"   Observed: {obs_mean:.4f} ± {obs_std:.4f}")
        
        # 2. Generate null distribution via label permutation
        print(f"\n[2/3] Generating null distribution ({self.n_permutations} permutations)...")
        perm_scores = []
        
        for i in range(self.n_permutations):
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{self.n_permutations}")
            
            # Permute labels
            y_perm = np.random.permutation(self.y)
            
            # Train model on permuted data
            model_perm = model_factory()
            perm_cv_scores = cross_val_score(model_perm, self.X, y_perm, cv=cv, scoring=scoring, n_jobs=-1)
            perm_scores.append(np.mean(perm_cv_scores))
        
        perm_scores = np.array(perm_scores)
        
        # 3. Compute p-value
        print(f"\n[3/3] Computing significance...")
        # Two-tailed test: how many permutations >= observed?
        p_value = (np.sum(perm_scores >= obs_mean) + 1) / (len(perm_scores) + 1)
        
        # Compute z-score
        perm_mean = np.mean(perm_scores)
        perm_std = np.std(perm_scores)
        z_score = (obs_mean - perm_mean) / perm_std if perm_std > 0 else np.inf
        
        print(f"   Null Mean: {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"   Z-score: {z_score:.2f}")
        print(f"   P-value: {p_value:.4f}")
        
        if p_value < 0.001:
            print(f"   ✅ Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"   ✅ Significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"   ✅ Significant (p < 0.05)")
        else:
            print(f"   ⚠️  Not significant (p >= 0.05)")
        
        # Store results
        result = {
            'test_name': test_name,
            'observed_mean': obs_mean,
            'observed_std': obs_std,
            'observed_scores': obs_scores,
            'null_mean': perm_mean,
            'null_std': perm_std,
            'null_distribution': perm_scores,
            'p_value': p_value,
            'z_score': z_score,
            'n_permutations': self.n_permutations,
            'scoring': scoring
        }
        
        self.results[test_name] = result
        return result
    
    def plot_null_distribution(self, test_name=None, save_path=None):
        """
        Plot null distribution with observed score.
        
        Parameters:
        -----------
        test_name : str
            Which test to plot (if None, plots all)
        save_path : str
            Path to save figure
        """
        if test_name is None:
            test_names = list(self.results.keys())
        else:
            test_names = [test_name]
        
        n_tests = len(test_names)
        fig, axes = plt.subplots(1, n_tests, figsize=(6*n_tests, 5))
        
        if n_tests == 1:
            axes = [axes]
        
        for idx, name in enumerate(test_names):
            if name not in self.results:
                print(f"Warning: No results for {name}")
                continue
            
            result = self.results[name]
            ax = axes[idx]
            
            # Plot null distribution
            ax.hist(result['null_distribution'], bins=30, alpha=0.7, 
                   color='lightgray', edgecolor='black', density=True, label='Null Distribution')
            
            # Fit and plot normal curve to null
            null_mean = result['null_mean']
            null_std = result['null_std']
            x = np.linspace(result['null_distribution'].min(), 
                           result['null_distribution'].max(), 100)
            ax.plot(x, stats.norm.pdf(x, null_mean, null_std), 
                   'k--', linewidth=2, label=f'Null N({null_mean:.3f}, {null_std:.3f}²)')
            
            # Plot observed score
            obs_mean = result['observed_mean']
            ax.axvline(obs_mean, color='red', linewidth=3, linestyle='-', 
                      label=f'Observed: {obs_mean:.3f}')
            
            # Add shaded region for significance
            x_fill = x[x >= obs_mean]
            if len(x_fill) > 0:
                y_fill = stats.norm.pdf(x_fill, null_mean, null_std)
                ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='red')
            
            # Annotations
            p_val = result['p_value']
            z_score = result['z_score']
            
            ax.set_xlabel(f"{result['scoring'].upper()} Score", fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f"[POINT 4] {name}\nZ={z_score:.2f}, p={p_val:.4f}", 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    def get_summary_table(self):
        """
        Generate summary table of all permutation tests.
        
        Returns:
        --------
        pd.DataFrame : Summary of all tests
        """
        if not self.results:
            print("No results to summarize. Run permutation tests first.")
            return None
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Test': name,
                'Metric': result['scoring'],
                'Observed': result['observed_mean'],
                'Null Mean': result['null_mean'],
                'Null Std': result['null_std'],
                'Z-score': result['z_score'],
                'P-value': result['p_value'],
                'Significant': '***' if result['p_value'] < 0.001 else 
                              '**' if result['p_value'] < 0.01 else 
                              '*' if result['p_value'] < 0.05 else 'ns'
            })
        
        df = pd.DataFrame(summary_data)
        return df.round(4)
    
    def generate_report(self, title="[POINT 4] PERMUTATION TEST RESULTS", save_plots=False):
        """
        Generate complete permutation test report.
        
        Parameters:
        -----------
        title : str
            Report title
        save_plots : bool
            Whether to save plots to disk
        """
        print("=" * 80)
        print(title)
        print("=" * 80)
        print(f"\nAddresses Reviewer Concern:")
        print("'Massive predictive pleiotropy within the feature space could render")
        print("even highly predictive models nonsignificant.'")
        print(f"\nPermutation tests verify that models perform significantly better than")
        print(f"chance given the covariance structure of features.\n")
        
        # Summary table
        summary_df = self.get_summary_table()
        if summary_df is not None:
            print("\nSUMMARY TABLE:")
            print("=" * 80)
            print(summary_df.to_string(index=False))
            print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # Plot null distributions
        print(f"\n{'='*80}")
        print("NULL DISTRIBUTION PLOTS")
        print("=" * 80)
        
        save_path = 'permutation_test_results.png' if save_plots else None
        self.plot_null_distribution(save_path=save_path)
        
        print("\n" + "=" * 80)
        print("REPORT COMPLETE")
        print("=" * 80)


def run_permutation_test_quick(model_factory, X, y, n_permutations=100, scoring=None, cv_folds=5):
    """
    Quick helper function to run a single permutation test.
    
    Parameters:
    -----------
    model_factory : callable
        Function returning fresh model instance
    X : array-like
        Features
    y : array-like
        Target
    n_permutations : int
        Number of permutations
    scoring : str
        Metric to use
    cv_folds : int
        CV folds
        
    Returns:
    --------
    dict : Test results
    """
    reporter = PermutationTestReporter(X, y, n_permutations=n_permutations, cv_folds=cv_folds)
    result = reporter.run_permutation_test(model_factory, test_name="Quick Test", scoring=scoring)
    reporter.plot_null_distribution()
    return result
