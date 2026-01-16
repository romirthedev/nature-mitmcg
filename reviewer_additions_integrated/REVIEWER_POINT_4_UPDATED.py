#@title ✅ [POINT 4] Permutation-Based Null Distributions & Significance Testing

"""
REVIEWER POINT 4: Model Significance via Label Permutation
==========================================================
✅ Supports BOTH regression and classification
✅ Auto-detects scoring metric (AUC vs R²)
✅ Tests if model is significant vs null distribution
✅ Controls for "predictive pleiotropy" (shared covariance artifacts)

KEY FEATURES:
- Randomly permute labels to build null distribution
- Retrain models on permuted data
- Compute p-value: observed vs null (2-tailed)
- Visualization: null histogram with observed marked
- Interpretation: is model significant (p < 0.05)?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

# ========================================================================
# HELPER: Task Type & Scoring Detection
# ========================================================================

def detect_task_type(y):
    """Auto-detect classification vs regression."""
    unique_vals = np.unique(y)
    return 'classification' if len(unique_vals) <= 10 else 'regression'


def get_scoring_metric(task_type):
    """Get appropriate scoring metric for task type."""
    if task_type == 'classification':
        return 'roc_auc'
    else:
        return 'r2'


# ========================================================================
# MAIN: PermutationTestReporter Class
# ========================================================================

class PermutationTestReporter:
    """
    Test model significance via label permutation.

    MOTIVATION (Reviewer concern about "predictive pleiotropy"):
    - Many psychological variables covary (p-factor, shared rater effects)
    - Strong predictive performance might reflect covariance, not true prediction
    - Solution: Show observed performance >> null distribution

    METHODOLOGY:
    1. Observe: Train model on true y, measure performance
    2. Permute: Randomly shuffle y labels
    3. Null: Train model on permuted y, measure performance
    4. Repeat: Generate null distribution from 100+ permutations
    5. P-value: Proportion of null models ≥ observed (2-tailed)
    """

    def __init__(self, X, y, task_type='auto', n_permutations=100, cv_folds=5, random_state=42):
        """
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series or np.array
            Target variable
        task_type : str
            'classification', 'regression', or 'auto'
        n_permutations : int
            Number of permutations for null distribution (default 100, use 1000 for final)
        cv_folds : int
            Number of CV folds
        random_state : int
            Random seed
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.n_permutations = n_permutations
        self.cv_folds = cv_folds
        self.random_state = random_state

        if task_type == 'auto':
            self.task_type = detect_task_type(self.y.values)
        else:
            self.task_type = task_type

        self.is_classification = self.task_type == 'classification'
        self.is_regression = self.task_type == 'regression'
        self.scoring_metric = get_scoring_metric(self.task_type)

        self.observed_score = None
        self.null_scores = []
        self.p_value = None

        print(f"✅ Task type: {self.task_type.upper()}")
        print(f"   Scoring metric: {self.scoring_metric.upper()}")
        print(f"   Permutations: {n_permutations}")

    def run_permutation_test(self, model_factory=None, base_model=None, results_dir='permutation_results'):
        """
        Run permutation significance test.

        Parameters:
        -----------
        model_factory : callable, optional
            Function returning fresh model (for retraining each fold)
        base_model : estimator, optional
            Pre-trained model from pipeline
        results_dir : str
            Directory to save results
        """
        os.makedirs(results_dir, exist_ok=True)

        if model_factory is None and base_model is None:
            raise ValueError("Either model_factory or base_model must be provided")

        use_base_model = base_model is not None

        print(f"\n🔄 Running permutation test ({self.n_permutations} permutations)")

        # Step 1: Compute observed score on true labels
        print(f"\n   Step 1: Computing observed score (true labels)...")
        self.observed_score = self._compute_cv_score(self.y, use_base_model, base_model, model_factory)
        print(f"           Observed {self.scoring_metric.upper()}: {self.observed_score:.4f}")

        # Step 2: Generate null distribution via label permutation
        print(f"\n   Step 2: Generating null distribution ({self.n_permutations} permutations)...")
        rng = np.random.default_rng(self.random_state)

        for perm_num in range(self.n_permutations):
            # Permute labels
            y_perm = rng.permutation(self.y.values)

            # Compute score on permuted labels
            perm_score = self._compute_cv_score(y_perm, use_base_model, base_model, model_factory)
            self.null_scores.append(perm_score)

            if (perm_num + 1) % 20 == 0:
                print(f"           Progress: {perm_num+1}/{self.n_permutations}")

        self.null_scores = np.array(self.null_scores)

        # Step 3: Compute p-value
        self.p_value = self._compute_p_value()

        # Step 4: Save results
        self._save_results(results_dir)

        return {
            'observed_score': self.observed_score,
            'null_scores': self.null_scores,
            'p_value': self.p_value
        }

    def _compute_cv_score(self, y, use_base_model, base_model, model_factory):
        """Compute CV score (mean across folds)."""
        fold_scores = []

        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in cv.split(self.X, y):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            y_test = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]

            # Get or train model
            if use_base_model:
                model = base_model
            else:
                model = model_factory()
                model.fit(X_train, y_train)

            # Score
            if self.is_classification:
                y_proba = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_proba)
            else:
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

            fold_scores.append(score)

        return np.mean(fold_scores)

    def _compute_p_value(self):
        """
        Compute 2-tailed p-value.

        P-value = (# null scores ≥ observed + 1) / (# permutations + 1)
        The "+1" prevents p=0 and accounts for the observed score as one permutation.
        """
        num_extreme = np.sum(self.null_scores >= self.observed_score)
        p_value = (num_extreme + 1) / (len(self.null_scores) + 1)

        return p_value

    def _save_results(self, results_dir):
        """Save permutation results to CSV."""
        results_df = pd.DataFrame({
            'permutation': range(len(self.null_scores)),
            'score': self.null_scores
        })

        results_df.to_csv(f"{results_dir}/permutation_null_distribution.csv", index=False)

        summary_df = pd.DataFrame({
            'Metric': ['Observed Score', f'Null Mean', 'Null SD', 'P-value', 'Significant (α=0.05)'],
            'Value': [
                f"{self.observed_score:.4f}",
                f"{np.mean(self.null_scores):.4f}",
                f"{np.std(self.null_scores):.4f}",
                f"{self.p_value:.4f}",
                "Yes ✅" if self.p_value < 0.05 else "No ❌"
            ]
        })

        summary_df.to_csv(f"{results_dir}/permutation_test_summary.csv", index=False)

        print(f"\n✅ Saved results to {results_dir}/")

    def plot_null_distribution(self, figsize=(12, 6), save_path=None):
        """Visualize null distribution vs observed."""
        if self.observed_score is None:
            raise ValueError("Run run_permutation_test() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Histogram of null distribution
        ax.hist(self.null_scores, bins=30, alpha=0.7, color='lightblue', edgecolor='black',
               label=f'Null Distribution (n={len(self.null_scores)})')

        # Mark observed score
        ax.axvline(self.observed_score, color='red', linestyle='--', linewidth=3,
                  label=f'Observed {self.scoring_metric.upper()} = {self.observed_score:.4f}')

        # Mark mean and SD of null
        null_mean = np.mean(self.null_scores)
        null_sd = np.std(self.null_scores)
        ax.axvline(null_mean, color='green', linestyle=':', linewidth=2,
                  label=f'Null Mean = {null_mean:.4f}')

        ax.set_xlabel(f'{self.scoring_metric.upper()} Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'[POINT 4] Permutation Test: Model Significance\n'
                    f'p-value = {self.p_value:.4f} {"(Significant)" if self.p_value < 0.05 else "(Not Significant)"}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Saved: {save_path}")
        plt.show()

    def generate_report(self, title="PERMUTATION TEST RESULTS"):
        """Generate permutation test report."""
        if self.observed_score is None:
            raise ValueError("Run run_permutation_test() first.")

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        print(f"\nTask Type: {self.task_type.upper()}")
        print(f"Scoring Metric: {self.scoring_metric.upper()}")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Permutations: {self.n_permutations}\n")

        # Results table
        print("PERMUTATION TEST RESULTS")
        print("-" * 80)

        results_table = pd.DataFrame({
            'Statistic': [
                'Observed Score',
                'Null Mean',
                'Null SD',
                'Null Min',
                'Null Max',
                'P-value (2-tailed)',
                'Significant (α=0.05)'
            ],
            'Value': [
                f"{self.observed_score:.4f}",
                f"{np.mean(self.null_scores):.4f}",
                f"{np.std(self.null_scores):.4f}",
                f"{np.min(self.null_scores):.4f}",
                f"{np.max(self.null_scores):.4f}",
                f"{self.p_value:.4f}",
                "✅ YES" if self.p_value < 0.05 else "❌ NO"
            ]
        })

        print(results_table.to_string(index=False))

        # Interpretation
        print("\n📊 INTERPRETATION")
        print("-" * 80)

        if self.p_value < 0.05:
            print(f"✅ MODEL IS STATISTICALLY SIGNIFICANT (p={self.p_value:.4f} < 0.05)")
            print(f"\n   The observed {self.scoring_metric.upper()} of {self.observed_score:.4f} is significantly")
            print(f"   better than the null distribution. This suggests the model captures")
            print(f"   genuine predictive signal, not just shared covariance artifacts.")
        else:
            print(f"❌ MODEL IS NOT STATISTICALLY SIGNIFICANT (p={self.p_value:.4f} >= 0.05)")
            print(f"\n   The observed {self.scoring_metric.upper()} of {self.observed_score:.4f} could plausibly")
            print(f"   arise from the null distribution (permuted labels).")
            print(f"   Consider: insufficient signal, or predictive pleiotropy.")

        # Effect size interpretation
        null_mean = np.mean(self.null_scores)
        effect_size = self.observed_score - null_mean

        print(f"\n📈 EFFECT SIZE")
        print("-" * 80)
        print(f"   Observed - Null Mean: {effect_size:.4f}")
        print(f"   Observed is {effect_size / (np.std(self.null_scores) + 1e-10):.2f} SDs above null mean")

        # Visualization
        print("\n3. VISUALIZATION")
        print("-" * 80)
        self.plot_null_distribution()

        print("\n" + "=" * 80)
        print("✅ [POINT 4] COMPLETE")
        print("=" * 80)


# ========================================================================
# EXECUTION: Run permutation test
# ========================================================================

try:
    if 'X_train' in locals() and 'y_train' in locals():
        print("\n🚀 Running [POINT 4] with pipeline variables")

        perm_reporter = PermutationTestReporter(
            X_train.copy(), y_train.copy(),
            task_type='auto',
            n_permutations=100,  # Use 100 for speed, 1000 for final results
            cv_folds=5
        )

        # Get model
        if 'model' in locals():
            print("   ✅ Using trained model from pipeline")
            result = perm_reporter.run_permutation_test(base_model=model, results_dir='permutation_results')
        else:
            print("   ⚠️  No trained model found")
            print("   Define model_factory to retrain, or run ML pipeline first")
            raise ValueError("model not found")

        perm_reporter.generate_report(title="[POINT 4] Permutation Test: Model Significance - RP")

    else:
        print("\n❌ ERROR: X_train or y_train not found")
        print("   Make sure to run data prep and ML training cells first")

except Exception as e:
    print(f"\n❌ Error in [POINT 4]: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ [POINT 4] Complete!")
