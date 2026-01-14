# How to Use Nested CV Reporter with Copy_of_ABCTestR.ipynb

This guide shows how to add the Nested CV Reporter to your existing ABCD analysis notebook in Google Colab.

---

## 📋 What This Does (In Plain English)

After you've already run your machine learning models in your notebook, you can use this tool to:
- Test model performance across different data splits (nested cross-validation)
- See confidence intervals for all performance metrics
- Create publication-quality visualizations
- Save all results automatically to Google Drive

---

## 🚀 Quick Integration (Add These Steps to Your Notebook)

### Step 1: Upload the Python File (One Time)

1. Go to your existing **Copy_of_ABCTestR.ipynb** in Colab
2. On the left side, click the **folder icon** (Files)
3. Click **upload** (arrow pointing up)
4. Select `nested_cv_reporter.py` from your computer
5. Wait for upload to complete

### Step 2: Add New Cells at the End of Your Notebook

After you've already run your main analysis, add these cells:

#### Cell 1: Import the Reporter
```python
# Add this import to your existing imports at the top (or add as new cell)
from nested_cv_reporter import NestedCVReporter, get_results_dir
import pickle
```

#### Cell 2: Prepare Your Data for Nested CV
```python
# After you've loaded your data (sample) and created target variable (y)
# Just add this to wherever you're ready to test model performance

# Prepare features (X) and target (y) for nested CV
# Adjust these column names to match YOUR notebook
X = sample.drop(columns=['your_target_column_name'])  # All features
y = sample['your_target_column_name']  # What you're predicting

print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: {y.value_counts().to_dict()}")
```

#### Cell 3: Run Nested CV Analysis
```python
# Define your model the same way you did before
def model_factory():
    from sklearn.linear_model import LogisticRegression  # Or your model
    return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Run nested CV
print("Starting Nested CV Analysis...")
reporter = NestedCVReporter(X, y, n_outer=5, random_state=42)
reporter.run_outer_cv(model_factory)

print("✅ Analysis complete! Generating report...")
reporter.generate_report(title="NESTED CV RESULTS - Your Analysis")
```

#### Cell 4 (Optional): Statistical Significance Test
```python
# Test if results are statistically significant
print("Running permutation test (100 label shuffles)...")
reporter.permutation_null_distribution(model_factory, n_permutations=100)
print("✅ Significance test complete!")
```

#### Cell 5 (Optional): Access Your Results
```python
import pandas as pd

results_dir = get_results_dir()
print(f"Results saved to: {results_dir}\n")

# Load and show fold metrics
metrics = pd.read_csv(f"{results_dir}/fold_metrics.csv")
print("FOLD METRICS:")
print(metrics)

# Load and show predictions
oof = pd.read_csv(f"{results_dir}/oof_predictions.csv")
print(f"\nOUT-OF-FOLD PREDICTIONS: {len(oof)} samples")
print(oof.head())
```

---

## 📍 Where to Add These Cells in Your Notebook

In your **Copy_of_ABCTestR.ipynb**:

1. Scroll to the **end** of your notebook
2. Add a new section header: `# === NESTED CV PERFORMANCE REPORTING ===`
3. Add the 5 cells above in order
4. Run them one by one (click ▶ on each cell)

---

## 🔄 Integration with Your Existing Code

### If You Already Have a Model
```python
# Instead of re-defining, just use what you already built
# Example: if you built 'your_model' earlier, do this:

def model_factory():
    from catboost import CatBoostClassifier  # Or RandomForest, etc.
    return CatBoostClassifier(
        iterations=100,
        random_state=42,
        verbose=0,
        use_best_model=True
    )

reporter = NestedCVReporter(X, y, n_outer=5)
reporter.run_outer_cv(model_factory)
```

### For Different Target Variables
If you have multiple outcome variables you tested:
```python
# Test outcome 1
y_depression = sample['top_10_depression']
reporter1 = NestedCVReporter(X, y_depression, n_outer=5)
reporter1.run_outer_cv(model_factory)
reporter1.generate_report(title="NESTED CV: Depression (Top 10%)")

# Test outcome 2  
y_anxiety = sample['anxiety_score']
reporter2 = NestedCVReporter(X, y_anxiety, n_outer=5)
reporter2.run_outer_cv(model_factory)
reporter2.generate_report(title="NESTED CV: Anxiety")
```

---

## 📊 Understanding the Output

After running the cells, you'll see:

### 1. Performance Table
Shows metrics like AUC, sensitivity, specificity with confidence intervals

### 2. Three Graphs
- **Calibration Curve** - How well the model predicts probabilities
- **Decision Curve** - Performance at different decision points  
- **Fold Distributions** - Consistency across data splits

### 3. Saved Results
Files automatically saved to Google Drive:
- `oof_predictions.csv` - Predictions for each sample
- `fold_metrics.csv` - Performance for each fold
- `permutation_null_metrics.csv` - Statistical significance results (if you ran it)

---

## 💾 Download Results from Colab

After running the analysis:

1. In Colab, click the **folder icon** (Files) on the left
2. Navigate to `nestedcv_results/` folder
3. Right-click any CSV file → **Download**

Or from Google Drive:
1. Open Google Drive
2. Find folder: `My Drive/nestedcv_results/`
3. Download the files you need

---

## 🎯 Complete Example Workflow

Here's what a complete addition to your notebook looks like:

```python
# ============================================================================
# SECTION: NESTED CV PERFORMANCE REPORTING
# ============================================================================

# Cell 1: Import
from nested_cv_reporter import NestedCVReporter, get_results_dir

# Cell 2: Prepare Data (adjust column names to YOUR data)
X = sample.drop(columns=['your_outcome_column'])
y = sample['your_outcome_column']
print(f"Ready: {X.shape}")

# Cell 3: Run Analysis
def model_factory():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=50, verbose=0, random_state=42)

reporter = NestedCVReporter(X, y, n_outer=5)
reporter.run_outer_cv(model_factory)
reporter.generate_report()

# Cell 4: Test Significance
reporter.permutation_null_distribution(model_factory, n_permutations=50)

# Cell 5: Inspect Results
import pandas as pd
results_dir = get_results_dir()
print(pd.read_csv(f"{results_dir}/fold_metrics.csv"))
```

---

## ❓ FAQ for Your Notebook

**Q: Can I use this with my existing CatBoost model?**  
A: Yes! Just define `model_factory()` to return a fresh CatBoost instance

**Q: Do I need to mount Google Drive?**  
A: Yes - add `from google.colab import drive; drive.mount('/content/drive')` at the top

**Q: Will this work with multiple outcome variables?**  
A: Yes! Run the reporter separately for each outcome

**Q: How do I use different feature sets?**  
A: Just create different X dataframes before creating the reporter

**Q: Can I adjust the number of folds?**  
A: Yes: `NestedCVReporter(X, y, n_outer=10)` for 10 folds instead of 5

**Q: My model has custom parameters - will they work?**  
A: Yes! Anything you can import and instantiate works

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| "ModuleNotFoundError: nested_cv_reporter" | Upload `nested_cv_reporter.py` file |
| Results not saving | Make sure you mounted Google Drive at the top |
| Model import error | Check that the model package is installed (it should be) |
| Slow performance | Reduce `n_outer` from 5 to 3, or reduce data size |
| Can't find results | Check Google Drive → My Drive → nestedcv_results |

---

## 🎓 What's Happening Under the Hood

1. **Splits data 5 times** into train/test sets
2. **Trains model 5 times** on different training data
3. **Tests on held-out data** (data the model never saw)
4. **Calculates metrics** for each test
5. **Computes confidence intervals** using statistics
6. **Creates visualizations** showing all results
7. **Saves everything** to Google Drive

This is different from your usual model because it:
- ✅ Never tests on training data
- ✅ Shows uncertainty (confidence intervals)
- ✅ Tests multiple times (shows consistency)
- ✅ Saves detailed predictions for each sample

---

## 📝 Notes

- **No changes to your existing code needed** - just add new cells at the end
- **All dependencies already installed** in Colab
- **Results are publication-ready** - can use tables/graphs in papers
- **Fully automatic** - just click play on each cell

---

*Last Updated: December 10, 2025*  
*Works seamlessly with Copy_of_ABCTestR.ipynb* ✨
