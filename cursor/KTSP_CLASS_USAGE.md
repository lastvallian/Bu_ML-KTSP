# KTSPClassifier Class - Usage Guide

## Overview
The `KTSPClassifier` class wraps the KTSP training and testing pipeline into a clean, scikit-learn-style interface.

## Key Fixes from Original Code

### 1. **Array Indexing** ✅
- **Wrong**: `X[self.selected_genes, :]` (selects rows instead of columns)
- **Correct**: `X[:, self.selected_genes]` (selects columns/genes)

### 2. **Variable Name Consistency** ✅
- **Wrong**: Mixed use of `self.selected_top_genes` and `self.selected_genes`
- **Correct**: Consistently use `self.selected_genes`

### 3. **Method Signatures** ✅
- **Wrong**: Missing `self` parameter in `evaluation()` method
- **Correct**: All methods properly include `self` as first parameter

### 4. **Syntax Errors** ✅
Fixed multiple syntax issues:
- **Wrong**: `def predict(self,X_test: np.ndarray)->np.ndarray,` (comma instead of colon)
- **Correct**: `def predict(self, X_test: np.ndarray) -> np.ndarray:`

### 5. **Gene Pair Scoring** ✅
- **Wrong**: `score_top_k_gene_pairs(X_train_sel, y_train, self.top_genes, 10)`
- **Correct**: `score_top_k_gene_pairs(X_train, y_train, self.selected_genes, K=self.top_pairs)`
  - The function expects gene indices, not the number of genes
  - Should use full X_train, not just selected genes

### 6. **Removed Invalid Code** ✅
- Removed: `self.X_test_selected = X_test[selected_genes, :]` from `fit()` method
  - Test data should not be referenced during training

### 7. **Visualization Method** ✅
- Fixed parameter passing and variable references
- Proper extraction of test results before visualization

---

## Usage Examples

### Example 1: Basic Usage

```python
from pipeline import KTSPClassifier, load_and_normalize_data, split_data_processing_labels

# Load data
X, y = load_and_normalize_data("data.txt")
X_train, X_test, y_train, y_test = split_data_processing_labels(X, y)

# Initialize and train classifier
clf = KTSPClassifier(top_genes=300, top_pairs=10)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
results = clf.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")

# Visualize
clf.visualize(X_test, y_test)
```

### Example 2: Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

# KTSPClassifier is already compatible with sklearn's cross-validation
clf = KTSPClassifier(top_genes=300, top_pairs=10)

# Note: For cross-validation, you need to use the raw sklearn interface
# which expects fit() to not subset data internally
```

### Example 3: Grid Search for Hyperparameters

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'top_genes': [100, 200, 300],
    'top_pairs': [5, 10, 15, 20]
}

# Initialize classifier
clf = KTSPClassifier()

# Note: To use with GridSearchCV, you may need to implement sklearn's 
# BaseEstimator and ClassifierMixin interfaces
```

### Example 4: Multiple Datasets

```python
datasets = {
    "Dataset1": "path/to/data1.txt",
    "Dataset2": "path/to/data2.txt",
}

for name, path in datasets.items():
    print(f"\nProcessing {name}...")
    
    # Load and split
    X, y = load_and_normalize_data(path)
    X_train, X_test, y_train, y_test = split_data_processing_labels(X, y)
    
    # Train and evaluate
    clf = KTSPClassifier(top_genes=300, top_pairs=10)
    clf.fit(X_train, y_train)
    
    results = clf.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## Class API Reference

### Constructor
```python
KTSPClassifier(top_genes=300, top_pairs=10)
```
**Parameters:**
- `top_genes` (int): Number of top variable genes to select
- `top_pairs` (int): Number of top gene pairs to use

### Methods

#### `fit(X_train, y_train)`
Train the classifier.
- **Returns**: `self` (for method chaining)

#### `predict(X_test)`
Predict class labels.
- **Returns**: `np.ndarray` of predicted labels

#### `predict_proba(X_test)`
Get decision scores.
- **Returns**: `np.ndarray` of decision scores

#### `evaluate(X_test, y_test)`
Evaluate model performance.
- **Returns**: `dict` with 'accuracy', 'confusion_matrix', 'roc_data'

#### `visualize(X_test, y_test)`
Visualize classification results (displays plots).

---

## Attributes

After calling `fit()`, the following attributes are available:

- `clf.selected_genes`: Indices of selected genes
- `clf.gene_pairs`: List of top gene pairs [(gene_i, gene_j, score), ...]
- `clf.model`: Trained HierarchicalClassifier instance

---

## Comparison: Class vs Pipeline

### Original Pipeline Approach
```python
# Requires manual orchestration
model, selected_genes, gene_pairs = training_pipeline(file_path, top_genes=300, K=10)
results = testing_pipeline(model, X_test, y_test, selected_genes, gene_pairs)
```

### New Class-Based Approach
```python
# Clean, sklearn-style interface
clf = KTSPClassifier(top_genes=300, top_pairs=10)
clf.fit(X_train, y_train)
results = clf.evaluate(X_test, y_test)
```

**Benefits of Class-Based Approach:**
1. ✅ Cleaner API
2. ✅ Encapsulated state (genes, pairs, model)
3. ✅ Easy to save/load (pickle-able)
4. ✅ Compatible with sklearn patterns
5. ✅ Reusable on multiple test sets

---

## Running the Code

The `pipeline.py` file now supports both approaches. Set the flag in main:

```python
USE_CLASS_BASED = True  # Use KTSPClassifier class
# or
USE_CLASS_BASED = False  # Use original pipeline functions
```

Then run:
```bash
python pipeline.py
```


