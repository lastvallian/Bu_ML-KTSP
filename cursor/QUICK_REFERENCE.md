# KTSPClassifier Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

```python
from pipeline import KTSPClassifier, load_and_normalize_data, split_data_processing_labels

# Load data
X, y = load_and_normalize_data("data.txt")
X_train, X_test, y_train, y_test = split_data_processing_labels(X, y)

# Train
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

---

## 📚 API Reference

### Constructor
```python
clf = KTSPClassifier(top_genes=300, top_pairs=10)
```

### Methods
| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `fit(X_train, y_train)` | Training data & labels | `self` | Train the classifier |
| `predict(X_test)` | Test data | `np.ndarray` | Predict class labels |
| `predict_proba(X_test)` | Test data | `np.ndarray` | Get decision scores |
| `evaluate(X_test, y_test)` | Test data & labels | `dict` | Evaluate performance |
| `visualize(X_test, y_test)` | Test data & labels | None | Display plots |

### Attributes (after `fit()`)
- `clf.selected_genes` - Indices of selected genes
- `clf.gene_pairs` - List of top gene pairs
- `clf.model` - Trained HierarchicalClassifier

---

## 🔍 Common Operations

### Access Selected Genes
```python
clf.fit(X_train, y_train)
print(f"Number of selected genes: {len(clf.selected_genes)}")
print(f"Gene indices: {clf.selected_genes}")
```

### Access Gene Pairs
```python
for i, (gene_i, gene_j, score) in enumerate(clf.gene_pairs):
    print(f"Pair {i+1}: Gene {gene_i} vs Gene {gene_j} (score: {score:.4f})")
```

### Get Detailed Results
```python
results = clf.evaluate(X_test, y_test)

accuracy = results['accuracy']
cm = results['confusion_matrix']
roc_data = results['roc_data']  # None for multi-class

if roc_data:
    auc_score = roc_data['auc']
    print(f"AUC: {auc_score:.4f}")
```

---

## ⚠️ Critical Fixes from Original Code

### 1. Array Indexing ✅
```python
# ❌ WRONG
X_sel = X[self.selected_genes, :]  # Selects rows, not columns!

# ✅ CORRECT
X_sel = X[:, self.selected_genes]  # Selects columns (genes)
```

### 2. Gene Pair Scoring ✅
```python
# ❌ WRONG
score_top_k_gene_pairs(X_train_sel, y_train, self.top_genes, 10)

# ✅ CORRECT
score_top_k_gene_pairs(X_train, y_train, self.selected_genes, K=self.top_pairs)
```

### 3. Method Signatures ✅
```python
# ❌ WRONG
def predict(self, X_test: np.ndarray) -> np.ndarray,  # Comma!
def evaluation(X_test, y_test)                         # No self!

# ✅ CORRECT
def predict(self, X_test: np.ndarray) -> np.ndarray:  # Colon!
def evaluate(self, X_test, y_test):                    # Has self!
```

---

## 📊 Results Dictionary Structure

```python
results = {
    'accuracy': 0.95,                           # float
    'confusion_matrix': np.array([[10, 2],      # np.ndarray
                                   [1, 12]]),
    'roc_data': {                               # dict or None
        'fpr': np.array([...]),                 # False positive rate
        'tpr': np.array([...]),                 # True positive rate
        'auc': 0.98                             # AUC score
    }
}
```

---

## 🎯 Usage Patterns

### Pattern 1: Single Dataset
```python
X, y = load_and_normalize_data("data.txt")
X_train, X_test, y_train, y_test = split_data_processing_labels(X, y)

clf = KTSPClassifier().fit(X_train, y_train)  # Method chaining
results = clf.evaluate(X_test, y_test)
```

### Pattern 2: Multiple Datasets
```python
clf = KTSPClassifier(top_genes=300, top_pairs=10)

for name, path in datasets.items():
    X, y = load_and_normalize_data(path)
    X_train, X_test, y_train, y_test = split_data_processing_labels(X, y)
    
    clf.fit(X_train, y_train)
    results = clf.evaluate(X_test, y_test)
    print(f"{name}: {results['accuracy']:.4f}")
```

### Pattern 3: Hyperparameter Tuning
```python
best_acc = 0
best_params = None

for top_genes in [100, 200, 300]:
    for top_pairs in [5, 10, 15]:
        clf = KTSPClassifier(top_genes=top_genes, top_pairs=top_pairs)
        clf.fit(X_train, y_train)
        results = clf.evaluate(X_test, y_test)
        
        if results['accuracy'] > best_acc:
            best_acc = results['accuracy']
            best_params = (top_genes, top_pairs)

print(f"Best: {best_params}, Accuracy: {best_acc:.4f}")
```

---

## 🐛 Troubleshooting

### Error: "Model not trained"
```python
# Forgot to call fit()
clf = KTSPClassifier()
y_pred = clf.predict(X_test)  # ❌ Error!

# Solution: Call fit() first
clf.fit(X_train, y_train)     # ✅ Train first
y_pred = clf.predict(X_test)  # ✅ Now works
```

### Error: "Shape mismatch"
```python
# Make sure test data has same number of genes as training data
print(X_train.shape)  # (n_train_samples, n_genes)
print(X_test.shape)   # (n_test_samples, n_genes) ← Must match!
```

### Error: "Index out of bounds"
```python
# selected_genes indices must be valid for both train and test data
max_gene_idx = np.max(clf.selected_genes)
print(f"Max gene index: {max_gene_idx}")
print(f"Data has {X_test.shape[1]} genes")
# max_gene_idx must be < X_test.shape[1]
```

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `pipeline.py` | Main implementation with KTSPClassifier class |
| `example_ktsp_class.py` | Working example script |
| `KTSP_CLASS_USAGE.md` | Comprehensive usage guide |
| `KTSP_CLASS_SUMMARY.md` | Summary of changes and fixes |
| `BEFORE_AFTER_COMPARISON.md` | Side-by-side code comparison |
| `QUICK_REFERENCE.md` | This file - quick reference |

---

## 🎓 Key Concepts

### Gene Expression Data Shape
```
Shape: (samples, genes)
       ↓         ↓
       rows    columns

Select genes → X[:, gene_indices]
Select samples → X[sample_indices, :]
```

### KTSP Workflow
```
1. Select top variable genes (by variance)
2. Score all gene pairs (accuracy of gene_i > gene_j)
3. Build hierarchical tree using top K pairs
4. Predict by traversing tree with gene comparisons
```

---

## ✅ Checklist

Before using KTSPClassifier:
- [ ] Data loaded and normalized
- [ ] Data split into train/test
- [ ] Classifier initialized with parameters
- [ ] Model trained with `fit()`

For prediction:
- [ ] Model is trained (`fit()` called)
- [ ] Test data has same number of genes as training data
- [ ] Test data is normalized (same way as training data)

---

## 🚦 Status Indicators

```python
# Check if trained
if clf.model is None:
    print("❌ Not trained yet")
else:
    print("✅ Ready to predict")
    print(f"   Genes: {len(clf.selected_genes)}")
    print(f"   Pairs: {len(clf.gene_pairs)}")
```

---

**Remember**: All array indexing is (samples, genes), so gene selection uses `X[:, gene_indices]`! 🎯


