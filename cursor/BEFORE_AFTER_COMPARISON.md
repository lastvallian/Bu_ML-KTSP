# Before & After: Code Comparison

This document shows your original code next to the corrected version for easy comparison.

---

## Complete Class Definition

### ❌ BEFORE (Your Original Code)

```python
class KTSPClassifier():
    def __init__(self,top_genes=300,top_pairs=10):
        self.top_genes=top_genes
        self.top_pairs=top_pairs
        self.selected_genes=None
        self.gene_pairs=None
        self.model=None
        
    def fit(self,X_train:np.ndarray,y_train:np.ndarray):
        self.selected_top_genes=select_top_variable_genes(    # ← Wrong variable name
        X_train, top_n=self.top_genes)
        
        X_train_sel=X_train[:,self.selected_genes]             # ← Using wrong variable
        #selected gene_pairs
        self.gene_pairs= score_top_k_gene_pairs(
        X_train_sel,                                            # ← Wrong: already subsetted
        y_train,
        self.top_genes,10)                                      # ← Wrong: should be gene indices
        #3.Build the classifier model
        self.model=build_hierarchical_tree(X_train_sel,y_train,self.gene_pairs)
        #self.model.fit(self, X_train[:, self.selected_genes], y_train, self.gene_pairs)
        self.X_test_selected = X_test[selected_genes, :]       # ← X_test doesn't exist!
        return self

    
    def predict(self,X_test: np.ndarray)->np.ndarray,         # ← Syntax error: comma
                X_sel= X_test[self.selected_genes,:]           # ← Wrong indexing
                y_pred,_=predict_with_tree(self.model,X_sel)   # ← Indentation issue
        return y_pred
    
    def predict_proba(
        self,X_test:np.ndarray) -> np.ndarray:
        X_sel=X_test[self.selected_genes,:],                   # ← Wrong indexing + comma
        _, decision_scores = predict_with_tree(self.model,X_sel)
        return  decision_scores
    
    def evaluation(X_test:np.ndarray, y_test:np.ndarray)      # ← Missing self + colon
        y_pred=self.predict(X_test)
        decision_scores=self.predict_proba(X_test)
        results = evaluate_model(y_test, y_pred, decision_scores)
        accuracy = results['accuracy']
        cm = results['confusion_matrix']
        roc_data = results['roc_data']
        return results,accuracy,cm,roc_data                    # ← Redundant returns
        
    def visualize(self,X_test,y_test):
        y_pred,scores= self.predict_proba(X_test,y_test),     # ← Wrong signature
        results=self.evaluation(X_test,y_test)                 # ← Should be evaluate
        visualize_results(X_test=X_test_selected,              # ← Undefined variable
                           gene_pairs=gene_pairs,               # ← Undefined variable
                           selected_genes=selected_genes,       # ← Undefined variable
                           y_pred=y_pred,
                           confusion_matrix=results["cm"],      # ← Wrong key
                           roc_data=results["roc_data"]):      # ← Syntax error: colon
```

### ✅ AFTER (Corrected Code)

```python
class KTSPClassifier:
    """
    K-TSP (Top-Scoring Gene Pairs) Classifier with automatic gene selection.
    
    This class encapsulates the entire KTSP training and prediction pipeline,
    including gene selection, gene pair scoring, and hierarchical classification.
    """
    
    def __init__(self, top_genes: int = 300, top_pairs: int = 10):
        """
        Initialize KTSP Classifier.
        
        Parameters:
            top_genes: Number of top variable genes to select
            top_pairs: Number of top gene pairs to use for classification
        """
        self.top_genes = top_genes
        self.top_pairs = top_pairs
        self.selected_genes = None
        self.gene_pairs = None
        self.model = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the KTSP classifier.
        
        Parameters:
            X_train: Training data (samples × genes)
            y_train: Training labels
            
        Returns:
            self: Trained classifier
        """
        # Step 1: Select top variable genes from training data
        self.selected_genes = select_top_variable_genes(       # ✓ Consistent naming
            X_train, top_n=self.top_genes
        )
        
        # Step 2: Extract selected genes (not used here, for reference)
        X_train_sel = X_train[:, self.selected_genes]          # ✓ Correct indexing
        
        # Step 3: Score and select top gene pairs
        self.gene_pairs = score_top_k_gene_pairs(
            X_train,                                            # ✓ Full data
            y_train, 
            self.selected_genes,                                # ✓ Gene indices
            K=self.top_pairs                                    # ✓ Named parameter
        )
        
        # Step 4: Build the hierarchical classifier model
        self.model = build_hierarchical_tree(
            X_train,                                            # ✓ Full data
            y_train, 
            self.gene_pairs
        )
        
        return self                                             # ✓ No invalid code
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:      # ✓ Correct syntax
        """
        Predict class labels for test data.
        
        Parameters:
            X_test: Test data (samples × genes)
            
        Returns:
            y_pred: Predicted class labels
        """
        if self.model is None:                                  # ✓ Safety check
            raise ValueError("Model not trained. Call fit() first.")
        
        # Predict using the trained model (model handles subsetting internally)
        y_pred, _ = predict_with_tree(self.model, X_test)      # ✓ Full data
        return y_pred
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray: # ✓ Correct syntax
        """
        Predict decision scores for test data.
        
        Parameters:
            X_test: Test data (samples × genes)
            
        Returns:
            decision_scores: Decision scores for each sample
        """
        if self.model is None:                                  # ✓ Safety check
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get decision scores from the trained model
        _, decision_scores = predict_with_tree(self.model, X_test) # ✓ Full data
        return decision_scores
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict: # ✓ Has self
        """
        Evaluate model performance on test data.
        
        Parameters:
            X_test: Test data (samples × genes)
            y_test: True test labels
            
        Returns:
            results: Dictionary containing accuracy, confusion matrix, and ROC data
        """
        # Get predictions and scores
        y_pred = self.predict(X_test)
        decision_scores = self.predict_proba(X_test)
        
        # Evaluate
        results = evaluate_model(y_test, y_pred, decision_scores)
        
        return results                                          # ✓ Single return
    
    def visualize(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Visualize classification results.
        
        Parameters:
            X_test: Test data (samples × genes)
            y_test: True test labels
        """
        if self.model is None:                                  # ✓ Safety check
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get predictions and evaluation results
        y_pred = self.predict(X_test)                           # ✓ Correct call
        results = self.evaluate(X_test, y_test)                 # ✓ Correct method name
        
        # Extract selected genes from test data
        X_test_selected = X_test[:, self.selected_genes]       # ✓ Defined variable
        
        # Visualize
        visualize_results(
            X_test_selected=X_test_selected,                    # ✓ Named parameter
            gene_pairs=self.gene_pairs,                         # ✓ Use self
            selected_genes=self.selected_genes,                 # ✓ Use self
            y_pred=y_pred,
            confusion_matrix=results['confusion_matrix'],       # ✓ Correct key
            roc_data=results['roc_data']                        # ✓ Correct syntax
        )
```

---

## Error Summary

| Issue | Line(s) | Fix |
|-------|---------|-----|
| Variable name mismatch | `fit()` | `self.selected_top_genes` → `self.selected_genes` |
| Wrong array indexing | `fit()`, `predict()`, `predict_proba()` | `X[genes, :]` → `X[:, genes]` |
| Wrong parameters | `fit()` - `score_top_k_gene_pairs()` | Pass full data + gene indices, not subset + count |
| Undefined variable | `fit()` | Remove `self.X_test_selected = X_test[...]` |
| Missing `self` | `evaluation()` | Add `self` as first parameter |
| Syntax error (comma) | `predict()` | `)→np.ndarray,` → `) -> np.ndarray:` |
| Indentation error | `predict()` | Fix function body indentation |
| Syntax error (comma) | `predict_proba()` | Remove trailing comma |
| Method name typo | `visualize()` | `self.evaluation` → `self.evaluate` |
| Undefined variables | `visualize()` | Use `self.gene_pairs`, `self.selected_genes` |
| Wrong dict key | `visualize()` | `results["cm"]` → `results['confusion_matrix']` |
| Syntax error (colon) | `visualize()` | Remove trailing colon in function call |
| Wrong signature | `visualize()` | `predict_proba(X, y)` → `predict_proba(X)` |

---

## Key Insights

### 1. Array Indexing in NumPy
```python
# For array shape (samples, genes):
X[row_indices, :]      # Select specific ROWS (samples)
X[:, col_indices]      # Select specific COLUMNS (genes)

# Your gene indices are COLUMN indices, not row indices!
X_test[:, self.selected_genes]  # ✓ Correct
X_test[self.selected_genes, :]  # ✗ Wrong - transposes the data!
```

### 2. Function Parameter Understanding
```python
def score_top_k_gene_pairs(
    X_train,                # Full training data needed
    y_train,
    selected_gene_indices,  # Array of gene indices to use
    K                       # Number of pairs to return
):
    # Function internally subsets: X_selected = X_train[:, selected_gene_indices]
    # So don't pre-subset the data!
```

### 3. Scope and Variable Lifetime
```python
def fit(self, X_train, y_train):
    # X_train exists here
    # X_test does NOT exist here - it's only available during prediction!
    self.X_test_selected = X_test[...]  # ✗ NameError
```

---

## Testing Your Code

To verify the corrections work, run:

```bash
# Test with example script
python example_ktsp_class.py

# Or use the main pipeline
python pipeline.py
```

Both should now work without errors! 🎉


