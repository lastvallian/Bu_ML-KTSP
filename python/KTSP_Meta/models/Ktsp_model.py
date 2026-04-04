#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
Hierarchical Classifier + K-TSP (Top-Scoring Gene Pairs) Pipeline
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# In[6]:


# ==============================
# FUNCTION 3: SELECT_TOP_VARIABLE_GENES
# ==============================

def select_top_variable_genes(X_train: np.ndarray, top_n: int = 300) -> np.ndarray:
    """
    Select top N most variable genes based on variance.
    
    Input:
        - X_train: training data (samples × genes)
        - top_n: number of top genes to select
    Output:
        - selected_gene_indices: array of gene indices
    """
    variances = np.var(X_train, axis=0)
    top_indices = np.argsort(variances)[::-1][:top_n]
    return np.sort(top_indices)


# In[7]:


# ==============================
# FUNCTION 4: SCORE_TOP_K_GENE_PAIRS
# ==============================

def score_top_k_gene_pairs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_gene_indices: np.ndarray,
    K: int = 10
) -> List[Tuple[int, int, float]]:
    """
    Score all gene pairs and select top K pairs.
    
    Input:
        - X_train: training data (samples × genes)
        - y_train: training labels
        - selected_gene_indices: indices of selected genes
        - K: number of top pairs to return
    Output:
        - top_k_gene_pairs: list of (gene_i, gene_j, score) tuples
    """
    X_selected = X_train[:, selected_gene_indices]
    n_genes = len(selected_gene_indices)
    n_samples = X_selected.shape[0]
    
    n_genes = X_selected.shape[1]
    n_samples = X_selected.shape[0]
    
    # Binarize labels to 0 and 1 for scoring
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 2:
        y_binary = (y_train == unique_labels[1]).astype(int)
    else:
        # For multi-class, use majority class as class 1
        label_counts = Counter(y_train)
        majority_class = label_counts.most_common(1)[0][0]
        y_binary = (y_train == majority_class).astype(int)
    
    pair_scores = []
    
    # Score all pairs
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            # Predict: if gene_i > gene_j → class 1, else class 0
            predictions = (X_selected[:, i] > X_selected[:, j]).astype(int)
            accuracy = np.mean(predictions == y_binary)
            pair_scores.append((selected_gene_indices[i], selected_gene_indices[j], accuracy))
    
    # Sort by score (descending) and select top K
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    return pair_scores[:K]


# In[8]:


# ==============================
# FUNCTION 5: BUILD_HIERARCHICAL_TREE
# ==============================

class TreeNode:
    """Node in hierarchical tree."""
    def __init__(self):
        self.gene_pair = None
        self.left = None
        self.right = None
        self.predicted_class = None
        self.is_leaf = False


class HierarchicalClassifier:
    """Hierarchical classifier using K-TSP gene pairs."""
    
    def __init__(self):
        self.root = None
        self.label_encoder = None
        self.gene_pairs = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, gene_pairs: List[Tuple[int, int, float]]):
        """
        Train hierarchical classifier.
        
        Input:
            - X: training data (samples × genes)
            - y: training labels
            - gene_pairs: list of (gene_i, gene_j, score) tuples
        """
        self.gene_pairs = gene_pairs
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.root = self._build_tree(X, y_encoded, 0)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, pair_idx: int) -> TreeNode:
        """Recursively build hierarchical tree."""
        node = TreeNode()
        
        # Base case: all samples have same class or no more pairs
        unique_classes = np.unique(y)
        if len(unique_classes) == 1 or pair_idx >= len(self.gene_pairs):
            node.is_leaf = True
            node.predicted_class = unique_classes[0]
            return node
        
        # Get current gene pair
        gene_i, gene_j, _ = self.gene_pairs[pair_idx]
        node.gene_pair = (gene_i, gene_j)
        
        # Split: gene_i > gene_j goes left (major class), else right
        split_mask = X[:, gene_i] > X[:, gene_j]
        X_left = X[split_mask]
        y_left = y[split_mask]
        X_right = X[~split_mask]
        y_right = y[~split_mask]
        
        # If one side is empty, make this a leaf
        if len(y_left) == 0 or len(y_right) == 0:
            node.is_leaf = True
            node.predicted_class = Counter(y).most_common(1)[0][0]
            return node
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X_left, y_left, pair_idx + 1)
        node.right = self._build_tree(X_right, y_right, pair_idx + 1)
        
        return node
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for samples."""
        predictions = []
        for sample in X:
            pred = self._predict_sample(self.root, sample)
            predictions.append(pred)
        return np.array(predictions)
    
    def _predict_sample(self, node: TreeNode, sample: np.ndarray) -> int:
        """Predict class for a single sample."""
        if node.is_leaf:
            return node.predicted_class
        
        gene_i, gene_j = node.gene_pair
        if sample[gene_i] > sample[gene_j]:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (decision scores)."""
        # For hierarchical classifier, use depth-based confidence
        scores = []
        for sample in X:
            score = self._get_decision_score(self.root, sample, depth=0)
            scores.append(score)
        return np.array(scores)
    
    def _get_decision_score(self, node: TreeNode, sample: np.ndarray, depth: int) -> float:
        """Get decision score based on tree depth."""
        if node.is_leaf:
            # Deeper nodes indicate higher confidence
            return 1.0 / (1.0 + depth * 0.1)
        
        gene_i, gene_j = node.gene_pair
        if sample[gene_i] > sample[gene_j]:
            return self._get_decision_score(node.left, sample, depth + 1)
        else:
            return self._get_decision_score(node.right, sample, depth + 1)


def build_hierarchical_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_k_gene_pairs: List[Tuple[int, int, float]]
) -> HierarchicalClassifier:
    """
    Build and train hierarchical classifier.
    
    Input:
        - X_train: training data (samples × genes)
        - y_train: training labels
        - top_k_gene_pairs: list of (gene_i, gene_j, score) tuples
    Output:
        - trained HierarchicalClassifier
    """
    classifier = HierarchicalClassifier()
    classifier.fit(X_train, y_train, top_k_gene_pairs)
    return classifier


# In[9]:


# ==============================
# FUNCTION 6: PREDICT_WITH_TREE
# ==============================

def predict_with_tree(
    trained_model: HierarchicalClassifier,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict using trained hierarchical classifier.
    
    Input:
        - trained_model: trained HierarchicalClassifier
        - X_test: test data (samples × genes)
    Output:
        - y_pred: predicted labels
        - decision_scores: decision scores
    """
    y_pred = trained_model.predict(X_test)
    decision_scores = trained_model.predict_proba(X_test)
    return y_pred, decision_scores


# In[10]:


# ==============================
# FUNCTION 7: EVALUATE_MODEL
# ==============================

def evaluate_model(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray
) -> Dict:
    """
    Evaluate model performance.
    
    Input:
        - y_test: true labels
        - y_pred: predicted labels
        - decision_scores: decision scores
    Output:
        - accuracy: accuracy score
        - confusion_matrix: confusion matrix
        - roc_data: dict with fpr, tpr, auc (only for binary classification)
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    roc_data = None
    unique_classes = len(np.unique(y_test))
    
    # ROC only for binary classification
    if unique_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, decision_scores)
        roc_auc = auc(fpr, tpr)
        roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'roc_data': roc_data
    }


# In[11]:


# ==============================
# FUNCTION 8: VISUALIZE_RESULTS
# ==============================

def visualize_results(
    X_test_selected: np.ndarray,
    gene_pairs: List[Tuple[int, int, float]],
    selected_genes: np.ndarray,
    y_pred: np.ndarray,
    confusion_matrix: np.ndarray,
    roc_data: Optional[Dict] = None
):
    """
    Visualize classification results.
    
    Input:
        - X_test_selected: test data with selected genes (samples × genes)
        - gene_pairs: list of (gene_i, gene_j, score) tuples
        - selected_genes: indices of selected genes
        - y_pred: predicted labels
        - confusion_matrix: confusion matrix
        - roc_data: dict with fpr, tpr, auc (optional)
    """
    fig, axes = plt.subplots(1, 3 if roc_data else 2, figsize=(15, 5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot ROC curve if provided
    if roc_data:
        axes[1].plot(roc_data['fpr'], roc_data['tpr'], 
                     label=f'ROC (AUC = {roc_data["auc"]:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        heatmap_ax = axes[2]
    else:
        heatmap_ax = axes[1]
    
    # Extract unique genes from gene pairs and map to selected_genes indices
    unique_genes_original = set()
    for gene_i, gene_j, _ in gene_pairs:
        unique_genes_original.add(gene_i)
        unique_genes_original.add(gene_j)
    
    # Map original gene indices to indices in selected_genes array
    gene_index_map = {gene: idx for idx, gene in enumerate(selected_genes)}
    gene_indices_in_selected = []
    for gene in unique_genes_original:
        if gene in gene_index_map:
            gene_indices_in_selected.append(gene_index_map[gene])
    
    # Plot heatmap of selected genes
    if len(gene_indices_in_selected) > 0:
        heatmap_data = X_test_selected[:, gene_indices_in_selected]
        sns.heatmap(heatmap_data.T, cmap='viridis', ax=heatmap_ax, 
                   cbar_kws={'label': 'Expression Level'})
        heatmap_ax.set_title(f'Gene Expression Heatmap\n({len(gene_indices_in_selected)} genes)')
        heatmap_ax.set_xlabel('Samples')
        heatmap_ax.set_ylabel('Genes')
    
    plt.tight_layout()
    plt.show()


# In[12]:


class KTSPClassifier:
    def __init__(self, top_genes: int = 300, top_pairs: int = 10):
        self.top_genes = top_genes
        self.top_pairs = top_pairs

        self.selected_genes = None
        self.gene_pairs = None
        self.model = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Step 1: select top variable genes
        self.selected_genes = select_top_variable_genes(
            X_train, top_n=self.top_genes
        )

        X_train_sel = X_train[:, self.selected_genes]

        # Step 2: score top K gene pairs
        self.gene_pairs = score_top_k_gene_pairs(
            X_train_sel, y_train,
            selected_gene_indices=np.arange(X_train_sel.shape[1]),
            K=self.top_pairs
        )

        # Step 3: build hierarchical classifier
        self.model = build_hierarchical_tree(
            X_train_sel, y_train, self.gene_pairs
        )

        return self
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_sel = X_test[:, self.selected_genes]
        y_pred, _ = predict_with_tree(self.model, X_sel)
        return y_pred
    
    def predict_proba(self,X_test:np.ndarray)->np.ndarray:
        scores = self.decision_function(X_test)
        probs = (scores - scores.min()) / (scores.max() - scores.min())
        return np.vstack([1 - probs, probs]).T
    
    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        X_sel = X_test[:, self.selected_genes]
        _, scores = predict_with_tree(self.model, X_sel)
        return scores

    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = self.predict(X_test)
        scores = self.decision_function(X_test)

        results = evaluate_model(y_test, y_pred, scores)
        return results
    
    def visualize(self, X_test: np.ndarray, y_test: np.ndarray):
        results = self.evaluate(X_test, y_test)

        X_test_sel = X_test[:, self.selected_genes]

        visualize_results(
            X_test_selected=X_test_sel,
            gene_pairs=self.gene_pairs,
            selected_genes=self.selected_genes,
            y_pred=self.predict(X_test),
            confusion_matrix=results["confusion_matrix"],
            roc_data=results["roc_data"]
        )


