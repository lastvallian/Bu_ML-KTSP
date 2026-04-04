#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==============================
# LABEL NORMALIZATION
# ==============================
from typing import List  
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

LABEL_MAP = {
        "tumor": ["tumor", "t", "cancer"],
        "normal": ["normal", "n", "control"],
        "all": ["all"],
        "aml": ["aml"],
        "adca": ["adca"],
        "mesothelioma": ["mesothelioma"],
        "dlbcl": ["dlbcl"],
        "fl": ["fl"]
        } 

def normalize_label(label: str) -> str:
    """Normalize label to standard format."""
    label = label.strip().lower()

    for normalized, variants in LABEL_MAP.items():
        if label in variants:
            return normalized

    return label
  


def _is_numeric_row(values: List[str]) -> bool:
    """Check if row contains only numeric values."""
    try:
        return all(pd.to_numeric(v, errors='coerce').notna().all() for v in values if str(v).strip())
    except:
        return False


def _generate_default_labels(n_samples: int) -> List[str]:
    """Generate default tumor/normal labels when none provided."""
    half = n_samples // 2
    return ["tumor"] * half + ["normal"] * (n_samples - half)


# ==============================
# FUNCTION 1: LOAD_AND_NORMALIZE_DATA
# ==============================

def load_and_normalize_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load gene expression matrix and normalize data.

    Input:
        - file_path: path to data file
    Output:
        - X: gene expression matrix (genes × samples)
        - y: labels as integers
    """
    df = pd.read_csv(file_path, sep=',', header=None)

    first_row = df.iloc[0].astype(str).tolist()

    # 是否第一行是标签
    labels_in_first_row = not all(
        v.replace('.', '').replace('-', '').isdigit()
        for v in first_row
    )

    if labels_in_first_row:
        labels = [normalize_label(l) for l in first_row]
        data = df.iloc[1:].values.astype(float)
    else:
        data = df.values.astype(float)
        n_samples = data.shape[1]
        labels = ["tumor"] * (n_samples // 2) + ["normal"] * (n_samples - n_samples // 2)

    # 保证 X 是 genes × samples
    if data.shape[0] < data.shape[1]:
        X = data
    else:
        X = data.T

    unique_labels = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("label unique:", np.unique(y))

    return X, y
   


# In[2]:


# ==============================
# FUNCTION 2: SPLIT_DATA_PROCESSING_LABELS
# ==============================

def split_data_processing_labels(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Input:
        - X: feature matrix
        - y: labels
        - test_size: proportion of test set
        - random_state: random seed
    Output:
        - X_train, X_test, y_train, y_test
    """
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X.T, y, test_size=test_size, random_state=random_state, stratify=y
    #)
    #return X_train, X_test, y_train, y_test

    #idx = np.arange(X.shape[1])

    #train_idx, test_idx = train_test_split(
    #   idx, test_size=test_size, random_state=random_state, stratify=y
    #)

    #X_train = X[:, train_idx]
    #X_test  = X[:, test_idx]
    #y_train = y[train_idx]
    #y_test  = y[test_idx]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
    )
    return X_train, X_test, y_train, y_test

