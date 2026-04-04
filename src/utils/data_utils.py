#!/usr/bin/env python
# coding: utf-8

# In[66]:


# ==============================
# LABEL NORMALIZATION
# ==============================
from typing import List  
from typing import Tuple
import pandas as pd
import numpy as np
import os
import json
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

#=================================================
#Function1:MANAGE DIFFERENT LABEL
#=================================================
class LabelManager:
    def __init__(self, storage_path="label_config.json"):
        self.storage_path=storage_path
        self.default_map={
            "tumor": ["tumor", "t", "cancer"],
            "normal": ["normal", "n", "control"],
            "all": ["all"],
            "aml": ["aml"],
            "adca": ["adca"],
            "mesothelioma": ["mesothelioma"],
            "dlbcl": ["dlbcl"],
            "fl": ["fl"]
        }
        self.label_map=self._load_config()
    def _load_config(self):
        if(os.path.exists(self.storage_path)):
            with open(self.storage_path,'r')as f:
                return json.load(f)
        return self.default_map
    def save_config(self):
        try:
            temp_data = json.dumps(self.label_map, indent=4)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                f.write(temp_data)
        except Exception as e:
             print(f"Error saving label config: {e}")
    def normalize(self,label:str)->str:
        label=str(label).strip().lower()
        #finding current mapping
        for normalized,variants in self.label_map.items():
            if label==normalized or label in variants:
                return normalized
        return label
    
    def set_classification_map(self, classes_list: List[str]):
        """
        according to classification to generate mapping
       for example ['adca', 'normal'] -> generate {"0": "adca", "1": "normal"}
        """
        self.label_map = {str(i): self.normalize(name) for i, name in enumerate(classes_list)}
        print(f"generate mapping: {self.label_map}")


# In[67]:


# ==============================
# FUNCTION 2: LOAD_AND_NORMALIZE_DATA
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
    def is_not_digit(v):
        s=str(v).strip().lower()
        #remove "-",".","e"
        clean_s=s.replace('.', '').replace('-', '').replace("e", '').replace('+', '')
        return not clean_s.isdigit()
    manager = LabelManager()
    df = pd.read_csv(file_path, sep=',', header=None)

    first_row = df.iloc[0,:].astype(str).tolist()
    first_row_has_text=any(not v.replace('.', '').replace('-', '').isdigit() for v in first_row)
    first_row_text_count = sum(1 for v in first_row if is_not_digit(v))
  
    #Check last row is non-digital 
    last_col_series = df.iloc[:, -1] # choose last column
    last_col_list = last_col_series.astype(str).tolist()
    last_col_has_text=any(not v.replace('.', '').replace('-', '').isdigit() for v in last_col_list)
    last_col_text_count = sum(1 for v in last_col_list if is_not_digit(v))
    # if the first row is tag
    labels_in_first_row = not all(
        v.replace('.', '').replace('-', '').isdigit()
        for v in first_row
    )
     
    # for A Label is the first row As TanData
    if first_row_has_text and not (last_col_has_text and not is_not_digit(last_col_list[0])):
        print(">>> Detecting TanDataSet mode (Labels in First Row)")
        labels=[manager.normalize(l)for l in first_row]
        #X=df.iloc[1:].values.astype(float).T
        X = df.iloc[1:, :].values.astype(float).T
        #features_name is the index for row
        feature_names=[f"Gene_{i}" for i in range(X.shape[1])]
    elif last_col_has_text:  
         print(">>> Detecting other DataSet mode (Labels in Last Column)")
         #Assuming the first row is the labels_names
         df_real = pd.read_csv(file_path, sep=None, engine='python')
         X=df_real.iloc[:,:-1].values.astype(float)
         feature_names = df_real.columns[:-1].tolist()
         labels=[manager.normalize(str(l)) for l in df_real.iloc[:, -1]]
    else: ##no labels default for TanData
         print(">>> not detecting text labels using spliting by half")
         X = df.values.astype(float)
         labels = ["tumor"]*(X.shape[0]//2) + ["normal"]*(X.shape[0]-X.shape[0]//2)
         feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    #raw_labels
    normalized_labels = [manager.normalize(l) for l in labels]
    unique_labels = sorted(list(set(normalized_labels)))
    # get mapping from the raw_labels    
    text_to_int_map = {l: i for i, l in enumerate(unique_labels)}
    try:
        y = np.array([text_to_int_map[l] for l in normalized_labels])
    except KeyError as e:
        print(f"!!! Error mapping label {e}. Available keys: {list(text_to_int_map.keys())}")
        raise
    #formatting: {"0": "mesothelioma", "1": "normal"}
    inference_map = {str(i): str(l) for l, i in text_to_int_map.items()}
    #Repair how to conver the X and y
    num_samples=len(y)
    print(f"num_samples:{num_samples}")
    
    #if rows of X is not for 
    #if rows of X is not for 
    if X.shape[0]!=num_samples:
        if X.shape[1]==num_samples:
            X=X.T
            print(f">>> Transposed X to align with y. New shape: {X.shape}")
        elif X.shape[1] == num_samples + 1:
            X = X[:, 1:].T
            print(f">>> Removed ID column and Transposed X. New shape: {X.shape}")
        elif X.shape[0] == num_samples + 1:
            X = X[1:, :]
            print(f">>> Removed ID row. New shape: {X.shape}")
        else:
            raise ValueError(f"Data alignment error: X shape {X.shape} cannot match y length {num_samples}")
    
    #define to label_map
    manager.label_map = inference_map
    manager.save_config() # save to label_config.json
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("label unique:", np.unique(y))
    print(f">>> [SUCCESS] Final Inference Map: {manager.label_map}")
    print(f"Final check before split - X: {X.shape}, y: {len(y)}")

    return X, y,feature_names


# In[69]:


# ==============================
# FUNCTION 3: SPLIT_DATA_PROCESSING_LABELS
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

