#!/usr/bin/env python
# coding: utf-8

# In[40]:


#!/usr/bin/env python
# coding: utf-8

"""
Module 5 & 6: Model persistence and inference utilities.

Module 5: exportModule()
    - Input : trained_model_object, feature_list
    - Output: inference_config dict with "model_path" and "features"
              and artifacts on disk:
                  * model.joblib
                  * metadata.json (for model parameters and meta info)
                  * inference_config.json

Module 6: run_on_user_input()
    - Input : new_sample_file (CSV or Salmon output), model_path
    - Output: dict with Prediction_Result, Confidence_Score, Visualization
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib 

try:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
except NameError:
    # if running on jupyter Notebook 
    current_dir = os.getcwd()

# get the parent_dir
parent_dir = os.path.dirname(current_dir)

# add parent_dir into  sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#
try:
    # loading from one of pacakge (call from run_pipeline.ipynb )
    from utils.data_utils import load_and_normalize_data
except ImportError:
    # loading from localfiles (loading from utils )
    from data_utils import load_and_normalize_data

import numpy as np
from joblib import dump, load


# In[41]:


def _default_export_dir() -> str:
    """
    Default base directory for exported models.

    Returns:
        str: Absolute path to the directory where models will be saved.
    """
    base_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def exportModule(trained_model_object: Any, 
                 feature_list: List[str],
                 model_name: str,
                label_map: Optional[Dict[str, Any]] = None,
                scaler_object: Optional[Any] = None,
                use_pca: bool = False
                ) -> Dict[str, Any]:
    """
    Module 5:
    Take a trained model object and feature list, and persist them
    in a self-contained inference package.

    Args:
        trained_model_object: Fitted model or sklearn-like pipeline.
        feature_list (List[str]): Ordered list of feature names used during training.

    Returns:
        Dict[str, Any]: inference_config dict with keys:
            - "model_path": path to persisted model (.joblib)
            - "features"  : list of feature names
    """
    export_root = _default_export_dir()

    model_class_name = trained_model_object.__class__.__name__
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    package_dir = os.path.join(export_root, model_name)
    os.makedirs(package_dir, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(package_dir, "model.joblib")
    dump(trained_model_object, model_path)
    
    scaler_path = None
    if scaler_object is not None:
        scaler_path = os.path.join(package_dir, "scaler.joblib")
        dump(scaler_object, scaler_path)

    # 2. Save metadata (model parameters, feature list, timestamps, etc.)
    metadata: Dict[str, Any] = {
        "model_class": model_class_name,
        "model_type": trained_model_object.__class__.__name__,
        "model_path": model_path,
        "scaler_path": scaler_path,      # if none for None
        "use_pca": use_pca,              # boolean value
        "features": feature_list,        # feature_list
        "label_map": label_map,          # dict：{"0": "ADCA", "1": "Normal"}
         "created_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    # Try to capture model hyper-parameters if available
    if hasattr(trained_model_object, "get_params"):
        try:
            metadata["model_params"] = trained_model_object.get_params()
        except Exception:
            # Gracefully skip if params cannot be serialized
            pass

    metadata_path = os.path.join(package_dir, "inference_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Inference configuration (minimal info needed at prediction time)
    inference_config: Dict[str, Any] = {
        "model_path": model_path,
        "features": feature_list,
        "model_class": model_name,
        "model_type": trained_model_object.__class__.__name__,
        "scaler_path": scaler_path,      # if none None
        "use_pca": use_pca,              # boolean values
        "label_map": label_map,          # dict：{"0": "ADCA", "1": "Normal"}
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    inference_config_path = os.path.join(package_dir, "inference_config.json")
    
    config_path = os.path.join(package_dir, "inference_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2)
    with open(inference_config_path, "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2)

    return inference_config


# In[44]:


def run_on_user_input(new_sample_file: str, model_path: str) -> Dict[str, Any]:
    """
    Module 6:
    Load a persisted model and run inference on new diagnostic data.

    Args:
        new_sample_file (str): Path to CSV / Salmon output file with features.
                               Labels are assumed unknown and will be ignored.
        model_path (str): Path to the persisted model file created by Module 5.

    Returns:
        Dict[str, Any]: {
            "Prediction_Result": List[int/str],
            "Confidence_Score": List[float] (same length as predictions),
            "Visualization": {
                "label_distribution": {label: count, ...}
            }
        }
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        config = json.load(f) 
    
    
    if model_path.endswith(".json"):
        with open(model_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        actual_model_path = config["model_path"]
        feature_list = config.get("features") # get feature_list
    else:
        # transfer to  .joblib 
        actual_model_path = model_path
        feature_list = None

    # 1. Load model
    model = joblib.load(actual_model_path)

    # 2. Load and normalize new data (ignore labels)
    if feature_list:
        df_new = pd.read_csv(new_sample_file)
        #real aligned for datafiles
        X_aligned = df_new.reindex(columns=feature_list, fill_value=0)
        X_values = X_aligned.values
    else:
        # no feature_list
        X, _ = load_and_normalize_data(new_sample_file)
    
    if "scaler_path" in config:
        scaler_file = config["scaler_path"]
        config_dir = os.path.dirname(model_path)
        scaler_full_path = os.path.join(config_dir, scaler_file)
       
        # Loading SubspaceModule transformer
        subspace_transformer = joblib.load(scaler_full_path)
        print(f"Transforming data from {X_values.shape[1]} to reduced dimensions...")
        
        # Calling subspace.transform()，execute PCA.transform
        # Finally X_values turn into  (n_samples, 40)
        
        X_values = subspace_transformer.transform(X_values)
        print(f"New shape: {X_values.shape}") # 应该是 (n_samples, 40)
        
    # 3. Run predictions
    preds = model.predict(X_values)
    
    
    
    # 4.Get the label_map from config 
    #  read json file for run_on_user_input even exportModule has defined label_map 
    raw_preds = np.array(preds).ravel()
    for p in raw_preds:
        # 处理可能的嵌套情况：如果是 ['0'] 或 array(['0'])，取里面的 '0'
        inner_val = p[0] if isinstance(p, (list, np.ndarray, pd.Series)) else p
    if hasattr(preds, "flatten"):
        preds_flat = preds.flatten()
    else:
        preds_flat = np.array(preds).flatten()
    label_map=config.get("label_map")
    print(f"DEBUG - label_map: {label_map}")
    
    final_predictions=[]

    str_val = str(inner_val).strip()
       
    if label_map and str_val in label_map:
            final_predictions.append(label_map[str_val])
    else:
            final_predictions.append(str_val)
    print(f"Predict_Result:{final_predictions}")    
    
    #Map transforming
    
    # 5. Compute confidence scores
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_values)
        if proba.ndim == 2:
            confidence = proba.max(axis=1)
        else:
            confidence = proba
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_values)
        scores = np.asarray(scores, dtype=float)
        # Simple min-max normalization to [0, 1] as a pseudo-confidence
        if scores.ndim == 1:
            # 2 mapping 
            # logic for：1 / (1 + exp(-x))
            confidence_mapped = 1 / (1 + np.exp(-np.abs(scores)))
            confidence = confidence_mapped
           
        else:
            # multiple classification (scores.ndim == 2)
            # using Softmax function
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            softmax_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            confidence = softmax_proba.max(axis=1)
        #Neither for proba nor decision_function 
    if confidence is None:
       confidence = np.ones(len(preds))    

    confidence_score = confidence.tolist() if confidence is not None else None

    # 6. Lightweight visualization data: label distribution for front-end plotting
    unique_labels, counts =np.unique(final_predictions, return_counts=True)
    label_distribution = {
        str(label): int(count) for label, count in zip(unique_labels, counts)
    }

    return {
        "Prediction_Result": final_predictions,
        "Confidence_Score": confidence_score,
        "Visualization": {
            "label_distribution": label_distribution
        },
    }

