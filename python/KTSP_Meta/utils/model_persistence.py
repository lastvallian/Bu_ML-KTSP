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

import numpy as np
from joblib import dump, load

from utils.data_utils import load_and_normalize_data


def _default_export_dir() -> str:
    """
    Default base directory for exported models.

    Returns:
        str: Absolute path to the directory where models will be saved.
    """
    base_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def exportModule(trained_model_object: Any, feature_list: List[str]) -> Dict[str, Any]:
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
    package_dir = os.path.join(export_root, f"{model_class_name}_{timestamp}")
    os.makedirs(package_dir, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(package_dir, "model.joblib")
    dump(trained_model_object, model_path)

    # 2. Save metadata (model parameters, feature list, timestamps, etc.)
    metadata: Dict[str, Any] = {
        "model_class": model_class_name,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "features": feature_list,
    }

    # Try to capture model hyper-parameters if available
    if hasattr(trained_model_object, "get_params"):
        try:
            metadata["model_params"] = trained_model_object.get_params()
        except Exception:
            # Gracefully skip if params cannot be serialized
            pass

    metadata_path = os.path.join(package_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Inference configuration (minimal info needed at prediction time)
    inference_config: Dict[str, Any] = {
        "model_path": model_path,
        "features": feature_list,
    }

    inference_config_path = os.path.join(package_dir, "inference_config.json")
    with open(inference_config_path, "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2)

    return inference_config


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

    # 1. Load model
    model = load(model_path)

    # 2. Load and normalize new data (ignore labels)
    X, _ = load_and_normalize_data(new_sample_file)

    # 3. Run predictions
    preds = model.predict(X)

    # 4. Compute confidence scores
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            confidence = proba.max(axis=1)
        else:
            confidence = proba
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        # Simple min-max normalization to [0, 1] as a pseudo-confidence
        if scores.ndim == 1:
            min_s, max_s = float(scores.min()), float(scores.max())
            if max_s > min_s:
                confidence = (scores - min_s) / (max_s - min_s)
            else:
                confidence = np.ones_like(scores)
        else:
            confidence = scores

    prediction_result = preds.tolist()
    confidence_score = confidence.tolist() if confidence is not None else None

    # 5. Lightweight visualization data: label distribution for front-end plotting
    unique_labels, counts = np.unique(preds, return_counts=True)
    label_distribution = {
        str(label): int(count) for label, count in zip(unique_labels, counts)
    }

    return {
        "Prediction_Result": prediction_result,
        "Confidence_Score": confidence_score,
        "Visualization": {
            "label_distribution": label_distribution
        },
    }


