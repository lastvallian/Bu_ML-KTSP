#!/usr/bin/env python
# coding: utf-8

"""
Dataset class for storing loaded and processed data
"""
import numpy as np
import pickle
import os
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Dataset:
    """Dataset object to store loaded and processed data"""
    X: np.ndarray
    y: np.ndarray
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    X_train_processed: Optional[np.ndarray] = None
    X_test_processed: Optional[np.ndarray] = None
    file_path: Optional[str] = None
    task_id: Optional[str] = None
    
    def save(self, file_path: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'Dataset':
        """Load dataset from disk"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def get_shape_info(self) -> dict:
        """Get shape information for debugging"""
        info = {
            "X_shape": self.X.shape if self.X is not None else None,
            "y_shape": self.y.shape if self.y is not None else None,
        }
        if self.X_train is not None:
            info["X_train_shape"] = self.X_train.shape
        if self.X_test is not None:
            info["X_test_shape"] = self.X_test.shape
        if self.X_train_processed is not None:
            info["X_train_processed_shape"] = self.X_train_processed.shape
        if self.X_test_processed is not None:
            info["X_test_processed_shape"] = self.X_test_processed.shape
        return info





