#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def evaluate_model(y_test, y_pred, y_proba=None):
    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred)
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    if y_proba is not None and len(y_proba.shape) == 2:
        y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        results["roc"] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr)
        }
    else:
        results["roc"] = None

    return results

