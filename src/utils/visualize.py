#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import roc_curve, auc


# In[6]:


def visualize_model(
     y_test:np.ndarray,
     y_pred:np.ndarray,
     y_proba:Optional[np.ndarray] = None,
     c_m:Optional[np.ndarray] = None,
     roc:Optional[np.ndarray] = None,
     X_test_selected:Optional[np.ndarray] = None,
     gene_pairs:Optional[List[Tuple[int,int,float]]] = None,
     selected_genes:Optional[np.ndarray] = None,
     clf=None,
     model:str=""):
     """
        Unified visualization for classification results.
        Includes confusion matrix, ROC, gene heatmap, and SVM weights.

        Parameters:
            y_test: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            c_m: Confusion matrix (optional)
            roc: dict with fpr, tpr, auc (optional)
            X_test_selected, gene_pairs, selected_genes: for gene heatmap (optional)
            clf: trained sklearn pipeline (for SVM feature weights)
            model_name: string for title
    """
    #----1.Step 1:Confusion Matrix & ROC ---
     if c_m is None:
        cm_plot = sk_cm(y_test, y_pred)
     else:
        cm_plot = c_m
     fig,axes = plt.subplots(1, 2, figsize=(12, 4))
     sns.heatmap(cm_plot, annot=True, fmt="d", cmap="Blues", ax=axes[0])
     axes[0].set_title("Confusion Matrix")
     axes[0].set_xlabel("Predicted")
     axes[0].set_ylabel("True")

     # Plot ROC curve if provided 
     if y_proba is not None:
        if y_proba.ndim==2:
            y_score=y_proba[:,1]
        else:
            y_score=y_proba
            fpr,tpr,_=roc_curve(y_test,y_score)
            roc=auc(fpr,tpr)
            axes[1].plot(fpr,tpr,label=f"AUC = {roc:.2f}")
            axes[1].plot([0, 1], [0, 1], "k--")
            axes[1].set_title("ROC Curve")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].legend()
     else:
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


        #-----------Linear model 
        try:
            clf=model.pipeline.names["clf"]
            if hasattr(clf,"coef_"):
                weights=clf.coef_[0]
                features=[f"feature_{i}" for i in range(len(weights))]
                df=pd.DataFrame({"Feature": features, "Weight": weights})
                df=df.sort_values("Weight", key=lambda x: np.abs(x), ascending=False).head(20)

                plt.figure(figsize=(10, 3))
                sns.heatmap(
                    df[["Weight"]].T,
                    annot=True,
                    cmap="coolwarm",
                    center=0 )
                plt.xticks(
                    range(len(df)),
                    df["Feature"],
                    rotation=45,
                    ha="right" )
                plt.yticks([0], ["Weight"])
                plt.title(f"Feature Weights – {model_name}")
                plt.tight_layout()
                plt.show()
        except Exception:
            pass

