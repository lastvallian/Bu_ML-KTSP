#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve, auc
import numpy as np

class SKLearnModelWrapper:
    """ Wrapper all the model"""
    def __init__(self,model):
        self.model=model
        self.pipeline=None
        
    def fit(self,X,y):
        self.pipeline=Pipeline([
            ("scaler",StandardScaler()),
            ("clf",self.model)
        ])
        self.pipeline.fit(X,y)
        return self
        
    def predict(self,X):
        return self.pipeline.predict(X)
    def predict_proba(self,X):
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf,"predict_proba"):
            return self.pipeline.predict_proba(X)
        elif hasattr(clf,"decision_function"):
            return self.pipeline.decision_function(X)
        else:
            raise ValuesError("Model does not support the output!")
        
          


# In[6]:


class ShrinkageCentroidClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self,shrinkage=2.0):
        self.shrinkage=shrinkage
    def fit(self,X,y):
        classes=np.unique(y)
        self.centroids={}
        self.overall_mean=X.mean(axis=0)
        
        for c in classes:
            Xc=X[y==c]
            mean_c=Xc.mean(axis=0)
            delta=mean_c-self.overall_mean
            delta_shrunk=np.sign(delta)*np.maximum(0,np.abs(delta)-self.shrinkage)
            self.centroids[c]=self.overall_mean+delta_shrunk
        return self
    def predict(self,X):
        preds=[]
        for x in X:
            dists = {c: np.linalg.norm(x - mu) for c, mu in self.centroids.items()}
            preds.append(min(dists, key=dists.get))
        return np.array(preds) 
    def decision_function(self, X):
        """
        Optional: negative distance to nearest centroid
        """
        scores = []
        for x in X:
            dists = [np.linalg.norm(x - mu) for mu in self.centroids.values()]
            scores.append(-min(dists))
        return np.array(scores)
        

