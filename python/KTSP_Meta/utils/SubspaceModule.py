#!/usr/bin/env python
# coding: utf-8

# In[3]:


from  sklearn.decomposition import PCA
import numpy as np 


# In[4]:


class SubspaceModule:
    def __init__(self,n_components=20):
        self.n_components=n_components
        self.pca0=0;
        self.pca1=0;
    def fit(self,X,y):
        X0=X[y==0];
        X1=X[y==1];
        
        ##Tha max_components should be less than the max amount of X0,X1
        max_c0 = min(X0.shape[0], X0.shape[1]) if X0.shape[0] > 0 else 0
        max_c1 = min(X1.shape[0], X1.shape[1]) if X1.shape[0] > 0 else 0
        
        
        n0 = max(1, min(self.n_components, max_c0)) if max_c0 > 0 else 0
        n1 = max(1, min(self.n_components, max_c1)) if max_c1 > 0 else 0
        print(f"DEBUG: PCA components adjusted to: Class0={n0}, Class1={n1}")
        
        if n0 > 0:
            self.pca0 = PCA(n_components=n0)
            self.pca0.fit(X0)
        
        if n1 > 0:
            self.pca1 = PCA(n_components=n1)
            self.pca1.fit(X1)
        return self
        
    def transform(self, X):
        # if PCA not initialized，skipped transform
        results = []
        if self.pca0 is not None:
            results.append(self.pca0.transform(X))
        if self.pca1 is not None:
            results.append(self.pca1.transform(X))
            
        if not results:
            return X
            
        return np.hstack(results)
    def fit_transform(self,X,y):
        return self.fit(X, y).transform(X)
    def explained_variance_ratio(self):
        return {
            "class0": self.pca0.explained_variance_ratio_,
            "class1": self.pca1.explained_variance_ratio_
        }

