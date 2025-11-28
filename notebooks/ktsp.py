#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from itertools import combinations

class KTopScoringPairs:
    def __init__(self, K=10):
        self.K = K
        self.pairs = []      # store top K  feature pair
        self.scores = []     # for score
        self.feature_names = None

    def fit(self, X, y):
        n_features = X.shape[0]
        # iterate for all features pairs
        scores = []
        pairs = []
        for i, j in combinations(range(n_features), 2):
            score = self._pair_score(X, y, i, j)
            scores.append(score)
            pairs.append((i,j))
        # 选 top K
        top_idx = np.argsort(scores)[-self.K:]
        self.pairs = [pairs[i] for i in top_idx]
        self.scores = [scores[i] for i in top_idx]

    def _pair_score(self, X, y, i, j):
        """simple scoring：for two features for the score 
           The proportion of sequential differences in the two types of samples
        """
        correct = 0
        for k in range(X.shape[1]):
            # Sample k
            if (X[i,k] > X[j,k] and y[k]==1) or (X[i,k] < X[j,k] and y[k]==0):
                correct += 1
        return correct / X.shape[1]

    def predict(self, X):
        preds = []
        for k in range(X.shape[1]):
            votes = 0
            for (i,j) in self.pairs:
                if X[i,k] > X[j,k]:
                    votes += 1
                else:
                    votes -= 1
            preds.append(1 if votes > 0 else 0)
        return np.array(preds)
    
    def decision_function(self, X):
        """
        X: shape = (n_genes, n_samples)
        return: decision scores, shape = (n_samples,)
        """
        scores = []
        for k in range(X.shape[1]):  # Iterate samples
            x = X[:, k]
            s = 0
            for (i, j) in self.pairs:
                if x[i] > x[j]:
                    s += 1
                else:
                    s -= 1
            scores.append(s)
        return np.array(scores)
                      


# In[ ]:




