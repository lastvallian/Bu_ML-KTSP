#!/usr/bin/env python
# coding: utf-8

# In[1]:


##KTSP_front# models_registry.py

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from models.multi_model_trainer import SKLearnModelWrapper
from models.Ktsp_model import KTSPClassifier
from models.multi_model_trainer import ShrinkageCentroidClassifier


MODELS = {
    "Linear SVM": lambda: SKLearnModelWrapper(
        SVC(kernel="linear", C=1, probability=True)
    ),
    "RBF SVM": lambda: SKLearnModelWrapper(
        SVC(kernel="rbf", C=5, gamma=0.05, probability=True)
    ),
    "Decision Tree": lambda: SKLearnModelWrapper(
        DecisionTreeClassifier()
    ),
    "Naive Bayes": lambda: SKLearnModelWrapper(
        GaussianNB()
    ),
    "kNN": lambda: SKLearnModelWrapper(
        KNeighborsClassifier(n_neighbors=5)
    ),
    "KTSP": lambda: KTSPClassifier(top_genes=300, top_pairs=10),
    "Custom SVMTrainer": lambda: SKLearnModelWrapper(
        ShrinkageCentroidClassifier(shrinkage=2.0)
    )
}

