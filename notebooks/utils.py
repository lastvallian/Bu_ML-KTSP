#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def normalize_label(l):
    """name for label"""
    l = l.strip().lower()
    # Prostate AND GCM
    if l in ["tumor", "t"]:
        return "tumor"
    elif l in ["normal", "n", "control"]:
        return "normal"
    # Leukemia
    elif l in ["all"]:
        return "all"
    elif l in ["aml"]:
        return "aml"
    #lung dataset
    elif l in["adca"]:
        return "adca"
    elif l in["mesothelioma"]:
        return "mesothelioma"
    # DLBCL dataset
    elif l in ["dlbcl"]:
        return "dlbcl"
    elif l in ["fl"]:
        return "fl"
    # Colon tumor/normal
    elif l in ["tumor", "cancer"]:
        return "tumor"
    elif l in ["normal", "control"]:
        return "normal"
    else:
        return l  # 保留原标签

def load_dataset(name,datasets):
    """
    read file according to the file，return matrix X、int label y、original label list labels and  unique_labels
    """
    filename = datasets[name]
    with open(filename, 'r') as f:
        lines = f.readlines()

    # the first line is tag or not（prostate2/prostate3）
    raw_labels = lines[0].strip().split(',')
    if all(l.replace('.', '').replace('-', '').isdigit() for l in raw_labels):
        # The first row is data not for label
        data = np.array([list(map(float, line.strip().split(','))) for line in lines])
        # For the dataset for file（prostate2, prostate3），assign the value for tumor/normal according to the number of sample
        # Assumed the the first some column n_tumor is tumor，the last some n_normal column normal
        n_samples = data.shape[1]
        # The demo example：The first half for tumor，the last half for normal
        half = n_samples // 2
        labels = ["tumor"] * half + ["normal"] * (n_samples - half)
    else:
        # The first row is label
        labels = [normalize_label(l) for l in raw_labels]
        data = np.array([list(map(float, line.strip().split(','))) for line in lines[1:]])
    
    #if the original data is (samples, features) then tranform to (features, samples)
    #for lines<rows,
    if data.shape[0] < data.shape[1]:
        data = data.T

    # last row is the tage
    #labels = data[-1, :]
    #X = data[:, :-1].T  # transform (features, samples)
        
    #int code
    unique_labels = list(sorted(set(labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels])
    
    return data, y, labels, unique_labels

#Select the top var genes like 300
def select_top_var_genes(X, top=300):
    vars = np.var(X, axis=1)
    idx = np.argsort(vars)[-top:]
    return X[idx, :], idx

