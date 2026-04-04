#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pipeline/Meta_pipeline.py
import sys
import os
sys.path.append(r'C:\Users\Administrator\KTSP\KTSP_Meta')

from utils.data_utils import(
    load_and_normalize_data,
    split_data_processing_labels
)
from utils.SubspaceModule import SubspaceModule
from utils.metrics import evaluate_model
from utils.visualize import visualize_model
class MetaPipeline():
    def __init__(self,model,test_size=0.2, random_state=40):
        self.model=model
        self.test_size=test_size
        self.random_state=random_state
    def run(self,file_path,model=None,visualize=True, use_subspace=False):
    
        #1.load data
        print("STEP 1: start loading data")
        X,y=load_and_normalize_data(file_path)
        print("STEP 2: data loaded")
        
        #2.Split data
        X_train,X_test,y_train,y_test=split_data_processing_labels(X,y,test_size=self.test_size,random_state=self.random_state)
        
        
        #3.Opional data
       
        if use_subspace:
            n_samples = X_train.shape[0]
            n_pca = min(20, int(n_samples * 0.5))# for half of the samples 
            subspace = SubspaceModule(n_components=n_pca)
            pca_models=[name for name in self.model.keys() if name!="KTSP"]
            if pca_models:
                print(f"Applying SubspaceModule for models:{pca_models}")
                X_train_sub = subspace.fit_transform(X_train,y_train)
                X_test_sub = subspace.transform(X_test)
            else:
                X_train_sub,X_test_sub=X_train,X_test
        else:
            X_train_sub,X_test_sub=X_train,X_test
           
        #4.loop models
        results={}
        
        for name,model in self.model.items():
            print(f"\n=== Training{name}===")
            
            print("STEP 3: start training")
            ##For Subspace
            if use_subspace and name!="KTSP":
                model_fit_X_train=X_train_sub
                model_fit_X_test=X_test_sub
            else:
                model_fit_X_train=X_train
                model_fit_X_test=X_test
            model.fit(model_fit_X_train,y_train)
            y_pred=model.predict(model_fit_X_test)
            scoreS=model.predict_proba(model_fit_X_test)
            print("STEP 4: training finished")
            
            results[name]=evaluate_model(y_test=y_test,
                                         y_pred=y_pred,
                                         y_proba=scoreS
            )
            
            if visualize:
                visualize_model(
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=scoreS,
                    c_m=results[name]["confusion_matrix"],
                    roc=results[name]["roc"]
                )
                
            print("STEP 5: prediction finished")
        
        return results
        

