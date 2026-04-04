#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pipeline/Meta_pipeline.py
import sys
import os

sys.path.append(r'C:\Users\Administrator\KTSP\KTSP_Meta')

from utils.data_utils import(
    load_and_normalize_data,
    split_data_processing_labels,
    LabelManager
)
from utils.SubspaceModule import SubspaceModule
from utils.metrics import evaluate_model
from utils.visualize import visualize_model
from utils.Module_persistency import exportModule
class MetaPipeline():
    def __init__(self,model,test_size=0.2, random_state=40):
        self.model=model
        self.test_size=test_size
        self.random_state=random_state
    def run(self,file_path,model=None,visualize=True, use_subspace=False):
     
        #1.load data
        print("STEP 1: start loading data")
        X,y,current_feature_names=load_and_normalize_data(file_path)
        import numpy as np
        unique_vals, counts_vals = np.unique(y, return_counts=True)
        print(f"@@@ [DEBUG] Label distribution: {dict(zip(unique_vals, counts_vals))}")
        
        #get current feature_names
        #current_feature_names = X.columns.tolist() 
        print(f"Detected {len(current_feature_names)} features")
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
            print(f"use_subspace:{use_subspace}")
            is_using_subspace=False  # Default as not using for subspace
            if use_subspace and name!="KTSP":
                is_using_subspace=use_subspace
                model_fit_X_train=X_train_sub
                model_fit_X_test=X_test_sub
                current_scaler = subspace
            else:
                model_fit_X_train=X_train
                model_fit_X_test=X_test
                current_scaler = None
            model.fit(model_fit_X_train,y_train)
            
            # --- added：integrated Module 5 ---
            # Exported Model package after  finishing trainning model
            print(f"STEP 4.5: Exporting {name} for inference...")
            #Get label_map
            manager = LabelManager()
            label_map = manager.label_map
            print(f"STEP 4.6: Exporting New label_map: {label_map}") 
            inf_config = exportModule(trained_model_object=model, 
                feature_list=current_feature_names, 
                model_name=name,
                label_map=label_map,         # tran for label_map
                scaler_object=current_scaler, # as for scaler_object
                use_pca=is_using_subspace                     )
            y_pred=model.predict(model_fit_X_test)
            scoreS=model.predict_proba(model_fit_X_test)
            print("STEP 4: training finished")
            
            results[name]=evaluate_model(y_test=y_test,
                                         y_pred=y_pred,
                                         y_proba=scoreS
            )
            
            # save the export_path 
            results[name]["inference_config"] = inf_config
            viz_data = None
            if visualize:
               viz_data = visualize_model(
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=scoreS,
                    c_m=results[name]["confusion_matrix"],
                    roc=results[name]["roc"],
                    X_test_selected=model_fit_X_test,
                    model=name
                )
            #storing visualization data
            results[name]["visualization"] = viz_data
                
            print("STEP 5: prediction finished")
        
        return results
        

