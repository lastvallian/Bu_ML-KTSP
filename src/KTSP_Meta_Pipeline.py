#!/usr/bin/env python
# coding: utf-8

# In[7]:


#pipeline/Meta_pipeline.py
from utils.data_utils import(
    load_and_normalize_data,
    split_data_processing_labels
)

class MetaPipeline():
    def __init__(self,model):
        self.model=model
        self.test_size=test_size
        self.random_state=random_state
    def run(self,file_path,visualize=True):
    
        #1.load data
       
        X_train,X_test,y_train,y_test=load_and_normalize_data(file_path)
        
        #2.Split data
        X_train,X_test,y_train,y_test=split_data_processing_labels(X,y,test_size=self.test_size,random_state=self.random_size)
        
        #3.loop models
        results={}
        
        for name ,models in self.models.items:
            print(f"\n=== Training{name}===")
            
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            scoreS=model.predict_proba(X_test)
            
            results[name]=evaluate_model(
            )
            
            if visualize:
                visualize_model(
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    confusion_matrix=results[name]["confusion_matrix"],
                    roc_data=results[name]["roc_data"]
                )
        
        return results
        
        


# In[ ]:




