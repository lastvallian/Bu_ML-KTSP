#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from pipe_line.KTSP_Meta_Pipeline import MetaPipeline
from models.multi_model_trainer import SKLearnModelWrapper
from models.multi_model_trainer import ShrinkageCentroidClassifier
from models.Ktsp_model import KTSPClassifier
import ray
import numpy as np
from utils.models_registry import MODELS
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.Module_persistency import exportModule
import os

# put the function outside  UnboundLocalError
def safe_convert(val):
    if hasattr(val, "tolist"): return val.tolist()
    if hasattr(val, "item"): return val.item()
    return val

def get_nested_value(obj, m_name, key):
    """
    Resolving the embeded dictionary value problems
    Supported for  obj['kNN']['accuracy'] 和 obj['accuracy'] 
    """
    if not isinstance(obj, dict):
        return getattr(obj, key, None)
    #resloving the emdbed data
    model_data = obj.get(m_name)
    if isinstance(model_data, dict):
        val= model_data.get(key)
        if val is not None:
            return val
    return obj.get(key)
##from Ray_cluster import RayPipelineExecutor
def run_ktsp_pipeline_single(
    model_name:str,
    #models_names:list[str],
    data_path:str,
    use_pca:bool
):
    print(">>> ORIGINAL PIPELINE STARTED")
    print(f"\n@@@ [DEBUG] running new version: {model_name} @@@")
   
    results_list=[]
    
  
    print(f"\n====Training {model_name}===")
    try:
        
        # 1. read from original CSV for (header)
        # nrows=0 just read from lables
        
        model_instance=MODELS[model_name]()
        model_dict = {model_name: model_instance}
        from utils.data_utils import load_and_normalize_data
        X_temp, y_temp, actual_feature_names = load_and_normalize_data(data_path)        
        #Using KTSP_Meta_Pipeline to Processing
        pipeline = MetaPipeline(model_dict)
        #Using Run function to process
        raw_result=pipeline.run(data_path,model=model_instance,use_subspace=use_pca)
        model_perf = raw_result.get(model_name, {})
        roc_info = model_perf.get("roc", {}) # using roc_data
        inference_info = model_perf.get("inference_config", {})
        #inference_info=exportModule(trained_model_object=model_instance,
           #                         feature_list=actual_feature_names,model_name=model_name)
        auc_val = 0.0
        if isinstance(roc_info, dict):
            auc_val = roc_info.get('auc', 0.0)
        elif isinstance(roc_info, (float, int)): # for error
            auc_val = roc_info    
        clean_result = {
            # --- coreprediction configuration ---
            "model_path": inference_info.get("model_path"),
            "features": actual_feature_names,                # for "features"
            "model_name": model_name, 
            "model_class": model_name,                      # for "model_class"
            "model_type": inference_info.get("model_type"),
            "scaler_path": inference_info.get("scaler_path"),
            "use_pca": inference_info.get("use_pca", False),
            "label_map": inference_info.get("label_map"),   # for：front label_map
            "created_at": inference_info.get("created_at"),
            
            # --- performance (display for console) ---
            "performance": {
                "accuracy": safe_convert(get_nested_value(raw_result, model_name, "accuracy")),
                "auc" : safe_convert(auc_val),
                "roc": {                         # transfer to front to draw
                        "fpr": safe_convert(roc_info.get('fpr')), 
                        "tpr": safe_convert(roc_info.get('tpr')),
                        "auc": safe_convert(auc_val)
                    },
                "confusion_matrix": safe_convert(get_nested_value(raw_result, model_name, "confusion_matrix")),
                "heatmap": safe_convert(get_nested_value(raw_result, model_name, "heatmap")),
                
            },
            
            # config_path
            "config_path": os.path.join("saved_models", model_name, "inference_config.json")
        }
        
            
        
        # simulate module6 calling for
        # if clean_result["model_path"]:
        #    test_inference = run_on_user_input(data_path, clean_result["model_path"])
        
        clean_result["config_path"] = os.path.join("saved_models", model_name, "inference_config.json")
        print(f"@@@ [DEBUG] {model_name} return for dict")
        
        return [clean_result] # return list of dict

    # --- for except ---
    except Exception as e:
        print(f"ERROR in model {model_name}: {str(e)}")
        
        # debugging for raw_result
        if 'raw_result' in locals():
            print(f"DEBUG: raw_result type is {type(raw_result)}")
        
        # print debug for tracying
        import traceback
        print(f"@@@ [DEBUG] {model_name} collapse!")
        traceback.print_exc()    
        
        return []
     


# In[16]:


def run_ktsp_pipeline_ray(models_names,data_path,use_pca,ray_address=None,num_cpus=None):
    print(">>> RAY PIPELINE STARTED")
    ##if not ray.is_initialized():
    ##    ray.init(ignore_reinit_error=True)
    from Ray_cluster import RayPipelineExecutor
    executor= RayPipelineExecutor(ray_address=ray_address, num_cpus=num_cpus)
    results=executor.execute_pipeline(
        file_path=data_path,
        models_names=models_names,
        use_pca=use_pca,
        task_id="task_"+str(np.random.randint(100000)))
    return results
     


# In[17]:


## Run for parallel pipeline to use run_single using efficiency of Server
def run_ktsp_pipeline_parallel(models_names, data_path, use_pca, max_workers=None):
    print(">>> ORIGINAL parallel_PIPELINE STARTED")
    
    # ensure models_names is list
    if isinstance(models_names, str):
        models_names = [models_names]

    # set parallel process numbers
    max_workers = max_workers or os.cpu_count()
    results = []
    
    # using ProcessPoolExecutor for paraelle computing
    # notice:under Windows make sure it was protected if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # committing for tasks
        futures = {
            executor.submit(run_ktsp_pipeline_single, model_name, data_path, use_pca): model_name
            for model_name in models_names
        }

        # collect results
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                # make sure result is lisk and  extending
                if result and isinstance(result, list):
                    results.extend(result)
                print(f"--- Model {model_name} finished parallel training ---")
            except Exception as e:
                print(f"ERROR in model {model_name} execution: {e}")
                
    return results

