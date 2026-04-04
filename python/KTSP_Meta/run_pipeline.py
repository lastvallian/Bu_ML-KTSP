#!/usr/bin/env python
# coding: utf-8

# In[44]:


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
import os

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
        ##model_name = model_name[0] if isinstance(model_name, list) else model_name
        #model_instance = {model_name: models[model_name]}
        model_instance=MODELS[model_name]()
        model_dict = {model_name: model_instance}
        #Using KTSP_Meta_Pipeline to Processing
        pipeline = MetaPipeline(model_dict)
        #Using Run function to process
        raw_result=pipeline.run(data_path,model=model_instance,use_subspace=use_pca)
        print(f"@@@ [DEBUG] pipeline.run return raw_result_type: {type(raw_result)}")
        print(f"@@@ [DEBUG] all tributes: {dir(raw_result)}")
        def safe_convert(val):
                # convert numpy data to protype datatype
                if hasattr(val, "tolist"): return val.tolist()
                if hasattr(val, "item"): return val.item()
                return val
        def get_value(obj, key_name):
            # 1.if it is dict ,to get value
            if isinstance(obj, dict):
                # some pipeline return { "Naive Bayes": { "accuracy": 0.9 } }
                # get the value
                val = obj.get(key_name)
                if val is None and model_name in obj:
                    val = obj[model_name].get(key_name)
                return val
            
            # 2. if object to get value using getattr
            return getattr(obj, key_name, None)
        clean_result = {
            "model": model_name,
            "accuracy": safe_convert(get_value(raw_result, "accuracy")),
            "roc": safe_convert(get_value(raw_result, "roc")),
            "confusion_matrix": safe_convert(get_value(raw_result, "confusion_matrix")),
            "heatmap": safe_convert(get_value(raw_result, "heatmap"))
            }     
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
     


# In[45]:


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
     


# In[46]:


def run_ktsp_pipeline_parallel(models_names, data_path, use_pca, max_workers=None):
    print(">>> ORIGINAL parallel_PIPELINE STARTED")
    if isinstance(models_names, str):
        modelsanames = [models_names]

    max_workers = max_workers or os.cpu_count()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_ktsp_pipeline_single, model_name, data_path, use_pca): model_name
                   for model_name in models_names}

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"ERROR in model {model_name} :{e}")
    return results    
     

