#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)
import sys
import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import run_pipeline 
from run_pipeline import run_ktsp_pipeline_parallel
import sys
from pydantic import BaseModel
from utils.Module_persistency import run_on_user_input

# 3. using lifespan instead @app.on_event("startup")
class TrainRequest(BaseModel):
    model_name: str
    data_path: str
    use_pca: bool
    custom_folder: str
@asynccontextmanager
async def lifespan(app: FastAPI):
    # operation for (Startup)
    print(">>> startup，has skipped  Ray Initilized (Ray Disabled)")
    yield
    # operation for (Shutdown)
    print(">>> System is closing...")
app = FastAPI(lifespan=lifespan)
origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"]
)

import numpy as np
@app.post("/upload")
async def upload_file(file:UploadFile=File(...)):
    #tempraily keep file
    temp_path=f"temp_{file.filename}" 
    with open(temp_path,"wb") as f:
        shutil.copyfileobj(file.file,f)
        return {"filename":file.filename,"save as":temp_path} 
class PipelineRequest(BaseModel):
    model_name:str
    data_path:str
    use_pca:str
@app.post("/run")
def execute_pipeline_task(
    file:UploadFile=File(...),
    models_names: str = Form(...),
    use_pca: str = Form("false")):
    ##req:PipelineRequest):
    
    #save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    model_list=json.loads(models_names)
    actual_use_pca = True if use_pca.lower() == "true" else False
    
   
    #Callrun.py as run_ktsp_pipeline_ray
    results=run_ktsp_pipeline_parallel(
        models_names=model_list,
        data_path= temp_path,
        use_pca=actual_use_pca
        ##ray_address=auto,
        ##num_cpus=num_cpus
    )
    def clean_json(obj):
        """clean all non json data"""
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean_json(i) for i in obj]
        return obj
     #flattened result
    flattened_results = []
    for r in results:
        item = r[0] if isinstance(r, list) and len(r) > 0 else r
        # make sure item clean for  JSON dict
        if isinstance(item, dict):
            print(f"DEBUG: item type is {type(item)}")
            flattened_results.append(clean_json(item))
    print(f"DEBUG: Flattened count: {len(flattened_results)}")
    if len(flattened_results) > 0:
        print(f"DEBUG: First item keys: {flattened_results[0].keys()}")         
    final_response = {
    "results": flattened_results,
    "accuracy": [r.get("performance", {}).get("accuracy", 0) for r in flattened_results if isinstance(r, dict)],
    "auc": [r.get("performance", {}).get("auc", 0) for r in flattened_results if isinstance(r, dict)]
    # ...
    } 
    cleaned_data = clean_json(final_response)
    print(f"DEBUG: FINAL JSON TO FRONTEND: { json.dumps(cleaned_data)[:500]}...")
    return{
        "results": cleaned_data,
        "accuracy": [r.get("performance", {}).get("accuracy", 0) for r in cleaned_data if isinstance(r, dict)],
        "auc": [r.get("performance", {}).get("auc", 0) for r in cleaned_data if isinstance(r, dict)],
        "confusion_matrix": [r.get("performance", {}).get("confusion_matrix", []) for r in cleaned_data if isinstance(r, dict)]
    }
    print(f"DEBUG ACC LIST: {cleaned_data['accuracy']}")
#train_predict
@app.post("/api/train_and_export")
async def train_endpoint(req: TrainRequest):
    # writing for saving models
    results = run_ktsp_pipeline_single(
        model_name=req.model_name,
        data_path=req.data_path,
        use_pca=req.use_pca,
        export_name=req.custom_folder  # saving_document
    )
    
    return {"status": "success", "results": results}

@app.post("/predict")
async def predict_endpoint( file:UploadFile=File(...),folder_name: str = Form(...)):
    """
    recieve the uploadfile by user,running Module6 and return result
    """
    # save the uploadfile temporaily
    temp_path=f"temp_{file.filename}"
    with open(temp_path,"wb")as buffer:
        shutil.copyfileobj(file.file,buffer)
    try:
        #Config_path
        config_path = os.path.join("saved_models", folder_name, "inference_config.json")
        # call run_on_user_input from Module_persistency
        results=run_on_user_input(temp_path, config_path)
        return {
            "status":"success",
            "data":results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. delete tempfile
        if os.path.exists(temp_path):
            os.remove(temp_path)

