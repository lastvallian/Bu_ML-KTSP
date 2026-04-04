#!/usr/bin/env python
# coding: utf-8

# In[35]:


#!/usr/bin/env python
# coding: utf-8

"""
Ray cluster configuration and distributed processing for KTSP pipeline
Provides robust, safe distributed computing with resource management
"""
import os
import sys
import json
import pickle
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import time
import os
current_directory = os.getcwd()
import torch



# Add project path
sys.path.append(os.path.dirname(os.path.abspath(current_directory)))

try:
    import ray
    from ray.util.client import ray as ray_client
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray is not installed. Install with: pip install ray[default]")

# Import required modules
from utils.data_utils import load_and_normalize_data, split_data_processing_labels
##from utils.dataset import Dataset
from utils.SubspaceModule import SubspaceModule
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from models.Ktsp_model import KTSPClassifier
from models.multi_model_trainer import SKLearnModelWrapper, ShrinkageCentroidClassifier
from utils.metrics import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[36]:


# Global Ray cluster manager instance
_ray_manager: Optional["RayClusterManager"] = None

def get_ray_manager(
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    **kwargs
) -> "RayClusterManager":
    """Get or create Ray cluster manager singleton"""
    global _ray_manager

    #num_gpus = 1 if torch.cuda.is_available() else 0
    num_gpus = 0
    if _ray_manager is None:
        _ray_manager = RayClusterManager(
            address=address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            **kwargs
        )
        _ray_manager.initialize()

    return _ray_manager    

class RayClusterManager:
    """Manages Ray cluster connection and resources"""
    def __init__(
        self,
        address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        object_store_memory: Optional[int] = None,
        runtime_env: Optional[Dict] = None,
        ignore_reinit_error: bool = True
    ):
        """
        Initialize Ray cluster manager
        
        Args:
            address: Ray cluster address (None for local)
            num_cpus: Number of CPUs to use (None for auto)
            num_gpus: Number of GPUs to use (None for auto)
            object_store_memory: Object store memory in bytes (None for auto)
            runtime_env: Runtime environment configuration
            ignore_reinit_error: Ignore reinitialization errors
        """
        self.address = address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.object_store_memory = object_store_memory
        self.runtime_env = runtime_env or {}
        self.ignore_reinit_error = ignore_reinit_error
        self.is_initialized = False
        self._ray_context = None
        
    def initialize(self) -> bool:
        """
        Initialize Ray cluster connection
        
        Returns:
            True if successful, False otherwise
        """
        if not RAY_AVAILABLE:
            logger.error("Ray is not available. Please install: pip install ray[default]")
            return False
        
        try:
            if self.address:
                # Connect to remote cluster
                logger.info(f"Connecting to Ray cluster at {self.address}")
                ray.init(
                    address=self.address,
                    runtime_env=self.runtime_env,
                    ignore_reinit_error=self.ignore_reinit_error
                )
            else:
                # Initialize local cluster
                init_kwargs = {
                    "ignore_reinit_error": self.ignore_reinit_error,
                    "runtime_env": self.runtime_env
                }
                
                if self.num_cpus is not None:
                    init_kwargs["num_cpus"] = self.num_cpus
                if self.num_gpus is not None:
                    init_kwargs["num_gpus"] = self.num_gpus
                if self.object_store_memory is not None:
                    init_kwargs["object_store_memory"] = self.object_store_memory
                
                logger.info(f"Initializing local Ray cluster with {init_kwargs}")
                ray.init(**init_kwargs)
            
            self.is_initialized = True
            logger.info(f"Ray cluster initialized. Resources: {ray.cluster_resources()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {str(e)}")
            self.is_initialized = False
            return False
    
    def shutdown(self):
        """Safely shutdown Ray cluster"""
        if RAY_AVAILABLE and self.is_initialized:
            try:
                ray.shutdown()
                self.is_initialized = False
                logger.info("Ray cluster shutdown successfully")
            except Exception as e:
                logger.warning(f"Error during Ray shutdown: {str(e)}")
    
    @contextmanager
    def cluster_context(self):
        """Context manager for Ray cluster"""
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Ray cluster")
        try:
            yield self
        finally:
            # Don't shutdown in context manager - let it persist
            pass
    
    def get_resources(self) -> Dict:
        """Get current cluster resources"""
        if not RAY_AVAILABLE or not self.is_initialized:
            return {}
        try:
            return {
                "available": ray.available_resources(),
                "cluster": ray.cluster_resources()
            }
        except:
            return {}



    # Ray remote functions for distributed processing
    # These are defined conditionally based on Ray availability

    if RAY_AVAILABLE:

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)


# In[37]:


#raw_cluster_adapter.py
from run_pipeline import run_ktsp_pipeline #original run_pipeline
from pipe_line.KTSP_Meta_Pipeline import MetaPipeline
from utils.models_registry import MODELS

#For New MetaPipeLine Wrapper
class LegacyPipelineWrapper:
    """ Adapter wrapper to call KSTP pipeline"""
    def __init__(self,use_pca=True):
        self.use_pca=use_pca
        
    def run(self,dataset_path:str,model_name:str,use_pca:str):
        #Create modelinstance
        model_instance = MODELS[model_name]()

        # each modelfor MetaPipeline
        pipeline = MetaPipeline(model_instance)
        return run_ktsp_pipeline(dataset_path,[model_name],use_pca=use_pca)


# In[38]:


@ray.remote(max_retries=2,num_cpus=2)
def ray_train_legacy(dataset_path:str,model_name: str,top_genes: int,top_pairs: int,
                     task_id:str,use_pca:bool):
    ##from raw_cluster_adapter import LegacyPipelineWrapper
    print(">>> TASK STARTED:", model_name)
    print("DEBUG worker model_name:", model_name) 
    wrapper=LegacyPipelineWrapper(use_pca=use_pca)
    results=wrapper.run(dataset_path,model_name,use_pca)
    
    #save task_id 
    import os,json
    results_dir='task_results'
    os.makedirs(results_dir,exist_ok=True)
    result_file=os.path.join(results_dir,f'result_{task_id}.json')
    with open(result_file,'w')as f:
        json.dump(results,f,indent=2)
    return {'status':'success','task_id':task_id,'results':result}
    


# In[39]:


class RayPipelineExecutor:
    """Execute KTSP pipeline using Ray cluster"""

    def __init__(
        self,
        ray_address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        timeout: int = 3600
    ):
        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.manager = None

    def initialize(self) -> bool:
        self.manager = get_ray_manager(
            address=self.ray_address,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus
        )
        return self.manager.is_initialized

    def execute_pipeline(
        self,
        file_path: str,
        models_names: List[str],
        use_pca: bool,
        task_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        
        #ensure models be the list
        if isinstance(models_names, str):
            models_names = [models_names]
        # Initialize Ray 
        if not self.manager or not self.manager.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Ray cluster")

        # commit tasks
        training_futures = {}

        total_models = len(models_names)
        print("DEBUG models_names:", models_names)
        print("DEBUG type:", type(models_names))

        for idx, model_name in enumerate(models_names):

            if progress_callback:
                progress = 50 + int((idx / total_models) * 20)
                progress_callback(progress, f"training_{model_name}")

            future = ray_train_legacy.remote(
                dataset_path=file_path,
                model_name=model_name,
                top_genes=300,
                top_pairs=10,
                task_id=task_id,
                use_pca=use_pca
            )

            training_futures[model_name] = future
        
        results_list = ray.get(list(training_futures.values()))
        
        aggregated_results = []
        for r in results_list:
            aggregated_results.append(r['results'])  # r['results'] 是 wrapper.run() 返回的结果


        # Save TaskResults
        results_dir = "task_results"
        os.makedirs(results_dir, exist_ok=True)

        #task_file = os.path.join(results_dir, f"task_{task_id}.json")
        result_file = os.path.join(results_dir, f"task_{task_id}.json")

        ##task_info = {
          #  "status": "submitted",
           # "task_id": task_id,
           # "models": list(training_futures.keys())
        #}

        with open(result_file, "w") as f:
            json.dump(aggregated_results, f, indent=2)

        return aggregated_results

    def shutdown(self):
        """Shutdown Ray cluster"""
        if self.manager:
            self.manager.shutdown()

    def execute_with_ray(
        file_path: str,
        models_names: List[str],
        use_pca: bool,
        task_id: str,
        ray_address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Convenience function to execute pipeline with Ray

        Args:
            file_path: Path to data file
            models_names: List of model names
            use_pca: Whether to use PCA
            task_id: Task identifier
            ray_address: Ray cluster address
            num_cpus: Number of CPUs
            progress_callback: Progress callback

        Returns:
            Results dictionary
        """
        executor = RayPipelineExecutor(
            ray_address=ray_address,
            num_cpus=num_cpus
        )

        try:
            return executor.execute_pipeline(
                file_path=file_path,
                models_names=models_names,
                use_pca=use_pca,
                task_id=task_id,
                progress_callback=progress_callback
            )
        finally:
            # Note: We don't shutdown here to keep cluster alive for future tasks
            pass

