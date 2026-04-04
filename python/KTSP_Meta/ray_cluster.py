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

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import ray
    from ray.util.client import ray as ray_client
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray is not installed. Install with: pip install ray[default]")

# Import required modules
from utils.data_utils import load_and_normalize_data, split_data_processing_labels
from utils.dataset import Dataset
from utils.SubspaceModule import SubspaceModule
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from models.Ktsp_model import KTSPClassifier
from models.multi_model_trainer import SKLearnModelWrapper, ShrinkageCentroidClassifier
from utils.metrics import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# Global Ray cluster manager instance
_ray_manager: Optional[RayClusterManager] = None


def get_ray_manager(
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    **kwargs
) -> RayClusterManager:
    """Get or create Ray cluster manager singleton"""
    global _ray_manager
    
    if _ray_manager is None:
        _ray_manager = RayClusterManager(
            address=address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            **kwargs
        )
        _ray_manager.initialize()
    
    return _ray_manager


# Ray remote functions for distributed processing
# These are defined conditionally based on Ray availability

if RAY_AVAILABLE:
    @ray.remote(max_retries=2, num_cpus=1)
    def ray_load_data(file_path: str, task_id: str) -> Dict:
        """
        Ray remote function: Load data and save as Dataset object
        
        Args:
            file_path: Path to data file
            task_id: Task identifier
            
        Returns:
            Dictionary with dataset path and status
        """
        try:
            # Load data
            X, y = load_and_normalize_data(file_path)
            
            # Create Dataset object
            dataset = Dataset(
                X=X,
                y=y,
                file_path=file_path,
                task_id=task_id
            )
            
            # Save dataset
            dataset_dir = 'datasets'
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.join(dataset_dir, f'dataset_{task_id}.pkl')
            dataset.save(dataset_path)
            
            return {
                'status': 'success',
                'dataset_path': dataset_path,
                'shape_info': dataset.get_shape_info(),
                'task_id': task_id
            }
        except Exception as e:
            logger.error(f"Error in ray_load_data: {str(e)}")
            raise


    @ray.remote(max_retries=2, num_cpus=2)
    def ray_preprocess_data(dataset_path: str, use_pca: bool, task_id: str) -> Dict:
        """
        Ray remote function: Execute preprocessing (PCA + normalization)
        
        Args:
            dataset_path: Path to Dataset object
            use_pca: Whether to use PCA
            task_id: Task identifier
            
        Returns:
            Dictionary with processed dataset path and status
        """
        try:
            # Load dataset
            dataset = Dataset.load(dataset_path)
            
            # Split data
            X_train, X_test, y_train, y_test = split_data_processing_labels(
                dataset.X, dataset.y, test_size=0.2, random_state=40
            )
            
            dataset.X_train = X_train
            dataset.X_test = X_test
            dataset.y_train = y_train
            dataset.y_test = y_test
            
            # Apply preprocessing
            if use_pca:
                n_samples = X_train.shape[0]
                n_pca = min(20, int(n_samples * 0.5))
                subspace = SubspaceModule(n_components=n_pca)
                X_train_processed = subspace.fit_transform(X_train, y_train)
                X_test_processed = subspace.transform(X_test)
            else:
                X_train_processed = X_train.copy()
                X_test_processed = X_test.copy()
            
            # Normalization
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_processed)
            X_test_processed = scaler.transform(X_test_processed)
            
            dataset.X_train_processed = X_train_processed
            dataset.X_test_processed = X_test_processed
            
            # Save updated dataset
            dataset.save(dataset_path)
            
            return {
                'status': 'success',
                'dataset_path': dataset_path,
                'use_pca': use_pca,
                'task_id': task_id
            }
        except Exception as e:
            logger.error(f"Error in ray_preprocess_data: {str(e)}")
            raise

    @ray.remote(max_retries=2, num_cpus=4)
    def ray_train_model(
        dataset_path: str,
        model_name: str,
        top_genes: int = 300,
        top_pairs: int = 10,
        task_id: str = None
    ) -> Dict:
        """
        Ray remote function: Train model
        
        Args:
            dataset_path: Path to Dataset object
            model_name: Name of model to train
            top_genes: Number of top genes (for KTSP)
            top_pairs: Number of top pairs (for KTSP)
            task_id: Task identifier
            
        Returns:
            Dictionary with model path, results, and status
        """
        try:
            # Load dataset
            dataset = Dataset.load(dataset_path)
            
            # Determine which data to use
            if model_name == "KTSP":
                X_train = dataset.X_train
                X_test = dataset.X_test
            else:
                X_train = dataset.X_train_processed if dataset.X_train_processed is not None else dataset.X_train
                X_test = dataset.X_test_processed if dataset.X_test_processed is not None else dataset.X_test
            
            # Initialize model
            models_dict = {
                "Linear SVM": SKLearnModelWrapper(SVC(kernel="linear", C=1, probability=True)),
                "RBF SVM": SKLearnModelWrapper(SVC(kernel="rbf", C=5, gamma=0.05, probability=True)),
                "Decision Tree": SKLearnModelWrapper(DecisionTreeClassifier()),
                "Naive Bayes": SKLearnModelWrapper(GaussianNB()),
                "kNN": SKLearnModelWrapper(KNeighborsClassifier(n_neighbors=5)),
                "KTSP": KTSPClassifier(top_genes=top_genes, top_pairs=top_pairs),
                "Custom SVMTrainer": SKLearnModelWrapper(ShrinkageCentroidClassifier(shrinkage=2.0))
            }
            
            if model_name not in models_dict:
                raise ValueError(f"Model '{model_name}' is not supported")
            
            # Train model
            model = models_dict[model_name]
            model.fit(X_train, dataset.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate model
            results = evaluate_model(
                y_test=dataset.y_test,
                y_pred=y_pred,
                y_proba=y_proba
            )
            
            # Save model
            models_dir = 'saved_models'
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'model_{task_id}_{model_name}.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Convert numpy arrays to lists for JSON serialization
            def to_json_safe(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: to_json_safe(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [to_json_safe(v) for v in obj]
                return obj
            
            results_json = to_json_safe(results)
            
            return {
                'status': 'success',
                'model_path': model_path,
                'model_name': model_name,
                'results': results_json,
                'task_id': task_id
            }
        except Exception as e:
            logger.error(f"Error in ray_train_model: {str(e)}")
            raise
else:
    # Placeholder functions when Ray is not available
    def ray_load_data(*args, **kwargs):
        raise RuntimeError("Ray is not available")
    
    def ray_preprocess_data(*args, **kwargs):
        raise RuntimeError("Ray is not available")
    
    def ray_train_model(*args, **kwargs):
        raise RuntimeError("Ray is not available")


class RayPipelineExecutor:
    """Execute KTSP pipeline using Ray cluster"""
    
    def __init__(
        self,
        ray_address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        timeout: int = 3600
    ):
        """
        Initialize Ray pipeline executor
        
        Args:
            ray_address: Ray cluster address (None for local)
            num_cpus: Number of CPUs per task
            num_gpus: Number of GPUs per task
            timeout: Task timeout in seconds
        """
        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.manager = None
        
    def initialize(self) -> bool:
        """Initialize Ray cluster"""
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
        """
        Execute full pipeline using Ray
        
        Args:
            file_path: Path to data file
            models_names: List of model names
            use_pca: Whether to use PCA
            task_id: Task identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with results
        """
        if not self.manager or not self.manager.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Ray cluster")
        
        try:
            # Step 1: Load data
            if progress_callback:
                progress_callback(10, 'loading_data')
            
            load_future = ray_load_data.remote(file_path, task_id)
            load_result = ray.get(load_future, timeout=self.timeout)
            dataset_path = load_result['dataset_path']
            
            if progress_callback:
                progress_callback(30, 'preprocessing')
            
            # Step 2: Preprocess data
            preprocess_future = ray_preprocess_data.remote(dataset_path, use_pca, task_id)
            preprocess_result = ray.get(preprocess_future, timeout=self.timeout)
            
            # Step 3: Train models in parallel
            all_results = []
            total_models = len(models_names)
            
            # Submit all training tasks
            training_futures = []
            for idx, model_name in enumerate(models_names):
                if progress_callback:
                    progress = 50 + int((idx / total_models) * 20)
                    progress_callback(progress, f'training_{model_name}')
                
                future = ray_train_model.remote(
                    dataset_path=dataset_path,
                    model_name=model_name,
                    top_genes=300,
                    top_pairs=10,
                    task_id=task_id
                )
                training_futures.append((model_name, future))
            
            # Collect results
            for idx, (model_name, future) in enumerate(training_futures):
                if progress_callback:
                    progress = 70 + int((idx / total_models) * 30)
                    progress_callback(progress, f'collecting_{model_name}')
                
                train_result = ray.get(future, timeout=self.timeout)
                
                all_results.append({
                    'model': model_name,
                    'accuracy': train_result['results'].get('accuracy'),
                    'roc': train_result['results'].get('roc'),
                    'confusion_matrix': train_result['results'].get('confusion_matrix'),
                    'model_path': train_result['model_path']
                })
            
            if progress_callback:
                progress_callback(100, 'complete')
            
            # Save results
            results_dir = 'task_results'
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, f'result_{task_id}.json')
            
            result_data = {
                'status': 'success',
                'task_id': task_id,
                'results': all_results,
                'dataset_path': dataset_path,
                'progress': 100,
                'step': 'complete',
                'executor': 'ray'
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return result_data
            
        except Exception as e:
            if "GetTimeoutError" in str(type(e)) or "TimeoutError" in str(type(e)):
            logger.error(f"Task {task_id} timed out after {self.timeout} seconds")
            raise TimeoutError(f"Pipeline execution timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise
    
    def shutdown(self):
        """Shutdown Ray cluster"""
        if self.manager:
            self.manager.shutdown()


def execute_with_ray(
    file_path: str,
    models_names: List[str],
    use_pca: bool,
 P   task_id: str,
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

