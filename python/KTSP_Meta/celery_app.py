#!/usr/bin/env python
# coding: utf-8

"""
Celery configuration and worker tasks for KTSP pipeline
"""
import os
import sys
import shutil
import json
import pickle
import numpy as np
from celery import Celery
from celery.result import AsyncResult
from typing import Dict, List, Optional

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Celery configuration
celery_app = Celery(
    'ktsp_meta',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    worker_prefetch_multiplier=1,
)

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


def _load_data_helper(file_path: str, task_id: str) -> Dict:
    """Helper function to load data (can be called directly or as a task)"""
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


@celery_app.task(bind=True, name='tasks.load_data')
def load_data_task(self, file_path: str, task_id: str) -> Dict:
    """
    Task 1: Load data and save as Dataset object
    
    Args:
        file_path: Path to the data file
        task_id: Unique task identifier
        
    Returns:
        Dictionary with dataset path and status
    """
    try:
        self.update_state(state='PROGRESS', meta={'step': 'loading_data', 'progress': 10})
        result = _load_data_helper(file_path, task_id)
        self.update_state(state='PROGRESS', meta={'step': 'data_loaded', 'progress': 20})
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


def _preprocess_data_helper(dataset_path: str, use_pca: bool, task_id: str) -> Dict:
    """Helper function to preprocess data"""
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
        # PCA preprocessing
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


@celery_app.task(bind=True, name='tasks.preprocess_data')
def preprocess_data_task(self, dataset_path: str, use_pca: bool, task_id: str) -> Dict:
    """
    Task 2: Execute preprocessing (PCA + normalization)
    
    Args:
        dataset_path: Path to saved Dataset object
        use_pca: Whether to use PCA preprocessing
        task_id: Unique task identifier
        
    Returns:
        Dictionary with processed dataset path and status
    """
    try:
        self.update_state(state='PROGRESS', meta={'step': 'preprocessing', 'progress': 30})
        self.update_state(state='PROGRESS', meta={'step': 'data_split', 'progress': 40})
        result = _preprocess_data_helper(dataset_path, use_pca, task_id)
        self.update_state(state='PROGRESS', meta={'step': 'preprocessing_complete', 'progress': 50})
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


def _train_model_helper(
    dataset_path: str, 
    model_name: str,
    top_genes: int = 300,
    top_pairs: int = 10,
    task_id: str = None
) -> Dict:
    """Helper function to train model"""
    # Load dataset
    dataset = Dataset.load(dataset_path)
    
    # Determine which data to use
    # For KTSP, use original data (not PCA processed)
    if model_name == "KTSP":
        X_train = dataset.X_train
        X_test = dataset.X_test
    else:
        # For other models, use processed data if available
        X_train = dataset.X_train_processed if dataset.X_train_processed is not None else dataset.X_train
        X_test = dataset.X_test_processed if dataset.X_test_processed is not None else dataset.X_test
    
    # Initialize model based on model_name
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
        raise ValueError(f"Model '{model_name}' is not supported. Supported: {list(models_dict.keys())}")
    
    # Initialize and train model
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


@celery_app.task(bind=True, name='tasks.train_ktsp_model')
def train_ktsp_model_task(
    self, 
    dataset_path: str, 
    model_name: str,
    top_genes: int = 300,
    top_pairs: int = 10,
    task_id: str = None
) -> Dict:
    """
    Task 3: Execute KTSP model training
    
    Args:
        dataset_path: Path to processed Dataset object
        model_name: Name of the model
        top_genes: Number of top genes to select
        top_pairs: Number of top pairs to use
        task_id: Unique task identifier
        
    Returns:
        Dictionary with model path, results, and status
    """
    try:
        self.update_state(state='PROGRESS', meta={'step': 'training_model', 'progress': 60})
        self.update_state(state='PROGRESS', meta={'step': 'initializing_model', 'progress': 70})
        result = _train_model_helper(dataset_path, model_name, top_genes, top_pairs, task_id)
        self.update_state(state='PROGRESS', meta={'step': 'model_trained', 'progress': 80})
        self.update_state(state='PROGRESS', meta={'step': 'evaluation_complete', 'progress': 90})
        self.update_state(state='PROGRESS', meta={'step': 'model_saved', 'progress': 100})
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='tasks.run_full_pipeline')
def run_full_pipeline_task(
    self,
    file_path: str,
    models_names: List[str],
    use_pca: bool,
    task_id: str
) -> Dict:
    """
    Complete pipeline task that chains all steps together
    
    Args:
        file_path: Path to data file
        models_names: List of model names to train
        use_pca: Whether to use PCA preprocessing
        task_id: Unique task identifier
        
    Returns:
        Dictionary with all results
    """
    try:
        # Step 1: Load data
        self.update_state(state='PROGRESS', meta={'step': 'loading_data', 'progress': 10})
        load_result = _load_data_helper(file_path, task_id)
        dataset_path = load_result['dataset_path']
        
        # Step 2: Preprocess data
        self.update_state(state='PROGRESS', meta={'step': 'preprocessing', 'progress': 30})
        preprocess_result = _preprocess_data_helper(dataset_path, use_pca, task_id)
        
        # Step 3: Train models
        all_results = []
        total_models = len(models_names)
        
        for idx, model_name in enumerate(models_names):
            progress = 50 + int((idx / total_models) * 50)
            self.update_state(
                state='PROGRESS',
                meta={'step': f'training_{model_name}', 'progress': progress}
            )
            
            train_result = _train_model_helper(
                dataset_path=dataset_path,
                model_name=model_name,
                top_genes=300,
                top_pairs=10,
                task_id=task_id
            )
            
            all_results.append({
                'model': model_name,
                'accuracy': train_result['results'].get('accuracy'),
                'roc': train_result['results'].get('roc'),
                'confusion_matrix': train_result['results'].get('confusion_matrix'),
                'model_path': train_result['model_path']
            })
        
        self.update_state(state='PROGRESS', meta={'step': 'complete', 'progress': 100})
        
        # Save results to file for status checking
        results_dir = 'task_results'
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f'result_{task_id}.json')
        
        result_data = {
            'status': 'success',
            'task_id': task_id,
            'results': all_results,
            'dataset_path': dataset_path,
            'progress': 100,
            'step': 'complete'
        }
        
        # Convert numpy arrays to lists for JSON
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
        
        with open(result_file, 'w') as f:
            json.dump(to_json_safe(result_data), f, indent=2)
        
        return result_data
    except Exception as e:
        error_msg = str(e)
        self.update_state(state='FAILURE', meta={'error': error_msg})
        
        # Save error to result file
        results_dir = 'task_results'
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f'result_{task_id}.json')
        error_data = {
            'status': 'failed',
            'task_id': task_id,
            'error': error_msg,
            'progress': 0,
            'step': 'error'
        }
        try:
            with open(result_file, 'w') as f:
                json.dump(error_data, f, indent=2)
        except:
            pass
        
        raise

