# KTSP Meta API Documentation

## Overview
This API provides asynchronous processing of KTSP (K-Top Scoring Pairs) machine learning pipeline tasks using Celery worker queues.

## Workflow
1. **Load data** → Save as Dataset object
2. **Commit tasks** → Submit to worker queue
3. **Worker preprocessing** → Execute PCA and normalization
4. **Worker training** → Execute KTSP model training
5. **Save model** → Persist trained model and results
6. **Return task_id** → Provide API endpoints for React frontend

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. POST /run
Submit a new pipeline task to the worker queue.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Data file (CSV format)
  - `models_names`: JSON string array of model names (e.g., `["KTSP"]`)
  - `use_pca`: String ("true" or "false")

**Response:**
```json
{
  "status": "submitted",
  "task_id": "uuid-string",
  "celery_task_id": "celery-uuid",
  "message": "Task submitted to worker queue.",
  "status_url": "/task/{task_id}/celery/{celery_task_id}",
  "check_status": "Use GET /task/{task_id}/celery/{celery_task_id} to check status"
}
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/run" \
  -F "file=@data.csv" \
  -F "models_names=[\"KTSP\"]" \
  -F "use_pca=false"
```

**Example (JavaScript/Fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('models_names', JSON.stringify(['KTSP']));
formData.append('use_pca', 'false');

const response = await fetch('http://localhost:8000/run', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Task ID:', data.task_id);
console.log('Status URL:', data.status_url);
```

---

### 2. GET /task/{task_id}/celery/{celery_task_id}
Check the status of a submitted task.

**Request:**
- Method: `GET`
- Path Parameters:
  - `task_id`: Task ID from `/run` response
  - `celery_task_id`: Celery task ID from `/run` response

**Response (Pending):**
```json
{
  "task_id": "uuid-string",
  "celery_task_id": "celery-uuid",
  "status": "pending",
  "message": "Task is waiting to be processed"
}
```

**Response (Processing):**
```json
{
  "task_id": "uuid-string",
  "celery_task_id": "celery-uuid",
  "status": "processing",
  "progress": 50,
  "step": "training_KTSP",
  "message": "Task is in progress: training_KTSP"
}
```

**Response (Completed):**
```json
{
  "task_id": "uuid-string",
  "celery_task_id": "celery-uuid",
  "status": "completed",
  "progress": 100,
  "results": [
    {
      "model": "KTSP",
      "accuracy": 0.95,
      "roc": {
        "fpr": [0.0, 0.1, ...],
        "tpr": [0.0, 0.9, ...],
        "auc": 0.98
      },
      "confusion_matrix": [[10, 2], [1, 9]],
      "model_path": "saved_models/model_uuid_KTSP.pkl"
    }
  ],
  "message": "Task completed successfully"
}
```

**Response (Failed):**
```json
{
  "task_id": "uuid-string",
  "celery_task_id": "celery-uuid",
  "status": "failed",
  "error": "Error message here",
  "message": "Task failed"
}
```

**Example:**
```bash
curl "http://localhost:8000/task/{task_id}/celery/{celery_task_id}"
```

**Example (JavaScript - Polling):**
```javascript
async function checkTaskStatus(taskId, celeryTaskId) {
  const response = await fetch(
    `http://localhost:8000/task/${taskId}/celery/${celeryTaskId}`
  );
  const status = await response.json();
  
  if (status.status === 'completed') {
    console.log('Results:', status.results);
    return status.results;
  } else if (status.status === 'processing') {
    console.log(`Progress: ${status.progress}% - ${status.step}`);
    // Poll again after 2 seconds
    setTimeout(() => checkTaskStatus(taskId, celeryTaskId), 2000);
  } else if (status.status === 'failed') {
    console.error('Task failed:', status.error);
  }
}
```

---

### 3. GET /task/{task_id}
Alternative endpoint to check task status (reads from result file).

**Request:**
- Method: `GET`
- Path Parameter: `task_id`

**Response:** Same format as `/task/{task_id}/celery/{celery_task_id}`

---

### 4. POST /upload
Upload a data file (optional - can also upload directly in `/run`).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Data file

**Response:**
```json
{
  "status": "success",
  "filename": "data.csv",
  "file_path": "uploads/temp_uuid.csv",
  "message": "File uploaded successfully"
}
```

---

### 5. GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "KTSP Meta API"
}
```

---

## Supported Models

The following models are supported:
- `"KTSP"` - K-Top Scoring Pairs classifier
- `"Linear SVM"` - Linear Support Vector Machine
- `"RBF SVM"` - RBF kernel Support Vector Machine
- `"Decision Tree"` - Decision Tree classifier
- `"Naive Bayes"` - Gaussian Naive Bayes
- `"kNN"` - k-Nearest Neighbors
- `"Custom SVMTrainer"` - Custom Shrinkage Centroid Classifier

**Example models_names:**
```json
["KTSP"]
```
or
```json
["KTSP", "Linear SVM", "Decision Tree"]
```

---

## Task Status Values

- `pending` - Task is waiting in queue
- `processing` - Task is being executed
- `completed` - Task finished successfully
- `failed` - Task encountered an error

---

## Progress Steps

Tasks progress through these steps:
1. `loading_data` - Loading data file
2. `data_loaded` - Data loaded successfully
3. `preprocessing` - Starting preprocessing
4. `data_split` - Splitting train/test data
5. `preprocessing_complete` - Preprocessing finished
6. `training_{model_name}` - Training specific model
7. `initializing_model` - Initializing model
8. `model_trained` - Model training complete
9. `evaluation_complete` - Model evaluation finished
10. `model_saved` - Model saved to disk
11. `complete` - All tasks finished

---

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200` - Success
- `500` - Server error (check error message in response)

Errors are included in the response body:
```json
{
  "detail": "Error message here"
}
```

---

## React Frontend Integration Example

```javascript
// Submit task
async function submitPipelineTask(file, models, usePCA) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('models_names', JSON.stringify(models));
  formData.append('use_pca', usePCA ? 'true' : 'false');
  
  const response = await fetch('http://localhost:8000/run', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Failed to submit task');
  }
  
  return await response.json();
}

// Poll for results
async function pollTaskStatus(taskId, celeryTaskId, onProgress, onComplete, onError) {
  const checkStatus = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/task/${taskId}/celery/${celeryTaskId}`
      );
      const status = await response.json();
      
      if (status.status === 'completed') {
        onComplete(status.results);
      } else if (status.status === 'processing') {
        onProgress(status.progress, status.step);
        setTimeout(checkStatus, 2000); // Check again in 2 seconds
      } else if (status.status === 'failed') {
        onError(status.error);
      } else {
        setTimeout(checkStatus, 1000); // Check again in 1 second
      }
    } catch (error) {
      onError(error.message);
    }
  };
  
  checkStatus();
}

// Usage
const { task_id, celery_task_id } = await submitPipelineTask(
  file, 
  ['KTSP'], 
  false
);

pollTaskStatus(
  task_id,
  celery_task_id,
  (progress, step) => {
    console.log(`Progress: ${progress}% - ${step}`);
    // Update UI progress bar
  },
  (results) => {
    console.log('Results:', results);
    // Display results
  },
  (error) => {
    console.error('Error:', error);
    // Show error message
  }
);
```

---

## Notes

- Tasks are processed asynchronously
- Results are saved to `task_results/` directory
- Models are saved to `saved_models/` directory
- Dataset objects are saved to `datasets/` directory
- Uploaded files are saved to `uploads/` directory
- Make sure Redis and Celery worker are running before submitting tasks





