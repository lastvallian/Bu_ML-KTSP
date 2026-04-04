# KTSP Meta - Worker Queue Setup Guide

## Overview
This project uses Celery for asynchronous task processing. The workflow includes:
1. Load data and save as Dataset object
2. Commit calculate tasks to worker queue
3. Worker execute preprocess PCA to normalization
4. Worker execute KTSP model and training
5. Save model and execute data
6. Return task_id and API for React frontend

## Prerequisites

1. **Redis** - Required for Celery broker and backend
   - Install Redis: https://redis.io/download
   - Start Redis server: `redis-server`

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Setup Instructions

### 1. Start Redis Server
```bash
# Windows (if installed)
redis-server

# Linux/Mac
redis-server
```

### 2. Start Celery Worker
Open a new terminal and run:
```bash
celery -A celery_app worker --loglevel=info
```

Or use the worker script:
```bash
python worker.py
```

### 3. Start FastAPI Server
Open another terminal and run:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /run
Submit a new pipeline task
- **Input**: File upload, models_names (JSON string), use_pca (string)
- **Output**: Returns task_id and celery_task_id

Example:
```bash
curl -X POST "http://localhost:8000/run" \
  -F "file=@data.csv" \
  -F "models_names=[\"KTSP\"]" \
  -F "use_pca=false"
```

### GET /task/{task_id}/celery/{celery_task_id}
Check task status
- **Input**: task_id and celery_task_id from /run response
- **Output**: Current status, progress, and results if completed

Example:
```bash
curl "http://localhost:8000/task/{task_id}/celery/{celery_task_id}"
```

### GET /task/{task_id}
Check task status (alternative endpoint)
- **Input**: task_id
- **Output**: Current status and results

### GET /health
Health check endpoint

## Task Flow

1. **Client submits task** → POST /run
   - Returns: `task_id` and `celery_task_id`

2. **Worker processes task**:
   - Step 1: Load data → Save as Dataset object
   - Step 2: Preprocess (PCA + Normalization)
   - Step 3: Train KTSP model(s)
   - Step 4: Save model and results

3. **Client checks status** → GET /task/{task_id}/celery/{celery_task_id}
   - Returns: Progress, status, and results when complete

## Directory Structure

```
KTSP_Meta/
├── celery_app.py          # Celery configuration and tasks
├── main.py                # FastAPI application
├── worker.py              # Worker startup script
├── requirements.txt       # Python dependencies
├── datasets/              # Saved Dataset objects
├── saved_models/          # Trained models
├── task_results/          # Task result files
└── uploads/               # Uploaded data files
```

## React Frontend Integration

### Example API Call
```javascript
// Submit task
const formData = new FormData();
formData.append('file', file);
formData.append('models_names', JSON.stringify(['KTSP']));
formData.append('use_pca', 'false');

const response = await fetch('http://localhost:8000/run', {
  method: 'POST',
  body: formData
});

const { task_id, celery_task_id, status_url } = await response.json();

// Poll for status
const checkStatus = async () => {
  const statusResponse = await fetch(status_url);
  const status = await statusResponse.json();
  
  if (status.status === 'completed') {
    console.log('Results:', status.results);
  } else if (status.status === 'processing') {
    console.log(`Progress: ${status.progress}% - ${status.step}`);
    setTimeout(checkStatus, 2000); // Check again in 2 seconds
  }
};

checkStatus();
```

## Troubleshooting

1. **Worker not starting**: Check Redis is running
2. **Tasks stuck**: Check worker logs for errors
3. **Import errors**: Ensure all dependencies are installed
4. **Port conflicts**: Change ports in uvicorn command

## Notes

- Tasks are processed asynchronously
- Results are saved to `task_results/` directory
- Models are saved to `saved_models/` directory
- Dataset objects are saved to `datasets/` directory





