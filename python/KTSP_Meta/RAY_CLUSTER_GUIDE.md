# Ray Cluster Integration Guide for KTSP Meta

## Overview

Ray cluster support has been added to the KTSP Meta project, providing distributed computing capabilities for robust and scalable machine learning pipeline processing. Ray enables parallel processing across multiple nodes/CPUs, making it ideal for large-scale data processing and model training.

## Features

✅ **Distributed Processing** - Process tasks across multiple nodes/CPUs  
✅ **Parallel Model Training** - Train multiple models simultaneously  
✅ **Resource Management** - Automatic resource allocation and monitoring  
✅ **Fault Tolerance** - Automatic retries and error recovery  
✅ **Safety Features** - Timeouts, resource limits, and error handling  
✅ **Integration** - Works alongside existing Celery system  

## Architecture

```
┌─────────────┐
│ FastAPI API │
└──────┬──────┘
       │
       ├───► Celery Worker (Redis)
       │
       └───► Ray Cluster
              ├─── Head Node
              ├─── Worker Node 1
              ├─── Worker Node 2
              └─── Worker Node N
```

## Installation

### 1. Install Ray

```bash
pip install ray[default]
```

Or update requirements:
```bash
pip install -r requirements.txt
```

### 2. Start Ray Cluster

#### Local Cluster (Single Machine)

**Windows:**
```bash
start_ray_cluster.bat
```

**Linux/Mac:**
```bash
chmod +x start_ray_cluster.sh
./start_ray_cluster.sh
```

**Manual:**
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

#### Remote Cluster (Multi-Node)

**On Head Node:**
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

**On Worker Nodes:**
```bash
ray start --address=<head-node-ip>:6379
```

### 3. Verify Ray Cluster

Check Ray dashboard: http://localhost:8265

Or check via API:
```bash
curl http://localhost:8000/ray/resources
```

## Usage

### API Endpoints

#### POST /run (with Ray)

Submit a task using Ray executor:

```bash
curl -X POST "http://localhost:8000/run" \
  -F "file=@data.csv" \
  -F "models_names=[\"KTSP\"]" \
  -F "use_pca=false" \
  -F "executor=ray"
```

**Parameters:**
- `executor`: "ray" or "celery" (default: "celery")
- `ray_address`: Optional Ray cluster address (e.g., "ray://head-node:10001")

**Response:**
```json
{
  "status": "submitted",
  "task_id": "uuid-here",
  "executor": "ray",
  "status_url": "/task/{task_id}",
  "check_status": "Use GET /task/{task_id} to check status"
}
```

#### GET /ray/resources

Get Ray cluster resource information:

```bash
curl http://localhost:8000/ray/resources
```

**Response:**
```json
{
  "status": "success",
  "resources": {
    "available": {
      "CPU": 8.0,
      "memory": 16000000000.0
    },
    "cluster": {
      "CPU": 8.0,
      "memory": 16000000000.0
    }
  }
}
```

#### POST /ray/initialize

Initialize Ray cluster:

```bash
curl -X POST "http://localhost:8000/ray/initialize" \
  -F "address=" \
  -F "num_cpus=8"
```

#### GET /health

Check health status including Ray:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "KTSP Meta API",
  "executors": {
    "celery": "available",
    "ray": "available"
  },
  "ray_cluster": {
    "status": "connected",
    "resources": {...}
  }
}
```

## Configuration

### Ray Configuration File (`ray_config.yaml`)

```yaml
cluster:
  address: null  # Ray cluster address
  num_cpus: null  # Auto-detect or specify
  num_gpus: null  # Auto-detect or specify
  
tasks:
  timeout: 3600  # Task timeout (seconds)
  max_retries: 2  # Retry attempts
  
safety:
  task_timeout: 3600  # Safety timeout
  monitor_resources: true  # Enable monitoring
```

### Environment Variables

```bash
export RAY_ADDRESS="ray://head-node:10001"
export RAY_NUM_CPUS=8
export RAY_NUM_GPUS=0
```

## Safety Features

### 1. **Resource Limits**
- Automatic CPU allocation per task
- Memory monitoring
- Object store memory limits

### 2. **Error Handling**
- Automatic retries (max 2 retries per task)
- Timeout protection (default 1 hour)
- Graceful error recovery

### 3. **Task Isolation**
- Each task runs in isolated Ray actor
- Resource cleanup after task completion
- No interference between tasks

### 4. **Monitoring**
- Real-time resource monitoring
- Task progress tracking
- Error logging

## Performance Benefits

### Parallel Processing

**Celery (Sequential):**
```
Task 1 → Task 2 → Task 3 → Task 4
Time: 4 × task_time
```

**Ray (Parallel):**
```
Task 1 ┐
Task 2 ├─→ All complete simultaneously
Task 3 │
Task 4 ┘
Time: 1 × task_time (with 4 CPUs)
```

### Example: Training Multiple Models

With Ray, multiple models train in parallel:

```python
# Sequential (Celery): ~4 minutes for 4 models
# Parallel (Ray): ~1 minute for 4 models (with 4 CPUs)
```

## Comparison: Celery vs Ray

| Feature | Celery | Ray |
|---------|--------|-----|
| **Distributed** | ✅ (with Redis) | ✅ (Native) |
| **Parallel Tasks** | Limited | ✅ Excellent |
| **Resource Management** | Manual | ✅ Automatic |
| **Fault Tolerance** | ✅ | ✅ |
| **Setup Complexity** | Medium | Low |
| **Best For** | Task queues | Distributed ML |

## Troubleshooting

### Ray Not Starting

```bash
# Check if Ray is installed
python -c "import ray; print(ray.__version__)"

# Check if port is available
netstat -an | grep 6379

# Start with verbose logging
ray start --head --verbose
```

### Connection Issues

```bash
# Check Ray cluster status
ray status

# Check dashboard
curl http://localhost:8265

# Restart Ray
ray stop
ray start --head
```

### Resource Errors

```bash
# Check available resources
ray status

# Reduce CPU allocation
# Edit ray_config.yaml: num_cpus: 4
```

### Task Failures

1. Check Ray logs: `ray logs`
2. Check task status: `GET /task/{task_id}`
3. Verify data file is accessible
4. Check resource availability

## Best Practices

### 1. **Resource Allocation**
- Allocate appropriate CPUs per task
- Monitor resource usage via dashboard
- Don't overallocate resources

### 2. **Error Handling**
- Always check task status
- Implement retry logic in client
- Monitor Ray cluster health

### 3. **Performance**
- Use Ray for parallel model training
- Use Celery for sequential tasks
- Balance between executors

### 4. **Safety**
- Set appropriate timeouts
- Monitor resource usage
- Clean up failed tasks

## React Frontend Integration

```javascript
// Submit task with Ray executor
async function submitWithRay(file, models, usePCA) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('models_names', JSON.stringify(models));
  formData.append('use_pca', usePCA ? 'true' : 'false');
  formData.append('executor', 'ray');  // Use Ray
  
  const response = await fetch('http://localhost:8000/run', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Check Ray cluster resources
async function checkRayResources() {
  const response = await fetch('http://localhost:8000/ray/resources');
  return await response.json();
}
```

## Advanced Configuration

### Custom Ray Address

```python
# In ray_cluster.py
manager = RayClusterManager(
    address="ray://192.168.1.100:10001",
    num_cpus=16,
    num_gpus=2
)
```

### Resource Limits Per Task

```python
@ray.remote(max_retries=2, num_cpus=8, num_gpus=1)
def ray_train_model(...):
    # Task with 8 CPUs and 1 GPU
    pass
```

## Monitoring

### Ray Dashboard

Access at: http://localhost:8265

Features:
- Real-time cluster status
- Resource usage graphs
- Task execution timeline
- Error logs

### API Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Resource check
curl http://localhost:8000/ray/resources
```

## Migration from Celery to Ray

1. **Start Ray cluster**: `./start_ray_cluster.sh`
2. **Update API calls**: Add `executor=ray` parameter
3. **Monitor performance**: Compare execution times
4. **Gradual migration**: Use both executors initially

## Support

For issues or questions:
1. Check Ray documentation: https://docs.ray.io
2. Check Ray dashboard for errors
3. Review logs: `ray logs`
4. Verify cluster status: `ray status`

---

**Status**: ✅ Ray Cluster Integration Complete and Production Ready





