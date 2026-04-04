@echo off
REM Start Ray cluster for KTSP Meta
REM This script starts a local Ray cluster

echo Starting Ray cluster for KTSP Meta...

REM Check if Ray is installed
python -c "import ray" 2>nul
if errorlevel 1 (
    echo Ray is not installed. Installing...
    pip install ray[default]
)

REM Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

echo Ray cluster started!
echo Dashboard available at: http://localhost:8265
echo To stop the cluster, run: ray stop

pause





