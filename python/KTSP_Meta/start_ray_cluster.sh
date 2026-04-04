#!/bin/bash

# Start Ray cluster for KTSP Meta
# This script starts a local Ray cluster

echo "Starting Ray cluster for KTSP Meta..."

# Check if Ray is installed
if ! command -v ray &> /dev/null; then
    echo "Ray is not installed. Installing..."
    pip install ray[default]
fi

# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Ray cluster started!"
echo "Dashboard available at: http://localhost:8265"
echo "To stop the cluster, run: ray stop"





