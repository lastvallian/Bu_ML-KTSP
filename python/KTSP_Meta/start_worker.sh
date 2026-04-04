#!/bin/bash
echo "Starting Celery Worker for KTSP Meta..."
celery -A celery_app worker --loglevel=info





