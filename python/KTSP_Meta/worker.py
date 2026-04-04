#!/usr/bin/env python
# coding: utf-8

"""
Celery worker startup script
Run this file to start the Celery worker: python worker.py
"""

from celery_app import celery_app

if __name__ == '__main__':
    celery_app.start()





