"""Celery application configuration for ExaSense background tasks."""

from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "exasense",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Tokyo",
    task_track_started=True,
    result_expires=3600,  # 1 hour
    # Queue routing: 3 queues for different worker types
    task_routes={
        "exasense.run_simulation": {"queue": "simulation"},
        "exasense.run_reconstruction": {"queue": "reconstruction"},
        "exasense.run_vlm_analysis": {"queue": "vlm"},
    },
    task_default_queue="simulation",
    # Worker prefetch: 1 task at a time for GPU workers (memory management)
    worker_prefetch_multiplier=1,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["src.tasks"])
