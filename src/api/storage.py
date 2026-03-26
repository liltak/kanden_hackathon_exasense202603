"""MinIO/S3 object storage client wrapper."""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

BUCKET_MESHES = "exasense-meshes"
BUCKET_REPORTS = "exasense-reports"
BUCKET_MODELS = "exasense-models"

_client = None


def _get_client():
    """Lazy-initialize MinIO client."""
    global _client
    if _client is None:
        from minio import Minio

        _client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _client


def ensure_buckets() -> None:
    """Create buckets if they don't exist."""
    client = _get_client()
    for bucket in [BUCKET_MESHES, BUCKET_REPORTS, BUCKET_MODELS]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("Created bucket: %s", bucket)


def upload_bytes(bucket: str, object_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to MinIO. Returns the object key."""
    client = _get_client()
    client.put_object(
        bucket,
        object_key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_key


def download_bytes(bucket: str, object_key: str) -> bytes:
    """Download object from MinIO as bytes."""
    client = _get_client()
    response = client.get_object(bucket, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_object(bucket: str, object_key: str) -> None:
    """Delete object from MinIO."""
    client = _get_client()
    client.remove_object(bucket, object_key)


def get_presigned_url(bucket: str, object_key: str, expires_hours: int = 1) -> str:
    """Get a presigned download URL."""
    from datetime import timedelta

    client = _get_client()
    return client.presigned_get_object(
        bucket, object_key, expires=timedelta(hours=expires_hours)
    )
