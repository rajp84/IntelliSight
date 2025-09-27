from __future__ import annotations

import os
from typing import Optional

from minio import Minio
from minio.error import S3Error
from io import BytesIO

# Default connection settings (update these as needed)
MINIO_ENDPOINT_DEFAULT = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY_DEFAULT = "minioadmin"
MINIO_SECRET_KEY_DEFAULT = "minioadmin123"
MINIO_SECURE_DEFAULT = False


_CLIENT: Optional[Minio] = None
_CLIENT_SECURE: Optional[bool] = None
_CLIENT_ENDPOINT: Optional[str] = None


def get_client() -> Minio:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    raw_endpoint = os.getenv("MINIO_ENDPOINT", MINIO_ENDPOINT_DEFAULT)
    access_key = os.getenv("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY_DEFAULT)
    secret_key = os.getenv("MINIO_SECRET_KEY", MINIO_SECRET_KEY_DEFAULT)
    secure_env = os.getenv("MINIO_SECURE")
    # Normalize endpoint: allow http(s)://host:port or host:port
    endpoint = raw_endpoint.strip().rstrip('/')
    secure = bool(MINIO_SECURE_DEFAULT)
    if endpoint.startswith("http://"):
        endpoint = endpoint[len("http://"):]
        secure = False
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://"):]
        secure = True
    # Env override takes precedence
    if secure_env is not None:
        secure = secure_env.lower() == "true"
    global _CLIENT_SECURE
    _CLIENT = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    _CLIENT_SECURE = secure
    _CLIENT_ENDPOINT = endpoint
    return _CLIENT


def ensure_bucket(bucket: str) -> None:
    client = get_client()
    found = client.bucket_exists(bucket)
    if not found:
        client.make_bucket(bucket)


def put_image_bytes(bucket: str, object_name: str, data: bytes, content_type: str = "image/jpeg") -> str:
    client = get_client()
    ensure_bucket(bucket)
    # MinIO expects a file-like object with a read() method
    bio = BytesIO(data)
    bio.seek(0)
    client.put_object(bucket, object_name, data=bio, length=len(data), content_type=content_type)
    # Return stored object path; URL resolution is handled by API layer
    return f"{bucket}/{object_name}"


def training_bucket_name(training_id: str) -> str:
    """Construct a per-training bucket name that includes the training document id.

    S3 bucket rules require lowercase and hyphens; avoid underscores.
    """
    prefix = os.getenv("MINIO_TRAINING_BUCKET_PREFIX", "training-")
    # Sanitize: lowercase, replace invalid chars with '-'
    base = f"{prefix}{training_id}".lower()
    sanitized = []
    for ch in base:
        if ch.isalnum() or ch == '-':
            sanitized.append(ch)
        else:
            sanitized.append('-')
    name = ''.join(sanitized).strip('-')
    # Enforce length limits 3-63
    if len(name) < 3:
        name = (name + "---")[:3]
    if len(name) > 63:
        name = name[:63]
    return name


def ensure_training_bucket(training_id: str) -> str:
    """Ensure a bucket for the given training id exists; return its name."""
    bucket = training_bucket_name(training_id)
    ensure_bucket(bucket)
    return bucket


def put_training_image_bytes(training_id: str, object_name: str, data: bytes, content_type: str = "image/jpeg") -> str:
    """Upload image bytes to the per-training bucket and return the object URL."""
    bucket = ensure_training_bucket(training_id)
    return put_image_bytes(bucket, object_name, data, content_type)


def _resolved_endpoint_url() -> str:
    # Build endpoint URL from stored/env configuration
    _ = get_client()
    endpoint = _CLIENT_ENDPOINT or "127.0.0.1:9000"
    secure = (_CLIENT_SECURE is True)
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}"


def public_object_url(bucket: str, object_name: str) -> str:
    base = _resolved_endpoint_url()
    return f"{base}/{bucket}/{object_name}"


def get_object_bytes(bucket: str, object_name: str) -> tuple[bytes, str]:
    client = get_client()
    resp = client.get_object(bucket, object_name)
    try:
        data = resp.read()
        content_type = resp.headers.get('Content-Type', 'application/octet-stream') if hasattr(resp, 'headers') else 'application/octet-stream'
        return data, content_type
    finally:
        resp.close()
        resp.release_conn()


def get_training_image_bytes(training_id: str, object_name: str) -> tuple[bytes, str]:
    bucket = training_bucket_name(training_id)
    return get_object_bytes(bucket, object_name)


def things_bucket_name() -> str:
    """Get the bucket name for the things collection."""
    return "things"


def ensure_things_bucket() -> None:
    """Ensure the things bucket exists."""
    client = get_client()
    bucket = things_bucket_name()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"[DEBUG] Created things bucket: {bucket}")
    except S3Error as e:
        print(f"[ERROR] Failed to create things bucket {bucket}: {e}")
        raise


def get_things_image_bytes(filename: str) -> bytes:
    """Get image bytes from the things bucket."""
    bucket = things_bucket_name()
    data, _ = get_object_bytes(bucket, filename)
    return data




