"""Integration-test infrastructure.

Every test in this directory is gated on ``RUN_INTEGRATION_TESTS=1`` so
that ``pytest`` from the repo root never accidentally fires real AWS
calls. The unit test suite continues to live under ``tests/`` (one
directory up) and is unaffected.
"""

from __future__ import annotations

import os

import pytest

from jina_sagemaker import Client

# The parent tests/conftest.py installs placeholder AWS credentials so the
# Stubber-based unit suite can import boto3/sagemaker without complaining.
# Those placeholders are picked up by the default boto3 credential chain
# AHEAD of AWS_PROFILE, so a developer who runs the integration suite with
# only ``AWS_PROFILE=...`` set would silently authenticate as the fake
# ``testing`` IAM principal. Clear the known placeholder values here, only
# under integration mode and only when they still match the placeholders
# (so a developer who explicitly set real AWS_ACCESS_KEY_ID=... is left
# untouched).
if os.environ.get("RUN_INTEGRATION_TESTS") == "1":
    for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if os.environ.get(_key) == "testing":
            del os.environ[_key]


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Skip every integration test unless ``RUN_INTEGRATION_TESTS=1``."""
    if os.environ.get("RUN_INTEGRATION_TESTS") == "1":
        return
    skip_marker = pytest.mark.skip(reason="Integration tests are opt-in. Set RUN_INTEGRATION_TESTS=1 to enable.")
    for item in items:
        item.add_marker(skip_marker)


def _resolve_endpoint(prefix: str) -> tuple[str, str] | None:
    """Return ``(endpoint_name, arn)`` from the env, or ``None`` if either
    is unset — caller should use this to skip its tests."""
    endpoint = os.environ.get(f"JINA_TEST_{prefix}_ENDPOINT")
    arn = os.environ.get(f"JINA_TEST_{prefix}_ARN")
    if not endpoint or not arn:
        return None
    return endpoint, arn


def _build_client(prefix: str, *, model: str | None = None) -> Client:
    pair = _resolve_endpoint(prefix)
    if pair is None:
        pytest.skip(f"JINA_TEST_{prefix}_ENDPOINT and JINA_TEST_{prefix}_ARN must both be set to run this test.")
    endpoint_name, arn = pair
    client = Client(region_name=os.environ.get("AWS_REGION", "us-east-1"))
    client.connect_to_endpoint(endpoint_name, arn, model=model)
    return client


@pytest.fixture
def v3_client() -> Client:
    return _build_client("V3")


@pytest.fixture
def v4_client() -> Client:
    return _build_client("V4")


@pytest.fixture
def clip_v2_client() -> Client:
    return _build_client("CLIP_V2")


@pytest.fixture
def reranker_client() -> Client:
    return _build_client("RERANKER")


@pytest.fixture
def reader_lm_v2_client() -> Client:
    # Real Marketplace ARNs for ReaderLM-v2 use a lowercased product-id
    # slug (e.g. reader-lm-v2-...), which does NOT match the case-
    # sensitive "ReaderLM-v2" substring in the detection table. Pin the
    # model explicitly so the fixture-built client always sends the
    # right request shape regardless of how the ARN happens to be cased.
    return _build_client("READER_LM_V2", model="ReaderLM-v2")
