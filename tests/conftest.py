"""Shared test fixtures.

Tests stub boto3 clients via ``botocore.stub.Stubber``; no real AWS calls are
made. We set safe placeholder credentials and a region here so that boto3 and
the sagemaker SDK can be imported and constructed without complaining about
missing config.
"""

import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

import pytest  # noqa: E402

from jina_sagemaker import Client  # noqa: E402


@pytest.fixture
def client():
    return Client(region_name="us-east-1")
