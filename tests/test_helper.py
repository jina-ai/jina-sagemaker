import csv
import os
import tempfile
import uuid
from unittest import mock

import pytest

from jina_sagemaker.helper import download_s3_folder, prefix_csv_with_ids

# Real 32-character hex UUIDs (no dashes) — uuid.UUID() accepts both forms.
REAL_UUID_1 = "f47ac10b58cc4372a5670e02b2c3d479"
REAL_UUID_2 = "550e8400e29b41d4a716446655440000"

SAMPLE_CSV_NO_IDS = """How is the weather today?
When are you open?"""

SAMPLE_CSV_WITH_IDS = f"""{REAL_UUID_1},How is the weather today?
{REAL_UUID_2},When are you open?"""


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_prefix_csv_adds_uuid_when_missing():
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", encoding="utf-8"
    ) as f:
        f.write(SAMPLE_CSV_NO_IDS)
        input_path = f.name

    try:
        output_path = prefix_csv_with_ids(input_path)
        rows = _read_rows(output_path)

        assert len(rows) == 2
        for row in rows:
            assert len(row) == 2
            uuid.UUID(row[0])  # raises ValueError if not a valid UUID
    finally:
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def test_prefix_csv_raises_on_empty_input():
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", encoding="utf-8"
    ) as f:
        input_path = f.name
        # leave file empty

    try:
        with pytest.raises(ValueError, match="empty"):
            prefix_csv_with_ids(input_path)
    finally:
        os.remove(input_path)


def test_prefix_csv_handles_blank_leading_line():
    """A CSV whose first line is blank must NOT crash on first_row[0]."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", encoding="utf-8"
    ) as f:
        f.write("\nHow is the weather today?\nWhen are you open?\n")
        input_path = f.name

    try:
        output_path = prefix_csv_with_ids(input_path)
        rows = _read_rows(output_path)

        # The blank leading line is dropped by csv.reader; every remaining
        # row gets a fresh UUID prefix.
        assert len(rows) >= 2
        for row in rows:
            if row:  # ignore any blank rows the reader emits
                uuid.UUID(row[0])
    finally:
        os.remove(input_path)
        if "output_path" in dir() and os.path.exists(output_path):
            os.remove(output_path)


def test_prefix_csv_preserves_existing_uuids():
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", encoding="utf-8"
    ) as f:
        f.write(SAMPLE_CSV_WITH_IDS)
        input_path = f.name

    try:
        output_path = prefix_csv_with_ids(input_path)
        rows = _read_rows(output_path)

        assert len(rows) == 2
        assert rows[0][0] == REAL_UUID_1
        assert rows[1][0] == REAL_UUID_2
    finally:
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


# ----------------------------------------------------------------------------
# download_s3_folder
# ----------------------------------------------------------------------------


def _mock_obj(key: str):
    o = mock.MagicMock()
    o.key = key
    return o


def _set_up_bucket(mock_resource, keys):
    """Configure ``boto3.resource("s3").Bucket(...)`` to return a Bucket
    whose ``objects.filter(...)`` yields one mock object per key.

    Returns the mock_bucket so the test can assert on ``download_file``.
    """
    mock_bucket = mock.MagicMock()
    mock_bucket.objects.filter.return_value = [_mock_obj(k) for k in keys]
    mock_resource.return_value.Bucket.return_value = mock_bucket
    return mock_bucket


def test_download_s3_folder_downloads_every_non_folder_key(tmp_path):
    with mock.patch("jina_sagemaker.helper.boto3.resource") as mock_resource:
        bucket = _set_up_bucket(
            mock_resource,
            keys=[
                "prefix/a.json",
                "prefix/b.json",
                "prefix/subdir/c.json",
            ],
        )

        download_s3_folder("s3://bucket/prefix", local_dir=str(tmp_path))

    assert bucket.download_file.call_count == 3
    calls = {call.args for call in bucket.download_file.call_args_list}
    assert ("prefix/a.json", os.path.join(str(tmp_path), "a.json")) in calls
    assert ("prefix/b.json", os.path.join(str(tmp_path), "b.json")) in calls
    assert (
        "prefix/subdir/c.json",
        os.path.join(str(tmp_path), "subdir", "c.json"),
    ) in calls


def test_download_s3_folder_skips_folder_marker_keys(tmp_path):
    with mock.patch("jina_sagemaker.helper.boto3.resource") as mock_resource:
        bucket = _set_up_bucket(
            mock_resource,
            keys=[
                "prefix/",  # folder marker — must be skipped
                "prefix/file.json",
                "prefix/subdir/",  # nested folder marker — must be skipped
                "prefix/subdir/nested.json",
            ],
        )

        download_s3_folder("s3://bucket/prefix", local_dir=str(tmp_path))

    # Only the two real files should hit download_file.
    assert bucket.download_file.call_count == 2
    keys_called = {call.args[0] for call in bucket.download_file.call_args_list}
    assert keys_called == {"prefix/file.json", "prefix/subdir/nested.json"}


def test_download_s3_folder_runs_in_parallel(tmp_path):
    """Every download_file call must hit the bucket within one ThreadPool
    cycle. We assert that by gating each mock call on a barrier — if the
    implementation were still serial, the barrier would never trip and
    the test would time out."""
    import threading

    n = 8
    barrier = threading.Barrier(n, timeout=5.0)

    def _gated_download(key, target):
        barrier.wait()

    with mock.patch("jina_sagemaker.helper.boto3.resource") as mock_resource:
        bucket = _set_up_bucket(
            mock_resource,
            keys=[f"prefix/f{i}.json" for i in range(n)],
        )
        bucket.download_file.side_effect = _gated_download

        download_s3_folder("s3://bucket/prefix", local_dir=str(tmp_path), max_workers=n)

    assert bucket.download_file.call_count == n


def test_download_s3_folder_propagates_worker_exceptions(tmp_path):
    def _boom(key, target):
        if key.endswith("bad.json"):
            raise RuntimeError("simulated S3 failure")

    with mock.patch("jina_sagemaker.helper.boto3.resource") as mock_resource:
        bucket = _set_up_bucket(
            mock_resource,
            keys=["prefix/ok.json", "prefix/bad.json"],
        )
        bucket.download_file.side_effect = _boom

        with pytest.raises(RuntimeError, match="simulated S3 failure"):
            download_s3_folder("s3://bucket/prefix", local_dir=str(tmp_path))


def test_download_s3_folder_handles_empty_listing(tmp_path):
    """An empty prefix should be a no-op, not an error."""
    with mock.patch("jina_sagemaker.helper.boto3.resource") as mock_resource:
        bucket = _set_up_bucket(mock_resource, keys=[])
        download_s3_folder("s3://bucket/prefix", local_dir=str(tmp_path))
    assert bucket.download_file.call_count == 0
