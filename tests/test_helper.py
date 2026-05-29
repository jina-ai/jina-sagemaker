import csv
import os
import tempfile
import uuid

import pytest

from jina_sagemaker.helper import prefix_csv_with_ids

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
