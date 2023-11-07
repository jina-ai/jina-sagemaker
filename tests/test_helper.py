import os
import tempfile

import pytest

from jina_sagemaker.helper import prefix_csv_with_ids

SAMPLE_CSV_CONTENT = """How is the weather today?
When are you open?"""


SAMPLE_CSV_CONTENT_WITH_IDS = """a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6,How is the weather today?
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6,When are you open?"""


@pytest.mark.parametrize("content", [SAMPLE_CSV_CONTENT, SAMPLE_CSV_CONTENT_WITH_IDS])
def test_prefix_csv_with_ids(content):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv") as f:
        input_file_name = f.name
        f.write(content.encode("utf-8"))

    try:
        output_file_name = prefix_csv_with_ids(input_file_name)

        # Check the output
        with open(output_file_name, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                # Assert each line has a UUID, comma, and then content
                assert len(line.split(",", 1)) == 2
                assert len(line.split(",", 1)[0]) == 32

    finally:
        os.remove(input_file_name)
        os.remove(output_file_name)
