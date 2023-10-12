import os
import tempfile

from jina_sagemaker.helper import prefix_csv_with_ids

SAMPLE_CSV_CONTENT = """How is the weather today?
When are you open?"""


def test_prefix_csv_with_ids():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        input_file_name = f.name
        f.write(SAMPLE_CSV_CONTENT.encode("utf-8"))

    with tempfile.NamedTemporaryFile(delete=False) as f:
        output_file_name = f.name

    try:
        prefix_csv_with_ids(input_file_name, output_file_name)

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
