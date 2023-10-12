import os
import tempfile
import uuid


def prefix_csv_with_ids(input_path, output_path):
    """Prefix each line in the CSV with a random ID."""

    with open(input_path, "r") as csv_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as temp_file:
        for line in csv_file:
            text = line.strip()
            uid = str(uuid.uuid4()).replace("-", "")[:32]
            temp_file.write(f"{uid},{text}\n")

    # Replace the original file with the temporary one
    os.replace(temp_file.name, output_path)
