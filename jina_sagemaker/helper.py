import os
import tempfile
import uuid

import boto3
import sagemaker


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


def get_role():
    try:
        return sagemaker.get_execution_role()
    except ValueError:
        print("Using default role: 'ServiceRoleSagemaker'.")
        return "ServiceRoleSagemaker"


def download_s3_folder(path: str, local_dir: str = None):
    """
    Download the contents of a folder directory
    Args:
        path: the s3 path
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket_name, s3_folder = path.replace("s3://", "").split("/", 1)
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)
