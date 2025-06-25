import csv
import os
import uuid

import boto3


def prefix_csv_with_ids(input_path: str) -> str:
    def is_uuid(val):
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False

    # output_path should be oldfilename_with_ids.csv
    output_path = input_path.replace(".csv", "_with_ids.csv")

    with open(input_path, mode="r", encoding="utf-8") as infile, open(
        output_path, mode="w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile, quoting=csv.QUOTE_ALL)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        first_row = next(reader, None)
        if first_row is None:
            return

        has_uuid = is_uuid(first_row[0]) if first_row else False

        infile.seek(0)
        for row in reader:
            if has_uuid:
                writer.writerow(row)
            else:
                writer.writerow([uuid.uuid4().hex] + row)

    print(f"Input file with ids created at {output_path}.")
    return output_path


def get_role():
    import sagemaker

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
