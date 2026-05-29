import csv
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

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

    with (
        open(input_path, mode="r", encoding="utf-8") as infile,
        open(output_path, mode="w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile, quoting=csv.QUOTE_ALL)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Input CSV {input_path!r} is empty")

        # csv.reader returns an empty list ([]) for a blank leading line —
        # treat that as "no UUID present" so we still prefix every data row
        # rather than crashing on first_row[0].
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


def download_s3_folder(
    path: str,
    local_dir: Optional[str] = None,
    max_workers: int = 16,
) -> None:
    """Download every object under an S3 prefix in parallel.

    Args:
        path: An ``s3://bucket/prefix`` URI. All objects under the prefix
            are downloaded; objects whose key ends in ``/`` (S3 folder
            markers) are skipped.
        local_dir: Local destination root. When set, each downloaded object
            is written to ``<local_dir>/<key-relative-to-prefix>``; when
            ``None`` the object key itself is used verbatim as the path
            (preserving the original behaviour of this helper).
        max_workers: Upper bound on concurrent ``S3.Client.download_file``
            calls. Capped to the actual object count so we do not spawn
            extra threads for tiny folders.

    Re-raises the first exception any worker thread encounters so partial
    failures surface as a single clean error to the caller, rather than a
    half-downloaded folder with no signal.

    Threading note: the low-level boto3 ``s3`` client is documented as
    thread-safe and is used for the worker downloads. boto3 *resources*
    (``boto3.resource("s3")``) are NOT thread-safe and are confined to
    the serial listing pass.
    """
    bucket_name, s3_folder = path.replace("s3://", "").split("/", 1)

    # Pass 1: enumerate the listing serially via the resource API (handy
    # paginated iterator) and create target directories. Directory
    # creation runs serially because parallel makedirs() against a shared
    # parent races; the listing + mkdir cost is negligible compared to
    # the downloads themselves.
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    tasks = []
    for obj in bucket.objects.filter(Prefix=s3_folder):
        if obj.key.endswith("/"):
            continue
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tasks.append((obj.key, target))

    if not tasks:
        return

    # Pass 2: parallel downloads through the low-level client. boto3
    # clients are thread-safe (per the boto3 docs); resources are not,
    # which is why we don't reuse ``bucket.download_file`` here.
    s3_client = boto3.client("s3")
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [
            pool.submit(s3_client.download_file, bucket_name, key, target)
            for key, target in tasks
        ]
        for fut in as_completed(futures):
            fut.result()
