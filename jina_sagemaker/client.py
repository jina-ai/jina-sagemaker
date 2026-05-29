"""Jina SageMaker client.

Provides the high-level :class:`Client` for interacting with Jina model
endpoints deployed on AWS SageMaker — real-time inference, batch transform,
asynchronous inference, and endpoint lifecycle management.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError, ParamValidationError

from .helper import download_s3_folder, get_role, prefix_csv_with_ids

log = logging.getLogger(__name__)


class InputType(Enum):
    DOCUMENT = "document"
    QUERY = "query"


class Task(Enum):
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    TEXT_MATCHING = "text-matching"
    CLASSIFICATION = "classification"
    SEPARATION = "separation"


@dataclass(frozen=True)
class ModelSpec:
    """Captures the request-payload conventions of a Jina SageMaker model.

    The ``family`` field is the single source of truth for payload-shape
    decisions in :meth:`Client.embed`, :meth:`Client.rerank`, and
    :meth:`Client.read`. Callers should never re-parse the ARN to make these
    decisions; do them via ``self._model_spec`` instead.
    """

    name: str
    family: str
    reader_model: Optional[str] = None


# Ordered detection table. The first slug that appears as a substring of the
# ARN wins, so longer / more-specific slugs come first to prevent collisions
# (e.g. ``jina-reranker-m0`` must come before the generic ``jina-reranker``
# entry, and ``jina-embeddings-v4`` before ``jina-embeddings``).
_DETECTION_TABLE: List[Tuple[str, ModelSpec]] = [
    ("jina-reranker-m0", ModelSpec("jina-reranker-m0", "reranker-m0")),
    ("jina-embeddings-v4", ModelSpec("jina-embeddings-v4", "embeddings-v4")),
    ("jina-embeddings-v3", ModelSpec("jina-embeddings-v3", "embeddings-v3")),
    ("jina-clip-v2", ModelSpec("jina-clip-v2", "clip-v2")),
    (
        "ReaderLM-v2",
        ModelSpec("ReaderLM-v2", "reader-lm", reader_model="ReaderLM-v2"),
    ),
    (
        "1500m",
        ModelSpec("reader-lm-1.5b", "reader-lm", reader_model="reader-lm-1.5b"),
    ),
    (
        "reader-lm-500m",
        ModelSpec("reader-lm-0.5b", "reader-lm", reader_model="reader-lm-0.5b"),
    ),
    ("jina-reranker", ModelSpec("jina-reranker", "reranker")),
    ("jina-colbert", ModelSpec("jina-colbert", "colbert")),
    ("jina-embeddings", ModelSpec("jina-embeddings", "embeddings-default")),
]


# Accepts the canonical ``ModelSpec.name`` as an explicit override when the
# caller passes ``model=`` to ``connect_to_endpoint`` / ``create_endpoint`` /
# ``create_async_endpoint``. The override path skips ARN detection entirely.
_OVERRIDE_TABLE: Dict[str, ModelSpec] = {
    spec.name: spec for _, spec in _DETECTION_TABLE
}


def _resolve_model(arn: str, override: Optional[str] = None) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``arn``, or for ``override`` if set.

    ``override`` must be a known canonical model name when supplied;
    unknown values raise :class:`ValueError` at connect time so the customer
    sees the problem immediately instead of getting a malformed payload at
    inference time.

    When ``override`` is ``None`` the ARN is matched against the detection
    table in order. An ARN that matches nothing falls back to
    ``embeddings-default`` — same behaviour as the pre-refactor substring
    matching.
    """
    if override is not None:
        if override not in _OVERRIDE_TABLE:
            known = sorted(_OVERRIDE_TABLE)
            raise ValueError(
                f"Unknown model override {override!r}. Known values: {known}"
            )
        return _OVERRIDE_TABLE[override]

    for slug, spec in _DETECTION_TABLE:
        if slug in arn:
            return spec
    return ModelSpec(arn, "embeddings-default")


class Client:
    def __init__(
        self,
        region_name: Optional[str] = None,
        client_args: Optional[dict] = None,
    ):
        import sagemaker

        client_args = dict(client_args or {})
        if region_name:
            client_args["region_name"] = region_name
        # Stored so methods that build their own boto3 clients (e.g.
        # ``read_async``'s ad-hoc S3 client) reuse the same region and
        # credentials.
        self._client_args = client_args

        self._sm_runtime_client = boto3.client("sagemaker-runtime", **client_args)
        self._sm_client = boto3.client("sagemaker", **client_args)
        # Thread region_name into the boto3 Session so the high-level sagemaker
        # SDK honours the customer-supplied region instead of silently falling
        # back to AWS_DEFAULT_REGION.
        self._sm_session = sagemaker.Session(
            boto_session=boto3.Session(region_name=client_args.get("region_name")),
            sagemaker_client=self._sm_client,
        )
        self._aas_client = boto3.client("application-autoscaling", **client_args)
        self._cw_client = boto3.client("cloudwatch", **client_args)

        self._endpoint_name: Optional[str] = None
        self._endpoint_config_name: Optional[str] = None
        self._model_name: Optional[str] = None
        # Connected-state attrs initialised here so methods called before
        # ``connect_to_endpoint`` raise the explicit "No endpoint connected"
        # message instead of AttributeError.
        self._variant_name: Optional[str] = None
        self._resource_id: Optional[str] = None
        self._arn: Optional[str] = None
        self._model_spec: Optional[ModelSpec] = None

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self._sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            return False
        return True

    def connect_to_endpoint(
        self,
        endpoint_name: str,
        arn: str,
        *,
        model: Optional[str] = None,
    ) -> None:
        """Connect this client to an existing SageMaker endpoint.

        Args:
            endpoint_name: Name of the deployed SageMaker endpoint.
            arn: The model package ARN backing the endpoint.
            model: Optional explicit model identifier (e.g.
                ``"jina-embeddings-v3"``, ``"ReaderLM-v2"``) that bypasses ARN
                substring detection. Use this when a custom or future ARN
                format prevents auto-detection from picking the correct
                request shape. Must be one of the known canonical model
                names; otherwise a :class:`ValueError` is raised here at
                connect time.
        """
        if not self._does_endpoint_exist(endpoint_name):
            raise Exception(f"Endpoint {endpoint_name} does not exist.")
        self._endpoint_name = endpoint_name
        self._variant_name = "AllTraffic"
        self._resource_id = "endpoint/{}/variant/{}".format(
            self._endpoint_name, self._variant_name
        )
        self._arn = arn
        self._model_spec = _resolve_model(arn, override=model)

    def create_async_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        s3_output_path: str,
        instance_type: str,
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
        success_topic: Optional[str] = None,
        error_topic: Optional[str] = None,
        wait: bool = True,
        poll_interval: int = 30,
        timeout: int = 3600,
        *,
        model: Optional[str] = None,
    ) -> None:
        """Create an asynchronous SageMaker endpoint from a model package ARN.

        Args:
            arn: The model package ARN.
            endpoint_name: The name of the endpoint.
            s3_output_path: S3 path for asynchronous inference results.
            instance_type: Instance type (e.g. ``ml.m5.xlarge``).
            n_instances: Number of instances to deploy.
            recreate: If ``True``, replace an existing endpoint with the same
                name.
            role: IAM role ARN for the model.
            success_topic: SNS topic ARN for successful inference
                notifications.
            error_topic: SNS topic ARN for error notifications.
            wait: Block until the endpoint reaches ``InService``.
            poll_interval: Seconds between status polls when ``wait=True``.
            timeout: Maximum seconds to wait for endpoint readiness.
            model: Explicit model identifier passed through to
                :meth:`connect_to_endpoint`.
        """
        if role is None:
            role = get_role()

        model_name = endpoint_name
        try:
            self._sm_client.delete_model(ModelName=model_name)
        except ClientError:
            pass

        self._sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role,
            Containers=[{"ModelPackageName": arn}],
        )
        self._model_name = model_name

        try:
            self._sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            pass

        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name, arn, model=model)
                self.delete_endpoint()
            else:
                raise Exception(
                    f"Endpoint {endpoint_name} already exists and recreate={recreate}."
                )

        async_inference_config: Dict = {
            "OutputConfig": {
                "S3OutputPath": s3_output_path,
            }
        }
        if success_topic or error_topic:
            async_inference_config["OutputConfig"]["NotificationConfig"] = {}
            if success_topic:
                async_inference_config["OutputConfig"]["NotificationConfig"][
                    "SuccessTopic"
                ] = success_topic
            if error_topic:
                async_inference_config["OutputConfig"]["NotificationConfig"][
                    "ErrorTopic"
                ] = error_topic

        self._sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": n_instances,
                }
            ],
            AsyncInferenceConfig=async_inference_config,
        )
        self._endpoint_config_name = endpoint_name

        self._sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )

        if wait:
            log.info(f"Waiting for endpoint {endpoint_name} to be InService...")
            start_time = time.time()
            while True:
                response = self._sm_client.describe_endpoint(EndpointName=endpoint_name)
                status = response["EndpointStatus"]

                if status == "InService":
                    log.info(f"Async endpoint {endpoint_name} is now InService.")
                    break
                elif status in ["Failed", "RollingBack"]:
                    raise Exception(f"Endpoint creation failed with status: {status}")

                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    raise TimeoutError(f"Endpoint {endpoint_name} creation timed out.")

                log.info(f"Endpoint {endpoint_name} status: {status}. Waiting...")
                time.sleep(poll_interval)

        self.connect_to_endpoint(endpoint_name, arn, model=model)

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        instance_type: str,
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
        *,
        model: Optional[str] = None,
    ) -> None:
        import sagemaker

        if role is None:
            role = get_role()

        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name, arn, model=model)
                self.delete_endpoint()
            else:
                raise Exception(
                    f"Endpoint {endpoint_name} already exists and recreate={recreate}."
                )

        try:
            self._sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            pass

        sm_model = sagemaker.ModelPackage(
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,
            model_package_arn=arn,
        )

        try:
            sm_model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)
        except ParamValidationError:
            sm_model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)

        self._endpoint_config_name = endpoint_name
        self.connect_to_endpoint(endpoint_name, arn, model=model)

    def register_scalable_target(self, max_capacity, min_capacity=1):
        return self._aas_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=self._resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

    def set_step_autoscaling(self, policy_name, policy_configuration: Dict):
        return self._aas_client.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace="sagemaker",
            ResourceId=self._resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="StepScaling",
            StepScalingPolicyConfiguration=policy_configuration,
        )

    def set_metric_alarm(self, policy_arn, **kwargs):
        kwargs["Dimensions"] = [
            {"Name": "EndpointName", "Value": self._endpoint_name},
            {"Name": "VariantName", "Value": self._variant_name},
        ]
        kwargs["AlarmActions"] = [policy_arn]
        return self._cw_client.put_metric_alarm(**kwargs)

    def create_transform_job(
        self,
        arn: str,
        n_instances: int,
        instance_type: str,
        input_path: str,
        output_path: str,
        role: Optional[str] = None,
        wait: bool = True,
        logs: bool = True,
        assemble_with: Optional[str] = None,
        max_payload: Optional[int] = None,
    ) -> Optional[str]:
        import sagemaker

        # wait=False kicks off the transform asynchronously and returns
        # immediately; a local output_path triggers an immediate
        # download_s3_folder which would land on empty/partial results.
        # Refuse the combination up front instead of silently producing
        # bad output.
        if not wait and not output_path.startswith("s3://"):
            raise ValueError(
                "create_transform_job(wait=False) requires an s3:// "
                "output_path. Use wait=True for a local output_path so the "
                "transform completes before files are downloaded."
            )

        if role is None:
            role = get_role()

        model = sagemaker.ModelPackage(
            name=arn.split("/")[-1],
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,
            model_package_arn=arn,
        )

        uid = uuid.uuid4().hex
        if not input_path.startswith("s3://"):
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path {input_path} does not exist.")
            csv_path_with_ids = prefix_csv_with_ids(input_path=input_path)
            s3_input_path = self._sm_session.upload_data(
                path=csv_path_with_ids, key_prefix=f"input/{uid}"
            )
            log.info(f"Input file uploaded to {s3_input_path}.")
        else:
            s3_input_path = input_path
            log.info(f"Input file is already on S3, using {s3_input_path}.")

        download_output_path = None
        if not output_path.startswith("s3://"):
            download_output_path = output_path
            output_path = os.path.join(
                "s3://",
                self._sm_session.default_bucket(),
                "output",
                model.name,
                uid,
            )

        transformer = model.transformer(
            instance_count=n_instances,
            instance_type=instance_type,
            output_path=output_path,
            strategy="MultiRecord",
            assemble_with=assemble_with,
            max_payload=max_payload,
        )

        transformer.transform(
            data=s3_input_path,
            content_type="text/csv",
            split_type="Line",
            wait=wait,
            logs=logs,
        )

        if download_output_path is not None:
            download_s3_folder(
                path=output_path,
                local_dir=download_output_path,
            )
            log.info(f"Output downloaded to {download_output_path}.")

        job_name = None
        if transformer.latest_transform_job is not None:
            job_name = transformer.latest_transform_job.name
        return job_name

    def _require_endpoint(self) -> None:
        if self._endpoint_name is None or self._model_spec is None:
            raise Exception("No endpoint connected. Run connect_to_endpoint() first.")

    def _reader_model_name(self) -> str:
        # Preserves pre-refactor behaviour: ARNs that didn't match the reader
        # patterns silently defaulted to "reader-lm-0.5b" in the request body.
        assert self._model_spec is not None  # _require_endpoint ran first
        return self._model_spec.reader_model or "reader-lm-0.5b"

    def read_async(self, prompt: str, input_s3_path: str):
        """Asynchronous variant of :meth:`read` using ``invoke_endpoint_async``.

        Args:
            prompt: Input prompt for the ReaderLM model.
            input_s3_path: S3 location where the input payload is uploaded
                before the async invocation. Must be an ``s3://`` URI the
                caller's credentials can write to.

        Returns:
            A dict with ``OutputLocation`` (the S3 URI of the async result)
            and ``InputLocation`` (the URI passed in).
        """
        self._require_endpoint()

        data = json.dumps(
            {
                "model": self._reader_model_name(),
                "prompt": prompt,
            }
        )

        # Reuse client_args so this s3 client honours the same region and
        # credentials as the rest of the Client.
        s3 = boto3.client("s3", **self._client_args)
        bucket_name, input_key = input_s3_path.replace("s3://", "").split("/", 1)
        s3.put_object(Bucket=bucket_name, Key=input_key, Body=data)

        response = self._sm_runtime_client.invoke_endpoint_async(
            EndpointName=self._endpoint_name,
            InputLocation=input_s3_path,
            ContentType="application/json",
        )

        return {
            "OutputLocation": response["OutputLocation"],
            "InputLocation": input_s3_path,
        }

    def read(self, prompt: str, stream: bool = False):
        """Send a prompt to a ReaderLM endpoint and return the response.

        Args:
            prompt: Input prompt for the model.
            stream: When ``True``, parse a server-sent-event style stream
                and return the list of decoded events.
        """
        self._require_endpoint()

        data = json.dumps(
            {
                "model": self._reader_model_name(),
                "prompt": prompt,
                "stream": stream,
            }
        )

        response = self._sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

        if stream:
            response_body = response["Body"]
            streamed_results = []

            for line in response_body.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data:"):
                        json_data = decoded_line[5:].strip()
                        try:
                            streamed_results.append(json.loads(json_data))
                        except json.JSONDecodeError:
                            pass

            return streamed_results
        else:
            response_body = response["Body"].read().decode()
            return json.loads(response_body)

    def embed(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        image_urls: Optional[Union[str, List[str]]] = None,
        image_bytes: Optional[Union[str, List[str]]] = None,
        pdf_url: Optional[str] = None,
        use_colbert: Optional[bool] = False,
        input_type: Optional[InputType] = InputType.DOCUMENT,
        task_type: Optional[Task] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = False,
        return_multivector: Optional[bool] = False,
    ):
        """Embed text, images, or a PDF.

        Args:
            texts: The text or texts to embed.
            image_urls: URL(s) of the image(s) to embed.
            image_bytes: Base64-encoded image bytes.
            pdf_url: URL of a PDF to embed. PDF cannot be mixed with other
                media types and is only supported on ``jina-embeddings-v4``.
            use_colbert: Use the ColBERT request shape.
            input_type: Treat texts as queries or documents (ColBERT only).
            task_type: Downstream task for v3/v4/clip-v2; ``None`` selects
                the default ``"text-matching"``.
            dimensions: Output dimensions (v3/v4/clip-v2).
            late_chunking: Apply the late-chunking technique (v3/v4).
            return_multivector: Return multi-vector output (v4 only).
        """
        self._require_endpoint()
        assert self._model_spec is not None  # _require_endpoint guarantees this
        spec = self._model_spec

        if use_colbert:
            if isinstance(texts, str):
                payload = {
                    "data": {"text": texts},
                    "parameters": {"input_type": input_type.value},
                }
            else:
                payload = {
                    "data": [{"text": text} for text in texts],
                    "parameters": {"input_type": input_type.value},
                }
            data = json.dumps(payload)
        else:
            data_obj: Dict = {"data": []}

            if spec.family in ("embeddings-v3", "embeddings-v4"):
                data_obj["parameters"] = {
                    "task": task_type.value if task_type else "text-matching",
                    "dimensions": dimensions,
                    "late_chunking": late_chunking,
                }
                if spec.family == "embeddings-v4":
                    data_obj["parameters"]["return_multivector"] = return_multivector
            elif spec.family == "clip-v2":
                data_obj["parameters"] = {
                    "task": task_type.value if task_type else "text-matching",
                    "dimensions": dimensions,
                }

            if texts:
                if isinstance(texts, str):
                    data_obj["data"] += [{"text": texts}]
                else:
                    data_obj["data"] += [{"text": text} for text in texts]

            if image_urls:
                key = "url" if spec.family == "clip-v2" else "image"
                if isinstance(image_urls, str):
                    data_obj["data"] += [{key: image_urls}]
                else:
                    data_obj["data"] += [{key: image_url} for image_url in image_urls]

            if image_bytes:
                key = "bytes" if spec.family == "clip-v2" else "image"
                if isinstance(image_bytes, str):
                    data_obj["data"] += [{key: image_bytes}]
                else:
                    data_obj["data"] += [{key: ib} for ib in image_bytes]

            if spec.family == "embeddings-v4" and pdf_url:
                data_obj["data"] = {"pdf": pdf_url}

            data = json.dumps(data_obj)

        response = self._sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

        resp = json.loads(response["Body"].read().decode())
        return resp["data"]

    def rerank(
        self,
        documents: List[Union[str, dict]],
        query: str,
        top_n: Optional[int] = None,
    ):
        self._require_endpoint()
        assert self._model_spec is not None

        normalized_documents = []
        for doc in documents:
            if isinstance(doc, str):
                normalized_documents.append({"text": doc})
            elif isinstance(doc, dict):
                normalized_documents.append(doc)
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")

        data: Dict = {
            "documents": normalized_documents,
            "query": query,
        }

        if top_n:
            data["top_n"] = min(top_n, len(normalized_documents))

        if self._model_spec.family == "reranker-m0":
            payload = json.dumps({"data": [data]})
        else:
            payload = json.dumps({"data": data})

        response = self._sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=payload,
        )

        resp = json.loads(response["Body"].read().decode())
        return resp["data"]

    def delete_endpoint(self) -> None:
        """Delete the endpoint, its config, and the model (if set)."""
        if self._endpoint_name is None:
            raise Exception("No endpoint connected.")

        try:
            self._sm_client.delete_endpoint(EndpointName=self._endpoint_name)
            log.info(f"Deleted endpoint: {self._endpoint_name}")
        except ClientError:
            log.info(f"Endpoint '{self._endpoint_name}' not found, skipping deletion.")

        if self._endpoint_config_name is not None:
            try:
                self._sm_client.delete_endpoint_config(
                    EndpointConfigName=self._endpoint_config_name
                )
                log.info(
                    f"Deleted endpoint configuration: {self._endpoint_config_name}"
                )
            except ClientError:
                log.info(
                    f"Endpoint configuration '{self._endpoint_config_name}' "
                    "not found, skipping deletion."
                )

        if self._model_name is not None:
            try:
                self._sm_client.delete_model(ModelName=self._model_name)
                log.info(f"Deleted model: {self._model_name}")
            except ClientError:
                log.info(f"Model '{self._model_name}' not found, skipping deletion.")

    def close(self) -> None:
        """Close the underlying boto3 clients.

        Older boto3 releases (pre-1.27) did not implement ``.close()`` on the
        low-level clients; in that case we log a single info line and
        continue rather than raising AttributeError, since this is a
        best-effort cleanup.
        """
        try:
            self._sm_runtime_client.close()
            self._sm_client.close()
        except AttributeError:
            log.info(
                "SageMaker client could not be closed; this can happen on "
                "very old boto3 versions where Client.close() was not yet "
                "implemented. Continuing."
            )
