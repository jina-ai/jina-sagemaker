import json
import logging
import os
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError, ParamValidationError

from .helper import download_s3_folder, get_role, prefix_csv_with_ids

logging.basicConfig(level=logging.INFO)
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


class Client:
    def __init__(
        self,
        region_name: Optional[str] = None,
        client_args: Optional[dict] = None,
    ):
        import sagemaker

        client_args = client_args or {}
        if region_name:
            client_args["region_name"] = region_name

        self._sm_runtime_client = boto3.client("sagemaker-runtime", **client_args)
        self._sm_client = boto3.client("sagemaker", **client_args)
        self._sm_session = sagemaker.Session(
            boto_session=boto3.Session(),
            sagemaker_client=self._sm_client,
        )
        self._aas_client = boto3.client("application-autoscaling", **client_args)
        self._cw_client = boto3.client("cloudwatch", **client_args)

        self._endpoint_name = None
        self._endpoint_config_name = None
        self._model_name = None

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self._sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            return False
        return True

    def connect_to_endpoint(self, endpoint_name: str, arn: str) -> None:
        if not self._does_endpoint_exist(endpoint_name):
            raise Exception(f"Endpoint {endpoint_name} does not exist.")
        self._endpoint_name = endpoint_name
        self._variant_name = "AllTraffic"
        self._resource_id = "endpoint/{}/variant/{}".format(
            self._endpoint_name, self._variant_name
        )
        self._arn = arn

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
    ) -> None:
        """
        Creates an asynchronous SageMaker endpoint from a model package ARN.

        Args:
            arn (str): The model package ARN.
            endpoint_name (str): The name of the endpoint.
            s3_output_path (str): S3 path where the asynchronous inference results will be stored.
            instance_type (str): The instance type for the endpoint (e.g., "ml.m5.xlarge").
            n_instances (int): The number of instances to deploy (default: 1).
            recreate (bool): Whether to recreate the endpoint if it already exists (default: False).
            role (Optional[str]): The IAM role ARN to associate with the model.
            success_topic (Optional[str]): SNS topic ARN for successful inference notifications (default: None).
            error_topic (Optional[str]): SNS topic ARN for error notifications (default: None).
            wait (bool): Whether to wait for the endpoint to be fully deployed (default: True).
            poll_interval (int): Interval in seconds to check the endpoint status (default: 30).
            timeout (int): Maximum time in seconds to wait for the endpoint to be deployed (default: 3600).
        """
        from botocore.exceptions import ClientError

        if role is None:
            role = get_role()

        model_name = endpoint_name
        try:
            self._sm_client.delete_model(ModelName=model_name)
        except ClientError as e:
            pass

        # Create model
        self._sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role,
            Containers=[{"ModelPackageName": arn}],
        )
        self._model_name = model_name

        # Delete existing endpoint configuration if it exists
        try:
            self._sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            pass

        # Check if the endpoint already exists
        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name, arn)
                self.delete_endpoint()
            else:
                raise Exception(
                    f"Endpoint {endpoint_name} already exists and recreate={recreate}."
                )

        # Create an endpoint configuration with AsyncInferenceConfig
        async_inference_config = {
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

        # Wait for the endpoint to become "InService" if `wait` is True
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

        self.connect_to_endpoint(endpoint_name, arn)

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        instance_type: str,
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
    ) -> None:
        import sagemaker

        if role is None:
            role = get_role()

        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name, arn)
                self.delete_endpoint()
            else:
                raise Exception(
                    f"Endpoint {endpoint_name} already exists and recreate={recreate}."
                )

        # Check if there is already endpoint config, if so delete it or it will block model.deploy
        try:
            self._sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            pass

        model = sagemaker.ModelPackage(
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,  # makes sure the right region is used
            model_package_arn=arn,
        )

        try:
            model.deploy(
                n_instances,
                instance_type,
                endpoint_name=endpoint_name,
            )
        except ParamValidationError:
            model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)

        self._endpoint_config_name = endpoint_name
        self.connect_to_endpoint(endpoint_name, arn)

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

        if role is None:
            role = get_role()

        model = sagemaker.ModelPackage(
            name=arn.split("/")[-1],
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,  # makes sure the right region is used
            model_package_arn=arn,
        )

        uid = uuid.uuid4().hex
        # if input path is a local path, upload to default s3 bucket
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
        # if output path is a local path, change to default s3 bucket,
        # add job name and random uuid
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

    def read_async(self, prompt: str, input_s3_path: str):
        """
        Asynchronous version of the read method that uses invoke_endpoint_async.

        Args:
            prompt (str): The input prompt for the model.
            input_s3_path (str): S3 path where the input data will be uploaded.
            output_s3_path (str): S3 path where the output data will be stored.

        """
        if self._endpoint_name is None:
            raise Exception("No endpoint connected. Run connect_to_endpoint() first.")

        model = "reader-lm-0.5b"
        if "1500m" in self._arn:
            model = "reader-lm-1.5b"
        elif "v2" in self._arn:
            model = "ReaderLM-v2"

        # Prepare the input payload
        data = json.dumps(
            {
                "model": model,
                "prompt": prompt,
            }
        )

        s3 = boto3.client("s3")
        bucket_name, input_key = input_s3_path.replace("s3://", "").split("/", 1)
        s3.put_object(Bucket=bucket_name, Key=input_key, Body=data)

        # Call the async endpoint
        response = self._sm_runtime_client.invoke_endpoint_async(
            EndpointName=self._endpoint_name,
            InputLocation=input_s3_path,
            ContentType="application/json",
        )

        # Return the response metadata, including the output location
        return {
            "OutputLocation": response["OutputLocation"],
            "InputLocation": input_s3_path,
        }

    def read(self, prompt: str, stream: bool = False):
        """
        Send a request to a ReaderLM and process the response.

        Args:
            prompt (str): The input prompt for the model.
            stream (bool, optional): Flag indicating whether the response should be
                processed as a stream. Defaults to False.

        """
        if self._endpoint_name is None:
            raise Exception(
                "No endpoint connected. " "Run connect_to_endpoint() first."
            )

        model = "reader-lm-0.5b"
        if "1500m" in self._arn:
            model = "reader-lm-1.5b"
        elif "v2" in self._arn:
            model = "ReaderLM-v2"

        data = json.dumps(
            {
                "model": model,
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
                    if decoded_line.startswith("data:"):  # Handle 'data:' prefix
                        json_data = decoded_line[5:].strip()
                        try:
                            streamed_results.append(json.loads(json_data))
                        except json.JSONDecodeError:
                            pass

            return streamed_results
        else:
            # For non-streamed responses, read the entire body
            response_body = response["Body"].read().decode()
            return json.loads(response_body)

    def embed(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        image_urls: Optional[Union[str, List[str]]] = None,
        image_bytes: Optional[Union[str, List[str]]] = None,
        pdf_url: Optional[str] = False,
        use_colbert: Optional[bool] = False,
        input_type: Optional[InputType] = InputType.DOCUMENT,
        task_type: Optional[Task] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = False,
        return_multivector: Optional[bool] = False,
    ):
        """
        Embeds the given texts.

        Parameters:
            - texts (Union[str, List[str]]): The text or texts to embed. Can be a single
            string or a list of strings.
            - image_urls (Optional[Union[str, List[str]]]): URLs of the images to embed. Can be a single
            URL or a list of URLs.
            - image_bytes (Optional[Union[str, List[str]]]): Bytes of the images to embed. Can be a single
            byte string or a list of byte strings.
            - pdf_url (Optional[str]): URLs of the PDF to embed. PDF cannot be mixed with other media types.
            - use_colbert (bool, optional): A flag indicating ColBERT model is used for embedding.
            - input_type (InputType, optional): The type of input texts, indicating whether
            they should be treated as documents or queries. This is only needed when use_colbert is True.
            - task_type (Task, optional): Select the downstream task for which the embeddings will be used. The model will return the optimized embeddings for that task. None meaning no specific task is needed.
            - dimensions (Optional[int], optional): Output dimensions. Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
            - late_chunking (Optional[bool], optional): Apply the late chunking technique to leverage the model's long-context capabilities for generating contextual chunk embeddings.
            - return_multivector (Optional[bool], optional): Whether to return multi vector output.
        """

        if self._endpoint_name is None:
            raise Exception(
                "No endpoint connected. " "Run connect_to_endpoint() first."
            )

        if not use_colbert:
            data = {"data": []}
            if "jina-embeddings-v3" in self._arn or "jina-embeddings-v4" in self._arn:
                data["parameters"] = {
                    "task": task_type.value if task_type else "text-matching",
                    "dimensions": dimensions,
                    "late_chunking": late_chunking,
                }
                if "jina-embeddings-v4" in self._arn:
                    data["parameters"]["return_multivector"] = return_multivector
            elif "jina-clip-v2" in self._arn:
                data["parameters"] = {
                    "task": task_type.value if task_type else "text-matching",
                    "dimensions": dimensions,
                }

            if texts:
                if isinstance(texts, str):
                    data["data"] += [{"text": texts}]
                else:
                    data["data"] += [{"text": text} for text in texts]

            if image_urls:
                key = "url" if "jina-clip-v2" in self._arn else "image"
                if isinstance(image_urls, str):
                    data["data"] += [{key: image_urls}]
                else:
                    data["data"] += [{key: image_url} for image_url in image_urls]

            if image_bytes:
                key = "bytes" if "jina-clip-v2" in self._arn else "image"
                if isinstance(image_bytes, str):
                    data["data"] += [{key: image_bytes}]
                else:
                    data["data"] += [
                        {key: image_bytes_item} for image_bytes_item in image_bytes
                    ]

            if "jina-embeddings-v4" in self._arn and pdf_url:
                data["data"] = {"pdf": pdf_url}

            data = json.dumps(data)
        else:
            if isinstance(texts, str):
                data = json.dumps(
                    {
                        "data": {"text": texts},
                        "parameters": {"input_type": input_type.value},
                    }
                )
            else:
                data = json.dumps(
                    {
                        "data": [{"text": text} for text in texts],
                        "parameters": {"input_type": input_type.value},
                    }
                )

        response = self._sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

        resp = json.loads(response["Body"].read().decode())
        return resp["data"]

    def rerank(
        self, documents: List[Union[str, dict]], query: str, top_n: Optional[int] = None
    ):
        if self._endpoint_name is None:
            raise Exception("No endpoint connected. Run connect_to_endpoint() first.")

        # Normalize input into list of dicts
        normalized_documents = []
        for doc in documents:
            if isinstance(doc, str):
                normalized_documents.append({"text": doc})
            elif isinstance(doc, dict):
                normalized_documents.append(doc)
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")

        data = {
            "documents": normalized_documents,
            "query": query,
        }

        if top_n:
            data["top_n"] = min(top_n, len(normalized_documents))

        if "jina-reranker-m0" in self._arn:
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
        """
        Deletes the endpoint, its configuration, and the associated model if their names are set.
        """

        if self._endpoint_name is None:
            raise Exception("No endpoint connected.")

        # Delete the endpoint
        try:
            self._sm_client.delete_endpoint(EndpointName=self._endpoint_name)
            log.info(f"Deleted endpoint: {self._endpoint_name}")
        except ClientError:
            log.info(f"Endpoint '{self._endpoint_name}' not found, skipping deletion.")

        # Delete the endpoint configuration
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
                    f"Endpoint configuration '{self._endpoint_config_name}' not found, skipping deletion."
                )

        # Delete the model
        if self._model_name is not None:
            try:
                self._sm_client.delete_model(ModelName=self._model_name)
                log.info(f"Deleted model: {self._model_name}")
            except ClientError:
                log.info(f"Model '{self._model_name}' not found, skipping deletion.")

    def close(self) -> None:
        try:
            self._sm_runtime_client.close()
            self._sm_client.close()
        except AttributeError:
            log.info(
                "SageMaker client could not be closed. "
                "This might be because you are using an old version of SageMaker."
            )
            raise
