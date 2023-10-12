import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError, ParamValidationError

from .helper import prefix_csv_with_ids

if TYPE_CHECKING:
    from sagemaker.serverless import ServerlessInferenceConfig


class Client:
    def __init__(self, region_name: Optional[str] = None, verbose=False):
        if not verbose:
            logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

        import sagemaker

        self.sm_runtime_client = boto3.client(
            "sagemaker-runtime", region_name=region_name
        )
        self.sm_client = boto3.client("sagemaker", region_name=region_name)
        self._sm_session = sagemaker.Session(sagemaker_client=self.sm_client)
        self.aas_client = boto3.client(
            "application-autoscaling", region_name=region_name
        )
        self.cw_client = boto3.client("cloudwatch", region_name=region_name)

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            return False
        return True

    def connect_to_endpoint(self, endpoint_name: str) -> None:
        if not self._does_endpoint_exist(endpoint_name):
            raise Exception(f"Endpoint {endpoint_name} does not exist.")
        self._endpoint_name = endpoint_name
        self._variant_name = "AllTraffic"
        self._resource_id = "endpoint/{}/variant/{}".format(
            self._endpoint_name, self._variant_name
        )

    def get_role(self):
        import sagemaker

        try:
            return sagemaker.get_execution_role()
        except ValueError:
            print("Using default role: 'ServiceRoleSagemaker'.")
            return "ServiceRoleSagemaker"

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        instance_type: str = "",
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
        sls_config: Optional["ServerlessInferenceConfig"] = None,
    ) -> None:
        import sagemaker

        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name)
                self.delete_endpoint()
            else:
                raise Exception(
                    f"Endpoint {endpoint_name} already exists and recreate={recreate}."
                )

        kwargs = {"model_package_arn": arn}

        if role is None:
            role = self.get_role()

        model = sagemaker.ModelPackage(
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,  # makes sure the right region is used
            **kwargs,
        )

        validation_params = dict(
            model_data_download_timeout=2400,
            container_startup_health_check_timeout=2400,
        )

        if (instance_type and sls_config) or (not instance_type and not sls_config):
            raise Exception(
                "Please specify either instance_type or sls_config, but not both. These parameters are mutually exclusive"
            )

        try:
            model.deploy(
                n_instances,
                instance_type,
                endpoint_name=endpoint_name,
                serverless_inference_config=sls_config,
                **validation_params,
            )
        except ParamValidationError:
            model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)

        self.connect_to_endpoint(endpoint_name)

    def register_scalable_target(self, max_capacity, min_capacity=1):
        response = self.aas_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=self._resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )
        return response

    def set_step_autoscaling(self, policy_name, policy_configuration: Dict):
        response = self.aas_client.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace="sagemaker",
            ResourceId=self._resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="StepScaling",
            StepScalingPolicyConfiguration=policy_configuration,
        )
        return response

    def set_metric_alarm(self, policy_arn, **kwargs):
        kwargs["Dimensions"] = [
            {"Name": "EndpointName", "Value": self._endpoint_name},
            {"Name": "VariantName", "Value": self._variant_name},
        ]
        kwargs["AlarmActions"] = [policy_arn]

        return self.cw_client.put_metric_alarm(**kwargs)

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
    ):
        import sagemaker

        if role is None:
            role = self.get_role()

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
            if os.path.exists(input_path):
                prefix_csv_with_ids(input_path=input_path, output_path=input_path)
                input_path = self._sm_session.upload_data(
                    path=input_path, key_prefix=f"input/{uid}"
                )

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
        )

        transformer.transform(
            data=input_path,
            content_type="text/csv",
            split_type="Line",
            wait=wait,
            logs=logs,
        )

        if download_output_path is not None:
            self.download_s3_folder(
                s3_path=output_path,
                local_dir=download_output_path,
            )
            print(f"Output downloaded to {download_output_path}.")

    def embed(self, texts: Union[str, List[str]]):
        if self._endpoint_name is None:
            raise Exception(
                "No endpoint connected. " "Run connect_to_endpoint() first."
            )

        if isinstance(texts, str):
            data = json.dumps({"data": {"text": texts}})
        else:
            data = json.dumps({"data": [{"text": text} for text in texts]})

        response = self.sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

        resp = json.loads(response["Body"].read().decode())
        return resp["data"]

    def delete_endpoint(self) -> None:
        if self._endpoint_name is None:
            raise Exception("No endpoint connected.")
        try:
            self.sm_client.delete_endpoint(EndpointName=self._endpoint_name)
        except ClientError:
            print("Endpoint not found, skipping deletion.")

        try:
            self.sm_client.delete_endpoint_config(
                EndpointConfigName=self._endpoint_name
            )
        except ClientError:
            print("Endpoint config not found, skipping deletion.")

    def close(self) -> None:
        try:
            self.sm_runtime_client.close()
            self.sm_client.close()
        except AttributeError:
            print(
                "SageMaker client could not be closed. This might be because you are using an old version of SageMaker."
            )
            raise

    @staticmethod
    def download_s3_folder(s3_path: str, local_dir: str = None):
        """
        Download the contents of a folder directory
        Args:
            s3_path: the s3 path
            local_dir: a relative or absolute directory path in the local file system
        """
        bucket_name, s3_folder = s3_path.replace("s3://", "").split("/", 1)
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
