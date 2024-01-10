import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError, ParamValidationError

from .helper import download_s3_folder, get_role, prefix_csv_with_ids


class Client:
    def __init__(
        self,
        region_name: Optional[str] = None,
        verbose=False,
        client_args: Optional[dict] = None,
    ):
        import sagemaker

        if not verbose:
            logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

        client_args = client_args or {}
        if region_name:
            client_args["region_name"] = region_name

        self._sm_runtime_client = boto3.client("sagemaker-runtime", **client_args)
        self._sm_client = boto3.client("sagemaker", **client_args)
        self._sm_session = sagemaker.Session(sagemaker_client=self._sm_client)
        self._aas_client = boto3.client("application-autoscaling", **client_args)
        self._cw_client = boto3.client("cloudwatch", **client_args)

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self._sm_client.describe_endpoint(EndpointName=endpoint_name)
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
                self.connect_to_endpoint(endpoint_name)
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

        self.connect_to_endpoint(endpoint_name)

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
            print(f"Input file uploaded to {s3_input_path}.")
        else:
            s3_input_path = input_path
            print(f"Input file is already on S3, using {s3_input_path}.")

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
            print(f"Output downloaded to {download_output_path}.")

        job_name = None
        if transformer.latest_transform_job is not None:
            job_name = transformer.latest_transform_job.name
        return job_name

    def _invoke_endpoint(self, texts: Union[str, List[str]]):
        if self._endpoint_name is None:
            raise Exception(
                "No endpoint connected. " "Run connect_to_endpoint() first."
            )

        if isinstance(texts, str):
            data = json.dumps({"data": {"text": texts}})
        else:
            data = json.dumps({"data": [{"text": text} for text in texts]})

        return self._sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

    def embed(self, texts: Union[str, List[str]]):
        response = self._invoke_endpoint(texts)
        resp = json.loads(response["Body"].read().decode())
        return resp["data"]

    def delete_endpoint(self) -> None:
        if self._endpoint_name is None:
            raise Exception("No endpoint connected.")
        try:
            self._sm_client.delete_endpoint(EndpointName=self._endpoint_name)
        except ClientError:
            print("Endpoint not found, skipping deletion.")

        try:
            self._sm_client.delete_endpoint_config(
                EndpointConfigName=self._endpoint_name
            )
        except ClientError:
            print("Endpoint config not found, skipping deletion.")

    def close(self) -> None:
        try:
            self._sm_runtime_client.close()
            self._sm_client.close()
        except AttributeError:
            print(
                "SageMaker client could not be closed. "
                "This might be because you are using an old version of SageMaker."
            )
            raise
