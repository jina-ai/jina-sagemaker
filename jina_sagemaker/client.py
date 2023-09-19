import json
from typing import Dict, List, Optional

import boto3
import sagemaker
from botocore.exceptions import ClientError, ParamValidationError
from sagemaker.serverless import ServerlessInferenceConfig


class Client:
    def __init__(self, region_name: Optional[str] = None):
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

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        instance_type: str = "",
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
        sls_config: Optional[ServerlessInferenceConfig] = None,
    ) -> None:
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
            try:
                role = sagemaker.get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

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
        content_type: str,
        split_type: str,
        strategy: str,
        role: Optional[str] = None,
    ):
        kwargs = {"model_package_arn": arn}

        if role is None:
            try:
                role = sagemaker.get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

        model = sagemaker.ModelPackage(
            role=role,
            model_data=None,
            sagemaker_session=self._sm_session,  # makes sure the right region is used
            **kwargs,
        )

        transformer = model.transformer(
            instance_count=n_instances,
            instance_type=instance_type,
            output_path=output_path,
            strategy=strategy,
        )

        transformer.transform(
            data=input_path,
            content_type=content_type,
            split_type=split_type,
        )

    def embed(
        self,
        model: str,
        texts: List[str],
    ):
        if self._endpoint_name is None:
            raise Exception(
                "No endpoint connected. " "Run connect_to_endpoint() first."
            )

        data = json.dumps({"model": model, "texts": texts})

        response = self.sm_runtime_client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=data,
        )

        return json.loads(response["Body"].read().decode())

    def delete_endpoint(self) -> None:
        if self._endpoint_name is None:
            raise Exception("No endpoint connected.")
        try:
            self.sm_client.delete_endpoint(EndpointName=self._endpoint_name)
        except:
            print("Endpoint not found, skipping deletion.")

        try:
            self.sm_client.delete_endpoint_config(
                EndpointConfigName=self._endpoint_name
            )
        except:
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
