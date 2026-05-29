"""Tests for jina_sagemaker.Client.

All AWS calls are stubbed via ``botocore.stub.Stubber``. The goal of this suite
is to lock the wire-format that every Client method emits for each model
family, so future refactors of model dispatch are guaranteed not to change
customer-visible request bodies.
"""

import json
from datetime import datetime, timezone
from io import BytesIO
from unittest import mock

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from jina_sagemaker import Client, InputType, Task

# ----------------------------------------------------------------------------
# ARN fixtures
# ----------------------------------------------------------------------------

ACCT = "123456789012"
REGION = "us-east-1"


def _arn(slug: str) -> str:
    return f"arn:aws:sagemaker:{REGION}:{ACCT}:model-package/{slug}/1"


ARN_V2 = _arn("jina-embeddings-v2-base-en")
ARN_V3 = _arn("jina-embeddings-v3")
ARN_V4 = _arn("jina-embeddings-v4")
ARN_CLIP_V2 = _arn("jina-clip-v2")
ARN_RERANKER = _arn("jina-reranker-v2-base-multilingual")
ARN_RERANKER_M0 = _arn("jina-reranker-m0")
ARN_READER_0_5B = _arn("reader-lm-500m")
ARN_READER_1_5B = _arn("reader-lm-1500m")
ARN_READER_LM_V2 = _arn("ReaderLM-v2")
ARN_COLBERT = _arn("jina-colbert-v1-en")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _streaming(payload: bytes) -> StreamingBody:
    return StreamingBody(BytesIO(payload), len(payload))


def _json_streaming(obj) -> StreamingBody:
    return _streaming(json.dumps(obj).encode("utf-8"))


def _connect(client: Client, arn: str, endpoint_name: str = "test-endpoint") -> None:
    """Bypass connect_to_endpoint() so individual tests don't have to stub
    describe_endpoint just to set up Client state."""
    from jina_sagemaker.client import _resolve_model

    client._endpoint_name = endpoint_name
    client._variant_name = "AllTraffic"
    client._resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    client._arn = arn
    client._model_spec = _resolve_model(arn)


def _describe_endpoint_response(name: str = "test-endpoint") -> dict:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return {
        "EndpointName": name,
        "EndpointArn": f"arn:aws:sagemaker:{REGION}:{ACCT}:endpoint/{name}",
        "EndpointConfigName": name,
        "EndpointStatus": "InService",
        "CreationTime": now,
        "LastModifiedTime": now,
    }


# ----------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------


def test_init_default(client):
    assert client._endpoint_name is None
    assert client._endpoint_config_name is None
    assert client._model_name is None


def test_init_with_region():
    c = Client(region_name="eu-west-1")
    assert c._sm_client.meta.region_name == "eu-west-1"
    assert c._sm_runtime_client.meta.region_name == "eu-west-1"
    assert c._aas_client.meta.region_name == "eu-west-1"
    assert c._cw_client.meta.region_name == "eu-west-1"


def test_init_with_client_args():
    c = Client(client_args={"region_name": "ap-southeast-1"})
    assert c._sm_client.meta.region_name == "ap-southeast-1"


# ----------------------------------------------------------------------------
# _does_endpoint_exist / connect_to_endpoint
# ----------------------------------------------------------------------------


def test_does_endpoint_exist_true(client):
    with Stubber(client._sm_client) as stub:
        stub.add_response(
            "describe_endpoint",
            _describe_endpoint_response("test-endpoint"),
            {"EndpointName": "test-endpoint"},
        )
        assert client._does_endpoint_exist("test-endpoint") is True


def test_does_endpoint_exist_false_on_client_error(client):
    with Stubber(client._sm_client) as stub:
        stub.add_client_error(
            "describe_endpoint",
            service_error_code="ValidationException",
        )
        assert client._does_endpoint_exist("test-endpoint") is False


def test_connect_to_endpoint_success(client):
    with Stubber(client._sm_client) as stub:
        stub.add_response(
            "describe_endpoint",
            _describe_endpoint_response("test-endpoint"),
            {"EndpointName": "test-endpoint"},
        )
        client.connect_to_endpoint("test-endpoint", ARN_V3)

    assert client._endpoint_name == "test-endpoint"
    assert client._variant_name == "AllTraffic"
    assert client._resource_id == "endpoint/test-endpoint/variant/AllTraffic"
    assert client._arn == ARN_V3


def test_connect_to_endpoint_not_found_raises(client):
    with Stubber(client._sm_client) as stub:
        stub.add_client_error(
            "describe_endpoint",
            service_error_code="ValidationException",
        )
        with pytest.raises(Exception, match="does not exist"):
            client.connect_to_endpoint("missing", ARN_V3)


# ----------------------------------------------------------------------------
# embed — locks the wire payload for every model family
# ----------------------------------------------------------------------------


def _invoke_response_data(rows):
    return {"data": rows}


def test_embed_no_endpoint_raises(client):
    with pytest.raises(Exception, match="No endpoint connected"):
        client.embed(texts="hi")


def test_embed_v3_text_default_task(client):
    _connect(client, ARN_V3)
    response_body = _json_streaming(_invoke_response_data([{"index": 0, "embedding": [0.1]}]))
    expected_body = json.dumps(
        {
            "data": [{"text": "hello"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {"Body": response_body, "ContentType": "application/json"},
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        result = client.embed(texts="hello")
    assert result == [{"index": 0, "embedding": [0.1]}]


def test_embed_v3_text_with_task_retrieval_query(client):
    _connect(client, ARN_V3)
    expected_body = json.dumps(
        {
            "data": [{"text": "q"}],
            "parameters": {
                "task": "retrieval.query",
                "dimensions": None,
                "late_chunking": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="q", task_type=Task.RETRIEVAL_QUERY)


def test_embed_v3_text_with_dimensions_and_late_chunking(client):
    _connect(client, ARN_V3)
    expected_body = json.dumps(
        {
            "data": [{"text": "x"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": 256,
                "late_chunking": True,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="x", dimensions=256, late_chunking=True)


def test_embed_text_list(client):
    _connect(client, ARN_V3)
    expected_body = json.dumps(
        {
            "data": [{"text": "a"}, {"text": "b"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts=["a", "b"])


def test_embed_v4_text_default_includes_return_multivector(client):
    _connect(client, ARN_V4)
    expected_body = json.dumps(
        {
            "data": [{"text": "hello"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
                "return_multivector": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="hello")


def test_embed_v4_with_return_multivector_true(client):
    _connect(client, ARN_V4)
    expected_body = json.dumps(
        {
            "data": [{"text": "h"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
                "return_multivector": True,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="h", return_multivector=True)


def test_embed_v4_pdf(client):
    _connect(client, ARN_V4)
    expected_body = json.dumps(
        {
            "data": {"pdf": "https://example.com/x.pdf"},
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
                "return_multivector": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(pdf_url="https://example.com/x.pdf")


def test_embed_clip_v2_image_url_uses_url_key(client):
    _connect(client, ARN_CLIP_V2)
    expected_body = json.dumps(
        {
            "data": [{"url": "https://example.com/cat.jpg"}],
            "parameters": {"task": "text-matching", "dimensions": None},
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(image_urls="https://example.com/cat.jpg")


def test_embed_clip_v2_image_bytes_uses_bytes_key(client):
    _connect(client, ARN_CLIP_V2)
    expected_body = json.dumps(
        {
            "data": [{"bytes": "BASE64=="}],
            "parameters": {"task": "text-matching", "dimensions": None},
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(image_bytes="BASE64==")


def test_embed_non_clip_image_url_uses_image_key(client):
    _connect(client, ARN_V4)
    expected_body = json.dumps(
        {
            "data": [{"image": "https://example.com/cat.jpg"}],
            "parameters": {
                "task": "text-matching",
                "dimensions": None,
                "late_chunking": False,
                "return_multivector": False,
            },
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(image_urls="https://example.com/cat.jpg")


def test_embed_v2_no_parameters_block(client):
    _connect(client, ARN_V2)
    expected_body = json.dumps({"data": [{"text": "hello"}]})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="hello")


def test_embed_colbert_query(client):
    _connect(client, ARN_COLBERT)
    expected_body = json.dumps({"data": {"text": "find this"}, "parameters": {"input_type": "query"}})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts="find this", use_colbert=True, input_type=InputType.QUERY)


def test_embed_colbert_document_default(client):
    _connect(client, ARN_COLBERT)
    expected_body = json.dumps(
        {
            "data": [{"text": "a"}, {"text": "b"}],
            "parameters": {"input_type": "document"},
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.embed(texts=["a", "b"], use_colbert=True)


# ----------------------------------------------------------------------------
# rerank
# ----------------------------------------------------------------------------


def test_rerank_no_endpoint_raises(client):
    with pytest.raises(Exception, match="No endpoint connected"):
        client.rerank(["a"], query="q")


def test_rerank_strings_normalized_to_dicts(client):
    _connect(client, ARN_RERANKER)
    expected_body = json.dumps({"data": {"documents": [{"text": "a"}, {"text": "b"}], "query": "q"}})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.rerank(["a", "b"], query="q")


def test_rerank_dicts_passthrough(client):
    _connect(client, ARN_RERANKER)
    docs = [{"text": "a", "meta": 1}, {"text": "b"}]
    expected_body = json.dumps({"data": {"documents": docs, "query": "q"}})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.rerank(docs, query="q")


def test_rerank_top_n_clamped_to_length(client):
    _connect(client, ARN_RERANKER)
    expected_body = json.dumps(
        {
            "data": {
                "documents": [{"text": "a"}, {"text": "b"}],
                "query": "q",
                "top_n": 2,
            }
        }
    )
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.rerank(["a", "b"], query="q", top_n=999)


def test_rerank_reranker_m0_wraps_data_in_list(client):
    _connect(client, ARN_RERANKER_M0)
    inner = {"documents": [{"text": "a"}], "query": "q"}
    expected_body = json.dumps({"data": [inner]})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {
                "Body": _json_streaming(_invoke_response_data([])),
                "ContentType": "application/json",
            },
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        client.rerank(["a"], query="q")


def test_rerank_rejects_unsupported_doc_type(client):
    _connect(client, ARN_RERANKER)
    with pytest.raises(ValueError, match="Unsupported document type"):
        client.rerank([42], query="q")


# ----------------------------------------------------------------------------
# read / read_async
# ----------------------------------------------------------------------------


def test_read_no_endpoint_raises(client):
    with pytest.raises(Exception, match="No endpoint connected"):
        client.read("hi")


@pytest.mark.parametrize(
    "arn,model_name",
    [
        (ARN_READER_0_5B, "reader-lm-0.5b"),
        (ARN_READER_1_5B, "reader-lm-1.5b"),
        (ARN_READER_LM_V2, "ReaderLM-v2"),
    ],
)
def test_read_model_dispatch(client, arn, model_name):
    _connect(client, arn)
    expected_body = json.dumps({"model": model_name, "prompt": "hi", "stream": False})
    response = {"output": "ok"}
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {"Body": _json_streaming(response), "ContentType": "application/json"},
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        assert client.read("hi") == response


def test_read_stream_true_parses_data_lines(client):
    _connect(client, ARN_READER_LM_V2)
    payload = b'data: {"token": "hello"}\ndata: {"token": " world"}\n\n'
    expected_body = json.dumps({"model": "ReaderLM-v2", "prompt": "hi", "stream": True})
    with Stubber(client._sm_runtime_client) as stub:
        stub.add_response(
            "invoke_endpoint",
            {"Body": _streaming(payload), "ContentType": "application/json"},
            {
                "EndpointName": "test-endpoint",
                "ContentType": "application/json",
                "Body": expected_body,
            },
        )
        result = client.read("hi", stream=True)
    assert result == [{"token": "hello"}, {"token": " world"}]


@pytest.mark.parametrize(
    "arn,model_name",
    [
        (ARN_READER_0_5B, "reader-lm-0.5b"),
        (ARN_READER_1_5B, "reader-lm-1.5b"),
        (ARN_READER_LM_V2, "ReaderLM-v2"),
    ],
)
def test_read_async_model_dispatch(client, arn, model_name):
    _connect(client, arn)
    expected_body = json.dumps({"model": model_name, "prompt": "hi"})
    input_s3 = "s3://bucket/prefix/input.json"

    with (
        Stubber(client._sm_runtime_client) as stub,
        mock.patch("jina_sagemaker.client.boto3.client") as boto_client,
    ):
        s3_mock = mock.MagicMock()
        boto_client.return_value = s3_mock
        stub.add_response(
            "invoke_endpoint_async",
            {"OutputLocation": "s3://bucket/output/result.out"},
            {
                "EndpointName": "test-endpoint",
                "InputLocation": input_s3,
                "ContentType": "application/json",
            },
        )

        result = client.read_async("hi", input_s3)

    boto_client.assert_called_once_with("s3", region_name="us-east-1")
    s3_mock.put_object.assert_called_once_with(Bucket="bucket", Key="prefix/input.json", Body=expected_body)
    assert result == {
        "OutputLocation": "s3://bucket/output/result.out",
        "InputLocation": input_s3,
    }


# ----------------------------------------------------------------------------
# delete_endpoint
# ----------------------------------------------------------------------------


def test_delete_endpoint_without_connect_raises(client):
    with pytest.raises(Exception, match="No endpoint connected"):
        client.delete_endpoint()


def test_delete_endpoint_full(client):
    _connect(client, ARN_V3)
    client._endpoint_config_name = "test-endpoint"
    client._model_name = "test-endpoint"

    with Stubber(client._sm_client) as stub:
        stub.add_response("delete_endpoint", {}, {"EndpointName": "test-endpoint"})
        stub.add_response("delete_endpoint_config", {}, {"EndpointConfigName": "test-endpoint"})
        stub.add_response("delete_model", {}, {"ModelName": "test-endpoint"})
        client.delete_endpoint()


def test_delete_endpoint_only_endpoint_set(client):
    _connect(client, ARN_V3)
    # endpoint_config_name and model_name remain None — neither sub-delete should fire.
    with Stubber(client._sm_client) as stub:
        stub.add_response("delete_endpoint", {}, {"EndpointName": "test-endpoint"})
        client.delete_endpoint()


# ----------------------------------------------------------------------------
# PR2 hardening: model dispatch override, region threading, close(),
# create_transform_job wait=False guard.
# ----------------------------------------------------------------------------


def test_resolve_model_detects_each_family():
    from jina_sagemaker.client import _resolve_model

    cases = [
        (ARN_V2, "embeddings-default"),
        (ARN_V3, "embeddings-v3"),
        (ARN_V4, "embeddings-v4"),
        (ARN_CLIP_V2, "clip-v2"),
        (ARN_RERANKER, "reranker"),
        (ARN_RERANKER_M0, "reranker-m0"),
        (ARN_READER_0_5B, "reader-lm"),
        (ARN_READER_1_5B, "reader-lm"),
        (ARN_READER_LM_V2, "reader-lm"),
        (ARN_COLBERT, "colbert"),
    ]
    for arn, family in cases:
        assert _resolve_model(arn).family == family, arn


def test_resolve_model_override_wins_over_arn():
    from jina_sagemaker.client import _resolve_model

    # ARN points at clip-v2 but the customer claims it's actually a v4 endpoint.
    spec = _resolve_model(ARN_CLIP_V2, override="jina-embeddings-v4")
    assert spec.family == "embeddings-v4"


def test_resolve_model_unknown_override_raises():
    from jina_sagemaker.client import _resolve_model

    with pytest.raises(ValueError, match="Unknown model override"):
        _resolve_model(ARN_V3, override="jina-embeddings-v999")


def test_connect_to_endpoint_with_model_override(client):
    with Stubber(client._sm_client) as stub:
        stub.add_response(
            "describe_endpoint",
            _describe_endpoint_response("test-endpoint"),
            {"EndpointName": "test-endpoint"},
        )
        client.connect_to_endpoint("test-endpoint", ARN_CLIP_V2, model="jina-embeddings-v4")
    assert client._model_spec is not None
    assert client._model_spec.family == "embeddings-v4"


def test_init_threads_region_into_sagemaker_session(monkeypatch):
    """The high-level sagemaker SDK must honour Client(region_name=...) too."""
    captured = {}

    def fake_session(*args, **kwargs):
        captured["region_name"] = kwargs.get("region_name")
        return mock.MagicMock()

    monkeypatch.setattr("jina_sagemaker.client.boto3.Session", fake_session)
    Client(region_name="eu-west-1")
    assert captured["region_name"] == "eu-west-1"


def test_init_stores_client_args_for_reuse():
    c = Client(region_name="ap-northeast-1")
    assert c._client_args["region_name"] == "ap-northeast-1"


def test_close_suppresses_attribute_error_on_old_boto3(client, caplog):
    """Older boto3 builds lacked client.close(); we must not re-raise."""
    import logging as _logging

    client._sm_runtime_client.close = mock.MagicMock(
        side_effect=AttributeError("'BotocoreClient' object has no attribute 'close'")
    )
    with caplog.at_level(_logging.INFO, logger="jina_sagemaker.client"):
        client.close()  # must not raise

    assert any("could not be closed" in rec.message for rec in caplog.records)


def test_close_succeeds_on_modern_boto3(client):
    runtime_close = mock.MagicMock()
    sm_close = mock.MagicMock()
    client._sm_runtime_client.close = runtime_close
    client._sm_client.close = sm_close

    client.close()

    runtime_close.assert_called_once_with()
    sm_close.assert_called_once_with()


def test_create_transform_job_rejects_wait_false_with_local_output(client):
    with pytest.raises(ValueError, match="requires an s3:// output_path"):
        client.create_transform_job(
            arn=ARN_V3,
            n_instances=1,
            instance_type="ml.m5.xlarge",
            input_path="s3://bucket/in.csv",
            output_path="/tmp/local-output",
            wait=False,
        )
