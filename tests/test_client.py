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
    client._endpoint_name = endpoint_name
    client._variant_name = "AllTraffic"
    client._resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    client._arn = arn


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
    response_body = _json_streaming(
        _invoke_response_data([{"index": 0, "embedding": [0.1]}])
    )
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
    expected_body = json.dumps(
        {"data": {"text": "find this"}, "parameters": {"input_type": "query"}}
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
    expected_body = json.dumps(
        {"data": {"documents": [{"text": "a"}, {"text": "b"}], "query": "q"}}
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

    boto_client.assert_called_once_with("s3")
    s3_mock.put_object.assert_called_once_with(
        Bucket="bucket", Key="prefix/input.json", Body=expected_body
    )
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
        stub.add_response(
            "delete_endpoint_config", {}, {"EndpointConfigName": "test-endpoint"}
        )
        stub.add_response("delete_model", {}, {"ModelName": "test-endpoint"})
        client.delete_endpoint()


def test_delete_endpoint_only_endpoint_set(client):
    _connect(client, ARN_V3)
    # endpoint_config_name and model_name remain None — neither sub-delete should fire.
    with Stubber(client._sm_client) as stub:
        stub.add_response("delete_endpoint", {}, {"EndpointName": "test-endpoint"})
        client.delete_endpoint()
