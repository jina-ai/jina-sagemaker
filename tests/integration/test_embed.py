"""Integration tests for the embed/* model families."""

from __future__ import annotations

import pytest

from jina_sagemaker import Task


@pytest.mark.usefixtures("v3_client")
class TestEmbeddingsV3:
    def test_embed_single_text(self, v3_client):
        result = v3_client.embed(texts="hello world")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0]["embedding"], list)
        assert len(result[0]["embedding"]) > 0

    def test_embed_text_list(self, v3_client):
        result = v3_client.embed(texts=["foo", "bar", "baz"])
        assert len(result) == 3

    def test_embed_with_task_and_dimensions(self, v3_client):
        result = v3_client.embed(
            texts="search query",
            task_type=Task.RETRIEVAL_QUERY,
            dimensions=128,
        )
        assert len(result[0]["embedding"]) == 128


class TestEmbeddingsV4:
    def test_embed_single_text(self, v4_client):
        result = v4_client.embed(texts="hello world")
        assert len(result) == 1
        assert "embedding" in result[0]

    def test_embed_with_return_multivector(self, v4_client):
        result = v4_client.embed(texts="hello", return_multivector=True)
        # v4 with return_multivector returns a list of vectors per input;
        # the exact shape is endpoint-version-dependent, just verify the
        # response is non-empty.
        assert result


class TestClipV2:
    def test_embed_text(self, clip_v2_client):
        result = clip_v2_client.embed(texts="a photo of a cat")
        assert len(result) == 1

    def test_embed_image_url(self, clip_v2_client):
        result = clip_v2_client.embed(
            image_urls="https://dummyimage.com/224x224/000/fff.jpg&text=test"
        )
        assert len(result) == 1
