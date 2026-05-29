"""Integration tests for the rerank model families."""

from __future__ import annotations


def test_rerank_string_documents(reranker_client):
    result = reranker_client.rerank(
        documents=[
            "Pandas are native to China.",
            "Bamboo is a fast-growing grass.",
            "The Eiffel Tower is in Paris.",
        ],
        query="what do pandas eat",
    )
    assert isinstance(result, list)
    assert len(result) == 3


def test_rerank_with_top_n(reranker_client):
    result = reranker_client.rerank(
        documents=["a", "b", "c", "d"],
        query="anything",
        top_n=2,
    )
    assert len(result) == 2
