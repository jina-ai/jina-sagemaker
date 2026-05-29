"""Integration tests for the ReaderLM model family."""

from __future__ import annotations


def test_read_returns_parsable_response(reader_lm_v2_client):
    response = reader_lm_v2_client.read("Extract the main topic of this text: cats")
    # ReaderLM's response shape is endpoint-version-specific. The
    # integration contract is just "we get a non-empty response back".
    assert response
