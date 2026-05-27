# =============================================================================
# tests/test_pipeline.py — Integration tests for RAGPipeline
# =============================================================================
# Run with:  pytest tests/test_pipeline.py -v
#
# WHAT IS AN INTEGRATION TEST?
#   Unit tests test one component in isolation (with everything else mocked).
#   Integration tests test how components work TOGETHER.
#
#   Here we test the full pipeline.query() flow:
#     question → embed → search → retrieve → generate → answer
#
#   We still mock the Anthropic API (no real calls) and redirect storage
#   to a temp directory (no polluting the real storage/ folder).
# =============================================================================

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pipeline import RAGPipeline
from components.loader import Document
from components.chunker import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_api_response(text="Mocked answer."):
    """Fake Anthropic API response."""
    mock_content = MagicMock()
    mock_content.text = text
    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage = mock_usage
    return mock_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_with_mocked_api(tmp_dir, monkeypatch):
    """
    A fully initialised RAGPipeline where:
    - The Anthropic API is mocked (no real calls)
    - Storage paths point to a temp directory (no real files written)
    - The embedding model is the real model (tests actual vector math)

    This is a realistic integration test that exercises every component
    except the external API.
    """
    import config
    import components.vector_store as vs_module

    vec_path = os.path.join(tmp_dir, "vectors.npz")
    chunk_path = os.path.join(tmp_dir, "chunks.json")
    monkeypatch.setattr(vs_module, "VECTOR_INDEX_PATH", vec_path)
    monkeypatch.setattr(vs_module, "CHUNKS_INDEX_PATH", chunk_path)
    monkeypatch.setattr(config, "VECTOR_INDEX_PATH", vec_path)
    monkeypatch.setattr(config, "CHUNKS_INDEX_PATH", chunk_path)

    with patch("components.generator.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = make_fake_api_response()
        pipeline = RAGPipeline()
        yield pipeline, MockAnthropic


@pytest.fixture
def indexed_pipeline(pipeline_with_mocked_api, tmp_dir, monkeypatch):
    """
    A pipeline that has already been indexed with a small set of documents.
    Use this for query() tests so you don't have to set up indexing in each test.
    """
    pipeline, MockAnthropic = pipeline_with_mocked_api

    # Inject 3 small documents directly into the pipeline
    # (bypasses disk I/O — we just load the vector store directly)
    from components.chunker import TextChunker
    from components.embedder import TextEmbedder

    docs = [
        Document(text="Python is a programming language created by Guido van Rossum.",
                 metadata={"filename": "python.txt", "extension": ".txt"}),
        Document(text="Machine learning allows computers to learn from data.",
                 metadata={"filename": "ml.txt", "extension": ".txt"}),
        Document(text="Docker packages your application into a container.",
                 metadata={"filename": "docker.txt", "extension": ".txt"}),
    ]
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_documents(docs)
    embeddings = pipeline.embedder.embed_chunks(chunks)
    pipeline.vector_store.add(chunks, embeddings)

    yield pipeline, MockAnthropic


# ---------------------------------------------------------------------------
# Tests: query()
# ---------------------------------------------------------------------------

class TestPipelineQuery:

    def test_query_returns_dict(self, indexed_pipeline):
        """query() should return a dict with 'answer', 'sources', 'question' keys."""
        pipeline, _ = indexed_pipeline
        result = pipeline.query("What is Python?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "question" in result

    def test_query_echoes_question(self, indexed_pipeline):
        """The returned dict should contain the original question."""
        pipeline, _ = indexed_pipeline
        q = "What is Python?"
        result = pipeline.query(q)
        assert result["question"] == q

    def test_query_returns_sources(self, indexed_pipeline):
        """Sources should be a list (possibly empty, but usually populated)."""
        pipeline, _ = indexed_pipeline
        result = pipeline.query("What is Python?")
        assert isinstance(result["sources"], list)

    def test_query_sources_have_correct_shape(self, indexed_pipeline):
        """Each source should contain chunk_id, source_file, similarity_score, text_preview."""
        pipeline, _ = indexed_pipeline
        result = pipeline.query("What is Python?")
        for source in result["sources"]:
            assert "chunk_id" in source
            assert "source_file" in source
            assert "similarity_score" in source
            assert "text_preview" in source

    def test_query_calls_api_once(self, indexed_pipeline):
        """The Anthropic API should be called exactly once per query."""
        pipeline, MockAnthropic = indexed_pipeline
        MockAnthropic.return_value.messages.create.reset_mock()
        pipeline.query("What is machine learning?")
        assert MockAnthropic.return_value.messages.create.call_count == 1

    def test_empty_question_returns_error(self, indexed_pipeline):
        """An empty question should return an error message without hitting the API."""
        pipeline, MockAnthropic = indexed_pipeline
        MockAnthropic.return_value.messages.create.reset_mock()
        result = pipeline.query("")
        assert "non-empty" in result["answer"].lower() or "please" in result["answer"].lower()
        MockAnthropic.return_value.messages.create.assert_not_called()

    def test_whitespace_question_rejected(self, indexed_pipeline):
        """A whitespace-only question should also be rejected."""
        pipeline, MockAnthropic = indexed_pipeline
        MockAnthropic.return_value.messages.create.reset_mock()
        result = pipeline.query("    ")
        MockAnthropic.return_value.messages.create.assert_not_called()

    def test_query_no_index_returns_graceful_message(self, pipeline_with_mocked_api):
        """
        If no index has been built yet, query() should return a friendly message
        telling the user to run --index first.
        """
        pipeline, _ = pipeline_with_mocked_api
        # vector_store is empty — no index loaded
        result = pipeline.query("What is Python?")
        assert "index" in result["answer"].lower() or "no documents" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Tests: index()
# ---------------------------------------------------------------------------

class TestPipelineIndex:

    def test_index_with_no_documents(self, pipeline_with_mocked_api, monkeypatch, tmp_dir):
        """
        If documents/ is empty, index() should log an error and return gracefully
        without crashing.
        """
        pipeline, _ = pipeline_with_mocked_api

        # Point the loader at an empty temp directory
        import config
        monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_dir)
        pipeline.loader.documents_dir = tmp_dir

        # Should not raise — just logs "no documents found"
        pipeline.index()
        assert pipeline.vector_store.vectors is None   # nothing was indexed
