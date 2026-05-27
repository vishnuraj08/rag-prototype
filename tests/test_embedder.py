# =============================================================================
# tests/test_embedder.py — Tests for TextEmbedder
# =============================================================================
# Run with:  pytest tests/test_embedder.py -v
#
# WHY "SLOW" TESTS?
#   The embedder loads the all-MiniLM-L6-v2 model (~80MB) into memory.
#   This takes 2–10 seconds on first run (after that it's cached and faster).
#   We mark these tests with @pytest.mark.slow so you can skip them when
#   you want fast feedback during development:
#
#       pytest tests/ -v -m "not slow"   → skips embedder tests
#       pytest tests/ -v                 → runs everything
#
# WHAT WE'RE TESTING:
#   - Does embed_chunks() return the right numpy shape?
#   - Are the embeddings unit-normalised (required for cosine sim = dot product)?
#   - Does embed_query() return a 1D vector of the right size?
#   - Are empty inputs handled gracefully?
#   - Does using the same model for chunks AND query put them in the same space?
# =============================================================================

import pytest
import numpy as np
from components.chunker import Chunk
from components.embedder import TextEmbedder


# Shared embedder instance — loaded once for the whole test session
# This avoids reloading the model for every individual test function.
@pytest.fixture(scope="module")
def embedder():
    """
    A real TextEmbedder (loads the actual model).
    scope="module" means: create once, reuse for all tests in this file.
    Without this, the model would reload for every test — very slow!
    """
    return TextEmbedder()


# Helper: make a simple Chunk
def make_chunk(text, idx=0):
    return Chunk(text=text, metadata={"filename": "test.txt"}, chunk_id=f"test_{idx}")


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.slow
class TestEmbedChunks:
    """Tests for embed_chunks() — the main indexing method."""

    def test_returns_numpy_array(self, embedder):
        """embed_chunks() should return a numpy ndarray."""
        chunks = [make_chunk("Hello world", 0)]
        result = embedder.embed_chunks(chunks)
        assert isinstance(result, np.ndarray)

    def test_output_shape_single_chunk(self, embedder):
        """
        1 chunk → shape (1, 384).
        The 384 is the embedding dimension of all-MiniLM-L6-v2.
        """
        chunks = [make_chunk("Python is a programming language.", 0)]
        result = embedder.embed_chunks(chunks)
        assert result.shape == (1, 384)

    def test_output_shape_multiple_chunks(self, embedder):
        """N chunks → shape (N, 384). One row per chunk."""
        texts = ["Python", "Machine learning", "Docker", "Numpy", "pytest"]
        chunks = [make_chunk(t, i) for i, t in enumerate(texts)]
        result = embedder.embed_chunks(chunks)
        assert result.shape == (len(texts), 384)

    def test_embeddings_are_unit_normalised(self, embedder):
        """
        We pass normalize_embeddings=True to the model, so every row should
        have L2 norm ≈ 1.0. This is what makes cosine similarity = dot product.

        L2 norm = sqrt(sum of squares of all values in the vector).
        For a unit vector: sqrt(0.21² + (-0.54)² + ...) = 1.0
        """
        chunks = [make_chunk("Test sentence.", i) for i in range(3)]
        result = embedder.embed_chunks(chunks)
        norms = np.linalg.norm(result, axis=1)   # norm of each row
        # All norms should be very close to 1.0 (floating point imprecision)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                   err_msg="Embeddings are not unit-normalised!")

    def test_empty_chunks_returns_empty_array(self, embedder):
        """
        No chunks in → empty array out (not a crash).
        Shape should be (0, 384) — zero rows, correct columns.
        """
        result = embedder.embed_chunks([])
        assert result.shape[0] == 0   # 0 embeddings
        assert result.shape[1] == 384  # correct dimensions

    def test_different_texts_produce_different_vectors(self, embedder):
        """
        Two completely different texts should produce different vectors.
        If all texts produced the same vector, search would be useless.
        """
        chunk_a = make_chunk("Python programming language", 0)
        chunk_b = make_chunk("The Eiffel Tower is in Paris", 1)
        result = embedder.embed_chunks([chunk_a, chunk_b])
        # The two rows should NOT be identical
        assert not np.allclose(result[0], result[1]), \
            "Different texts produced identical embeddings!"

    def test_similar_texts_produce_similar_vectors(self, embedder):
        """
        Semantically similar texts should produce similar vectors.
        Their dot product (= cosine similarity for unit vectors) should be high.

        This is the WHOLE POINT of semantic embeddings — unlike keyword search,
        "Python programming" and "coding in Python" should be considered similar
        even though they use different words.
        """
        chunk_a = make_chunk("Python is a programming language", 0)
        chunk_b = make_chunk("Python is used for coding", 1)
        result = embedder.embed_chunks([chunk_a, chunk_b])

        similarity = float(np.dot(result[0], result[1]))  # dot product of unit vectors = cosine sim
        assert similarity > 0.7, f"Similar texts have low similarity: {similarity:.3f}"


@pytest.mark.slow
class TestEmbedQuery:
    """Tests for embed_query() — used at query time."""

    def test_returns_numpy_array(self, embedder):
        """embed_query() should return a numpy array."""
        result = embedder.embed_query("What is Python?")
        assert isinstance(result, np.ndarray)

    def test_output_is_1d_vector(self, embedder):
        """A single query → 1D vector of shape (384,)."""
        result = embedder.embed_query("What is Python?")
        assert result.shape == (384,)   # 1D, not 2D like embed_chunks returns

    def test_query_is_unit_normalised(self, embedder):
        """embed_query() also uses normalize_embeddings=True, so norm ≈ 1.0."""
        result = embedder.embed_query("What is machine learning?")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Query vector norm is {norm:.6f}, expected 1.0"

    def test_query_and_chunk_in_same_space(self, embedder):
        """
        A query about Python and a chunk about Python should have high similarity.
        This confirms we're using the same model for both — essential for RAG to work.
        If query and chunks were embedded with different models, search would fail.
        """
        chunk = make_chunk("Python is a high-level programming language.", 0)
        chunk_embedding = embedder.embed_chunks([chunk])[0]   # 1D vector (row 0)
        query_embedding = embedder.embed_query("What is Python?")

        # dot product of two unit vectors = cosine similarity
        similarity = float(np.dot(query_embedding, chunk_embedding))
        assert similarity > 0.6, \
            f"Query and relevant chunk have low similarity: {similarity:.3f}. " \
            f"Are you using the same model for both?"
