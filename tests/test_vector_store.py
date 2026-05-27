# =============================================================================
# tests/test_vector_store.py — Tests for VectorStore
# =============================================================================
# Run with:  pytest tests/test_vector_store.py -v
#
# WHAT WE'RE TESTING:
#   - Does add() store chunks and vectors correctly?
#   - Does search() return results sorted by similarity?
#   - Does the MIN_SIMILARITY_SCORE threshold filter work?
#   - Does save() + load() round-trip correctly?
#   - Are edge cases (empty store, fewer chunks than top_k) handled?
# =============================================================================

import os
import pytest
import numpy as np

from components.chunker import Chunk
from components.vector_store import VectorStore


# Helper: create unit-normalised random vectors (required for cosine sim = dot product)
def make_unit_vectors(n, dim=384, seed=0):
    np.random.seed(seed)
    raw = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# Helper: create a simple Chunk
def make_chunk(text="test", filename="test.txt", idx=0):
    return Chunk(text=text, metadata={"filename": filename}, chunk_id=f"{filename}_{idx}")


class TestVectorStoreAdd:
    """Tests for the add() method."""

    def test_add_populates_store(self, empty_vector_store):
        """After add(), vectors and chunks should be set."""
        store = empty_vector_store
        chunks = [make_chunk(idx=i) for i in range(3)]
        vectors = make_unit_vectors(3)
        store.add(chunks, vectors)
        assert store.vectors is not None
        assert len(store.chunks) == 3

    def test_add_mismatched_sizes_raises(self, empty_vector_store):
        """
        If you pass 3 chunks but 5 vectors, something is wrong — should raise ValueError.
        This is a safety check to prevent silent data corruption.
        """
        store = empty_vector_store
        chunks = [make_chunk(idx=i) for i in range(3)]
        vectors = make_unit_vectors(5)   # 5 vectors, 3 chunks → mismatch
        with pytest.raises(ValueError):
            store.add(chunks, vectors)

    def test_add_stores_correct_vector_shape(self, empty_vector_store):
        """Stored vectors should have shape (n_chunks, embedding_dim)."""
        store = empty_vector_store
        n, dim = 4, 384
        chunks = [make_chunk(idx=i) for i in range(n)]
        vectors = make_unit_vectors(n, dim)
        store.add(chunks, vectors)
        assert store.vectors.shape == (n, dim)


class TestVectorStoreSearch:
    """Tests for the search() method — the most important part of the whole system."""

    def test_search_returns_list(self, populated_vector_store):
        """search() should always return a list."""
        store, chunks, vectors = populated_vector_store
        query = make_unit_vectors(1)[0]   # single query vector
        results = store.search(query, top_k=3)
        assert isinstance(results, list)

    def test_search_returns_chunk_score_tuples(self, populated_vector_store):
        """Each result should be a (Chunk, float) tuple."""
        store, chunks, vectors = populated_vector_store
        query = make_unit_vectors(1)[0]
        results = store.search(query, top_k=2)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            chunk, score = item
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_search_results_sorted_descending(self, populated_vector_store):
        """Results must be sorted highest similarity first."""
        store, chunks, vectors = populated_vector_store
        query = make_unit_vectors(1)[0]
        results = store.search(query, top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results are not sorted by score!"

    def test_search_respects_top_k(self, populated_vector_store):
        """search() should return at most top_k results."""
        store, chunks, vectors = populated_vector_store
        query = make_unit_vectors(1)[0]
        results = store.search(query, top_k=2)
        assert len(results) <= 2

    def test_search_fewer_chunks_than_top_k(self, empty_vector_store):
        """
        If the store has 2 chunks but top_k=5, return only 2 (don't crash).
        """
        store = empty_vector_store
        chunks = [make_chunk(idx=i) for i in range(2)]
        vectors = make_unit_vectors(2)
        store.add(chunks, vectors)
        query = make_unit_vectors(1)[0]
        results = store.search(query, top_k=5)
        assert len(results) <= 2

    def test_search_empty_store_returns_empty(self, empty_vector_store):
        """Searching an empty store should return an empty list, not crash."""
        query = make_unit_vectors(1)[0]
        results = empty_vector_store.search(query, top_k=3)
        assert results == []

    def test_search_finds_identical_vector(self, empty_vector_store):
        """
        If we search with a vector that exactly matches a stored vector,
        that chunk should come back as the top result with score ≈ 1.0.
        (Dot product of two identical unit vectors = 1.0)
        """
        store = empty_vector_store
        chunks = [make_chunk("exact match", idx=0), make_chunk("unrelated", idx=1)]
        vectors = make_unit_vectors(2)
        store.add(chunks, vectors)

        # Use the first stored vector as the query — it should be the top hit
        query = vectors[0].copy()
        results = store.search(query, top_k=1)
        assert len(results) == 1
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_id == "test.txt_0"
        assert top_score > 0.99   # should be very close to 1.0

    def test_search_filters_by_min_similarity(self, empty_vector_store, monkeypatch):
        """
        Chunks with similarity below MIN_SIMILARITY_SCORE should be excluded.
        We monkeypatch the config value to test the filtering logic.
        """
        import components.vector_store as vs_module
        monkeypatch.setattr(vs_module, "MIN_SIMILARITY_SCORE", 0.99)

        store = empty_vector_store
        chunks = [make_chunk(idx=i) for i in range(3)]
        vectors = make_unit_vectors(3, seed=99)
        store.add(chunks, vectors)

        # A random query is unlikely to be very similar to any chunk
        query = make_unit_vectors(1, seed=77)[0]
        results = store.search(query, top_k=3)
        # With threshold=0.99, most random vectors won't qualify
        for _, score in results:
            assert score >= 0.99


class TestVectorStorePersistence:
    """Tests for save() and load() — make sure the index survives to disk and back."""

    def test_save_creates_files(self, empty_vector_store, tmp_dir, monkeypatch):
        """
        After save(), both files (vectors.npz and chunks.json) should exist on disk.
        We monkeypatch the config paths to use a temp directory so we don't pollute
        the real storage/ folder.
        """
        import config
        vec_path = os.path.join(tmp_dir, "vectors.npz")
        chunk_path = os.path.join(tmp_dir, "chunks.json")
        monkeypatch.setattr(config, "VECTOR_INDEX_PATH", vec_path)
        monkeypatch.setattr(config, "CHUNKS_INDEX_PATH", chunk_path)
        # Re-import constants that were already bound
        import components.vector_store as vs_module
        monkeypatch.setattr(vs_module, "VECTOR_INDEX_PATH", vec_path)
        monkeypatch.setattr(vs_module, "CHUNKS_INDEX_PATH", chunk_path)

        store = empty_vector_store
        chunks = [make_chunk(idx=i) for i in range(2)]
        vectors = make_unit_vectors(2)
        store.add(chunks, vectors)
        store.save()

        assert os.path.exists(vec_path), "vectors.npz was not created"
        assert os.path.exists(chunk_path), "chunks.json was not created"

    def test_load_restores_data(self, empty_vector_store, tmp_dir, monkeypatch):
        """
        After save() + load(), the restored store should have the same
        vectors shape and same number of chunks.
        """
        import config
        import components.vector_store as vs_module
        vec_path = os.path.join(tmp_dir, "vectors.npz")
        chunk_path = os.path.join(tmp_dir, "chunks.json")
        monkeypatch.setattr(vs_module, "VECTOR_INDEX_PATH", vec_path)
        monkeypatch.setattr(vs_module, "CHUNKS_INDEX_PATH", chunk_path)

        # Save
        n = 4
        store = empty_vector_store
        chunks = [make_chunk(f"chunk {i}", idx=i) for i in range(n)]
        vectors = make_unit_vectors(n)
        store.add(chunks, vectors)
        store.save()

        # Load into a brand new store
        new_store = VectorStore()
        monkeypatch.setattr(vs_module, "VECTOR_INDEX_PATH", vec_path)
        monkeypatch.setattr(vs_module, "CHUNKS_INDEX_PATH", chunk_path)
        result = new_store.load()

        assert result is True, "load() should return True on success"
        assert new_store.vectors.shape == (n, 384)
        assert len(new_store.chunks) == n
        assert new_store.chunks[0].text == "chunk 0"

    def test_load_returns_false_when_no_files(self, tmp_dir, monkeypatch):
        """load() should return False (not crash) when no saved index exists."""
        import components.vector_store as vs_module
        monkeypatch.setattr(vs_module, "VECTOR_INDEX_PATH", os.path.join(tmp_dir, "nope.npz"))
        monkeypatch.setattr(vs_module, "CHUNKS_INDEX_PATH", os.path.join(tmp_dir, "nope.json"))
        store = VectorStore()
        result = store.load()
        assert result is False
