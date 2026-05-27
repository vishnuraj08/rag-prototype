# =============================================================================
# tests/conftest.py — Shared test fixtures
# =============================================================================
# conftest.py is a special pytest file — fixtures defined here are automatically
# available to ALL test files in this directory without importing them.
#
# A "fixture" is a reusable piece of test setup. Instead of repeating
#   chunker = TextChunker()
# in every test function, you define it once here and pytest injects it.
# =============================================================================

import os
import tempfile
import pytest
import numpy as np

from components.loader import Document
from components.chunker import TextChunker, Chunk
from components.vector_store import VectorStore


# -----------------------------------------------------------------------------
# Text / Document fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_text():
    """A plain paragraph of text — used across multiple tests."""
    return (
        "Python is a high-level programming language known for its simplicity. "
        "It was created by Guido van Rossum in 1991. Python supports multiple "
        "programming paradigms including procedural, object-oriented, and functional. "
        "It is widely used in web development, data science, artificial intelligence, "
        "and automation. The language emphasises code readability and clean syntax."
    )


@pytest.fixture
def sample_document(sample_text):
    """A Document object wrapping sample_text — mimics what DocumentLoader produces."""
    return Document(
        text=sample_text,
        metadata={
            "filename": "test_doc.txt",
            "filepath": "/fake/path/test_doc.txt",
            "file_size_bytes": len(sample_text),
            "extension": ".txt",
        }
    )


@pytest.fixture
def multi_document(sample_text):
    """Two Document objects — for testing multi-document processing."""
    doc1 = Document(text=sample_text, metadata={"filename": "doc1.txt", "extension": ".txt"})
    doc2 = Document(text="Machine learning is a subset of artificial intelligence. "
                         "It allows systems to learn from data without being explicitly programmed.",
                    metadata={"filename": "doc2.txt", "extension": ".txt"})
    return [doc1, doc2]


# -----------------------------------------------------------------------------
# Chunker fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def chunker():
    """A TextChunker with small sizes so we can test chunking behaviour easily."""
    return TextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def default_chunker():
    """A TextChunker with production defaults from config.py."""
    return TextChunker()


# -----------------------------------------------------------------------------
# Vector store fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def empty_vector_store():
    """A fresh, empty VectorStore."""
    return VectorStore()


@pytest.fixture
def populated_vector_store():
    """
    A VectorStore pre-loaded with 5 fake chunks and matching fake vectors.
    Uses random unit-normalised vectors so cosine similarity works correctly.
    """
    store = VectorStore()
    dim = 384   # must match all-MiniLM-L6-v2 output dimension

    # Create 5 fake Chunk objects
    chunks = [
        Chunk(text=f"This is chunk number {i}.", metadata={"filename": "fake.txt"}, chunk_id=f"fake_{i}")
        for i in range(5)
    ]

    # Create 5 random unit-normalised vectors (so cosine sim = dot product)
    np.random.seed(42)   # seed makes tests deterministic
    raw = np.random.randn(5, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms   # normalise to unit length

    store.add(chunks, vectors)
    return store, chunks, vectors


# -----------------------------------------------------------------------------
# Temporary directory fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """
    A real temporary directory that is cleaned up after each test.
    Use this whenever a test needs to read/write actual files.
    """
    with tempfile.TemporaryDirectory() as d:
        yield d
