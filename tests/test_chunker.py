# =============================================================================
# tests/test_chunker.py — Tests for TextChunker
# =============================================================================
# Run with:  pytest tests/test_chunker.py -v
#
# WHAT WE'RE TESTING:
#   - Does chunking produce the right number of chunks?
#   - Is overlap working correctly?
#   - Are edge cases (empty text, short text) handled?
#   - Is metadata correctly copied from Document to Chunk?
#   - Are chunk IDs unique?
# =============================================================================

import pytest
from components.loader import Document
from components.chunker import TextChunker, Chunk


class TestTextChunkerInit:
    """Tests for the __init__ method — does it set up correctly?"""

    def test_step_calculation(self):
        """step = chunk_size - chunk_overlap. This is how the window advances."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.step == 450

    def test_overlap_must_be_less_than_size(self):
        """
        If overlap >= size, the window never moves forward → infinite loop.
        The chunker should raise ValueError to prevent this.
        """
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)   # equal → error

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)   # greater → error

    def test_valid_config_does_not_raise(self):
        """A valid config should initialise without errors."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20


class TestChunkDocuments:
    """Tests for chunk_documents() — the main public method."""

    def test_returns_list_of_chunks(self, chunker, sample_document):
        """Result should be a list of Chunk objects."""
        chunks = chunker.chunk_documents([sample_document])
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_document_list(self, chunker):
        """No documents in → no chunks out."""
        chunks = chunker.chunk_documents([])
        assert chunks == []

    def test_multiple_documents_combined(self, chunker, multi_document):
        """Chunks from all documents should be returned in a single flat list."""
        chunks = chunker.chunk_documents(multi_document)
        # Both docs should contribute chunks — result is one combined list
        assert len(chunks) > 0
        # Source filenames should appear in chunk metadata
        filenames = {c.metadata["filename"] for c in chunks}
        assert "doc1.txt" in filenames
        assert "doc2.txt" in filenames

    def test_chunk_ids_are_unique(self, chunker, multi_document):
        """Every chunk across all documents must have a unique ID."""
        chunks = chunker.chunk_documents(multi_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found!"

    def test_chunk_ids_include_filename(self, chunker, sample_document):
        """Chunk IDs should be like 'filename_0', 'filename_1', ..."""
        chunks = chunker.chunk_documents([sample_document])
        for chunk in chunks:
            assert chunk.chunk_id.startswith("test_doc.txt_")


class TestChunkSingleDocument:
    """Tests for _chunk_document() behaviour on a single document."""

    def test_text_shorter_than_chunk_size(self):
        """
        If the document is shorter than CHUNK_SIZE, it should produce exactly 1 chunk
        containing the full text.
        """
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        short_text = "This is a short document."
        doc = Document(text=short_text, metadata={"filename": "short.txt"})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].text == short_text

    def test_exact_chunk_size_boundary(self):
        """A text exactly equal to chunk_size should produce exactly 1 chunk."""
        chunk_size = 100
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=10)
        text = "A" * chunk_size
        doc = Document(text=text, metadata={"filename": "exact.txt"})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) == 1

    def test_chunks_respect_chunk_size(self):
        """No chunk should be longer than chunk_size characters."""
        chunk_size = 100
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=20)
        long_text = "Hello world. " * 50   # ~650 characters
        doc = Document(text=long_text, metadata={"filename": "long.txt"})
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size

    def test_overlap_means_adjacent_chunks_share_text(self):
        """
        With overlap=20, the end of chunk[0] should appear at the start of chunk[1].
        This is the whole point of overlap — no sentence gets cut off at a boundary.
        """
        chunker = TextChunker(chunk_size=50, chunk_overlap=20)
        # 'A'*50 + 'B'*50 = 100 chars total
        text = "A" * 50 + "B" * 50
        doc = Document(text=text, metadata={"filename": "overlap_test.txt"})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) >= 2
        # The end of chunk[0] should appear at the start of chunk[1]
        end_of_chunk0 = chunks[0].text[-20:]
        start_of_chunk1 = chunks[1].text[:20]
        assert end_of_chunk0 == start_of_chunk1

    def test_whitespace_only_chunks_are_skipped(self):
        """Chunks that are only whitespace should not be included in results."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "Hello     " + " " * 10 + "World"
        doc = Document(text=text, metadata={"filename": "whitespace.txt"})
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_empty_text_produces_no_chunks(self):
        """An empty document should produce no chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(text="", metadata={"filename": "empty.txt"})
        chunks = chunker.chunk_documents([doc])
        assert chunks == []

    def test_metadata_inherited_from_document(self, chunker, sample_document):
        """Each chunk should carry the parent document's metadata fields."""
        chunks = chunker.chunk_documents([sample_document])
        for chunk in chunks:
            # Document metadata should be present in every chunk
            assert chunk.metadata["filename"] == "test_doc.txt"
            assert chunk.metadata["extension"] == ".txt"

    def test_chunk_metadata_has_position_info(self, chunker, sample_document):
        """Chunks should also contain their own position metadata."""
        chunks = chunker.chunk_documents([sample_document])
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_start_char" in chunk.metadata
            assert "chunk_end_char" in chunk.metadata

    def test_chunk_indices_are_sequential(self, chunker, sample_document):
        """chunk_index should be 0, 1, 2, ... within a document."""
        chunks = chunker.chunk_documents([sample_document])
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_coverage(self):
        """
        The union of all chunks should cover the entire original text.
        No character from the original should be missing.
        (With overlap, characters near boundaries appear in multiple chunks — that's fine.)
        """
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog. " * 5
        doc = Document(text=text, metadata={"filename": "coverage.txt"})
        chunks = chunker.chunk_documents([doc])
        # Verify first chunk starts at beginning and last chunk ends at the end
        assert chunks[0].metadata["chunk_start_char"] == 0
        assert chunks[-1].metadata["chunk_end_char"] == len(text)
