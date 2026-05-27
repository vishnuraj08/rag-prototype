# =============================================================================
# tests/test_loader.py — Tests for DocumentLoader
# =============================================================================
# Run with:  pytest tests/test_loader.py -v
# =============================================================================

import os
import pytest
from components.loader import DocumentLoader, Document


class TestDocumentLoader:

    def test_loads_txt_file(self, tmp_dir):
        """A .txt file should be loaded and returned as a Document."""
        path = os.path.join(tmp_dir, "hello.txt")
        with open(path, "w") as f:
            f.write("Hello world!")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Hello world!"
        assert docs[0].metadata["filename"] == "hello.txt"

    def test_loads_md_file(self, tmp_dir):
        """A .md file should also be loaded."""
        path = os.path.join(tmp_dir, "notes.md")
        with open(path, "w") as f:
            f.write("# Heading\nSome content.")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert len(docs) == 1
        assert "Heading" in docs[0].text

    def test_skips_unsupported_extensions(self, tmp_dir):
        """
        .pdf, .png, .csv etc. should be skipped — the loader only handles .txt and .md.
        """
        for name in ["image.png", "data.csv", "archive.zip"]:
            open(os.path.join(tmp_dir, name), "w").close()

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert docs == []

    def test_mixed_files_loads_only_supported(self, tmp_dir):
        """With a mix of file types, only .txt and .md should come back."""
        open(os.path.join(tmp_dir, "ignore.png"), "w").close()
        with open(os.path.join(tmp_dir, "keep.txt"), "w") as f:
            f.write("Keep me.")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "keep.txt"

    def test_empty_directory(self, tmp_dir):
        """An empty folder should return an empty list."""
        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert docs == []

    def test_nonexistent_directory(self, tmp_dir):
        """A directory that doesn't exist should return empty list, not crash."""
        loader = DocumentLoader(documents_dir=os.path.join(tmp_dir, "does_not_exist"))
        docs = loader.load()
        assert docs == []

    def test_metadata_contains_filename(self, tmp_dir):
        """Metadata should include filename, filepath, extension, and size."""
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Test content")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        meta = docs[0].metadata

        assert meta["filename"] == "test.txt"
        assert meta["extension"] == ".txt"
        assert "filepath" in meta
        assert meta["file_size_bytes"] > 0

    def test_returns_document_objects(self, tmp_dir):
        """Returned items should be Document instances."""
        with open(os.path.join(tmp_dir, "doc.txt"), "w") as f:
            f.write("content")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert all(isinstance(d, Document) for d in docs)

    def test_multiple_files_all_loaded(self, tmp_dir):
        """Multiple .txt files should all be loaded."""
        for i in range(3):
            with open(os.path.join(tmp_dir, f"file{i}.txt"), "w") as f:
                f.write(f"Content of file {i}")

        loader = DocumentLoader(documents_dir=tmp_dir)
        docs = loader.load()
        assert len(docs) == 3
