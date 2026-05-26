# =============================================================================
# components/loader.py — STEP 1: DOCUMENT LOADER
# =============================================================================
# PURPOSE: Read raw text from files on disk and return them as Python objects.
#
# WHY THIS STEP EXISTS:
# Before doing anything smart, we need to GET the data. The loader is the
# "mouth" of the pipeline — it ingests raw files.
#
# WHAT IT RETURNS:
# A list of Document objects, each containing:
#   - text: the raw string content of the file
#   - metadata: info about the file (name, path, size, etc.)
#
# HOW TO EXTEND LATER:
# - Add PDF support by adding an elif for ".pdf" and using PyMuPDF
# - Add web scraping by adding a URL-based loader
# - Add Google Docs, Notion, etc.
# =============================================================================

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS

# Get a logger for this module.
# NOTE: we do NOT call logging.basicConfig() here — that belongs only in the
# entry point (main.py). Calling basicConfig in a module would override the
# root logger configuration set by whoever imports us.
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# DATA MODEL: What a "Document" looks like in our system
# -----------------------------------------------------------------------------

@dataclass
class Document:
    """
    Represents one loaded document.
    
    A dataclass is like a simple container — think of it as a labeled box.
    @dataclass auto-generates __init__, __repr__, etc. for us.
    
    Attributes:
        text     : The full raw text content of the document
        metadata : A dictionary of info about the document (filename, path, etc.)
                   We store metadata so we can later tell the user WHICH document
                   an answer came from — important for trust and debugging.
    """
    text: str                              # The actual document content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Info about the doc


# -----------------------------------------------------------------------------
# LOADER CLASS
# -----------------------------------------------------------------------------

class DocumentLoader:
    """
    Loads documents from a directory on disk.
    
    Design principle: Keep loaders SIMPLE and FOCUSED.
    This one only handles text files. Each file type gets its own logic.
    """

    def __init__(self, documents_dir: str = DOCUMENTS_DIR):
        """
        Initialize the loader with the directory to scan.
        
        Args:
            documents_dir: Path to folder containing your documents.
                           Defaults to the 'documents/' folder next to config.py
        """
        self.documents_dir = documents_dir
        logger.info(f"DocumentLoader initialized. Looking in: {self.documents_dir}")

    def load(self) -> List[Document]:
        """
        Scan the documents directory and load all supported files.
        
        Returns:
            A list of Document objects, one per file.
            Empty list if no supported files found.
        """
        documents = []  # We'll collect all loaded docs here

        # Check if the documents folder actually exists
        if not os.path.exists(self.documents_dir):
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return documents  # Return empty list — don't crash

        # os.listdir() gives us all filenames in the folder
        # os.path.join() builds the full path: "documents/" + "file.txt" → "documents/file.txt"
        all_files = os.listdir(self.documents_dir)
        logger.info(f"Found {len(all_files)} total files in directory")

        for filename in all_files:
            filepath = os.path.join(self.documents_dir, filename)

            # os.path.splitext splits "report.txt" into ("report", ".txt")
            # We only care about the extension (index [1])
            _, extension = os.path.splitext(filename)

            # Skip files we don't support (images, PDFs for now, etc.)
            if extension.lower() not in SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping unsupported file type: {filename}")
                continue

            # Try to load the file — if it fails, log and move on (don't crash)
            try:
                doc = self._load_text_file(filepath, filename)
                documents.append(doc)
                logger.info(f"Loaded: {filename} ({len(doc.text)} characters)")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_text_file(self, filepath: str, filename: str) -> Document:
        """
        Load a single plain text file.
        
        The underscore prefix (_) is a Python convention meaning "private method"
        — it's used internally, not meant to be called from outside this class.
        
        Args:
            filepath : Full path to the file
            filename : Just the filename (for metadata)
            
        Returns:
            A Document with the file's text and metadata
        """
        # 'utf-8' handles most text including unicode characters
        # 'errors=ignore' skips any unreadable characters instead of crashing
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Build metadata dictionary — this info travels with the chunks later
        # so the user knows WHERE an answer came from
        metadata = {
            "filename": filename,
            "filepath": filepath,
            "file_size_bytes": os.path.getsize(filepath),
            "extension": os.path.splitext(filename)[1],
        }

        return Document(text=text, metadata=metadata)
