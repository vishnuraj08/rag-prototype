# =============================================================================
# components/chunker.py — STEP 2: TEXT CHUNKER
# =============================================================================
# PURPOSE: Split large documents into smaller, overlapping pieces called "chunks".
#
# WHY THIS STEP EXISTS:
# Problem 1 — LLMs have token limits: You can't feed a 100-page PDF into
#   Claude in one shot. We need to break it into small digestible pieces.
#
# Problem 2 — Precision: Even if the LLM could handle 100 pages, most of it
#   would be irrelevant to the query. We want to feed ONLY the relevant parts.
#
# THE OVERLAP TRICK:
# If we split at exactly every 500 chars, a sentence might get cut in half.
# Chunk 1: "The capital of France is"   ← incomplete!
# Chunk 2: "Paris which is also..."     ← missing context!
#
# With overlap (e.g. 50 chars), chunk 2 starts 50 chars before chunk 1 ended:
# Chunk 1: "The capital of France is"
# Chunk 2: "France is Paris which is also..."  ← context preserved!
#
# WHAT IT RETURNS:
# A list of Chunk objects, each containing:
#   - text: the chunk text
#   - metadata: inherited from the parent document + chunk-specific info
#   - chunk_id: unique identifier
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from components.loader import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# DATA MODEL: What a "Chunk" looks like in our system
# -----------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    Represents one piece of a document, ready to be embedded.
    
    Attributes:
        text       : The text content of this chunk
        metadata   : Document metadata + chunk-specific info (position, etc.)
        chunk_id   : Unique string ID like "filename_0", "filename_1", etc.
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""


# -----------------------------------------------------------------------------
# CHUNKER CLASS
# -----------------------------------------------------------------------------

class TextChunker:
    """
    Splits Document objects into smaller Chunk objects using a sliding window.
    
    The "sliding window" approach:
    - Start at position 0
    - Take CHUNK_SIZE characters → that's chunk 1
    - Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) characters
    - Take CHUNK_SIZE characters again → that's chunk 2
    - Repeat until end of document
    
    Visual example with CHUNK_SIZE=10, CHUNK_OVERLAP=3:
    Text: "ABCDEFGHIJKLMNOPQRST"
    Chunk 1: "ABCDEFGHIJ"         (position 0..10)
    Chunk 2: "HIJKLMNOPQ"         (position 7..17)  ← 3-char overlap with chunk 1
    Chunk 3: "NOPQRST"            (position 14..end)
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Args:
            chunk_size    : Max characters per chunk
            chunk_overlap : How many characters to repeat between consecutive chunks
        """
        # Validate that overlap is smaller than chunk size — otherwise the
        # sliding window would never move forward (infinite loop)
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # The "step" is how far we advance the window each iteration
        # If size=500, overlap=50 → step=450 (we move 450 chars forward each time)
        self.step = chunk_size - chunk_overlap

        logger.info(
            f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}, step={self.step}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Process a list of documents and return all their chunks combined.
        
        Args:
            documents: List of Document objects from the loader
            
        Returns:
            A flat list of all Chunk objects across all documents
        """
        all_chunks = []

        for document in documents:
            # Chunk each document individually
            doc_chunks = self._chunk_document(document)
            all_chunks.extend(doc_chunks)  # extend() adds all items to the list
            logger.info(
                f"Document '{document.metadata.get('filename', 'unknown')}' "
                f"→ {len(doc_chunks)} chunks"
            )

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks using the sliding window approach.
        
        Args:
            document: A single Document object
            
        Returns:
            List of Chunk objects for this document
        """
        chunks = []
        text = document.text

        # Get the filename to use as part of the chunk ID
        # dict.get() is safe — returns "unknown" if key doesn't exist
        filename = document.metadata.get("filename", "unknown")

        # --- The core sliding window loop ---
        # start = current position in the text
        # We loop while there's still text to process
        chunk_index = 0  # Counter for chunk numbering within this document

        start = 0
        while start < len(text):
            # Calculate where this chunk ends
            # min() ensures we don't go past the end of the text
            end = min(start + self.chunk_size, len(text))

            # Extract the chunk text
            chunk_text = text[start:end]

            # Skip empty or whitespace-only chunks
            # strip() removes whitespace from both ends
            if chunk_text.strip():

                # Build chunk metadata: copy doc metadata + add chunk-specific fields
                # dict(**a, **b) merges two dicts (Python 3.5+)
                chunk_metadata = {
                    **document.metadata,          # Inherit all doc metadata
                    "chunk_index": chunk_index,   # Position within this doc (0, 1, 2...)
                    "chunk_start_char": start,    # Character position in original doc
                    "chunk_end_char": end,        # End position
                    "chunk_size": len(chunk_text),
                }

                # Create a unique ID: "myfile.txt_0", "myfile.txt_1", etc.
                chunk_id = f"{filename}_{chunk_index}"

                chunk = Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Advance the window by `step` characters
            start += self.step

            # If we've reached the end, stop
            if end == len(text):
                break

        return chunks
