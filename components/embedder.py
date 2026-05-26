# =============================================================================
# components/embedder.py — STEP 3: EMBEDDER
# =============================================================================
# PURPOSE: Convert text into vectors (lists of numbers) that capture MEANING.
#
# WHY THIS STEP EXISTS:
# Computers can't compare "Paris" and "capital of France" as strings —
# they look completely different. But an embedding model converts both to
# vectors that are mathematically CLOSE to each other because they're
# semantically similar.
#
# WHAT IS AN EMBEDDING VECTOR?
# A vector is just a list of floating point numbers, e.g.:
#   "Paris" → [0.21, -0.54, 0.88, 0.13, ..., -0.32]  (384 numbers)
#   "capital of France" → [0.19, -0.51, 0.85, 0.17, ..., -0.29]
#
# These two vectors are close in "vector space" — that's how we find
# semantically similar text later in the retrieval step.
#
# MODEL USED:
# "all-MiniLM-L6-v2" from sentence-transformers:
#   - 384-dimensional output vectors
#   - Only ~80MB download
#   - Runs on CPU (no GPU needed)
#   - Trained specifically for semantic similarity tasks
# =============================================================================

import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from components.chunker import Chunk
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Converts text chunks into dense vector embeddings using a local model.
    
    The model runs LOCALLY on your machine — no API calls, no cost.
    The first time you run it, it downloads the model (~80MB).
    After that, it's cached and instant.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Load the embedding model into memory.
        
        This happens ONCE when you create the embedder.
        Subsequent calls to embed() are fast because the model is already loaded.
        
        Args:
            model_name: HuggingFace model identifier or local path
        """
        logger.info(f"Loading embedding model: {model_name}")
        logger.info("(First run downloads ~80MB — subsequent runs use cache)")

        # SentenceTransformer loads the model weights into RAM
        # It handles downloading, caching, and inference automatically
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Store the embedding dimension for reference
        # We'll use this to verify vectors have the right shape later
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Embed a list of Chunk objects into a 2D numpy array.
        
        Args:
            chunks: List of Chunk objects from the chunker
            
        Returns:
            A 2D numpy array of shape (num_chunks, embedding_dim)
            Each ROW is the embedding vector for one chunk.
            
            Example with 10 chunks and 384-dim model:
            Shape: (10, 384) — a matrix of 10 rows × 384 columns
        """
        if not chunks:
            logger.warning("No chunks to embed — returning empty array")
            # Return an empty 2D array with correct dimensions
            return np.zeros((0, self.embedding_dim))

        # Extract just the text strings from the Chunk objects
        # This is a list comprehension: [expression for item in iterable]
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Embedding {len(texts)} chunks...")

        # model.encode() converts a list of strings to a numpy array of vectors
        # show_progress_bar=True gives a nice progress bar in the terminal
        # batch_size controls how many texts are processed at once (memory tradeoff)
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,       # Process 32 chunks at a time
            normalize_embeddings=True,  # Normalize to unit length (makes cosine similarity = dot product)
        )

        # embeddings shape: (num_chunks, embedding_dim)
        # e.g., (47, 384) if we have 47 chunks
        logger.info(f"Embedding complete. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a vector.
        
        IMPORTANT: We use the SAME model for both chunks and queries.
        This ensures they live in the same "vector space" and can be compared.
        Using different models for indexing vs querying would give garbage results.
        
        Args:
            query: The user's search question
            
        Returns:
            1D numpy array of shape (embedding_dim,)
            e.g., shape (384,) — a single vector
        """
        logger.info(f"Embedding query: '{query[:80]}...' " if len(query) > 80 else f"Embedding query: '{query}'")

        # encode() can handle a single string or a list
        # normalize_embeddings=True keeps consistent with how we embedded chunks
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        # Result is a 1D array: [0.21, -0.54, 0.88, ..., -0.32]
        return embedding
