# =============================================================================
# components/vector_store.py — STEP 4: VECTOR STORE
# =============================================================================
# PURPOSE: Store vectors + their associated chunks, and search them efficiently.
#
# WHY THIS STEP EXISTS:
# After embedding, we have thousands of vectors. When a user asks a question,
# we need to find which vectors are most SIMILAR to the query vector — fast.
#
# HOW SIMILARITY WORKS (Cosine Similarity):
# Two vectors are "similar" if they point in roughly the same direction.
#
#  Vector A:  → → →       (pointing right)
#  Vector B:  → → ↗       (pointing right + slightly up) = high similarity
#  Vector C:  ↓ ↓ ↓       (pointing down)               = low similarity
#
# Cosine similarity = cos(angle between vectors)
#   - 1.0 means identical direction (same meaning)
#   - 0.0 means perpendicular (unrelated)
#   - -1.0 means opposite direction (opposite meaning)
#
# IMPLEMENTATION:
# We use PURE NUMPY — no external vector DB needed for base version.
# This is the most transparent implementation — you can see exactly
# what's happening in the math.
#
# HOW TO EXTEND:
# Swap this class out with a FAISS, Pinecone, or Chroma store later
# without touching anything else — that's the power of modular design.
# =============================================================================

import os
import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from components.chunker import Chunk
from config import VECTOR_INDEX_PATH, CHUNKS_INDEX_PATH, MIN_SIMILARITY_SCORE

logger = logging.getLogger(__name__)


# A SearchResult is a tuple of (Chunk, similarity_score)
# We define it as a type alias for readability
SearchResult = Tuple[Chunk, float]


class VectorStore:
    """
    A simple in-memory vector store backed by numpy arrays.
    
    Stores:
      - self.vectors  : 2D numpy array (num_chunks × embedding_dim)
      - self.chunks   : List of Chunk objects (parallel to vectors)
    
    The i-th row of self.vectors corresponds to self.chunks[i].
    
    Persistence:
      - vectors are saved to a .npz file (numpy's compressed format)
      - chunks are saved to a .json file
    """

    def __init__(self):
        # Start empty — gets populated by add() or load()
        self.vectors: np.ndarray = None   # Will be (N, D) array
        self.chunks: List[Chunk] = []     # Parallel list of Chunk objects
        logger.info("VectorStore initialized (empty)")

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.
        
        Args:
            chunks     : List of Chunk objects
            embeddings : 2D numpy array, shape (len(chunks), embedding_dim)
        """
        # Sanity check: number of chunks must match number of embedding rows
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )

        self.chunks = chunks
        self.vectors = embeddings  # Store the raw numpy matrix

        logger.info(f"VectorStore populated: {len(chunks)} chunks, vectors shape: {embeddings.shape}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Find the top_k most similar chunks to the query vector.
        
        This is the CORE of RAG — given a query, find relevant context.
        
        THE MATH (don't worry, it's just matrix multiplication):
        
        Since all our vectors are normalized (length = 1.0), cosine similarity
        between two vectors A and B equals their dot product:
            cosine_similarity(A, B) = A · B  (when both are normalized)
        
        To compare query against ALL chunks at once, we do:
            scores = vectors_matrix @ query_vector
            
        This gives us one similarity score per chunk in a single fast operation.
        
        Args:
            query_vector : 1D numpy array, shape (embedding_dim,)
            top_k        : Number of results to return
            
        Returns:
            List of (Chunk, score) tuples, sorted by score descending
        """
        if self.vectors is None or len(self.chunks) == 0:
            logger.warning("VectorStore is empty — nothing to search")
            return []

        # --- THE SIMILARITY CALCULATION ---
        # self.vectors shape: (N, D)  — N chunks, D dimensions each
        # query_vector shape: (D,)    — one query, D dimensions
        #
        # Matrix multiplication (N,D) @ (D,) → (N,)
        # Result: one similarity score for each of the N chunks
        similarity_scores = self.vectors @ query_vector

        # np.argsort returns indices that would sort the array ascending
        # [::-1] reverses it to get descending order (highest similarity first)
        # Example: scores=[0.3, 0.9, 0.1, 0.7] → argsort=[2,0,3,1] → reversed=[1,3,0,2]
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Collect top results, filtering by minimum score threshold.
        # We iterate ALL sorted indices (not just top_k * 2) so that we never
        # silently miss valid results when many chunks score below the threshold.
        results = []
        for idx in sorted_indices:
            score = float(similarity_scores[idx])  # Convert numpy float to Python float

            # Once scores drop below the threshold, everything after will be worse
            # (array is sorted descending), so we can stop early
            if score < MIN_SIMILARITY_SCORE:
                break

            results.append((self.chunks[idx], score))

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        logger.info(f"Search returned {len(results)} results (top score: {results[0][1]:.3f if results else 'N/A'})")
        return results

    def save(self) -> None:
        """
        Persist vectors and chunks to disk so we don't re-embed on every run.
        
        Files created:
        - vectors.npz  : Compressed numpy array (binary, fast to load)
        - chunks.json  : Human-readable JSON with all chunk text + metadata
        """
        # Create the storage directory if it doesn't exist
        # exist_ok=True means "don't error if it already exists"
        os.makedirs(os.path.dirname(VECTOR_INDEX_PATH), exist_ok=True)

        # Save the numpy array in compressed format
        # np.savez_compressed can store multiple arrays in one file
        # We use keyword argument "vectors" so we can retrieve by name
        np.savez_compressed(VECTOR_INDEX_PATH, vectors=self.vectors)
        logger.info(f"Vectors saved to: {VECTOR_INDEX_PATH}")

        # Convert Chunk objects to plain dictionaries for JSON serialization
        # JSON can't directly serialize custom Python objects — it only handles
        # dicts, lists, strings, numbers, booleans, and None
        chunks_data = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in self.chunks
        ]

        # json.dump() writes a Python object as JSON to a file
        # indent=2 makes it human-readable (pretty printed)
        with open(CHUNKS_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunks saved to: {CHUNKS_INDEX_PATH}")

    def load(self) -> bool:
        """
        Load previously saved vectors and chunks from disk.
        
        Returns:
            True if successfully loaded, False if no saved index exists
        """
        # Check if both files exist before trying to load
        if not os.path.exists(VECTOR_INDEX_PATH) or not os.path.exists(CHUNKS_INDEX_PATH):
            logger.info("No saved index found — need to run indexing first")
            return False

        # Load the numpy array from the .npz file
        # np.load returns a dict-like object; we retrieve by the key "vectors"
        data = np.load(VECTOR_INDEX_PATH)
        self.vectors = data["vectors"]
        logger.info(f"Vectors loaded. Shape: {self.vectors.shape}")

        # Load the chunks from JSON and reconstruct Chunk objects
        with open(CHUNKS_INDEX_PATH, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        # Reconstruct Chunk objects from the plain dictionaries
        self.chunks = [
            Chunk(
                text=item["text"],
                metadata=item["metadata"],
                chunk_id=item["chunk_id"],
            )
            for item in chunks_data
        ]

        logger.info(f"Chunks loaded: {len(self.chunks)} chunks")
        return True
