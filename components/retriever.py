# =============================================================================
# components/retriever.py — STEP 5: RETRIEVER
# =============================================================================
# PURPOSE: Given a user's query, embed it and return the most relevant chunks.
#
# WHY THIS STEP EXISTS AS SEPARATE FROM VECTOR_STORE?
# The VectorStore is a "dumb" storage layer — it just stores and searches vectors.
# The Retriever is the "smart" layer that:
#   1. Takes a human query (string)
#   2. Converts it to a vector (using the embedder)
#   3. Asks the vector store for similar chunks
#   4. Formats and returns the results
#
# This separation makes each piece swappable:
# - Swap VectorStore: numpy → FAISS → Pinecone (no changes to Retriever)
# - Swap Embedder: MiniLM → OpenAI embeddings (no changes to VectorStore)
# - Add re-ranking: just modify the Retriever (no changes to others)
#
# HOW TO EXTEND:
# - Add a re-ranker (cross-encoder) to improve result quality
# - Add keyword filtering (hybrid search)
# - Add metadata filtering ("only from document X")
# =============================================================================

import logging
from typing import List, Tuple
from components.embedder import TextEmbedder
from components.vector_store import VectorStore, SearchResult
from config import TOP_K

logger = logging.getLogger(__name__)


class Retriever:
    """
    Ties together the Embedder and VectorStore to answer the question:
    "Given this query, what text chunks are most relevant?"
    """

    def __init__(self, embedder: TextEmbedder, vector_store: VectorStore):
        """
        Args:
            embedder     : TextEmbedder instance (converts query to vector)
            vector_store : VectorStore instance (finds similar vectors)
            
        Note: We pass these in rather than creating them here.
        This is called "Dependency Injection" — it makes components
        testable and swappable.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info(f"Retriever initialized (top_k={TOP_K})")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        """
        Main retrieval method — takes a query string, returns relevant chunks.
        
        Steps:
        1. Embed the query → vector
        2. Search vector store → (chunk, score) pairs
        3. Return results
        
        Args:
            query  : The user's question as a plain string
            top_k  : How many chunks to return (overrides config default)
            
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by relevance.
            Most relevant chunk first.
        """
        logger.info(f"Retrieving top {top_k} chunks for: '{query}'")

        # Step 1: Convert the query string to a vector
        # IMPORTANT: Same model as used for chunks — must be identical!
        query_vector = self.embedder.embed_query(query)

        # Step 2: Ask the vector store to find similar chunks
        results = self.vector_store.search(query_vector, top_k=top_k)

        # Step 3: Log what we found (helpful for debugging)
        if results:
            logger.info(f"Retrieved {len(results)} chunks:")
            for i, (chunk, score) in enumerate(results):
                # Show a preview of each retrieved chunk + its similarity score
                preview = chunk.text[:80].replace("\n", " ")
                logger.info(f"  [{i+1}] score={score:.3f} | {preview}...")
        else:
            logger.warning("No relevant chunks found for this query")

        return results

    def get_context_string(self, results: List[SearchResult]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.
        
        The LLM needs a clear, structured context block it can reason over.
        We label each chunk with its source and rank so the LLM can cite them.
        
        Example output:
            [Source: myfile.txt | Chunk 1 | Score: 0.87]
            Paris is the capital of France...
            
            [Source: myfile.txt | Chunk 2 | Score: 0.74]
            France is a country in Western Europe...
        
        Args:
            results: List of (Chunk, score) tuples from retrieve()
            
        Returns:
            Formatted string to pass as context to the generator
        """
        if not results:
            return "No relevant context found."

        context_parts = []

        for rank, (chunk, score) in enumerate(results, start=1):
            # Extract source filename from metadata
            source = chunk.metadata.get("filename", "unknown source")

            # Format a labeled block for this chunk
            header = f"[Source: {source} | Chunk {rank} | Relevance: {score:.2f}]"
            context_parts.append(f"{header}\n{chunk.text}")

        # Join all chunks with a blank line separator
        # "\n\n---\n\n" is a visual separator the LLM can understand
        return "\n\n---\n\n".join(context_parts)
