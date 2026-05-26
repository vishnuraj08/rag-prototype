# =============================================================================
# pipeline.py — THE ORCHESTRATOR
# =============================================================================
# PURPOSE: Connect all 6 components into two coherent operations:
#   1. index()  — Load → Chunk → Embed → Store (done once per document set)
#   2. query()  — Embed query → Retrieve → Generate → Return answer
#
# WHY A SEPARATE PIPELINE?
# Each component (loader, chunker, etc.) knows how to do ONE thing.
# The pipeline knows how to CONNECT them in the right order.
# This is called the "Facade pattern" — one clean interface over many parts.
#
# DIAGRAM:
#
#   index():
#   Documents → [Loader] → [Chunker] → [Embedder] → [VectorStore.save()]
#
#   query():
#   Question → [Embedder] → [VectorStore.search()] → [Generator] → Answer
#
# Think of pipeline.py as the "director" and each component as an "actor".
# =============================================================================

import logging
from components.loader import DocumentLoader
from components.chunker import TextChunker
from components.embedder import TextEmbedder
from components.vector_store import VectorStore
from components.retriever import Retriever
from components.generator import Generator
from config import TOP_K

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    The complete RAG pipeline.
    
    Usage:
        pipeline = RAGPipeline()
        
        # One-time setup: index your documents
        pipeline.index()
        
        # Repeat as needed: query
        answer = pipeline.query("What is the capital of France?")
    """

    def __init__(self):
        """
        Initialize all components.
        
        Note: The embedding model is loaded here, which might take a moment
        the first time (downloading ~80MB). Subsequent runs are fast.
        """
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)

        # Instantiate each component
        # Each one gets its config from config.py automatically
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()      # ← Model loads here
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.generator = Generator()

        logger.info("All components initialized successfully")

    # -------------------------------------------------------------------------
    # PHASE 1: INDEXING
    # -------------------------------------------------------------------------

    def index(self) -> None:
        """
        Index all documents in the documents/ folder.
        
        This is the "offline" phase — you run this once (or when docs change).
        The result is saved to disk so you don't re-embed on every query.
        
        Steps:
            1. Load documents from documents/ folder
            2. Split each document into chunks
            3. Embed each chunk into a vector
            4. Store vectors in the VectorStore
            5. Save everything to disk
        """
        logger.info("\n--- PHASE 1: INDEXING ---")

        # Step 1: Load documents
        logger.info("Step 1/4: Loading documents...")
        documents = self.loader.load()

        # If no documents found, stop early with a helpful message
        if not documents:
            logger.error(
                "No documents found! "
                "Please add .txt or .md files to the 'documents/' folder."
            )
            return

        # Step 2: Split documents into chunks
        logger.info("Step 2/4: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.error("No chunks created — check your documents are not empty")
            return

        # Step 3: Embed all chunks
        logger.info("Step 3/4: Embedding chunks...")
        embeddings = self.embedder.embed_chunks(chunks)

        # Step 4: Add to vector store and save
        logger.info("Step 4/4: Saving to vector store...")
        self.vector_store.add(chunks, embeddings)
        self.vector_store.save()

        # Print a nice summary
        logger.info("\n" + "=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info(f"  Documents indexed : {len(documents)}")
        logger.info(f"  Total chunks      : {len(chunks)}")
        logger.info(f"  Embedding shape   : {embeddings.shape}")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # PHASE 2: QUERYING
    # -------------------------------------------------------------------------

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Answer a question using the indexed documents.
        
        This is the "online" phase — called every time a user asks something.
        
        Args:
            question : The user's question as a plain string
            top_k    : How many context chunks to retrieve
            
        Returns:
            A dictionary with:
                "answer"   : The generated text answer
                "sources"  : List of source chunks that were used
                "question" : The original question (for reference)
        """
        # Guard: reject empty or whitespace-only questions before doing any work
        if not question or not question.strip():
            logger.warning("query() called with an empty question")
            return {
                "answer": "Please provide a non-empty question.",
                "sources": [],
                "question": question,
            }

        logger.info(f"\n--- PHASE 2: QUERYING ---")
        logger.info(f"Question: {question}")

        # Load the saved index if it's not already in memory
        # This handles the case where you start a new Python session
        if self.vector_store.vectors is None:
            logger.info("Loading saved vector index from disk...")
            success = self.vector_store.load()

            if not success:
                # Index doesn't exist yet — user needs to run indexing first
                return {
                    "answer": "No documents have been indexed yet. Please run indexing first.",
                    "sources": [],
                    "question": question,
                }

        # Step A: Retrieve relevant chunks
        logger.info("Retrieving relevant chunks...")
        results = self.retriever.retrieve(question, top_k=top_k)

        if not results:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
                "question": question,
            }

        # Step B: Format context for the LLM
        context = self.retriever.get_context_string(results)

        # Step C: Generate answer
        logger.info("Generating answer...")
        answer = self.generator.generate(question, context)

        # Prepare source info for the caller (useful for showing citations)
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.metadata.get("filename", "unknown"),
                "similarity_score": round(score, 3),
                "text_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
            }
            for chunk, score in results
        ]

        # Return a structured result dictionary
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
