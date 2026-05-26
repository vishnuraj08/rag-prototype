# =============================================================================
# config.py — THE BRAIN OF CONFIGURATION
# =============================================================================
# This is the single place where you control every parameter in the system.
# Want to experiment? Change values HERE. No need to hunt across files.
#
# Think of this like a control panel — all knobs and dials in one place.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# PATHS — Where things live on disk
# -----------------------------------------------------------------------------

# Where your source documents (text files, etc.) are stored
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# Where we'll save the vector index so we don't re-compute on every run
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# The file that stores our saved vectors (numpy format)
VECTOR_INDEX_PATH = os.path.join(STORAGE_DIR, "vectors.npz")

# The file that stores the text chunks alongside their vectors
CHUNKS_INDEX_PATH = os.path.join(STORAGE_DIR, "chunks.json")


# -----------------------------------------------------------------------------
# CHUNKING — How we split documents
# -----------------------------------------------------------------------------

# How many characters per chunk?
# Too large → LLM context gets bloated with irrelevant text
# Too small → chunks lose meaning / context
CHUNK_SIZE = 500

# How many characters of overlap between consecutive chunks?
# Overlap ensures important sentences at chunk boundaries aren't cut in half
# Example: if chunk 1 ends mid-sentence, chunk 2 starts a bit earlier to catch it
CHUNK_OVERLAP = 50


# -----------------------------------------------------------------------------
# EMBEDDINGS — How we convert text to vectors
# -----------------------------------------------------------------------------

# Which sentence-transformer model to use for creating embeddings
# "all-MiniLM-L6-v2" is small (80MB), fast, and works great for RAG
# Other options you can swap in later:
#   - "all-mpnet-base-v2"         → more accurate, slower (420MB)
#   - "paraphrase-MiniLM-L3-v2"  → fastest, least accurate
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# RETRIEVAL — How we search for relevant chunks
# -----------------------------------------------------------------------------

# How many chunks to retrieve for each query?
# More chunks = more context for the LLM, but also more noise and token cost
# Start with 3-5, tune based on your use case
TOP_K = 3

# Minimum similarity score (0.0 to 1.0) to include a chunk
# Chunks below this threshold are considered irrelevant and excluded
# 0.0 = include everything, 1.0 = only exact matches
MIN_SIMILARITY_SCORE = 0.2


# -----------------------------------------------------------------------------
# GENERATION — How we call the LLM
# -----------------------------------------------------------------------------

# Claude model to use for generating answers
CLAUDE_MODEL = "claude-sonnet-4-6"

# Max tokens the LLM can generate in its response
MAX_TOKENS = 1024

# The system prompt — tells Claude what role to play
# You can customize this to change the tone, language, behavior, etc.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY 
on the provided context. If the answer is not in the context, say 
"I don't have enough information in the provided documents to answer that."
Do not make up information."""


# -----------------------------------------------------------------------------
# SUPPORTED FILE TYPES — What the loader can read
# -----------------------------------------------------------------------------

# Add more extensions here later (e.g., ".pdf", ".md", ".docx")
SUPPORTED_EXTENSIONS = [".txt", ".md"]
