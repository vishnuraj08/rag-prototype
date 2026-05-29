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

# Which device to run the embedding model on.
#
#   "auto" → use GPU (CUDA) if available, fall back to CPU automatically
#   "cpu"  → always use CPU (safe on any machine, slower)
#   "cuda" → force GPU (will fail if CUDA is not available)
#
# GPU is ~10-20x faster for embedding large document sets.
# For small document sets (<1000 chunks) the difference is minimal.
#
# IMPORTANT: Make sure your PyTorch build matches your CUDA version.
#   CPU only:  pip install torch --index-url https://download.pytorch.org/whl/cpu
#   CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121
#   CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118
DEVICE = "auto"


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

# Which LLM provider to use.
# Switch this one value to change the entire generation backend.
#
# Options:
#   "anthropic" → calls Claude via Anthropic API (needs ANTHROPIC_API_KEY)
#   "ollama"    → calls a local model via Ollama (no key needed, runs on your machine)
LLM_PROVIDER = "ollama"

# Which model to use. The right value depends on which provider you chose above.
#
# For LLM_PROVIDER = "anthropic":
#   "claude-sonnet-4-6"          ← fast and capable (recommended)
#   "claude-haiku-4-5-20251001"  ← fastest and cheapest
#   "claude-opus-4-6"            ← most capable, slowest
#
# For LLM_PROVIDER = "ollama":
#   "llama3.2"    ← Meta's Llama 3.2 (you already have this — run `ollama ls` to verify)
#   "llama3"      ← Meta's Llama 3 (older, slightly larger)
#   "mistral"     ← Mistral 7B (fast, lightweight)
#   "phi3"        ← Microsoft Phi-3 (very small, runs on weak hardware)
#   Run `ollama ls` to see which models you have downloaded locally.
LLM_MODEL = "llama3.2"

# Ollama server URL — only used when LLM_PROVIDER = "ollama"
# Default is localhost:11434 which is where Ollama runs after `ollama serve`
OLLAMA_BASE_URL = "http://localhost:11434"

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
