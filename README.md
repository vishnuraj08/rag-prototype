# RAG Prototype — Learning Guide

A minimal, heavily-commented Retrieval Augmented Generation (RAG) system
built for learning. Every component is modular and swappable.

## Architecture

```
INDEXING (offline, run once):
Documents → Loader → Chunker → Embedder → VectorStore → Disk

QUERYING (online, run per question):
Question → Embedder → VectorStore → Retriever → Generator → Answer
```

## Project Structure

```
rag_prototype/
├── config.py              ← All configuration in one place (start here!)
├── pipeline.py            ← Connects all components
├── main.py                ← CLI entry point
├── components/
│   ├── loader.py          ← STEP 1: Read files from disk
│   ├── chunker.py         ← STEP 2: Split text into chunks
│   ├── embedder.py        ← STEP 3: Convert text → vectors
│   ├── vector_store.py    ← STEP 4: Store & search vectors
│   ├── retriever.py       ← STEP 5: Find relevant chunks
│   └── generator.py       ← STEP 6: Call LLM with context
├── documents/             ← Put your .txt or .md files here
│   └── sample.txt
├── storage/               ← Auto-created, stores the vector index
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your API Key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Add Your Documents
Put `.txt` or `.md` files in the `documents/` folder.
A `sample.txt` about AI is already included to test with.

### 4. Index Your Documents (one time)
```bash
python main.py --index
```

### 5. Ask Questions
```bash
# Single question
python main.py --query "What is machine learning?"

# Interactive mode (ask multiple questions)
python main.py --interactive

# Index and then go interactive in one command
python main.py --index --interactive
```

## Tuning Guide (all in config.py)

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Larger = more context per chunk, but less precise |
| `CHUNK_OVERLAP` | 50 | Larger = fewer missed boundaries, more redundancy |
| `TOP_K` | 3 | More chunks = more context, but also more noise |
| `MIN_SIMILARITY_SCORE` | 0.2 | Higher = stricter relevance filter |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Swap for accuracy/speed tradeoff |
| `CLAUDE_MODEL` | claude-sonnet-4-... | Swap for different Claude model |

## How to Extend

### Add PDF Support
In `components/loader.py`, add:
```python
elif extension == ".pdf":
    import fitz  # pip install pymupdf
    doc = fitz.open(filepath)
    text = "".join(page.get_text() for page in doc)
```

### Swap Vector Store (e.g., use FAISS)
Replace `vector_store.py`'s `search()` method with FAISS index lookup.
No other file needs to change!

### Add Re-ranking
In `components/retriever.py`, after `self.vector_store.search()`,
add a cross-encoder re-ranking step before returning results.

### Add Streaming
In `components/generator.py`, use `self.client.messages.stream()` instead
of `create()` and yield tokens as they come.

## Running with Docker

Docker lets you run this project without installing Python, pip, or any dependencies
on your machine. Everything runs inside an isolated container.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Setup
```bash
# 1. Copy the environment template and add your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-your-key-here

# 2. Build the Docker image (first time takes ~5 minutes — downloads PyTorch)
docker-compose build

# 3. Index your documents
docker-compose run --rm rag python main.py --index

# 4. Ask a question
docker-compose run --rm rag python main.py --query "What is machine learning?"

# 5. Interactive mode
docker-compose run --rm rag python main.py --interactive
```

### How volumes work
| Volume | What it does |
|--------|-------------|
| `./documents` | Put your .txt / .md files here — the container reads them |
| `./storage` | Vector index is saved here — persists across container restarts |
| `model-cache` | Embedding model cached here — only downloaded once |

---

## Learning Path

1. Read `config.py` first — understand all the parameters
2. Follow the data flow: `loader.py` → `chunker.py` → `embedder.py`
3. Understand vector search: `vector_store.py` (the cosine similarity math)
4. See how it connects: `pipeline.py`
5. Run it: `main.py --index --interactive`
6. Try changing `CHUNK_SIZE` in config.py and re-index — see how answers change!
