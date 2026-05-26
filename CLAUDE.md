# CLAUDE.md — Project Context & Session Memory
# =============================================
# This file is read by Claude at the start of every Cowork session.
# It contains everything Claude needs to know to pick up where we left off.
# Keep this updated after every session.
#
# Last updated: May 2026

---

## Who I Am

**Name:** Vishnuraj  
**Email:** yadavvishnuraj@gmail.com  
**GitHub:** github.com/vishnuraj08  
**Role:** Founder + AI/ML Developer + Software Developer (Startup)

---

## This Project: `rag_prototype`

A **Retrieval Augmented Generation (RAG)** system built in Python for learning.

### What RAG does
```
INDEXING (run once):
  Your documents → Loader → Chunker → Embedder → VectorStore → saved to disk

QUERYING (run per question):
  Question → Embedder → VectorStore → Retriever → Claude API → Answer
```

### File map
```
rag_prototype/
├── CLAUDE.md                  ← YOU ARE HERE — session memory
├── config.py                  ← All config in one place (start here)
├── pipeline.py                ← Connects all 6 components (the orchestrator)
├── main.py                    ← CLI entry point (--index, --query, --interactive)
├── components/
│   ├── loader.py              ← Step 1: Read .txt / .md files from disk
│   ├── chunker.py             ← Step 2: Split text into overlapping chunks
│   ├── embedder.py            ← Step 3: Convert text → vectors (all-MiniLM-L6-v2)
│   ├── vector_store.py        ← Step 4: Store & search vectors (numpy cosine sim)
│   ├── retriever.py           ← Step 5: Embed query + find relevant chunks
│   └── generator.py           ← Step 6: Call Claude API with context → answer
├── documents/                 ← Put .txt or .md files here to index
├── storage/                   ← Auto-generated vector index (gitignored)
├── requirements.txt           ← anthropic, sentence-transformers, numpy, torch
├── Dockerfile                 ← Docker image (python:3.11-slim, CPU torch)
├── docker-compose.yml         ← Local dev container setup with volumes
├── .env.example               ← Template for environment variables
├── .gitignore                 ← Excludes .env, storage/, __pycache__, etc.
├── .dockerignore              ← Excludes .env, storage/, .git from image
├── CODE_REVIEW_CHANGES.md     ← All 7 bug fixes explained with concepts
└── GIT_AND_DOCKER_GUIDE.md    ← Full Git + Docker teaching doc
```

### Tech stack
- **Python 3.11**
- **sentence-transformers** — local embedding model (`all-MiniLM-L6-v2`, 384-dim)
- **numpy** — vector math / cosine similarity
- **torch** — required by sentence-transformers (CPU-only)
- **anthropic SDK** — calls Claude for answer generation
- **Claude model:** `claude-sonnet-4-6`

---

## Current State

### ✅ Done
- [x] All 7 code review bugs fixed (see `CODE_REVIEW_CHANGES.md`)
- [x] `.gitignore` created and configured
- [x] `.env.example` created
- [x] `Dockerfile` created (python:3.11-slim, non-root user, CPU torch)
- [x] `docker-compose.yml` created (volumes for documents/, storage/, model-cache)
- [x] `.dockerignore` created
- [x] `README.md` updated with Docker section
- [x] `CODE_REVIEW_CHANGES.md` — teaching doc for all 7 fixes
- [x] `GIT_AND_DOCKER_GUIDE.md` — teaching doc for git + Docker concepts
- [x] Git init done (via Cowork sandbox)

### ⏳ Pending (Vishnuraj to do)
- [ ] **Push to GitHub** — run these commands in Git Bash:
  ```bash
  cd "C:\Users\vishnuraj\Documents\Projects\rag_prototype"
  git init
  git branch -m main
  git config user.name "Vishnuraj"
  git config user.email "yadavvishnuraj@gmail.com"
  git add .
  git status        # verify storage/ and .env are NOT listed
  git commit -m "chore: initial project setup with Docker and git"
  # Then create repo on github.com/new → name: rag_prototype
  git remote add origin https://github.com/vishnuraj08/rag_prototype.git
  git push -u origin main
  ```
- [ ] Add more documents to `documents/` and test the pipeline end-to-end
- [ ] Write unit tests (at minimum for `VectorStore.search()` and `TextChunker`)

### 🔜 Next logical steps (when ready)
- Add PDF support in `loader.py` (use `pymupdf` / `fitz`)
- Swap numpy VectorStore → FAISS for scale
- Add streaming responses in `generator.py`
- Add a simple web UI (FastAPI + HTML or Streamlit)
- Containerize with multi-stage Docker build for smaller image

---

## Cowork Setup (for this machine)

### Plugins installed
| Plugin | Skills available |
|--------|-----------------|
| Engineering | `/engineering:code-review`, `/engineering:system-design`, `/engineering:architecture`, `/engineering:debug`, `/engineering:documentation`, `/engineering:deploy-checklist`, `/engineering:standup`, `/engineering:tech-debt` |
| Small Business | `/small-business:business-pulse`, `/small-business:monday-brief`, `/small-business:cash-flow-snapshot`, and 20+ more |
| Product Management | `/product-management:competitive-brief`, `/product-management:roadmap-update`, `/product-management:write-spec`, and more |

### Connectors connected
- **GitHub** — connected (MCP tools not yet surfacing in sessions — workaround: share local folder)

### Folder access
- `C:\Users\vishnuraj\Documents\Projects\rag_prototype` — connected and active

### Known issues / quirks
- GitHub MCP connector is connected but not surfacing tool calls in sessions yet. Workaround: share project folder directly via Cowork folder access.
- Git operations from the Linux sandbox on the Windows-mounted NTFS folder have file-lock conflicts. Run git commands from Git Bash on Windows instead.

---

## How Claude should approach this project

1. **Read this file first** at the start of every session — it's the source of truth.
2. **Update this file** at the end of every session — add completed items, new pending items, new decisions.
3. When writing code: follow the existing style — heavy inline comments, clear docstrings, modular components.
4. When fixing bugs: document changes in `CODE_REVIEW_CHANGES.md`.
5. When adding major features: create a short doc or update `GIT_AND_DOCKER_GUIDE.md`.
6. Always teach Vishnuraj **why** we're doing something, not just what.

---

## Session Log

### Session 1 — May 2026
- Set up Cowork for Founder + AI/ML Developer profile
- Installed Engineering, Small Business, Product Management plugins
- Connected GitHub connector
- Reviewed `rag_prototype` codebase — found and fixed 7 bugs
- Created full Docker setup (Dockerfile, docker-compose.yml, .env.example, .dockerignore)
- Initialized git repo
- Created teaching docs: `CODE_REVIEW_CHANGES.md`, `GIT_AND_DOCKER_GUIDE.md`
- Explained: git workflow, Docker concepts, f-strings, exception handling, defensive programming, logging architecture, sorted early-exit iteration
- **Blocker:** Git push pending — Vishnuraj needs to run commands in Git Bash

---
