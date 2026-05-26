# Git & Docker — Complete Step-by-Step Guide

Everything we did, why we did it, and what you need to understand.
Read this alongside the actual files — each section maps to a real file in your project.

---

## PART 1 — GIT

### What is Git and why do we use it?

Git is a **version control system** — it tracks every change you make to your code over time.
Think of it like Google Docs "version history" but for your entire project.

Without Git:
- You rename files as `main_v2.py`, `main_final.py`, `main_FINAL_v2.py` to "save" old versions
- You can't easily share code with teammates
- One bad change can break everything with no way back

With Git:
- Every "save" (called a **commit**) is permanent and named
- You can go back to any previous state instantly
- You can work in parallel (branches) and merge changes
- GitHub hosts your repo online so it's backed up and shareable

---

### Key Git concepts (you'll use these every day)

| Term | What it means |
|------|--------------|
| **Repository (repo)** | A project folder that Git is tracking |
| **Commit** | A saved snapshot of your code at a point in time |
| **Branch** | A parallel version of the code (you start on `main`) |
| **Remote** | A copy of the repo on a server (like GitHub) |
| **Push** | Send your local commits to the remote |
| **Pull** | Get remote changes down to your local machine |
| **Staging area** | Where you prepare files before committing them |

---

### What we ran (step by step)

#### Step 1: `git init`
```bash
cd rag_prototype
git init
```
**What it does:** Creates a hidden `.git/` folder inside your project.
That folder IS the repository — it stores every commit, every branch, every history entry.
Without it, Git doesn't know your folder exists.

**Analogy:** Opening a brand new notebook and writing your name on the cover.

---

#### Step 2: `git config` (set your identity)
```bash
git config user.name "Vishnuraj"
git config user.email "yadavvishnuraj@gmail.com"
```
**What it does:** Every commit records who made it. This sets your identity for this repo.
Git refuses to commit if it doesn't know who you are.

---

#### Step 3: Creating `.gitignore`
Before committing anything, we created `.gitignore`. This file tells Git:
"Never track these files — pretend they don't exist."

**Why `.gitignore` before the first commit?**
If you commit a file and later add it to `.gitignore`, Git still tracks it
(it's already in history). You'd have to manually untrack it. Adding `.gitignore`
*first* means Git never picks up those files at all.

Most important entries in our `.gitignore`:
```
.env          ← YOUR API KEY — never commit this
storage/      ← vector index — auto-generated, can be re-created
__pycache__/  ← Python compiled files — auto-generated
.venv/        ← virtual environment — each dev creates their own
```

**The golden rule:** If a file contains secrets OR can be re-generated, it belongs in `.gitignore`.

---

#### Step 4: `git add .`
```bash
git add .
```
**What it does:** Moves all changed files into the **staging area** (also called the index).
The staging area is like a "loading dock" — you decide what goes into the next commit.

```
Working directory       Staging area         Repository
(your files)      →     (git add)      →     (git commit)
```

The `.` means "add everything in the current directory".
You can also add specific files: `git add main.py`

---

#### Step 5: `git commit`
```bash
git commit -m "feat: initial project setup with git and Docker"
```
**What it does:** Takes everything in the staging area and saves it as a permanent snapshot.
The `-m` flag lets you write the commit message inline.

**Commit message convention (we follow [Conventional Commits](https://www.conventionalcommits.org/)):**
```
<type>: <short description>

Types:
  feat     → new feature
  fix      → bug fix
  docs     → documentation only
  refactor → code restructuring (no behaviour change)
  chore    → setup, tooling, config
  test     → adding tests
```

Good: `fix: resolve IndexError in vector_store search`
Bad: `stuff`, `changes`, `asdfg`

---

#### Step 6: `git remote add origin`
```bash
git remote add origin https://github.com/vishnuraj08/rag_prototype.git
```
**What it does:** Tells your local repo where the "remote" copy lives (GitHub).
`origin` is just the conventional name for your primary remote.

---

#### Step 7: `git push`
```bash
git push -u origin main
```
**What it does:** Sends all your local commits to GitHub.
`-u origin main` sets the default remote and branch so future pushes only need `git push`.

---

### Your daily Git workflow (the loop you'll use forever)

```bash
# 1. Make changes to your code

# 2. See what changed
git status

# 3. Review the actual changes
git diff

# 4. Stage the changes you want to commit
git add .           # stage everything
git add main.py     # or stage specific files

# 5. Commit with a meaningful message
git commit -m "feat: add PDF loader support"

# 6. Push to GitHub
git push
```

---

## PART 2 — DOCKER

### What is Docker and why do we use it?

**The problem Docker solves:** "It works on my machine."

You build a Python app. It works perfectly on your laptop.
Your colleague clones it, runs it, and gets errors because:
- They have Python 3.9, you have 3.11
- They're missing a system library
- Their OS is different (Windows vs Linux)

Docker packages your app + its entire environment into a **container** — a self-contained,
isolated box that runs identically on any machine that has Docker installed.

---

### Core Docker concepts

```
┌─────────────────────────────────────────────────────────┐
│                    DOCKER CONCEPTS                       │
├──────────────┬──────────────────────────────────────────┤
│ Dockerfile   │ Recipe — instructions for building        │
│              │ the image. Like a shopping list.          │
├──────────────┼──────────────────────────────────────────┤
│ Image        │ The built result of the Dockerfile.       │
│              │ A frozen snapshot of your app + env.      │
│              │ Like a class in Python.                   │
├──────────────┼──────────────────────────────────────────┤
│ Container    │ A running instance of an image.           │
│              │ Like an object (instance of a class).     │
│              │ You can run many containers from one image │
├──────────────┼──────────────────────────────────────────┤
│ Volume       │ Shared storage between host and container │
│              │ Files here survive container deletion.    │
├──────────────┼──────────────────────────────────────────┤
│ docker-      │ Tool for running multi-container apps.    │
│ compose      │ Replaces long `docker run` commands with  │
│              │ a simple config file.                     │
└──────────────┴──────────────────────────────────────────┘
```

---

### File 1: `Dockerfile` — explained line by line

```dockerfile
FROM python:3.11-slim
```
**What:** Start from the official Python 3.11 slim image.
**Why slim?** The full image has many tools you don't need. Slim = smaller image = faster.
**Why pin 3.11?** "latest" could become 3.14 tomorrow and break your app. Pin the version.

---

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*
```
**What:** Install C compilers (`gcc`, `g++`).
**Why:** Some Python packages (including parts of numpy/torch) are written in C and need
a compiler to build from source on Linux.
**The `&&` and `\` pattern:** Chains commands in a single RUN instruction.
Each RUN creates a new image layer. Chaining = fewer layers = smaller image.
**The `rm -rf` at the end:** Deletes the apt package cache. Without this, the cache bloats
the layer by 50–100MB even though you'll never use it again.

---

```dockerfile
WORKDIR /app
```
**What:** Set the working directory to `/app` for all following instructions.
**Why `/app`?** Just a convention. Could be anything. `/app` is widely used in the industry.
**Effect:** `COPY . .` now copies to `/app/`, and `CMD ["python", ...]` runs from `/app/`.

---

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
```
**The critical layer caching trick:**
Docker builds images as stacks of layers. If a layer's inputs haven't changed,
Docker reuses the cached version. If they have changed, that layer and every layer
after it must be rebuilt.

```
Layer 1: FROM python:3.11-slim           ← rarely changes
Layer 2: apt-get install gcc             ← rarely changes
Layer 3: COPY requirements.txt           ← only changes when you add/remove packages
Layer 4: pip install                     ← only rebuilds when Layer 3 changes
Layer 5: COPY . .                        ← changes every time you edit code
```

If we did `COPY . .` before `pip install`, editing `main.py` would force
a full pip reinstall on every build — minutes wasted. By copying requirements first,
most rebuilds skip straight to Layer 5.

**CPU-only PyTorch:** `--index-url https://download.pytorch.org/whl/cpu`
This installs the CPU-only version of PyTorch (~200MB) instead of the GPU version (~2GB).
Since this prototype runs on CPU anyway, no need for the massive GPU version.

---

```dockerfile
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
```
**What:** Create a non-root user and switch to it.
**Why:** Containers run as root by default. If your app has a security vulnerability,
an attacker exploiting it would have root access inside the container. Running as a
non-privileged user limits the blast radius.
**Industry standard:** Always run containers as non-root in production.

---

```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
```
**PYTHONUNBUFFERED=1:** Python buffers stdout/stderr by default. In Docker, this means
your `print()` statements and log messages might not appear immediately in `docker logs`.
Setting this to 1 disables buffering — you see output in real time.

**PYTHONDONTWRITEBYTECODE=1:** Stops Python from creating `.pyc` compiled files inside
the container. These waste space and can cause confusion.

---

```dockerfile
CMD ["python", "main.py", "--help"]
```
**What:** The default command when you run a container without specifying what to do.
**Note:** `CMD` can be overridden at runtime:
```bash
docker run rag-prototype python main.py --interactive
```
The `CMD` in the Dockerfile is just the default, not mandatory.

---

### File 2: `docker-compose.yml` — explained

```yaml
volumes:
  - ./documents:/app/documents
  - ./storage:/app/storage
  - model-cache:/home/appuser/.cache
```

**Bind mounts** (`./documents:/app/documents`):
Your local folder IS the container's folder — they're the same directory.
Edit a file on your host → the container sees it instantly.
This is how you add documents to the RAG system without rebuilding.

**Named volumes** (`model-cache:/home/appuser/.cache`):
Managed by Docker. The embedding model (~80MB) is downloaded on first run
and cached here. Without this volume, every new container would re-download the model.

```yaml
stdin_open: true
tty: true
```
Required for `python main.py --interactive`. Without these, the container can't
receive keyboard input and the interactive mode doesn't work.

---

### File 3: `.dockerignore` — explained

Same concept as `.gitignore` but for Docker build context.

When you run `docker build .`, Docker first sends ALL files in `.` to the Docker daemon.
Files in `.dockerignore` are excluded.

Most important exclusions:
```
.env      ← NEVER put secrets inside an image. Pass them at runtime.
storage/  ← Runtime data — mounted as a volume, not baked into the image
.git/     ← Git history has no place in a container (and can be large)
.venv/    ← The container has its own Python environment
```

**Critical security rule:** Secrets (`.env`, API keys, passwords) must NEVER go inside
a Docker image. Images can be shared, pushed to registries, inspected.
Always pass secrets at runtime via `--env-file .env` or environment variables.

---

### File 4: `.env.example` — explained

You never commit `.env` (it has your real API key). But you need to tell other developers
(or your future self) what variables are needed.

`.env.example` is the solution: a template with fake/placeholder values that IS committed.

```bash
# Developer workflow:
cp .env.example .env        # copy template
nano .env                   # fill in real values
```

This is the industry-standard pattern for managing environment variables in projects.

---

## PART 3 — HOW IT ALL FITS TOGETHER

```
Your machine
├── rag_prototype/           ← Your project folder (tracked by Git)
│   ├── .git/               ← Git's brain (hidden, don't touch)
│   ├── .gitignore          ← "Git, ignore these"
│   ├── .env                ← Your secrets (NOT in Git, NOT in Docker image)
│   ├── .env.example        ← Template (IS in Git, IS in Docker image? No — .dockerignore)
│   ├── Dockerfile          ← Recipe for building the image
│   ├── docker-compose.yml  ← How to run the container
│   ├── .dockerignore       ← "Docker, ignore these when building"
│   ├── documents/          ← Your .txt files (bind-mounted into container)
│   └── storage/            ← Vector index (bind-mounted into container)
│
├── Docker Engine
│   ├── Image: rag-prototype ← Built from Dockerfile (your app + Python + libs)
│   └── Volume: model-cache  ← Embedding model (80MB, persists across containers)
│
└── GitHub
    └── github.com/vishnuraj08/rag_prototype  ← Remote backup of your code
```

---

## PART 4 — QUICK REFERENCE CHEATSHEET

### Git commands you'll use daily
```bash
git status              # What files have changed?
git diff                # What exactly changed in those files?
git add .               # Stage all changes
git add <file>          # Stage a specific file
git commit -m "msg"     # Save a snapshot with a message
git push                # Send commits to GitHub
git pull                # Get latest changes from GitHub
git log --oneline       # See commit history (one line each)
git checkout -b feature # Create and switch to a new branch
```

### Docker commands you'll use daily
```bash
docker-compose build              # Build the image
docker-compose run --rm rag python main.py --index       # Index docs
docker-compose run --rm rag python main.py --interactive # Interactive mode
docker-compose run --rm rag python main.py --query "Q"   # Single query

docker images                     # List all images on your machine
docker ps                         # List running containers
docker ps -a                      # List all containers (including stopped)
docker system prune               # Clean up unused images/containers (frees disk)
```

---

*Guide created during Cowork session — May 2026*
*Every file described here is in your `rag_prototype/` folder.*
