# =============================================================================
# Dockerfile — Instructions for building the RAG Prototype container image
# =============================================================================
#
# WHAT IS DOCKER?
#   Docker packages your app + its entire environment (Python version,
#   libraries, OS config) into a single "image". Anyone who runs this image
#   gets EXACTLY the same environment — no "works on my machine" problems.
#
# WHAT IS AN IMAGE vs a CONTAINER?
#   Image  = the blueprint / template (like a class in Python)
#   Container = a running instance of that image (like an object)
#   You build an image once; you can run many containers from it.
#
# HOW DOCKER BUILDS WORK (layers):
#   Each instruction (FROM, RUN, COPY, etc.) creates a new "layer".
#   Docker CACHES layers — if nothing changed, it reuses the cached layer.
#   This is why we COPY requirements.txt and install deps BEFORE copying
#   source code: dependencies rarely change, source code changes often.
#   Correct order = fast rebuilds.
#
# BUILD THIS IMAGE:
#   docker build -t rag-prototype .
#
# RUN A CONTAINER:
#   docker run -it --env-file .env rag-prototype
# =============================================================================


# -----------------------------------------------------------------------------
# Stage: Base image
# -----------------------------------------------------------------------------
# FROM sets the starting point — we inherit everything from this official image.
# python:3.11-slim is the lightweight Debian-based Python image.
# "slim" = no unnecessary packages installed (smaller image = faster builds/pulls).
# We pin to 3.11 (not just "python") for reproducibility — "latest" can break things.
FROM python:3.11-slim

# LABEL adds metadata — useful for identifying images in a registry
LABEL maintainer="vishnuraj08"
LABEL description="RAG Prototype — Retrieval Augmented Generation system"
LABEL version="1.0"


# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
# RUN executes a shell command inside the image during build.
# We chain commands with && to keep it in ONE layer (fewer layers = smaller image).
# `rm -rf /var/lib/apt/lists/*` cleans the apt cache — saves ~50MB in the image.
#
# gcc and g++ are C compilers needed to build some Python packages from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
# WORKDIR sets the directory all subsequent commands run in.
# It also creates the directory if it doesn't exist.
# Using /app is a common convention for application code in Docker.
WORKDIR /app


# -----------------------------------------------------------------------------
# Python dependencies — installed BEFORE copying source code (cache trick)
# -----------------------------------------------------------------------------
# COPY <source-on-host> <destination-in-image>
# We copy ONLY requirements.txt first (not the whole project).
# Why? Docker caches each layer. If requirements.txt hasn't changed,
# this entire `pip install` layer is reused from cache — saves minutes!
COPY requirements.txt .

# Install PyTorch CPU-only first (much smaller than the GPU version)
# The --index-url tells pip to download from PyTorch's CPU-only wheel server
# This avoids downloading the massive CUDA version (~2GB vs ~200MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements.txt
# --no-cache-dir prevents pip from caching downloaded wheels (saves disk space in image)
RUN pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Application source code
# -----------------------------------------------------------------------------
# Now copy the rest of the project into the image.
# This layer changes often (every code edit) — that's fine because
# the slow pip install layers above are already cached.
COPY . .


# -----------------------------------------------------------------------------
# Directory setup
# -----------------------------------------------------------------------------
# Ensure the documents and storage directories exist.
# Even if they're empty, we need them for volumes to mount correctly.
RUN mkdir -p documents storage

# -----------------------------------------------------------------------------
# Pre-download the embedding model into the image
# -----------------------------------------------------------------------------
# Without this, the model is downloaded at runtime on first use (~90MB from HuggingFace).
# Baking it into the image means:
#   1. The image works completely OFFLINE (no internet needed at runtime)
#   2. Container startup is instant — no download wait
#   3. The other PC just needs the image file — nothing else
#
# TRANSFORMERS_OFFLINE=1 tells HuggingFace to never try the internet after this.
# HF_HUB_DISABLE_SYMLINKS_WARNING=1 suppresses the Windows symlink warning.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1


# -----------------------------------------------------------------------------
# Security: run as non-root user
# -----------------------------------------------------------------------------
# By default, Docker containers run as root — a security risk.
# If someone exploits your app, they'd have root access to the container.
# Best practice: create a dedicated user and switch to it.
#
# useradd -m = create home directory too
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser


# -----------------------------------------------------------------------------
# Runtime environment
# -----------------------------------------------------------------------------
# ENV sets environment variables that will be available at runtime.
# PYTHONUNBUFFERED=1 ensures Python output (print, logging) appears immediately
# in Docker logs instead of being buffered — essential for seeing log output.
# PYTHONDONTWRITEBYTECODE=1 stops Python from creating .pyc files in the container.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# -----------------------------------------------------------------------------
# Default command
# -----------------------------------------------------------------------------
# CMD defines what runs when you start a container without specifying a command.
# ["python", "main.py", "--help"] shows usage by default — safe and informative.
# Override at runtime: docker run -it rag-prototype python main.py --interactive
CMD ["python", "main.py", "--help"]
