# Use Python 3.6 base
FROM python:3.6-slim

ENV DEBIAN_FRONTEND=noninteractive

# system deps for compiling / wheel compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran wget git curl \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip etc.
RUN pip install --upgrade pip setuptools wheel

# COPY from build context (repo root) — requirements.txt must be at repo root
COPY requirements.txt /tmp/requirements.txt

# Try installing; fallback to older pip if needed
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    (pip install --no-cache-dir "pip==20.3.4" && pip install --no-cache-dir -r /tmp/requirements.txt)

# install debugpy for debugger support
RUN pip install debugpy

# create non-root user used by Codespaces
RUN useradd -m codespace
USER codespace
WORKDIR /workspaces/${GITHUB_REPOSITORY:-workspace}
