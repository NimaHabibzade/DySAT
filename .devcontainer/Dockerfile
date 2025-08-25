FROM python:3.6-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran wget git curl \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    (pip install --no-cache-dir "pip==20.3.4" && pip install --no-cache-dir -r /tmp/requirements.txt)

RUN pip install debugpy

RUN useradd -m codespace
USER codespace
WORKDIR /workspaces/${GITHUB_REPOSITORY:-workspace}
