FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN /opt/venv/bin/python -m pip install --no-cache-dir .

ENTRYPOINT ["/opt/venv/bin/python", "-m", "tiktok2026"]
