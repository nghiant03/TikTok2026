FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python${PYTHON_VERSION} python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir .

ENTRYPOINT ["python3.11", "-m", "tiktok2026"]
