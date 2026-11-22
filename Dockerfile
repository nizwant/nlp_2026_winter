# syntax=docker/dockerfile:1
#
# One image for everything: vLLM (offline LLM() or `vllm serve`), analysis, notebooks, tests.
# Python packages come from uv.lock, so this image and a native `uv sync` install the same
# versions. Build and use it through compose.yaml; see README.md.
#
# Base: CUDA 13.0 devel. The torch/vLLM wheels bring their own CUDA 13 runtime libraries; the
# toolkit is here for kernels that are compiled on first use (FlashInfer, Triton). CUDA 13.0
# needs NVIDIA driver >= 580 on the host. Do not move to 13.1+ without checking both drivers.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git less ninja-build tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /uvx /usr/local/bin/

# The venv lives outside /workspace so that bind-mounting the repo does not hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    PATH=/opt/venv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

# Dependencies first (cached layer), then the project itself in editable mode.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
 && chmod -R a+rX /opt/python /opt/venv

# Containers run as the host user (see compose.yaml), whose UID may not exist in the image.
# HOME and the caches are world-writable so any UID works.
#   /hf     Hugging Face cache, bind-mounted from the host so weights are downloaded once
#   /cache  vLLM / torch.compile / Triton / FlashInfer caches, a named volume
ENV HOME=/home/dev \
    HF_HOME=/hf \
    XDG_CACHE_HOME=/cache \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1
RUN mkdir -p /home/dev /hf /cache \
 && chmod 1777 /home/dev /hf /cache

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
