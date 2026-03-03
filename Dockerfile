# syntax=docker/dockerfile:1
# ── Base: CUDA 12.4 runtime — matches cu124 torch wheel index ────────────────
# Using runtime (not devel) saves ~3.5 GB. PyTorch ships its own cuDNN.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
# deadsnakes PPA provides Python 3.11 + 3.12 on Ubuntu 22.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev \
       python3.11 python3.11-venv python3.11-dev \
       ffmpeg libsndfile1 \
       build-essential \
       curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# ── uv — copy from pinned official image (no curl script, no cargo, <1 s) ────
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /usr/local/bin/

# Tell uv: use system Python already installed above, never download
# UV_LINK_MODE=copy is required inside Docker (no hardlinks across overlay layers)
ENV PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    UV_PYTHON_PREFERENCE=system \
    UV_LINK_MODE=copy

WORKDIR /app

# ── Dependency specs — copied first so this layer is cached ──────────────────
# The venv-install layer below only re-runs when these files change.
COPY nemo/pyproject.toml          nemo/uv.lock          nemo/.python-version          /app/nemo/
COPY translate-gemma/pyproject.toml translate-gemma/uv.lock translate-gemma/.python-version /app/translate-gemma/
COPY qwen3-tts/pyproject.toml     qwen3-tts/uv.lock     qwen3-tts/.python-version     /app/qwen3-tts/
COPY whisper/pyproject.toml       whisper/uv.lock       whisper/.python-version       /app/whisper/

# ── Install all venvs ─────────────────────────────────────────────────────────
# --mount=type=cache: downloaded wheels are cached between builds but NOT baked
# into the final image. Rebuild after a lockfile change re-uses cached wheels
# (no re-download of torch). Code-only changes never reach this layer at all.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --project /app/nemo          && \
    uv sync --frozen --no-dev --project /app/translate-gemma && \
    uv sync --frozen --no-dev --project /app/qwen3-tts     && \
    uv sync --frozen --no-dev --project /app/whisper

# ── Source code — separate layer so code edits don't bust the dep cache ───────
COPY run_pipeline.py pipeline_utils.py pipeline_paths.py  /app/
COPY nemo/       /app/nemo/
COPY translate-gemma/ /app/translate-gemma/
COPY qwen3-tts/  /app/qwen3-tts/
COPY whisper/    /app/whisper/

# ── Runtime env ───────────────────────────────────────────────────────────────
# OLLAMA_HOST points to the separate ollama container in docker-compose
ENV OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_BIN=ollama

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
