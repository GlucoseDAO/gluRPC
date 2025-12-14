# Modern, compact Dockerfile for gluRPC service
# Uses published PyPI package
FROM python:3.13-slim-trixie

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install build dependencies for packages that need compilation (statsforecast)
# Keep the image relatively small by cleaning up after
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
        wget \
        && \
    rm -rf /var/lib/apt/lists/*

# Install SNET daemon (override version via build-arg)
ARG SNETD_VERSION
RUN set -eux; \
    latest="$(curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")' || true)"; \
    version="${SNETD_VERSION:-${latest:-v3.1.6}}"; \
    tmpdir="$(mktemp -d)"; \
    cd "$tmpdir"; \
    wget "https://github.com/singnet/snet-daemon/releases/download/${version}/snet-daemon-${version}-linux-amd64.tar.gz"; \
    tar -xvf "snet-daemon-${version}-linux-amd64.tar.gz"; \
    mv "snet-daemon-${version}-linux-amd64/snetd" /usr/local/bin/snetd; \
    rm -rf "$tmpdir"

# Install glurpc from PyPI
# Specify version or use latest
ARG GLURPC_VERSION=0.5.5
RUN uv pip install --system "glurpc>=${GLURPC_VERSION}"

# Create directories for cache and logs
# These can be mounted as volumes for persistence
RUN mkdir -p /app/cache_storage /app/logs && \
    chmod 755 /app/cache_storage /app/logs

# Define volumes for external mounting
VOLUME ["/app/cache_storage", "/app/logs"]

# Environment variables with defaults
# --- Cache Configuration ---
ENV MAX_CACHE_SIZE=128 \
    ENABLE_CACHE_PERSISTENCE=True

# --- Data Processing Configuration ---
ENV MINIMUM_DURATION_MINUTES="" \
    MAXIMUM_WANTED_DURATION=""

# --- API Configuration ---
ENV ENABLE_API_KEYS=False

# --- Model and Inference Configuration ---
ENV NUM_COPIES_PER_DEVICE=2 \
    BACKGROUND_WORKERS_COUNT=4 \
    BATCH_SIZE=32 \
    NUM_SAMPLES=10

# --- Timeout Configuration ---
ENV INFERENCE_TIMEOUT_GPU=600.0 \
    INFERENCE_TIMEOUT_CPU=7200.0

# --- Queue Configuration ---
ENV MAX_INFERENCE_QUEUE_SIZE=64 \
    MAX_CALC_QUEUE_SIZE=8192

# --- Logging Configuration ---
ENV LOG_LEVEL_ROOT=INFO \
    LOG_LEVEL_LOGIC=INFO \
    LOG_LEVEL_ENGINE=INFO \
    LOG_LEVEL_CORE=INFO \
    LOG_LEVEL_APP=INFO \
    LOG_LEVEL_STATE=INFO \
    LOG_LEVEL_CACHE=INFO \
    LOG_LEVEL_LOCKS=ERROR

# Expose ports for both gRPC and REST
# 7003 for gRPC, 8000 for REST
EXPOSE 7003 8000

# Health check using the REST endpoint
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command to run the combined service
CMD ["glurpc-combined", "--combined", "--no-daemon"]
