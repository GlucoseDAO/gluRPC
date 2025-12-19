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

# Create SNET directories
RUN mkdir -p /app/etcd /app/snetd_configs /app/.certs && \
    chmod 755 /app/etcd /app/snetd_configs /app/.certs

# Install SNET daemon (override version via build-arg)
ARG SNETD_VERSION
RUN set -eux; \
    latest="$(curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')"; \
    version="${SNETD_VERSION:-${latest}}"; \
    wget -O /usr/local/bin/snetd "https://github.com/singnet/snet-daemon/releases/download/${version}/snetd-linux-amd64-${version}"; \
    chmod +x /usr/local/bin/snetd

# Install glurpc from PyPI
# Specify version or use latest

ARG GLUCOSEDAO_GLUCOBENCH_VERSION=0.4.1
RUN uv pip install --system "glucosedao-glucobench>=${GLUCOSEDAO_GLUCOBENCH_VERSION}"
ARG GLURPC_VERSION=0.5.11
RUN uv pip install --system "glurpc>=${GLURPC_VERSION}"

# Copy run_glurpc_service.py (not included in PyPI package)
# This script is used by the glurpc-combined entrypoint
COPY run_glurpc_service.py /usr/local/lib/python3.13/site-packages/run_glurpc_service.py

# Copy documentation and license
COPY LICENSE README.md /app/

# Copy SNET daemon configuration files (will be used as defaults)
COPY snetd_configs /app/snetd_configs_default

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create directories for cache and logs
# These can be mounted as volumes for persistence
RUN mkdir -p /app/cache_storage /app/logs && \
    chmod 755 /app/cache_storage /app/logs

# Define volumes for external mounting
VOLUME ["/app/cache_storage", "/app/logs", "/app/etcd", "/app/snetd_configs", "/app/.certs"]

# Environment variables with defaults
# --- Cache Configuration ---
ENV MAX_CACHE_SIZE=128 \
    ENABLE_CACHE_PERSISTENCE=True

# --- Data Processing Configuration ---
# Note: MINIMUM_DURATION_MINUTES defaults to 540 (model requirement: 5min * (96+12))
# Note: MAXIMUM_WANTED_DURATION defaults to 1080 (2x minimum)
# Uncomment and set if you want to override:
# ENV MINIMUM_DURATION_MINUTES=540
# ENV MAXIMUM_WANTED_DURATION=1080

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
    LOG_LEVEL_LOCKS=ERROR \
    VERBOSE=False

# Expose ports for both gRPC and REST
# 7003 for gRPC, 8000 for REST, 7000 for SNET daemon
EXPOSE 7003 8000 7000

# Health check using the REST endpoint
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command to run the combined service (no daemon by default)
CMD ["glurpc-combined", "--combined"]
