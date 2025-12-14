#!/usr/bin/env bash
set -euo pipefail

# Docker build script for gluRPC
# Usage: ./build-docker.sh [VERSION] [OPTIONS]
#   VERSION: Package version to install (default: from pyproject.toml)
#   OPTIONS: Additional docker build options

# Configuration
ORGANIZATION="glucosedao"
IMAGE_NAME="glurpc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get version from pyproject.toml if not provided
if [ -f "${SCRIPT_DIR}/pyproject.toml" ]; then
    DEFAULT_VERSION=$(grep -m 1 '^version = ' "${SCRIPT_DIR}/pyproject.toml" | cut -d'"' -f2)
else
    DEFAULT_VERSION="latest"
fi

VERSION="${1:-${DEFAULT_VERSION}}"
shift || true  # Remove first argument if it exists

# Build arguments
BUILD_ARGS=("$@")

echo "=========================================="
echo "Building gluRPC Docker Image"
echo "=========================================="
echo "Organization: ${ORGANIZATION}"
echo "Image:        ${IMAGE_NAME}"
echo "Version:      ${VERSION}"
echo "Build dir:    ${SCRIPT_DIR}"
echo "=========================================="

# Build with multiple tags
docker build \
    --build-arg GLURPC_VERSION="${VERSION}" \
    -t "${ORGANIZATION}/${IMAGE_NAME}:${VERSION}" \
    -t "${ORGANIZATION}/${IMAGE_NAME}:latest" \
    -t "${IMAGE_NAME}:local" \
    "${BUILD_ARGS[@]}" \
    "${SCRIPT_DIR}"

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Tagged images:"
echo "  - ${ORGANIZATION}/${IMAGE_NAME}:${VERSION}"
echo "  - ${ORGANIZATION}/${IMAGE_NAME}:latest"
echo "  - ${IMAGE_NAME}:local"
echo "=========================================="
echo ""
echo "Run with:"
echo "  docker run -p 7003:7003 -p 8000:8000 ${IMAGE_NAME}:local"
echo "  docker run -p 7003:7003 -p 8000:8000 ${ORGANIZATION}/${IMAGE_NAME}:${VERSION}"
echo "=========================================="
