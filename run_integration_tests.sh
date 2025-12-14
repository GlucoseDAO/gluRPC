#!/bin/bash
set -e

# Script to run integration tests against a running Docker container

echo "========================================"
echo "Integration Tests for gluRPC Docker"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is running
echo "Checking if Docker container is running..."
if ! docker ps | grep -q glurpc-service; then
    echo -e "${RED}Error: glurpc-service container is not running${NC}"
    echo ""
    echo "Please start the Docker container first:"
    echo "  docker-compose up -d"
    echo ""
    echo "Or run without -d to see logs:"
    echo "  docker-compose up"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Docker container is running${NC}"
echo ""

# Check if REST endpoint is responding
echo "Checking REST endpoint (localhost:8000)..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}Error: REST endpoint not responding${NC}"
    echo "Container may still be starting up. Wait a moment and try again."
    exit 1
fi
echo -e "${GREEN}✓ REST endpoint is responding${NC}"
echo ""

# Check if gRPC endpoint is responding
echo "Checking gRPC endpoint (localhost:7003)..."
if ! nc -z localhost 7003 2>/dev/null; then
    echo -e "${YELLOW}Warning: gRPC endpoint check inconclusive (nc not available or port not open)${NC}"
    echo "Continuing anyway..."
else
    echo -e "${GREEN}✓ gRPC endpoint is listening${NC}"
fi
echo ""

# Run the integration tests
echo "Running integration tests..."
echo ""

uv run pytest tests/test_combined_container.py -v -s --tb=short "$@"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================"
    echo "All integration tests passed!"
    echo "========================================${NC}"
else
    echo -e "${RED}========================================"
    echo "Some integration tests failed!"
    echo "========================================${NC}"
fi

exit $TEST_EXIT_CODE
