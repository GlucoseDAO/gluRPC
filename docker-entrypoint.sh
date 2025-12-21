#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== gluRPC Service Entrypoint ===${NC}"

# Ensure logs directory exists with proper permissions
LOGS_DIR="/app/logs"
echo -e "${YELLOW}Checking logs directory...${NC}"
if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
    chmod 755 "$LOGS_DIR"
    echo -e "${GREEN}  ✓ Created logs directory: $LOGS_DIR${NC}"
else
    echo -e "${GREEN}  ✓ Logs directory exists: $LOGS_DIR${NC}"
fi

echo -e "${YELLOW}Directory permissions:${NC}"
ls -lhd "$LOGS_DIR" || echo "  ⚠ Could not read logs directory permissions"
echo ""

# Populate snetd_daemon directory if empty or missing essential files
# Simply copy each artifact if it doesn't exist on the host mount
DAEMON_DIR="/app/snetd_daemon"
ARTIFACTS_DIR="/app/snetd_daemon_artifacts"

if [ -d "$DAEMON_DIR" ] && [ -d "$ARTIFACTS_DIR" ]; then
    echo -e "${YELLOW}Checking SNET daemon artifacts...${NC}"
    
    # Copy Dockerfile if missing
    if [ ! -f "$DAEMON_DIR/Dockerfile.snetd" ] && [ -f "$ARTIFACTS_DIR/Dockerfile.snetd" ]; then
        cp "$ARTIFACTS_DIR/Dockerfile.snetd" "$DAEMON_DIR/"
        echo -e "  ${GREEN}✓${NC} Copied Dockerfile.snetd"
    fi
    
    # Copy docker-compose.full.yml if missing
    if [ ! -f "$DAEMON_DIR/docker-compose.full.yml" ] && [ -f "$ARTIFACTS_DIR/docker-compose.full.yml" ]; then
        cp "$ARTIFACTS_DIR/docker-compose.full.yml" "$DAEMON_DIR/"
        echo -e "  ${GREEN}✓${NC} Copied docker-compose.full.yml"
    fi
    
    # Create snetd_configs directory and copy config
    mkdir -p "$DAEMON_DIR/snetd_configs"
    if [ ! -f "$DAEMON_DIR/snetd_configs/snetd.sepolia.json" ] && [ -f "$ARTIFACTS_DIR/snetd.sepolia.json" ]; then
        cp "$ARTIFACTS_DIR/snetd.sepolia.json" "$DAEMON_DIR/snetd_configs/"
        echo -e "  ${GREEN}✓${NC} Copied snetd.sepolia.json"
    fi
    
    # Copy documentation if missing
    if [ ! -f "$DAEMON_DIR/daemon_ssl.md" ] && [ -f "$ARTIFACTS_DIR/daemon_ssl.md" ]; then
        cp "$ARTIFACTS_DIR/daemon_ssl.md" "$DAEMON_DIR/"
        echo -e "  ${GREEN}✓${NC} Copied daemon_ssl.md"
    fi
    
    if [ ! -f "$DAEMON_DIR/snet_doc.md" ] && [ -f "$ARTIFACTS_DIR/snet_doc.md" ]; then
        cp "$ARTIFACTS_DIR/snet_doc.md" "$DAEMON_DIR/"
        echo -e "  ${GREEN}✓${NC} Copied snet_doc.md"
    fi
    
    if [ ! -f "$DAEMON_DIR/README.md" ] && [ -f "$ARTIFACTS_DIR/README.md" ]; then
        cp "$ARTIFACTS_DIR/README.md" "$DAEMON_DIR/"
        echo -e "  ${GREEN}✓${NC} Copied README.md"
    fi
    
    # Create certs directory if missing
    if [ ! -d "$DAEMON_DIR/certs" ]; then
        mkdir -p "$DAEMON_DIR/certs"
        echo -e "  ${GREEN}✓${NC} Created certs/ directory"
    fi
    
    echo -e "${GREEN}✓ SNET daemon directory ready${NC}"
fi

# Ensure we're in the correct working directory
cd /app || {
    echo -e "${RED}ERROR: Could not change to /app directory${NC}"
    exit 1
}
echo -e "${GREEN}Working directory: $(pwd)${NC}"
echo ""

# Execute the main command
echo -e "${GREEN}Starting gluRPC service...${NC}"
exec "$@"
