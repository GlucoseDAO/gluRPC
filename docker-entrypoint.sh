#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== gluRPC SNET Daemon Entrypoint ===${NC}"

# Create ETCD data directories for different networks
ETCD_BASE_DIR="/app/etcd"
NETWORKS=("sepolia" "ropsten" "mainnet" "testnet")

echo -e "${YELLOW}Creating ETCD data directories...${NC}"
for network in "${NETWORKS[@]}"; do
    if [ ! -d "${ETCD_BASE_DIR}/${network}" ]; then
        mkdir -p "${ETCD_BASE_DIR}/${network}"
        chmod 755 "${ETCD_BASE_DIR}/${network}"
        echo "  ✓ Created: ${ETCD_BASE_DIR}/${network}"
    else
        echo "  ✓ Directory already exists: ${ETCD_BASE_DIR}/${network}"
    fi
done

# Copy SNET daemon config files if they don't exist in mounted config directory
CONFIG_SOURCE="/app/snetd_configs_default"
CONFIG_TARGET="/app/snetd_configs"

if [ -d "$CONFIG_SOURCE" ]; then
    echo -e "${YELLOW}Checking SNET daemon configuration files...${NC}"
    mkdir -p "$CONFIG_TARGET"
    
    for config_file in "$CONFIG_SOURCE"/*; do
        if [ -f "$config_file" ]; then
            filename=$(basename "$config_file")
            target_file="$CONFIG_TARGET/$filename"
            
            if [ ! -f "$target_file" ]; then
                cp "$config_file" "$target_file"
                echo -e "  ${GREEN}✓${NC} Copied: $filename"
            else
                echo -e "  ${YELLOW}→${NC} Exists: $filename (skipping)"
            fi
        fi
    done
fi

# Check SSL certificate configuration
CERTS_DIR="/app/.certs"
if [ -d "$CERTS_DIR" ]; then
    echo -e "${YELLOW}Checking SSL certificates...${NC}"
    
    # Check if certificates directory is empty or missing required files
    if [ ! -f "$CERTS_DIR/fullchain.pem" ] || [ ! -f "$CERTS_DIR/privkey.pem" ]; then
        if [ "${REQUIRE_SSL:-false}" = "true" ]; then
            echo -e "${RED}ERROR: SSL certificates required but not found!${NC}"
            echo -e "${YELLOW}To generate SSL certificates, follow the instructions in:${NC}"
            
            # Copy SSL documentation to certs directory
            SSL_DOC_SOURCE="/app/snetd_configs_default/daemon_ssl.md"
            SSL_DOC_TARGET="$CERTS_DIR/HOW_TO_GENERATE_SSL_CERTS.md"
            
            if [ -f "$SSL_DOC_SOURCE" ] && [ ! -f "$SSL_DOC_TARGET" ]; then
                cp "$SSL_DOC_SOURCE" "$SSL_DOC_TARGET"
                echo -e "${GREEN}Copied SSL documentation to: $SSL_DOC_TARGET${NC}"
            fi
            
            echo ""
            echo -e "${YELLOW}Required certificate files:${NC}"
            echo "  - $CERTS_DIR/fullchain.pem"
            echo "  - $CERTS_DIR/privkey.pem"
            echo ""
            echo -e "${YELLOW}Please mount your SSL certificates and restart the container.${NC}"
            echo -e "${YELLOW}Or set REQUIRE_SSL=false to run without SSL.${NC}"
            exit 1
        else
            echo -e "${YELLOW}  ⚠ SSL certificates not found (REQUIRE_SSL=false, continuing)${NC}"
        fi
    else
        echo -e "${GREEN}  ✓ SSL certificates found${NC}"
    fi
fi

# Display configuration summary
echo -e "${GREEN}=== Configuration Summary ===${NC}"
echo "  ETCD Base Directory: $ETCD_BASE_DIR"
echo "  Config Directory: $CONFIG_TARGET"
echo "  Certs Directory: $CERTS_DIR"
echo "  Require SSL: ${REQUIRE_SSL:-false}"
echo ""

# Execute the main command
echo -e "${GREEN}Starting application...${NC}"
exec "$@"
