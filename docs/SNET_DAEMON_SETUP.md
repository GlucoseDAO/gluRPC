# SNET Daemon Integration Guide

This guide explains how to use the integrated SNET (SingularityNET) daemon with gluRPC.

## Overview

The gluRPC Docker image now includes the SNET daemon (`snetd`) which enables blockchain-based payments and integration with the SingularityNET marketplace.

### What's Included

The Docker container includes:
- **SNET daemon binary** (`snetd`) - Latest version from GitHub releases
- **Custom entrypoint script** - Automatic setup of directories and configurations
- **Default configuration files** - Template configs for different networks
- **Persistent volumes** - Separate storage for ETCD, configs, certificates, cache, and logs
- **Flexible deployment** - Run with or without daemon, with or without SSL

## Quick Start

### 1. Directory Structure

The container uses the following directories:

```
/app/
├── etcd/                    # ETCD data storage (persistent)
│   ├── sepolia/            # Sepolia testnet data
│   ├── ropsten/            # Ropsten testnet data
│   ├── mainnet/            # Mainnet data
│   └── testnet/            # Generic testnet data
├── snetd_configs/          # Daemon configuration files (editable)
├── .certs/                 # SSL certificates
├── cache_storage/          # gluRPC cache
└── logs/                   # Application logs
```

### 2. Volume Mounts

The docker-compose.yml includes these persistent volumes:

#### Named Volumes (Managed by Docker)

- **glurpc-cache**: gluRPC prediction cache
  - Container path: `/app/cache_storage`
  - Persistent across container rebuilds
  
- **glurpc-etcd**: ETCD blockchain payment channel data ⚠️ **CRITICAL - DO NOT DELETE**
  - Container path: `/app/etcd`
  - Contains payment channel state and transaction data
  - Loss of this data means loss of access to payment channels

#### Host Directory Mounts (Editable)

- **./snetd_configs** → `/app/snetd_configs`
  - Daemon configuration files (JSON)
  - Editable on host, automatically copied from defaults on first run
  
- **./certs** → `/app/.certs`
  - SSL certificates (fullchain.pem, privkey.pem)
  - Optional, required only if `REQUIRE_SSL=true`
  
- **./logs** → `/app/logs`
  - Application and daemon logs
  - Easy access for monitoring and debugging
  
- **./api-keys.txt** → `/app/api-keys.txt` (read-only)
  - API keys for REST endpoint authentication
  - One key per line

#### Volume Management

```bash
# List volumes
docker volume ls | grep glurpc

# Inspect ETCD volume (check size, location)
docker volume inspect glurpc-etcd

# DANGER: Only if you want to completely reset payment channels
docker volume rm glurpc-etcd  # ⚠️ Will lose payment channel data!

# Backup ETCD data
docker run --rm -v glurpc-etcd:/data -v $(pwd):/backup \
  alpine tar czf /backup/etcd-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restore ETCD data
docker run --rm -v glurpc-etcd:/data -v $(pwd):/backup \
  alpine sh -c "cd /data && tar xzf /backup/etcd-backup-YYYYMMDD.tar.gz"
```

### 3. Docker Entrypoint Script

The container uses a custom entrypoint script (`/docker-entrypoint.sh`) that runs before the main application:

#### What the Entrypoint Does

1. **Creates ETCD network directories** (if they don't exist):
   - `/app/etcd/sepolia`
   - `/app/etcd/ropsten`
   - `/app/etcd/mainnet`
   - `/app/etcd/testnet`

2. **Copies default configuration files** from `/app/snetd_configs_default/` to `/app/snetd_configs/`:
   - Only copies if files don't exist (won't overwrite your edits)
   - Includes: `snetd.sepolia.json`, `snet_doc.md`, `daemon_ssl.md`

3. **Checks SSL certificates** (if `REQUIRE_SSL=true`):
   - Verifies `fullchain.pem` and `privkey.pem` exist
   - If missing, copies SSL documentation and exits with error
   - If `REQUIRE_SSL=false`, continues without SSL

4. **Displays configuration summary** and starts the application

#### Entrypoint Logs

On container startup, you'll see output like:

```
=== gluRPC SNET Daemon Entrypoint ===
Creating ETCD data directories...
  ✓ Created: /app/etcd/sepolia
  ✓ Created: /app/etcd/ropsten
  ✓ Created: /app/etcd/mainnet
  ✓ Created: /app/etcd/testnet
Checking SNET daemon configuration files...
  ✓ Copied: snetd.sepolia.json
  ✓ Copied: snet_doc.md
  ✓ Copied: daemon_ssl.md
Checking SSL certificates...
  ⚠ SSL certificates not found (REQUIRE_SSL=false, continuing)
=== Configuration Summary ===
  ETCD Base Directory: /app/etcd
  Config Directory: /app/snetd_configs
  Certs Directory: /app/.certs
  Require SSL: false

Starting application...
```

### 4. Configuration

#### Edit Configuration Files

After first run, edit the configuration files in `./snetd_configs/`:

```bash
cd snetd_configs
nano snetd.sepolia.json  # For testnet
nano snetd.mainnet.json  # For mainnet (create if needed)
```

Replace these placeholders:

- `<YOUR_API_KEY>`: Your Alchemy API key
- `<ORGANIZATION_ID>`: Your SingularityNET organization ID
- `<SERVICE_ID>`: Your service ID
- Update `daemon_group_name`, `private_key_for_metering`, etc.

See `snetd_configs/snet_doc.md` for detailed configuration instructions.

### 5. Building and Running with Docker Compose

#### First Time Setup

```bash
# 1. Clone repository
cd gluRPC

# 2. Create necessary host directories (optional, Docker will create them)
mkdir -p snetd_configs certs logs

# 3. First run - initializes default configs
docker-compose up -d

# 4. Check logs for entrypoint output
docker-compose logs glurpc

# 5. Stop container to edit configs
docker-compose down

# 6. Edit configuration files
nano snetd_configs/snetd.sepolia.json
# Replace placeholders: <YOUR_API_KEY>, organization_id, service_id, etc.

# 7. (Optional) Set up SSL certificates
# See section 6 below

# 8. Start with daemon enabled
# Edit docker-compose.yml to add:
#   command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.sepolia.json"]

docker-compose up -d
```

#### Docker Compose Configuration

The provided `docker-compose.yml` includes:

```yaml
version: '3.8'

services:
  glurpc:
    image: glucosedao/glurpc:latest
    container_name: glurpc-service
    restart: unless-stopped
    
    ports:
      - "127.0.0.1:7003:7003"  # gRPC
      - "8000:8000"            # REST API
      # Optional ports (commented out):
      # - "127.0.0.1:7000:7000"   # SNET daemon
      # - "127.0.0.1:80:80"       # HTTP (for Let's Encrypt)
      # - "127.0.0.1:2379:2379"   # ETCD client (for external access)
      # - "127.0.0.1:2380:2380"   # ETCD peer (for cluster config)
    
    volumes:
      - glurpc-cache:/app/cache_storage:z
      - glurpc-etcd:/app/etcd:z
      - ./logs:/app/logs:z
      - ./snetd_configs:/app/snetd_configs:z
      - ./certs:/app/.certs:z
      - ./api-keys.txt:/app/api-keys.txt:ro,Z
    
    environment:
      REQUIRE_SSL: "False"
      ENABLE_API_KEYS: "True"
      # ... other settings

volumes:
  glurpc-cache:
    name: glurpc-cache
  glurpc-etcd:
    name: glurpc-etcd
    driver: local
```

#### Building from Source

If you want to build the image locally:

```bash
# Build with default SNET daemon version (latest)
docker build -t glucosedao/glurpc:latest .

# Build with specific SNET daemon version
docker build --build-arg SNETD_VERSION=v6.2.0 -t glucosedao/glurpc:latest .

# Build with specific gluRPC version
docker build --build-arg GLURPC_VERSION=0.5.5 -t glucosedao/glurpc:latest .

# Build with both
docker build \
  --build-arg SNETD_VERSION=v6.2.0 \
  --build-arg GLURPC_VERSION=0.5.5 \
  -t glucosedao/glurpc:latest .
```

#### Running without Docker Compose

```bash
# Run with default settings (no daemon)
docker run -d \
  -p 8000:8000 \
  -p 127.0.0.1:7003:7003 \
  -v glurpc-cache:/app/cache_storage \
  -v glurpc-etcd:/app/etcd \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/snetd_configs:/app/snetd_configs \
  -v $(pwd)/certs:/app/.certs \
  --name glurpc-service \
  glucosedao/glurpc:latest

# Run with daemon enabled
docker run -d \
  -p 8000:8000 \
  -p 127.0.0.1:7003:7003 \
  -p 127.0.0.1:7000:7000 \
  -v glurpc-cache:/app/cache_storage \
  -v glurpc-etcd:/app/etcd \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/snetd_configs:/app/snetd_configs \
  -v $(pwd)/certs:/app/.certs \
  -e REQUIRE_SSL=false \
  --name glurpc-service \
  glucosedao/glurpc:latest \
  glurpc-combined --combined --daemon-config /app/snetd_configs/snetd.sepolia.json
```

### 6. SSL Certificates

#### Option A: Let's Encrypt (Recommended for Production)

```bash
# Generate certificates on your host
sudo certbot certonly --standalone -d your-domain.com

# Mount the certificates in docker-compose.yml
# Uncomment and modify:
# - /etc/letsencrypt:/app/.certs:ro,Z
```

#### Option B: Development Mode (No SSL)

Set `REQUIRE_SSL=false` in docker-compose.yml (default).

If you start with `REQUIRE_SSL=true` and certificates are missing, the container will:
1. Copy SSL setup instructions to `./certs/HOW_TO_GENERATE_SSL_CERTS.md`
2. Exit with instructions

#### Option C: Existing Let's Encrypt Certificates

Mount your existing certificates directly:

```yaml
# In docker-compose.yml
volumes:
  # Mount the entire letsencrypt directory (certificates are symlinks)
  - /etc/letsencrypt:/etc/letsencrypt:ro
  
# Update your daemon config paths:
# "ssl_cert": "/etc/letsencrypt/live/your-domain.com/fullchain.pem"
# "ssl_key": "/etc/letsencrypt/live/your-domain.com/privkey.pem"
```

#### Option D: Custom Certificate Location

```yaml
# In docker-compose.yml - replace the certs mount
volumes:
  - /path/to/your/certs:/app/.certs:ro

# Your certificate directory should contain:
# - fullchain.pem
# - privkey.pem
```

### 7. Running with SNET Daemon

To start the service **with** SNET daemon:

```bash
# Edit your configuration files first
nano snetd_configs/snetd.sepolia.json

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f glurpc
```

To start **without** SNET daemon (default):

The default command already includes `--no-daemon`:

```bash
docker-compose up -d
```

### 8. Running with Custom Daemon Config

You can override the command to use a specific daemon configuration:

```bash
# In docker-compose.yml, change CMD:
command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.sepolia.json"]

# Or run directly:
docker run glucosedao/glurpc glurpc-combined \
  --combined \
  --daemon-config /app/snetd_configs/snetd.sepolia.json
```

## Environment Variables

Add these to docker-compose.yml environment section:

```yaml
environment:
  # SSL requirement for SNET daemon
  REQUIRE_SSL: "false"  # Set to "true" for production with SSL
  
  # Other gluRPC settings...
  ENABLE_API_KEYS: "True"
  MAX_CACHE_SIZE: 128
```

## Port Configuration

All ports in docker-compose.yml are configured for security and flexibility:

### Active Ports (Exposed by Default)

- **8000**: REST API (gluRPC)
  - Bound to `0.0.0.0` (all interfaces) - accessible externally
  - Used for HTTP REST API endpoints
  
- **7003**: gRPC service (gluRPC)
  - Bound to `127.0.0.1` (localhost only) - local access only
  - Used for internal gRPC communication

### Optional Ports (Commented Out)

Uncomment these in `docker-compose.yml` as needed:

- **7000**: SNET daemon endpoint
  - Bound to `127.0.0.1:7000:7000` (localhost only)
  - Used for blockchain payment verification
  - Uncomment if you want external services to access the daemon

- **80**: HTTP port for Let's Encrypt
  - Bound to `127.0.0.1:80:80` (localhost only)
  - Required for automatic SSL certificate generation/renewal
  - Only needed if using Let's Encrypt inside the container

- **2379**: ETCD client port
  - Bound to `127.0.0.1:2379:2379` (localhost only)
  - Used for external ETCD client access
  - Uncomment if you want to inspect/manage ETCD data externally
  - Useful for debugging payment channel state

- **2380**: ETCD peer port
  - Bound to `127.0.0.1:2380:2380` (localhost only)
  - Used for ETCD cluster peer communication
  - Uncomment if setting up multi-node ETCD cluster
  - Required for high-availability ETCD configurations

### Example: Exposing ETCD for External Access

```yaml
ports:
  - "127.0.0.1:7003:7003"
  - "8000:8000"
  - "127.0.0.1:2379:2379"  # Uncommented

# Now you can access ETCD from host:
# etcdctl --endpoints=http://localhost:2379 get "" --prefix
```

### Example: Multi-Node ETCD Cluster Setup

For a high-availability payment channel storage:

```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:7003:7003"
  - "8000:8000"
  - "127.0.0.1:7000:7000"    # Daemon
  - "127.0.0.1:2379:2379"    # Client
  - "127.0.0.1:2380:2380"    # Peer

# Update daemon config for cluster mode:
# "payment_channel_storage_server": {
#   "cluster": "storage-1=http://node1:2380,storage-2=http://node2:2380,storage-3=http://node3:2380",
#   "peer_port": 2380,
#   ...
# }
```

### Security Considerations

- **Localhost binding** (`127.0.0.1`): Only accessible from the host machine
- **All interfaces** (`0.0.0.0`): Accessible from any network interface
- Use firewall rules for additional protection
- Never expose ETCD ports to public internet without authentication

## Container Management

### Common Operations

```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# View logs (follow mode)
docker-compose logs -f glurpc

# View logs (last 100 lines)
docker-compose logs --tail=100 glurpc

# Restart service
docker-compose restart glurpc

# Rebuild and restart (after config changes)
docker-compose up -d --force-recreate

# Enter container shell
docker exec -it glurpc-service bash

# Check container status
docker-compose ps

# View resource usage
docker stats glurpc-service
```

### Updating the Container

```bash
# Pull latest image
docker-compose pull

# Restart with new image
docker-compose up -d

# Check new version
docker exec -it glurpc-service snetd version
```

### Managing Volumes

```bash
# List volumes
docker volume ls

# Inspect volume details
docker volume inspect glurpc-etcd
docker volume inspect glurpc-cache

# Backup ETCD volume
docker run --rm \
  -v glurpc-etcd:/source \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/etcd-$(date +%Y%m%d-%H%M%S).tar.gz -C /source .

# Backup cache volume
docker run --rm \
  -v glurpc-cache:/source \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/cache-$(date +%Y%m%d-%H%M%S).tar.gz -C /source .

# Restore ETCD volume from backup
docker run --rm \
  -v glurpc-etcd:/target \
  -v $(pwd)/backups:/backup \
  alpine sh -c "cd /target && tar xzf /backup/etcd-TIMESTAMP.tar.gz"

# Clean up old volumes (DANGEROUS - make backups first!)
docker-compose down -v  # Removes volumes defined in compose
```

### Configuration Updates

```bash
# After editing config files on host
nano snetd_configs/snetd.sepolia.json

# Restart container to apply changes
docker-compose restart glurpc

# Verify config is loaded
docker exec -it glurpc-service cat /app/snetd_configs/snetd.sepolia.json

# Check if daemon picked up new config (if running)
docker-compose logs glurpc | grep -i config
```

## Important Notes

### ETCD Data Persistence

⚠️ **CRITICAL**: The `/app/etcd` directory contains blockchain payment channel data. 

**DO NOT DELETE** this directory or the `glurpc-etcd` volume, or you will lose access to payment channels and prevent token withdrawals.

### Network-Specific Data

Each blockchain network has its own ETCD directory:

- `/app/etcd/sepolia`: Sepolia testnet
- `/app/etcd/mainnet`: Ethereum mainnet
- `/app/etcd/ropsten`: Ropsten testnet
- `/app/etcd/testnet`: Generic testnet

The `data_dir` in your daemon config should point to the appropriate directory:

```json
{
  "payment_channel_storage_server": {
    "data_dir": "/app/etcd/sepolia"
  }
}
```

### Switching Networks

To switch from testnet to mainnet:

1. Stop the container
2. Create/edit `snetd_configs/snetd.mainnet.json`
3. Update `blockchain_network_selected` to `"main"`
4. Update RPC endpoints to mainnet
5. Change `data_dir` to `/app/etcd/mainnet`
6. Re-register organization and service on mainnet
7. Update command to use new config
8. Restart container

See `snetd_configs/snet_doc.md` for detailed switching instructions.

## Troubleshooting

### Check SNET Daemon Status

```bash
# Enter the container
docker exec -it glurpc-service bash

# Check if snetd is installed
snetd version

# Check daemon configuration
cat /app/snetd_configs/snetd.sepolia.json

# Check ETCD directories
ls -la /app/etcd/
```

### View Daemon Logs

The daemon logs are integrated with the main application logs:

```bash
docker-compose logs -f glurpc
```

### SSL Certificate Issues

If you get SSL errors:

1. Verify certificates exist:
   ```bash
   ls -la ./certs/
   ```

2. Certificates should include:
   - `fullchain.pem`
   - `privkey.pem`

3. If using Let's Encrypt, mount the full `/etc/letsencrypt` directory

### ETCD Connection Issues

If payment channel storage fails:

1. Check `data_dir` path in configuration
2. Verify directory permissions: `chmod 755 /app/etcd/*`
3. Check ETCD ports are not blocked (2379, 2380)
4. Verify ETCD is actually running:
   ```bash
   docker exec -it glurpc-service ps aux | grep etcd
   ```
5. Check if ETCD client port is accessible:
   ```bash
   # Uncomment port 2379 in docker-compose.yml, then:
   curl http://localhost:2379/version
   ```

### Container Won't Start

1. Check entrypoint logs:
   ```bash
   docker-compose logs glurpc
   ```

2. Verify required volumes exist:
   ```bash
   docker volume ls | grep glurpc
   ```

3. Check for SSL requirement:
   ```bash
   # If REQUIRE_SSL=true, verify certificates exist:
   ls -la ./certs/
   ```

4. Try running with shell access:
   ```bash
   docker run --rm -it \
     -v glurpc-etcd:/app/etcd \
     -v $(pwd)/snetd_configs:/app/snetd_configs \
     glucosedao/glurpc:latest /bin/bash
   ```

### Permission Issues

If you see permission errors with mounted directories:

```bash
# Check SELinux context (if on RHEL/CentOS/Fedora)
ls -laZ ./snetd_configs
ls -laZ ./logs

# Fix SELinux labels
sudo chcon -Rt svirt_sandbox_file_t ./snetd_configs
sudo chcon -Rt svirt_sandbox_file_t ./logs
sudo chcon -Rt svirt_sandbox_file_t ./certs

# Or use :z/:Z flags in docker-compose.yml (already included)
```

### Daemon Not Starting

If the SNET daemon isn't starting when configured:

1. Verify command in docker-compose.yml includes daemon config:
   ```yaml
   command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.sepolia.json"]
   ```

2. Check daemon config syntax:
   ```bash
   docker exec -it glurpc-service cat /app/snetd_configs/snetd.sepolia.json | python -m json.tool
   ```

3. Test daemon binary:
   ```bash
   docker exec -it glurpc-service snetd version
   docker exec -it glurpc-service snetd help serve
   ```

4. Check if service endpoint is accessible:
   ```bash
   # From inside container
   docker exec -it glurpc-service curl http://localhost:7003
   ```

## Documentation

- Full SNET daemon configuration: `snetd_configs/snet_doc.md`
- SSL setup instructions: `snetd_configs/daemon_ssl.md`
- gluRPC documentation: See main README.md
- Dockerfile: See `Dockerfile` for build details
- Entrypoint script: See `docker-entrypoint.sh` for initialization logic

## Advanced Topics

### Custom Entrypoint Behavior

You can override entrypoint behavior with environment variables:

```yaml
environment:
  REQUIRE_SSL: "true"    # Enforce SSL certificate presence
  # Add custom variables as needed
```

### Multi-Network Setup

Run multiple instances for different networks:

```yaml
# docker-compose.yml
services:
  glurpc-sepolia:
    image: glucosedao/glurpc:latest
    container_name: glurpc-sepolia
    ports:
      - "8001:8000"
      - "127.0.0.1:7004:7003"
    volumes:
      - glurpc-etcd-sepolia:/app/etcd
      - ./snetd_configs:/app/snetd_configs
    command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.sepolia.json"]
  
  glurpc-mainnet:
    image: glucosedao/glurpc:latest
    container_name: glurpc-mainnet
    ports:
      - "8002:8000"
      - "127.0.0.1:7005:7003"
    volumes:
      - glurpc-etcd-mainnet:/app/etcd
      - ./snetd_configs:/app/snetd_configs
    command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.mainnet.json"]

volumes:
  glurpc-etcd-sepolia:
  glurpc-etcd-mainnet:
```

### External ETCD Cluster

To use an external ETCD cluster instead of embedded:

1. Set up external ETCD cluster
2. Modify daemon config:
   ```json
   {
     "payment_channel_storage_server": {
       "enabled": false
     },
     "payment_channel_storage_client": {
       "connection_timeout": "5s",
       "request_timeout": "3s",
       "endpoints": ["http://etcd1:2379", "http://etcd2:2379"]
     }
   }
   ```

3. Update docker-compose.yml networking to reach external ETCD

### Building Custom Images

If you need to customize the image:

```dockerfile
# custom.Dockerfile
FROM glucosedao/glurpc:latest

# Add custom tools
RUN apt-get update && apt-get install -y your-package

# Add custom scripts
COPY custom-script.sh /usr/local/bin/

# Keep the entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["glurpc-combined", "--combined", "--no-daemon"]
```

Build and use:
```bash
docker build -f custom.Dockerfile -t glucosedao/glurpc:custom .
```

### Health Checks and Monitoring

The container includes a built-in health check:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' glurpc-service

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' glurpc-service
```

Add external monitoring:

```yaml
# docker-compose.yml with Prometheus metrics
services:
  glurpc:
    # ... existing config
    labels:
      - "prometheus.scrape=true"
      - "prometheus.port=8000"
      - "prometheus.path=/metrics"
```

## Example: Complete Setup

```bash
# 1. Clone and prepare
cd gluRPC

# 2. Create directories
mkdir -p snetd_configs certs logs

# 3. Start once to get default configs
docker-compose up -d
docker-compose down

# 4. Edit configuration
nano snetd_configs/snetd.sepolia.json
# Replace <YOUR_API_KEY> with Alchemy key
# Replace organization_id, service_id, etc.

# 5. (Optional) Generate SSL certificates
sudo certbot certonly --standalone -d your-domain.com

# 6. Start with daemon
# Modify docker-compose.yml command:
# command: ["glurpc-combined", "--combined", "--daemon-config", "/app/snetd_configs/snetd.sepolia.json"]

docker-compose up -d

# 7. Check status
docker-compose logs -f glurpc
```

## Support

For issues specific to:
- **SNET daemon**: See SingularityNET documentation
- **gluRPC service**: See main project documentation
- **Integration issues**: Check container logs and configuration files
