# gluRPC Deployment Modes

This document explains the two deployment contexts for gluRPC with SNET daemon.

## Overview

gluRPC can be deployed in two modes:
1. **Starter Mode**: Service only (no daemon) - for local development
2. **Full Stack Mode**: Service + SNET Daemon - for production with blockchain integration

## Deployment Modes

### Mode 1: Starter (Service Only)

**Location**: Project root  
**Compose File**: `docker-compose.yml`  
**Purpose**: Initial setup, local development, artifact generation

```bash
# From project root
docker-compose up
```

**What it does**:
- Starts gluRPC service (gRPC + REST)
- Auto-populates `snetd_daemon/` directory with:
  - `Dockerfile.snetd` - Daemon container definition
  - `docker-compose.full.yml` - Full stack orchestration
  - `snetd_configs/snetd.sepolia.json` - Configuration template
  - `daemon_ssl.md`, `snet_doc.md`, `README.md` - Documentation
  - `certs/` - Empty directory for SSL certificates

**Mount points**:
- `./logs` → `/app/logs`
- `./api_keys_list` → `/app/api_keys_list`
- `./snetd_daemon` → `/app/snetd_daemon`

---

### Mode 2: Full Stack (Service + Daemon)

**Location**: `snetd_daemon/` subdirectory  
**Compose File**: `docker-compose.full.yml`  
**Purpose**: Production deployment with blockchain integration

```bash
# From snetd_daemon/ subdirectory
cd snetd_daemon
docker-compose -f docker-compose.full.yml up --build
```

**What it does**:
- Starts gluRPC service (gRPC + REST)
- Starts SNET daemon (blockchain/payment handling)
- Connects daemon to service via Docker network (`http://glurpc:7003`)

**Mount points (relative to snetd_daemon/)**:
- `../logs` → `/app/logs` (service container)
- `../api_keys_list` → `/app/api_keys_list` (service container)
- `.` (current dir) → `/app/snetd_daemon` (service container)
- `./snetd_configs` → `/opt/singnet/snetd_configs` (daemon container)
- `./certs` → `/opt/singnet/.certs` (daemon container)

---

## Complete Workflow

### Step 1: Initialize (Starter Mode)

```bash
# From project root
docker-compose up

# Wait for artifacts to be generated
# Press Ctrl+C when ready
```

Output:
```
✓ SNET daemon directory initialized
✓ Copied Dockerfile.snetd
✓ Copied docker-compose.full.yml
✓ Copied snetd.sepolia.json to snetd_configs/
✓ Copied documentation files
✓ Created certs/ directory
```

### Step 2: Configure

```bash
# Edit daemon configuration
vim snetd_daemon/snetd_configs/snetd.sepolia.json

# Add SSL certificates
cp /path/to/fullchain.pem snetd_daemon/certs/
cp /path/to/privkey.pem snetd_daemon/certs/
```

Key configuration points:
- `service_endpoint`: `"http://glurpc:7003"` (uses Docker network)
- `ssl_cert`: `"/opt/singnet/.certs/fullchain.pem"`
- `ssl_key`: `"/opt/singnet/.certs/privkey.pem"`
- `data_dir`: `"/opt/singnet/etcd/sepolia"`

### Step 3: Deploy (Full Stack Mode)

```bash
# From snetd_daemon/ directory
cd snetd_daemon
docker-compose -f docker-compose.full.yml up --build -d
```

Or from project root:
```bash
docker-compose -f snetd_daemon/docker-compose.full.yml up --build -d
```

### Step 4: Verify

```bash
# Check services
docker ps

# View logs
docker-compose -f snetd_daemon/docker-compose.full.yml logs -f

# Test REST API
curl http://localhost:8000/health

# Check daemon
docker exec glurpc-snetd pgrep snetd
```

---

## Context-Aware Behavior

The entrypoint script (`docker-entrypoint.sh`) automatically detects the deployment context:

### In Starter Mode:
- Detects empty/missing `snetd_daemon/` files
- Copies artifacts from `/app/snetd_daemon_artifacts/`
- Preserves existing user configs (won't overwrite)
- Shows "Next steps" instructions

### In Full Stack Mode:
- Detects existing `snetd_daemon/` files
- Skips artifact copying
- Shows "Ready for full stack deployment"
- Both containers share logs and cache volumes

---

## File Structure

```
gluRPC/
├── docker-compose.yml              # Starter mode
├── Dockerfile                      # Service image (includes daemon artifacts)
├── docker-entrypoint.sh            # Smart initialization
├── logs/                           # Shared logs
├── api_keys_list                   # API keys
└── snetd_daemon/                   # Daemon build directory
    ├── docker-compose.full.yml     # Full stack mode
    ├── Dockerfile.snetd            # Daemon image
    ├── README.md                   # Detailed guide
    ├── daemon_ssl.md               # SSL setup
    ├── snet_doc.md                 # SNET docs
    ├── snetd.sepolia.json          # Config template
    ├── snetd_configs/              # Generated on first run
    │   └── snetd.sepolia.json
    └── certs/                      # SSL certificates (user-provided)
        ├── fullchain.pem
        └── privkey.pem
```

---

## Troubleshooting

### "snetd_daemon/ directory empty"
**Context**: Starter mode  
**Solution**: Normal on first run. Artifacts will be auto-generated.

### "Missing Dockerfile.snetd"
**Context**: Full stack mode  
**Solution**: Run starter mode first to generate artifacts.

### "Connection refused to gRPC service"
**Context**: Full stack mode  
**Solution**: Check `service_endpoint` in daemon config uses `http://glurpc:7003`

### "SSL certificate not found"
**Context**: Full stack mode  
**Solution**: Copy certificates to `snetd_daemon/certs/`

---

## Quick Reference

| Task | Command | Location |
|------|---------|----------|
| Generate artifacts | `docker-compose up` | Project root |
| Edit config | `vim snetd_daemon/snetd_configs/snetd.sepolia.json` | Project root |
| Add SSL certs | `cp *.pem snetd_daemon/certs/` | Project root |
| Run full stack | `docker-compose -f docker-compose.full.yml up --build` | `snetd_daemon/` |
| View logs | `docker-compose -f docker-compose.full.yml logs -f` | `snetd_daemon/` |
| Stop services | `docker-compose -f docker-compose.full.yml down` | `snetd_daemon/` |

---

## Why Two Modes?

**Separation of Concerns**:
- Starter mode generates artifacts without circular dependencies
- Full stack mode uses generated artifacts for production deployment

**Flexibility**:
- Developers can use starter mode without blockchain complexity
- Production deployments use full stack with all components

**Maintainability**:
- Single source of truth for daemon artifacts (committed in `snetd_daemon/`)
- No code generation at runtime (except initial artifact copying)
- User configs preserved across container restarts

