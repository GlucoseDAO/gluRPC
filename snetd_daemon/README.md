# SNET Daemon Build Directory

This directory contains everything needed to build and run the SNET daemon as a separate container.

## Files

- `Dockerfile.snetd` - Daemon container definition
- `docker-compose.full.yml` - Full stack orchestration (service + daemon)
- `snetd_configs/` - SNET daemon JSON configuration files (auto-populated on first run)
- `certs/` - SSL certificates directory (mount your certs here)
- `daemon_ssl.md` - SSL certificate setup guide
- `snet_doc.md` - SNET daemon documentation

## Architecture

The daemon runs in a separate container from the main gluRPC service:
- **Daemon container**: Handles blockchain/payment operations
- **Service container**: Handles inference and API requests
- **Communication**: Daemon connects to service via `http://glurpc:7003`

## Quick Start

### Step 1: Initialize (First Time Only)

From the project root, start the service to populate this directory:

```bash
docker-compose up
```

This will:
- Start the gluRPC service
- Auto-populate `snetd_configs/` with configuration templates
- Create the `certs/` directory

### Step 2: Configure

Edit the daemon configuration:

```bash
vim snetd_daemon/snetd_configs/snetd.sepolia.json
```

Key settings:
- `service_endpoint`: Should be `http://glurpc:7003` (Docker network)
- `ssl_cert`: Path to SSL certificate (`/opt/singnet/.certs/fullchain.pem`)
- `ssl_key`: Path to SSL key (`/opt/singnet/.certs/privkey.pem`)
- `payment_channel_storage_server.data_dir`: ETCD data path (`/opt/singnet/etcd/sepolia`)

### Step 3: Add SSL Certificates

Copy your SSL certificates to the `certs/` directory:

```bash
cp /path/to/fullchain.pem snetd_daemon/certs/
cp /path/to/privkey.pem snetd_daemon/certs/
```

See `daemon_ssl.md` for detailed instructions on generating certificates with Let's Encrypt.

### Step 4: Run Full Stack

From this directory:

```bash
cd snetd_daemon
docker-compose -f docker-compose.full.yml up --build
```

Or from the project root, use the same compose file:

```bash
docker-compose -f snetd_daemon/docker-compose.full.yml up --build
```

## Configuration Details

### Service Endpoint

The daemon needs to reach the gRPC service. In Docker Compose, use the service name:

```json
{
  "service_endpoint": "http://glurpc:7003"
}
```

### SSL Certificates

Required files in `certs/`:
- `fullchain.pem` - Full certificate chain
- `privkey.pem` - Private key

The daemon configuration should reference:
```json
{
  "ssl_cert": "/opt/singnet/.certs/fullchain.pem",
  "ssl_key": "/opt/singnet/.certs/privkey.pem"
}
```

### ETCD Storage

The daemon uses ETCD for payment channel storage:
```json
{
  "payment_channel_storage_server": {
    "data_dir": "/opt/singnet/etcd/sepolia",
    ...
  }
}
```

This is persisted in the `glurpc-etcd` Docker volume.

## Standalone Daemon (Without docker-compose.full.yml)

You can also build and run just the daemon:

```bash
cd snetd_daemon

# Build
docker build -t glurpc-snetd:local -f Dockerfile.snetd .

# Run
docker run -d \
  --name glurpc-snetd \
  -p 127.0.0.1:7001:7001 \
  -v glurpc-etcd:/opt/singnet/etcd \
  -v $(pwd)/snetd_configs:/opt/singnet/snetd_configs \
  -v $(pwd)/certs:/opt/singnet/.certs \
  glurpc-snetd:local
```

## Troubleshooting

### Daemon fails to start

**Error**: `connection refused to gRPC service`

**Solution**: 
- Ensure gRPC service is running: `docker ps | grep glurpc-service`
- Check service endpoint in config uses `http://glurpc:7003`
- Try `network_mode: "host"` if Docker networking issues persist

**Error**: `SSL certificate not found`

**Solution**:
- Check certificates exist in `certs/` directory
- Verify paths in daemon config match: `/opt/singnet/.certs/`
- See `daemon_ssl.md` for certificate generation

### ETCD startup issues

**Error**: `etcd data directory locked`

**Solution**:
- Stop all daemon containers: `docker-compose -f docker-compose.full.yml down`
- Check for orphaned processes: `docker ps -a | grep snetd`
- Remove lock: `docker volume rm glurpc-etcd` (WARNING: deletes payment channel data)

### Config file not found

**Error**: `config file does not exist`

**Solution**:
- Ensure you ran the main service once: `cd .. && docker-compose up`
- Check `snetd_configs/` directory is populated
- Manually copy from templates: `cp snetd.sepolia.json snetd_configs/`

## Logs and Debugging

View daemon logs:
```bash
docker-compose -f docker-compose.full.yml logs -f snetd
```

Check daemon process:
```bash
docker exec glurpc-snetd pgrep snetd
```

View ETCD data:
```bash
docker exec glurpc-snetd ls -la /opt/singnet/etcd/sepolia/
```

## Network Modes

### Docker Bridge Network (Default)

Uses Docker's internal DNS:
- Service endpoint: `http://glurpc:7003`
- Requires `depends_on` in docker-compose

### Host Network Mode

Daemon sees host's localhost:
- Service endpoint: `http://localhost:7003`
- Simpler networking but less isolated
- Add `network_mode: "host"` to docker-compose

## Production Considerations

1. **SSL Certificates**: Use Let's Encrypt with auto-renewal
2. **ETCD Backups**: Regularly backup `glurpc-etcd` volume
3. **Monitoring**: Set up health check alerts for daemon
4. **Resource Limits**: Add memory/CPU limits in docker-compose
5. **Secrets**: Don't commit private keys in configs

## References

- `daemon_ssl.md` - SSL certificate setup
- `snet_doc.md` - SNET daemon documentation
- Project root `README.md` - Overall project documentation
- [SNET Daemon GitHub](https://github.com/singnet/snet-daemon)

