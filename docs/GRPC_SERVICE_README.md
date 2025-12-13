# gluRPC - Combined REST/gRPC Glucose Prediction Service

## Overview

gluRPC is a high-performance glucose prediction service that provides both REST and gRPC interfaces for processing continuous glucose monitoring (CGM) data and generating probabilistic forecasts using deep learning models (Gluformer - Transformer-based architecture).

This service is compatible with **SingularityNET** (SNET) and can be deployed as an SNET service with payment channel support.

## Features

- **Dual Interface**: Both REST (FastAPI) and gRPC endpoints
- **Multi-format CGM data ingestion** with automatic format detection
- **ML-powered glucose prediction** using Gluformer Transformer models
- **Uncertainty quantification** via Monte Carlo Dropout
- **Intelligent two-tier caching** (hot LRU memory + persistent disk)
- **Priority-based background processing** with concurrent request deduplication
- **Real-time plot generation** with fan charts for uncertainty visualization
- **Client disconnect detection** and graceful request cancellation
- **SNET daemon integration** for blockchain-based payments

## Architecture

```
┌─────────────────────────────────────────────────┐
│          SNET Daemon (Port 7000)                │
│     Payment channels + Authentication           │
└────────────────┬────────────────────────────────┘
                 │ Passthrough
                 ▼
┌─────────────────────────────────────────────────┐
│      gRPC Service (Port 7003)                   │
│   - ConvertToUnified                            │
│   - ProcessUnified                              │
│   - DrawPlot                                    │
│   - QuickPlot                                   │
│   - ManageCache                                 │
│   - CheckHealth                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│      REST Service (Port 8000)                   │
│   FastAPI endpoints (same functionality)        │
└─────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Install Dependencies

Using uv (recommended):
```bash
cd /data/sources/glucosedao/gluRPC
uv sync
```

Using pip:
```bash
pip install -e .
```

### Build Protocol Buffers

Generate Python gRPC code from proto files:
```bash
./buildproto.sh
```

Or manually:
```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/glurpc.proto
```

## Running the Service

### Option 1: Combined REST + gRPC (Recommended)

Run both REST and gRPC services together:

```bash
python run_glurpc_service.py
```

This will start:
- REST service on port 8000
- gRPC service on port 7003

### Option 2: gRPC Only

```bash
python run_glurpc_service.py --grpc-only
```

### Option 3: REST Only

```bash
python run_glurpc_service.py --rest-only
```

### Option 4: Standalone gRPC Service

```bash
python -m service.glurpc_service --grpc-port 7003
```

### Option 5: Standalone REST Service

```bash
uv run glurpc-server
# or
python -m uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

## Running with SNET Daemon

### Prerequisites

1. Install SNET Daemon from [releases](https://github.com/singnet/snet-daemon/releases)
2. Configure your SNET organization and service ID

### Configuration

Edit `snetd_configs/snetd.ropsten.json` or `snetd_configs/snetd.mainnet.json`:

```json
{
   "DAEMON_END_POINT": "0.0.0.0:7000",
   "BLOCKCHAIN_NETWORK_SELECTED": "ropsten",
   "IPFS_END_POINT": "http://ipfs.singularitynet.io:80",
   "PASSTHROUGH_ENDPOINT": "http://localhost:7003",
   "ORGANIZATION_ID": "your-organization",
   "SERVICE_ID": "your-service-id",
   "PAYMENT_CHANNEL_STORAGE_SERVER": {
       "DATA_DIR": "/opt/singnet/etcd/ropsten"
   }
}
```

### Start with Daemon

```bash
python run_glurpc_service.py --daemon-config snetd_configs/snetd.ropsten.json
```

Or use default configs:
```bash
python run_glurpc_service.py
```

The daemon will automatically start and route requests through payment channels.

## API Key Authentication

For REST and gRPC endpoints (except public endpoints like ConvertToUnified and CheckHealth):

1. Create `api_keys_list` file with one key per line
2. Set `ENABLE_API_KEYS=1` environment variable (default: enabled)
3. Include API key in requests:
   - **REST**: `X-API-Key` header
   - **gRPC**: `x-api-key` metadata

## Testing

### Test gRPC Service

```bash
python test_glurpc_service.py <path_to_cgm_file.csv> [api_key] [endpoint]
```

Example:
```bash
python test_glurpc_service.py data/sample.csv test_api_key localhost:7003
```

Auto mode (non-interactive):
```bash
python test_glurpc_service.py data/sample.csv auto
```

### Test REST Service

See existing integration tests in `tests/` directory:
```bash
uv run pytest tests/test_integration.py -v
```

## gRPC Service Specification

### Proto Definition

See `service/service_spec/glurpc.proto` for full protocol buffer definitions.

### Available RPCs

1. **ConvertToUnified** (public)
   - Input: Raw CGM file bytes
   - Output: Unified CSV format

2. **ProcessUnified** (auth required)
   - Input: Base64-encoded unified CSV
   - Output: Dataset handle + warnings

3. **DrawPlot** (auth required)
   - Input: Handle + sample index
   - Output: Plotly JSON figure

4. **QuickPlot** (auth required)
   - Input: Base64-encoded unified CSV
   - Output: Plotly JSON figure + warnings

5. **ManageCache** (auth required)
   - Input: Action (flush/info/delete/save/load) + optional handle
   - Output: Operation result

6. **CheckHealth** (public)
   - Input: Empty
   - Output: Service health metrics

## Disconnect Handling

Both REST and gRPC interfaces implement disconnect detection:

### REST (Middleware)
- `DisconnectMiddleware` attaches per-request disconnect event
- Monitors `request.is_disconnected()` with timeout protection
- Cancels background tasks on client disconnect

### gRPC (Context Monitoring)
- Monitors `context.is_active()` for disconnect detection
- Creates disconnect futures for graceful cancellation
- Integrates with existing `TaskRegistry` and `DisconnectTracker`

Both methods replicate the same behavior:
1. Detect client disconnect
2. Set disconnect future/event
3. Cancel pending inference/calculation tasks
4. Clean up resources

## Environment Variables

- `ENABLE_API_KEYS`: Enable/disable API key authentication (default: 1)
- `LOG_LEVEL_*`: Configure logging levels for different modules
- `MAX_CACHE_SIZE`: Maximum cache size
- `NUM_SAMPLES`: Number of Monte Carlo samples for predictions

See `src/glurpc/config.py` for full list.

## Deployment

### Docker

See `deployment/docker/Dockerfile` for containerization.

### Kubernetes

See `deployment/kubernetes/` for K8s manifests.

### Systemd

See `deployment/systemd/glurpc.service` for systemd service file.

## Service Registry

Port configuration in `service/__init__.py`:

```python
registry = {
    "glurpc_service": {
        "grpc": 7003,
        "rest": 8000,
    },
}
```

## Troubleshooting

### gRPC Connection Refused
- Ensure gRPC service is running on the correct port
- Check firewall rules
- Verify `PASSTHROUGH_ENDPOINT` in SNET daemon config

### API Key Authentication Failed
- Verify `api_keys_list` file exists
- Check API key is passed in metadata (gRPC) or header (REST)
- Ensure `ENABLE_API_KEYS` is set correctly

### Model Loading Failed
- Check model files are downloaded from HuggingFace
- Verify sufficient memory/VRAM
- Check logs in `logs/` directory

### Service Overloaded
- Monitor queue sizes via `/health` endpoint
- Increase queue capacity in config
- Scale horizontally with load balancer

## Contributing

See main project README for contribution guidelines.

## License

See LICENSE file for details.

## Links

- **REST API Documentation**: See `API_REFERENCE.md`
- **Architecture Documentation**: See `HLA.md` and `THREADING_ARCHITECTURE.md`
- **SingularityNET**: https://singularitynet.io/
- **gRPC**: https://grpc.io/

