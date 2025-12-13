# gluRPC SNET Service Structure - Comparison with Example Service

## Overview

This document maps the gluRPC service structure to the SingularityNET example-service template, showing how each component corresponds.

## File Structure Comparison

### Example Service → gluRPC Service

```
example-service/                    gluRPC/
├── service/                        ├── service/
│   ├── __init__.py                 │   ├── __init__.py (registry with ports)
│   ├── common.py                   │   ├── common.py (arg parsing, main_loop)
│   ├── example_service.py          │   ├── glurpc_service.py (main servicer)
│   └── service_spec/               │   └── service_spec/
│       └── example_service.proto   │       ├── glurpc.proto
│                                   │       ├── glurpc_pb2.py (generated)
│                                   │       └── glurpc_pb2_grpc.py (generated)
├── run_example_service.py          ├── run_glurpc_service.py
├── test_example_service.py         ├── test_glurpc_service.py
├── buildproto.sh                   ├── buildproto.sh
├── requirements.txt                ├── pyproject.toml (uv project)
└── snetd_configs/                  └── snetd_configs/
    └── snetd.ropsten.json              ├── snetd.ropsten.json
                                        └── snetd.mainnet.json
```

## Component Mapping

### 1. Service Registry (`service/__init__.py`)

**Example Service:**
```python
registry = {
    "example_service": {
        "grpc": 7003,
    },
}
```

**gluRPC Service:**
```python
registry = {
    "glurpc_service": {
        "grpc": 7003,
        "rest": 8000,  # Additional REST endpoint
    },
}
```

### 2. Service Implementation

**Example Service: `example_service.py`**
- Simple `CalculatorServicer` with 4 arithmetic methods
- Direct implementation of proto RPCs
- No external dependencies

**gluRPC Service: `glurpc_service.py`**
- `GlucosePredictionServicer` with 6 main RPCs
- Wraps existing REST core logic (`glurpc.core`)
- Integrates with background processor, model manager, caches
- **Implements disconnect detection** via `_create_disconnect_future()`
- API key authentication via gRPC metadata
- Service overload protection

### 3. Disconnect Handling (Key Difference)

#### Example Service
- No disconnect handling (simple request-response)

#### gluRPC Service - REST Middleware
```python
# glurpc/middleware.py - DisconnectMiddleware
async def dispatch(self, request: Request, call_next):
    disconnect_event = asyncio.Event()
    
    async def watch_disconnect():
        while True:
            await asyncio.sleep(0.05)
            if await asyncio.wait_for(
                request.is_disconnected(), 
                timeout=0.1
            ):
                disconnect_event.set()
                return
    
    watcher = asyncio.create_task(watch_disconnect())
    request.state.disconnect_event = disconnect_event
    # ... proceed with request
```

#### gluRPC Service - gRPC Context Monitoring
```python
# service/glurpc_service.py - GlucosePredictionServicer
def _create_disconnect_future(self, context: grpc.ServicerContext):
    loop = asyncio.get_event_loop()
    disconnect_future = loop.create_future()
    
    async def watch_disconnect():
        while context.is_active():
            await asyncio.sleep(0.1)
        # Context no longer active = disconnected
        if not disconnect_future.done():
            disconnect_future.set_result(True)
            logger.info("gRPC: Client disconnected")
    
    asyncio.create_task(watch_disconnect())
    return disconnect_future
```

**Both methods achieve:**
1. Monitor client connection status
2. Create future/event that fires on disconnect
3. Pass to background tasks for graceful cancellation
4. Cleanup resources via `DisconnectTracker.unregister_request()`

### 4. Runner Script

**Example Service: `run_example_service.py`**
- Starts SNET daemon (optional)
- Starts gRPC service
- Basic process supervision

**gluRPC Service: `run_glurpc_service.py`**
- Starts SNET daemon (optional)
- **Starts both gRPC AND REST services** (configurable)
- Options: `--grpc-only`, `--rest-only`, `--no-daemon`
- More flexible deployment options

### 5. Proto Definitions

**Example Service:**
```protobuf
service Calculator {
    rpc add(Numbers) returns (Result) {}
    rpc sub(Numbers) returns (Result) {}
    rpc mul(Numbers) returns (Result) {}
    rpc div(Numbers) returns (Result) {}
}
```

**gluRPC Service:**
```protobuf
service GlucosePrediction {
    rpc ConvertToUnified(ConvertToUnifiedRequest) returns (ConvertToUnifiedResponse) {}
    rpc ProcessUnified(ProcessUnifiedRequest) returns (ProcessUnifiedResponse) {}
    rpc DrawPlot(PlotRequest) returns (PlotResponse) {}
    rpc QuickPlot(QuickPlotRequest) returns (QuickPlotResponse) {}
    rpc ManageCache(CacheManagementRequest) returns (CacheManagementResponse) {}
    rpc CheckHealth(HealthRequest) returns (HealthResponse) {}
}
```

### 6. Authentication

**Example Service:**
- No built-in authentication
- Relies entirely on SNET daemon payment channels

**gluRPC Service:**
- Optional API key authentication
- Checks gRPC metadata: `x-api-key`
- Public endpoints: `ConvertToUnified`, `CheckHealth`
- Protected endpoints: All others
- Falls back to SNET daemon for blockchain payments

## Integration with Existing REST Service

The gluRPC service **does not replace** the existing REST service. Instead:

1. **Existing REST endpoints remain functional** (`glurpc.app`)
2. **gRPC service wraps the same core logic** (`glurpc.core`)
3. **Both can run simultaneously** on different ports
4. **Shared components:**
   - `glurpc.core`: Action handlers (convert, process, plot)
   - `glurpc.engine`: ModelManager, BackgroundProcessor
   - `glurpc.state`: Caches, DisconnectTracker, TaskRegistry
   - `glurpc.logic`: Data processing and ML inference

## Deployment Scenarios

### Scenario 1: REST Only (Current)
```bash
python -m uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

### Scenario 2: gRPC Only (SNET Service)
```bash
python run_glurpc_service.py --grpc-only --daemon-config snetd_configs/snetd.mainnet.json
```

### Scenario 3: Combined REST + gRPC (Hybrid)
```bash
python run_glurpc_service.py
```
- REST on port 8000 (for web apps, direct API access)
- gRPC on port 7003 (for SNET marketplace)
- SNET daemon on port 7000 (payment channels)

### Scenario 4: Docker/Kubernetes
Use combined mode with environment-based configuration.

## Key Enhancements Over Example Service

1. **Disconnect Detection**: Both REST and gRPC implement graceful disconnect handling
2. **Dual Interface**: REST and gRPC simultaneously
3. **Background Processing**: Priority queue, task deduplication, overload protection
4. **Caching**: Two-tier (memory + disk) with automatic persistence
5. **API Key Authentication**: Optional layer before SNET daemon
6. **Health Monitoring**: Comprehensive metrics endpoint
7. **Production Ready**: Logging, monitoring, queue management

## Testing

### Example Service Test
```bash
python test_example_service.py
# Interactive prompts for method and inputs
```

### gluRPC Service Test
```bash
# With actual CGM data file
python test_glurpc_service.py data/sample.csv [api_key] [endpoint]

# Auto mode
python test_glurpc_service.py data/sample.csv auto
```

## Summary

The gluRPC service follows the SNET example-service pattern but extends it significantly:

- ✅ **Compatible with SNET daemon** (payment channels, blockchain)
- ✅ **Maintains existing REST functionality** (no breaking changes)
- ✅ **Adds gRPC interface** (for SNET marketplace integration)
- ✅ **Implements disconnect tracking** (graceful cancellation like REST middleware)
- ✅ **Production-grade features** (caching, queuing, monitoring)
- ✅ **Flexible deployment** (REST-only, gRPC-only, or combined)

This makes gluRPC a **hybrid service** that can serve both traditional web clients (REST) and blockchain-based AI marketplace clients (gRPC + SNET).

