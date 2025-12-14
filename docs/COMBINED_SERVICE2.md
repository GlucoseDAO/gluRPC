# Combined gRPC + REST Service

This document explains how to use the combined gRPC + REST service that runs both protocols in the same process.

## Overview

The combined service (`combined_service.py`) runs both gRPC and REST APIs in the same Python process, sharing:
- **ModelManager** (singleton): All models are loaded once and shared
- **BackgroundProcessor** (singleton): Single task queue and worker pool
- **Cache** (singleton): Inference and plot caches are shared
- **Core Logic**: Both APIs use the same business logic functions

## Benefits

1. **Resource Efficiency**: Models loaded once, shared between both APIs
2. **Memory Savings**: Single cache, single model instances
3. **Simpler Deployment**: One process instead of two
4. **Consistent State**: Both APIs see the same cache and model state
5. **Lower Latency**: No inter-process communication overhead

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Combined Service Process                    │
│                                                          │
│  ┌─────────────┐              ┌─────────────┐          │
│  │ gRPC Server │              │ REST Server │          │
│  │  (port      │              │  (FastAPI)  │          │
│  │   7003)     │              │  (port      │          │
│  └──────┬──────┘              └──────┬──────┘          │
│         │                            │                  │
│         └────────────┬───────────────┘                  │
│                      │                                   │
│         ┌────────────▼────────────┐                     │
│         │   Core Business Logic   │                     │
│         │  (glurpc.core.*)        │                     │
│         └────────────┬────────────┘                     │
│                      │                                   │
│         ┌────────────▼────────────┐                     │
│         │   Shared Singletons     │                     │
│         │  - ModelManager         │                     │
│         │  - BackgroundProcessor  │                     │
│         │  - InferenceCache       │                     │
│         │  - PlotCache            │                     │
│         └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Running Combined Service

**Recommended way** (uses combined mode by default in the future):

```bash
uv run python run_glurpc_service.py --combined
```

This starts:
- gRPC server on port 7003
- REST server on port 8000
- Both in the same process

### Custom Ports

```bash
# Start with custom ports
python -m service.combined_service --grpc-port 7777 --rest-port 8888
```

### With SNET Daemon

```bash
# Combined service with SNET daemon
uv run python run_glurpc_service.py --combined --daemon-config snetd_configs/snetd.ropsten.json
```

## Testing

Test both endpoints are working:

```bash
uv run python test_combined_service.py
```

This will:
1. Test REST health endpoint
2. Test REST convert endpoint
3. Test gRPC health endpoint
4. Test gRPC convert endpoint

## Comparison: Separate vs Combined

### Separate Processes (Original)

```bash
# Starts 2 separate processes
uv run python run_glurpc_service.py
```

**Pros:**
- Independent scaling
- Isolation (one crash doesn't affect the other)

**Cons:**
- 2x memory usage (models loaded twice)
- 2x initialization time
- Separate caches (inconsistent state)
- More complex deployment

### Combined Process (New)

```bash
# Starts 1 process with both servers
uv run python run_glurpc_service.py --combined
```

**Pros:**
- 50% memory usage (models loaded once)
- Faster startup
- Shared cache (consistent state)
- Simpler deployment

**Cons:**
- Cannot scale independently
- Single point of failure

## Implementation Details

### Thread Safety

- **gRPC**: Runs in a ThreadPoolExecutor (10 workers by default)
- **REST**: Runs in a daemon thread with uvicorn
- **Singletons**: All singletons (ModelManager, BackgroundProcessor, Caches) use asyncio locks for thread safety

### Lifecycle

1. **Startup**:
   - gRPC server initializes (loads models, starts background processor)
   - REST server starts in a daemon thread (reuses already-initialized singletons)
   - Main thread blocks on gRPC server

2. **Runtime**:
   - Both servers handle requests concurrently
   - All requests share the same model instances and caches

3. **Shutdown**:
   - SIGINT/SIGTERM triggers gRPC shutdown
   - REST thread terminates (daemon=True)
   - Background processor and models are cleaned up

### API Endpoints

Both gRPC and REST expose the same endpoints:

| Endpoint | gRPC Method | REST Path | Auth Required |
|----------|-------------|-----------|---------------|
| Health Check | `CheckHealth` | `GET /health` | No |
| Convert to Unified | `ConvertToUnified` | `POST /convert_to_unified` | No |
| Process Unified | `ProcessUnified` | `POST /process_unified` | Yes |
| Draw Plot | `DrawPlot` | `POST /draw_a_plot` | Yes |
| Quick Plot | `QuickPlot` | `POST /quick_plot` | Yes |
| Cache Management | `ManageCache` | `POST /cache_management` | Yes |

## When to Use Combined vs Separate

**Use Combined Service When:**
- Running on single machine/container
- Resource efficiency is important
- Want consistent cache state
- Deployment simplicity matters

**Use Separate Processes When:**
- Need independent scaling (e.g., more REST replicas than gRPC)
- Want isolation between protocols
- Running in Kubernetes with separate services
- Need independent monitoring/restart

## Migration Guide

### From Separate to Combined

Replace:
```bash
uv run python run_glurpc_service.py
```

With:
```bash
uv run python run_glurpc_service.py --combined
```

### From Combined to Separate

Replace:
```bash
uv run python run_glurpc_service.py --combined
```

With:
```bash
uv run python run_glurpc_service.py
```

No code changes needed - just a command-line flag!
