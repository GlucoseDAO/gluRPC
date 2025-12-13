# gluRPC Architecture Diagram - Combined REST/gRPC Service

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Client Applications                            │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Web Apps    │  │ Python/JS    │  │  SNET Marketplace        │   │
│  │  (REST)      │  │  Clients     │  │  (Blockchain Payments)   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │
└─────────┼──────────────────┼───────────────────────┼──────────────────┘
          │                  │                       │
          │ REST             │ REST/gRPC             │ gRPC
          │ (HTTP)           │                       │
          ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    gluRPC Combined Service                               │
│                                                                           │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │   REST Service (8000)      │  │   SNET Daemon (7000)                │ │
│  │   ┌──────────────────────┐ │  │   ┌──────────────────────────────┐ │ │
│  │   │ FastAPI Application  │ │  │   │ Payment Channels             │ │ │
│  │   │ - /convert_to_unified│ │  │   │ - Blockchain Auth            │ │ │
│  │   │ - /process_unified   │ │  │   │ - IPFS Metadata              │ │ │
│  │   │ - /draw_a_plot       │ │  │   │ - Passthrough Routing        │ │ │
│  │   │ - /quick_plot        │ │  │   └─────────┬────────────────────┘ │ │
│  │   │ - /cache_management  │ │  │             │                       │ │
│  │   │ - /health            │ │  │             │ Passthrough           │ │
│  │   └──────────────────────┘ │  │             ▼                       │ │
│  │   Middleware:              │  └────────────────────────────────────┘ │
│  │   - DisconnectMiddleware   │                                          │
│  │   - RequestCounterMiddleware│  ┌────────────────────────────────────┐ │
│  └────────────┬───────────────┘  │   gRPC Service (7003)               │ │
│               │                  │   ┌──────────────────────────────┐  │ │
│               │                  │   │ GlucosePredictionServicer    │  │ │
│               │                  │   │ - ConvertToUnified           │  │ │
│               │                  │   │ - ProcessUnified             │  │ │
│               │                  │   │ - DrawPlot                   │  │ │
│               │                  │   │ - QuickPlot                  │  │ │
│               │                  │   │ - ManageCache                │  │ │
│               │                  │   │ - CheckHealth                │  │ │
│               │                  │   └──────────────────────────────┘  │ │
│               │                  │   Context Monitoring:               │ │
│               │                  │   - Disconnect detection            │ │
│               │                  │   - API key auth (metadata)         │ │
│               │                  └────────────┬───────────────────────┘ │
│               │                               │                          │
│               └───────────────────────────────┴─────────────────┐       │
│                                                                   │       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Core Business Logic Layer                        │ │
│  │                          (glurpc.core)                               │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │ convert_to_      │  │ parse_and_       │  │ generate_plot_   │  │ │
│  │  │ unified_action   │  │ schedule         │  │ from_handle      │  │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  │  ┌──────────────────┐                                               │ │
│  │  │ quick_plot_      │                                               │ │
│  │  │ action           │                                               │ │
│  │  └──────────────────┘                                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                   │                                      │
│       ┌───────────────────────────┼───────────────────────────────┐    │
│       │                           │                               │    │
│       ▼                           ▼                               ▼    │
│  ┌─────────┐              ┌──────────────┐                ┌──────────┐ │
│  │ Logic   │              │   Engine     │                │  State   │ │
│  │(logic.py)│             │ (engine.py)  │                │(state.py)│ │
│  │         │              │              │                │          │ │
│  │Data Proc│              │ModelManager  │                │Caches    │ │
│  │ML Infer │              │Background    │                │Tracking  │ │
│  │Plot Calc│              │Processor     │                │Registry  │ │
│  └─────────┘              │Priority Queue│                └──────────┘ │
│                           └──────────────┘                             │
└───────────────────────────────────────────────────────────────────────┘
```

## Disconnect Detection Flow

### REST Path (Middleware)
```
Client Request
    │
    ▼
DisconnectMiddleware.dispatch()
    │
    ├─► Create disconnect_event (asyncio.Event)
    │
    ├─► Start watch_disconnect() task
    │   └─► Poll request.is_disconnected() every 50ms
    │       └─► Set disconnect_event on disconnect
    │
    ├─► Attach to request.state.disconnect_event
    │
    └─► Pass to core action handler
            │
            ▼
        generate_plot_from_handle(disconnect_future=...)
            │
            └─► TaskRegistry.wait_for_result()
                └─► Monitors disconnect_future
                    └─► Cancels task if future.done()
```

### gRPC Path (Context Monitoring)
```
Client Request
    │
    ▼
GlucosePredictionServicer.DrawPlot()
    │
    ├─► Get gRPC context
    │
    ├─► Call _create_disconnect_future(context)
    │   │
    │   ├─► Create disconnect_future (asyncio.Future)
    │   │
    │   └─► Start watch_disconnect() task
    │       └─► Poll context.is_active() every 100ms
    │           └─► Set disconnect_future when inactive
    │
    └─► Pass to core action handler
            │
            ▼
        generate_plot_from_handle(disconnect_future=...)
            │
            └─► TaskRegistry.wait_for_result()
                └─► Monitors disconnect_future
                    └─► Cancels task if future.done()
```

### Shared Cancellation Handler
```
TaskRegistry.wait_for_result(disconnect_future)
    │
    ├─► Wait for inference result OR disconnect
    │   └─► asyncio.wait([result_future, disconnect_future])
    │
    ├─► If disconnect_future completes first:
    │   │
    │   ├─► Cancel inference task
    │   ├─► Remove from priority queue
    │   ├─► Cleanup resources
    │   └─► Raise asyncio.CancelledError
    │
    └─► Return result if completed successfully
```

## Deployment Scenarios

### Scenario 1: REST Only (Backward Compatible)
```bash
python -m uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```
- Original behavior
- No gRPC
- No SNET daemon

### Scenario 2: gRPC Only (SNET Service)
```bash
python run_glurpc_service.py --grpc-only --daemon-config snetd_configs/snetd.mainnet.json
```
- SNET daemon + gRPC
- Blockchain payments
- No REST

### Scenario 3: Combined (Hybrid)
```bash
python run_glurpc_service.py
```
- REST on 8000
- gRPC on 7003
- Optional SNET daemon
- Maximum flexibility

## Data Flow Example: Quick Plot

```
1. Client sends QuickPlot request
   │
   ├─► REST: POST /quick_plot (CSV in body)
   │   └─► DisconnectMiddleware attaches disconnect_event
   │
   └─► gRPC: QuickPlot RPC (CSV in message)
       └─► _create_disconnect_future(context)

2. Both paths converge at core.quick_plot_action()
   │
   ├─► parse_and_schedule() - Process CSV
   │   ├─► logic.get_handle_and_df() - Parse CSV
   │   ├─► logic.analyse_and_prepare_df() - Validate
   │   ├─► Check InferenceCache (hot + disk)
   │   └─► BackgroundProcessor.enqueue_inference() if miss
   │
   └─► generate_plot_from_handle() - Generate plot
       ├─► Wait for inference (with disconnect monitoring)
       ├─► Get PredictionsData from cache
       ├─► Check PlotCache
       └─► logic.calculate_plot_data() if miss

3. Return response
   │
   ├─► REST: JSON with plot_data dict
   │
   └─► gRPC: QuickPlotResponse protobuf message
```

## Key Design Decisions

1. **Shared Core Logic**: Both REST and gRPC use identical action handlers
   - Eliminates code duplication
   - Ensures consistent behavior
   - Simplifies testing and maintenance

2. **Interface-Specific Concerns**: 
   - REST: Middleware-based disconnect detection
   - gRPC: Context-based disconnect detection
   - Both produce same disconnect_future for core logic

3. **Backward Compatibility**:
   - Existing REST service unchanged
   - New gRPC service is additive
   - Optional SNET daemon

4. **Production Features**:
   - Queue overload protection
   - Request deduplication
   - Graceful cancellation
   - Comprehensive logging
   - Health monitoring

5. **SNET Integration**:
   - Follows example-service pattern
   - SNET daemon handles payments
   - Service focuses on prediction logic

## File Organization

```
gluRPC/
├── src/glurpc/              # Core service (REST + shared logic)
│   ├── app.py               # FastAPI application
│   ├── core.py              # Shared action handlers
│   ├── engine.py            # Model management
│   ├── logic.py             # ML inference
│   ├── state.py             # Caches & tracking
│   ├── middleware.py        # REST disconnect detection
│   └── ...
├── service/                 # gRPC-specific code
│   ├── __init__.py          # Service registry
│   ├── common.py            # Shared utilities
│   ├── glurpc_service.py    # gRPC servicer
│   └── service_spec/        # Protocol buffers
│       ├── glurpc.proto     # Service definition
│       ├── glurpc_pb2.py    # Generated (messages)
│       └── glurpc_pb2_grpc.py # Generated (stubs)
├── run_glurpc_service.py    # Combined runner
├── test_glurpc_service.py   # gRPC test client
├── buildproto.sh            # Proto compilation
└── snetd_configs/           # SNET daemon configs
    ├── snetd.ropsten.json
    └── snetd.mainnet.json
```

