# High-Level Architecture (HLA) - GluRPC

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Structure](#module-structure)
4. [Core Components](#core-components)
5. [Data Flow & Lifecycle](#data-flow--lifecycle)
6. [Design Patterns](#design-patterns)
7. [Scalability & Performance](#scalability--performance)
8. [Security](#security)

---

## Overview

**GluRPC** is a high-performance glucose prediction service that processes continuous glucose monitoring (CGM) data and generates probabilistic forecasts using deep learning models. The service provides RESTful APIs for data conversion, processing, caching, and visualization.

### Key Features
- **Multi-format CGM data ingestion** with automatic conversion to unified format
- **ML-powered glucose prediction** using Gluformer (Transformer-based) models
- **Uncertainty quantification** via Monte Carlo Dropout
- **Intelligent caching** with memory + disk persistence and superset matching
- **Background processing** with priority-based task scheduling
- **Real-time plot generation** with fan charts for uncertainty visualization
- **API key authentication** (optional)

### Technology Stack
- **Framework**: FastAPI (async)
- **ML**: PyTorch, Darts TimeSeries
- **Data Processing**: Polars, Pandas, NumPy
- **Plotting**: Plotly
- **Persistence**: Parquet (DataFrames) + Pickle (metadata)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                      │
│                            (app.py)                              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ├─► API Endpoints (REST)
                 │   ├─ /convert_to_unified (public)
                 │   ├─ /process_unified (auth)
                 │   ├─ /draw_a_plot (auth)
                 │   ├─ /quick_plot (auth)
                 │   ├─ /cache_management (auth)
                 │   └─ /health (public)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Action Layer                         │
│                            (core.py)                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐│
│  │ convert_to_      │  │ process_and_     │  │ generate_plot_││
│  │ unified_action   │  │ cache            │  │ from_handle   ││
│  └──────────────────┘  └──────────────────┘  └───────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┬─────────────┐
    ▼             ▼             ▼              ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│ Logic   │  │ Engine  │  │  State   │  │ Schemas │  │  Config  │
│ Module  │  │ Module  │  │  Module  │  │ Module  │  │  Module  │
│(logic.py│  │(engine. │  │ (state.  │  │(schemas │  │ (config. │
│)        │  │ py)     │  │  py)     │  │ .py)    │  │  py)     │
└─────────┘  └─────────┘  └──────────┘  └─────────┘  └──────────┘
```

### Architecture Layers

1. **Presentation Layer** (app.py)
   - FastAPI endpoints
   - Request validation (Pydantic)
   - API key authentication
   - Response serialization

2. **Business Logic Layer** (core.py)
   - Action orchestration
   - Cache coordination
   - Background task scheduling

3. **Service Layer**
   - **Processing Service** (logic.py): Data transformation, ML inference, plotting
   - **Engine Service** (engine.py): Model lifecycle, background workers
   - **State Service** (state.py): Caching, task coordination

4. **Data Layer**
   - In-memory cache (dict-based)
   - Disk persistence (Parquet + Pickle)
   - Metadata indexing

---

## Module Structure

### 1. app.py - FastAPI Application

**Purpose**: HTTP API layer and application lifecycle management

**Key Components**:
- `lifespan()`: Async context manager for startup/shutdown
  - Initializes `ModelManager` (loads ML models)
  - Starts `BackgroundProcessor` (workers)
  - Graceful shutdown handling

- `require_api_key()`: Dependency for protected endpoints

**Endpoints**:
| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/convert_to_unified` | POST | No | Convert CGM files to unified format |
| `/process_unified` | POST | Yes | Process CSV and cache for inference |
| `/draw_a_plot` | POST | Yes | Generate plot for cached dataset |
| `/quick_plot` | POST | Yes | One-shot: process + plot last sample |
| `/cache_management` | POST | Yes | Manage cache (flush/info/delete/save/load) |
| `/health` | GET | No | Service health metrics |

**Lifecycle**:
```
Startup → Initialize Models → Start Background Workers → Serve Requests
                                                              ↓
Shutdown ← Stop Workers ← Signal Handler (SIGINT/SIGTERM) ← Runtime
```

---

### 2. core.py - Core Business Logic

**Purpose**: Orchestrates actions by coordinating services

**Key Functions**:

#### `convert_to_unified_action(content_base64: str)`
- Parses arbitrary CGM file formats
- Returns unified CSV format
- Uses `logic.convert_logic()`

#### `process_and_cache(content_base64: str, force_calculate: bool)`
**Workflow**:
1. Parse CSV → compute content hash (handle)
2. **Cache Check**:
   - Direct hit: return handle immediately
   - Superset match: find larger cached dataset covering requested timerange
   - Miss: proceed to processing
3. **Data Processing**:
   - Interpolate gaps
   - Synchronize timestamps
   - Validate duration requirements
   - Create inference dataset (Darts TimeSeries)
4. **Cache Storage**:
   - Store dataset, scalers, model config
   - Initialize result DataFrame (empty, will be filled by workers)
   - Assign unique version ID for race condition handling
5. **Background Inference**:
   - Enqueue low-priority inference task
   - Returns handle immediately (non-blocking)

#### `generate_plot_from_handle(handle: str, index: int)`
**Workflow**:
1. Validate handle exists in cache
2. Check if plot data calculated:
   - **Hit**: return immediately
   - **Miss**: 
     - Check if forecasts exist
     - If not: enqueue **high-priority** inference
     - Wait for calculation completion
3. Render plot using Plotly

#### `quick_plot_action(content_base64: str)`
**Workflow**:
- Calls `process_and_cache()` with minimum duration
- Calls `generate_plot_from_handle()` for last index (0)
- Returns base64-encoded PNG

**API Key Management**:
- `APIKeyManager`: Singleton for key loading/verification
- Loads keys from `api_keys_list` file at startup
- Legacy compatibility functions for backward compatibility

---

### 3. engine.py - Model & Worker Management

**Purpose**: ML model lifecycle and background task execution

#### Class: `InferenceWrapper`
**Responsibility**: Thread-safe model loading with configuration validation

```python
Model State = (GluformerModelConfig, Gluformer)
```

**Key Methods**:
- `load_if_needed(required_config)`: Lazy loading with config matching
  - If current config != required config → reload model
  - Thread-safe (uses lock)
- `run_inference(dataset, config, ...)`: Execute prediction with validation

#### Class: `ModelManager` (Singleton)
**Responsibility**: Pool of model instances with queue-based access control

**Initialization**:
1. Download model from HuggingFace Hub
2. Create N copies (1 for CPU, `NUM_COPIES_PER_DEVICE * GPU_count` for CUDA)
3. Distribute copies across GPUs (round-robin)
4. Warm up each copy with default config
5. Add all copies to asyncio.Queue

**Usage Pattern** (Context Manager):
```python
async with model_manager.acquire(1) as [wrapper]:
    forecasts = await run_inference(...)
```

**Statistics Tracking**:
- Queue length
- Average fulfillment time
- VRAM usage (per GPU)
- Total requests/errors

#### Class: `BackgroundProcessor` (Singleton)
**Responsibility**: Background task execution via worker pools

**Architecture**:
```
Inference Queue (Priority)  →  [Inference Workers] → Calc Queue (Priority)
       ↓                              ↓                      ↓
  (handle, indices)         Run ML Prediction     (handle, index, forecast)
                                  ↓
                          Store forecasts in cache
                                  ↓
                    Enqueue calculation tasks (one per index)
```

**Two Worker Types**:

##### 1. Inference Workers
- **Count**: `NUM_COPIES` (one per model copy)
- **Queue Item**: `(priority, neg_timestamp, handle, indices)`
- **Priority**:
  - `0` = High (interactive, specific indices requested)
  - `1` = Low (background, full dataset)

**Workflow**:
1. Acquire data from `DataCache`
2. Check if forecasts already exist (via Polars DataFrame)
3. If missing:
   - Acquire model from `ModelManager`
   - Run inference for **entire dataset** (batch processing)
   - Flatten forecasts: `(N, 12, 10)` → `(N, 120)` for Polars storage
   - **Version Check** (race condition handling):
     - If cache version changed → discard result (unless larger dataset)
     - If overwriting smaller result → update version, invalidate old tasks
   - Store forecasts in DataFrame column
   - Persist to disk (if enabled)
4. Enqueue calculation tasks:
   - **High-priority requests**: enqueue only requested indices (priority 0)
   - **Background requests**: 
     - Index 0 (last sample) → priority 0
     - All others → priority 1 (newest to oldest)

##### 2. Calculation Workers
- **Count**: `BACKGROUND_WORKERS_COUNT` (default 4)
- **Queue Item**: `(priority, neg_timestamp, handle, index, forecasts, version)`

**Workflow**:
1. Acquire data from `DataCache`
2. **Version Validation**: Check if `task_version == cache_version`
   - Mismatch → notify error, drop task
3. Check if already calculated (via `is_calculated` flag)
4. Execute `calculate_plot_data()`:
   - Inverse transform forecasts (denormalize)
   - Retrieve true future values from dataset
   - Compute median forecast
   - Generate KDE-based fan charts (uncertainty bands)
5. Update Polars DataFrame (atomic):
   - Update plot data columns for this index
   - Set `is_calculated = True`
   - Persist to disk
6. Notify waiting requests via `TaskRegistry`

**Graceful Shutdown**:
- Set `shutdown_started` flag
- Cancel all worker tasks
- Wait with 5s timeout

---

### 4. logic.py - Data Processing & Inference Logic

**Purpose**: Pure business logic for data transformation and ML execution

#### Data Processing Functions

##### `parse_csv_content(content_base64: str) → pl.DataFrame`
- Decode base64 → write to temp file
- Use `cgm_format.FormatParser` to auto-detect and parse format
- Returns unified DataFrame with columns: `[datetime, glucose, sequence_id, ...]`

##### `compute_handle(unified_df: pl.DataFrame) → str`
- Serialize DataFrame to CSV
- Compute SHA-256 hash
- **Content-addressable**: identical data → identical handle

##### `create_dataset_from_df(unified_df: pl.DataFrame) → Dict`
**Processing Pipeline**:
1. **Gap Interpolation**: Fill missing values (max 15 min gap)
2. **Timestamp Synchronization**: Align to 5-minute intervals
3. **Quality Validation**: Check duration, detect issues (→ warning flags)
4. **Feature Engineering**:
   - Extract temporal features: day, month, year, hour, minute, second
   - Encode categorical features (sequence_id)
5. **Segmentation**: Split by `id_segment` (continuous sequences)
6. **Darts TimeSeries Creation**:
   - Target series: glucose values
   - Future covariates: temporal features
   - Static covariates: sequence ID
7. **Scaling**: StandardScaler for target and covariates
8. **Dataset Creation**: `SamplingDatasetInferenceDual`
   - Input: 96 time steps (8 hours)
   - Output: 12 time steps (1 hour)
9. **Model Config Inference**: Detect feature dimensions from dataset

**Returns**:
```python
{
    'dataset': SamplingDatasetInferenceDual,
    'scaler_target': ScalerCustom,
    'model_config': GluformerModelConfig,
    'warning_flags': ProcessingWarning
}
```

#### ML Inference Functions

##### `load_model(model_config, model_path, device) → ModelState`
- Instantiate `Gluformer` with config
- Load state dict from file
- Move to device (CPU/CUDA)
- **Set to train mode** (critical for MC Dropout)

##### `run_inference_full(dataset, model_state, batch_size, num_samples) → np.ndarray`
- Validate model config matches dataset requirements
- Run `model.predict()` with MC Dropout:
  - `num_samples=10`: 10 stochastic forward passes per input
  - `batch_size=32`: process 32 samples simultaneously
- Returns: `(N, 12, 10)` array (N samples, 12 time steps, 10 MC samples)

#### Calculation & Rendering Functions

##### `calculate_plot_data(forecasts, dataset, scalers, index) → PlotData`
**Steps**:
1. **Denormalization**: Inverse scale transform
2. **Retrieve Ground Truth**: Extract true future values from dataset
3. **Median Forecast**: Compute 50th percentile across MC samples
4. **Fan Chart Generation** (KDE-based):
   - For each forecast time step:
     - Fit Gaussian KDE to 10 MC samples
     - Generate probability density curve
     - Normalize to [0, 1]
     - Assign color with opacity gradient (closer = darker)

##### `render_plot(plot_data) → bytes`
- Create Plotly figure with:
  - **Fan charts**: Filled probability density fans
  - **True values**: Blue line (past + future)
  - **Median forecast**: Red line
- Export to PNG (1000x600px)

---

### 5. state.py - State Management

**Purpose**: Centralized state with thread-safe access

#### Class: `StateManager` (Singleton)
**Responsibility**: Application-wide flags

- `shutdown_started`: Boolean flag for graceful shutdown coordination

#### Class: `DataCache` (Singleton)
**Responsibility**: Two-tier cache with persistence

**Architecture**:
```
Memory Cache (Dict)  ←→  Disk Storage (Parquet + Pickle)
       ↓                         ↓
  Fast Access              Persistence Layer
  (Hot Data)               (Cold Data)
       ↓                         ↓
   FIFO Eviction       Metadata Index (index.pkl)
```

**Data Structure**:
```python
{
    handle: {
        'dataset': SamplingDatasetInferenceDual,
        'scalers': {'target': ScalerCustom},
        'model_config': GluformerModelConfig,
        'warning_flags': ProcessingWarning,
        'timestamp': datetime,
        'start_time': datetime,  # For superset matching
        'end_time': datetime,    # For superset matching
        'version': str,          # UUID for race condition detection
        'data_df': pl.DataFrame  # Results storage
    }
}
```

**Result DataFrame Schema**:
```python
{
    'index': Int32,              # Negative indexing: 0=last, -(N-1)=first
    'forecast': List[Float64],   # Flattened (12*10=120 values)
    'true_values_x': List[Int32],
    'true_values_y': List[Float64],
    'median_x': List[Int32],
    'median_y': List[Float64],
    'fan_charts': List[Struct],  # KDE data
    'is_calculated': Boolean
}
```

**Key Methods**:

##### `get(handle) → Optional[Dict]`
1. Check memory cache
2. If not found and persistence enabled → load from disk
3. Auto-evict if memory full (FIFO)

##### `set(handle, data)`
1. Evict oldest if cache full
2. Assign unique version ID
3. Store in memory
4. Persist to disk (if enabled):
   - Save DataFrame as Parquet
   - Save metadata as Pickle
   - Update metadata index

##### `find_superset(start_time, end_time) → Optional[str]`
**Superset Matching Logic**:
- Find cached dataset where:
  - `cached_end_time == requested_end_time` (aligned by end)
  - `cached_start_time <= requested_start_time` (covers full range)
- **Use Case**: User requests last 2 hours, but we have 8 hours cached → reuse

##### Persistence Methods:
- `save_to_disk(handle)`: Manual save (on-demand)
- `load_from_disk(handle)`: Manual load (on-demand)
- `clear_cache()`: Full flush (memory + disk)
- `delete_handle(handle)`: Remove specific entry

**Thread Safety**:
- All async methods use `async with self._lock`
- Sync methods (`get_sync`, `set_sync`) for use within locked context

#### Class: `TaskRegistry` (Singleton)
**Responsibility**: Async coordination for plot generation

**Problem**: Multiple requests for same plot may arrive while calculation in progress

**Solution**: Future-based notification system

**Data Structure**:
```python
{
    (handle, index): [Future1, Future2, ...]  # Multiple waiters
}
```

**Workflow**:
1. Request arrives for plot
2. Check if calculated
3. If not:
   - Register future in registry
   - Trigger calculation (if not already running)
   - `await future`
4. Calculation worker completes → `notify_success(handle, index)`
5. All waiting futures resolve → return results

**Methods**:
- `wait_for_result(handle, index)`: Block until ready
- `notify_success(handle, index)`: Wake up all waiters (success)
- `notify_error(handle, index, error)`: Wake up all waiters (error)
- `cancel_all_for_handle(handle)`: Cancel on cache eviction
- `cancel_all()`: Global flush

---

### 6. schemas.py - API Data Models

**Purpose**: Pydantic models for request/response validation

**Key Models**:

- `ProcessRequest`: CSV upload (base64)
- `UnifiedResponse`: Handle + warnings
- `PlotRequest`: Handle + index
- `QuickPlotResponse`: PNG (base64) + warnings
- `ConvertResponse`: Unified CSV content
- `HealthResponse`: Service metrics
- `CacheManagementResponse`: Cache operation result

**Configuration**: All models use `frozen=True` (immutable)

---

### 7. config.py - Configuration

**Purpose**: Centralized configuration with environment variable support

**Categories**:

1. **Data Processing**:
   - `STEP_SIZE_MINUTES = 5`
   - `MINIMUM_DURATION_MINUTES = 540` (9 hours)
   - `MAXIMUM_WANTED_DURATION = 1080` (18 hours)

2. **Cache**:
   - `MAX_CACHE_SIZE = 128`
   - `ENABLE_CACHE_PERSISTENCE = True`

3. **Security**:
   - `ENABLE_API_KEYS = False`

4. **ML Inference**:
   - `NUM_COPIES_PER_DEVICE = 2` (2 models per GPU)
   - `BACKGROUND_WORKERS_COUNT = 4`
   - `BATCH_SIZE = 32`
   - `NUM_SAMPLES = 10` (MC Dropout samples)

**Validation**: Ensures minimum duration meets model requirements

---

### 8. data_classes.py - Domain Data Models

**Purpose**: Pydantic models for domain logic

**Key Models**:

- `GluformerInferenceConfig`: Input parameters for processing pipeline
- `GluformerModelConfig`: Model architecture specification
- `PlotData`: Aggregated data for rendering
- `FanChartData`: Single KDE distribution slice

**Schema Definitions**:
- `RESULT_SCHEMA`: Polars schema for result DataFrame

---

## Data Flow & Lifecycle

### Scenario 1: Quick Plot Request (First Time)

```
1. Client → POST /quick_plot {csv_base64}
             ↓
2. core.quick_plot_action()
             ↓
3. core.process_and_cache()
   ├─ Parse CSV → compute handle (SHA-256)
   ├─ Check cache → MISS
   ├─ logic.create_dataset_from_df()
   │  ├─ Interpolate, sync, validate
   │  ├─ Create Darts dataset
   │  └─ Infer model config
   ├─ DataCache.set(handle, data)
   │  ├─ Store in memory
   │  ├─ Initialize empty result DataFrame
   │  └─ Persist to disk
   └─ BackgroundProcessor.enqueue_inference(handle, priority=1)
             ↓
4. core.generate_plot_from_handle(handle, index=0)
   ├─ Check result cache → MISS (not calculated)
   ├─ Check forecast cache → MISS (not computed)
   ├─ BackgroundProcessor.enqueue_inference(handle, priority=0, indices=[0])
   └─ TaskRegistry.wait_for_result(handle, 0)
             ↓
5. [Background] Inference Worker
   ├─ ModelManager.acquire() → get model
   ├─ Run inference for FULL dataset (N samples)
   ├─ Store forecasts in DataCache[handle]['data_df']['forecast']
   ├─ BackgroundProcessor.enqueue_calc(handle, index=0, priority=0)
   └─ BackgroundProcessor.enqueue_calc(handle, index=-(N-1) to -1, priority=1)
             ↓
6. [Background] Calculation Worker
   ├─ logic.calculate_plot_data(forecasts[0], ...)
   │  ├─ Denormalize
   │  ├─ Compute median
   │  └─ Generate fan charts (KDE)
   ├─ Update DataCache[handle]['data_df'] (row 0)
   └─ TaskRegistry.notify_success(handle, 0)
             ↓
7. core.generate_plot_from_handle (resumed)
   ├─ Retrieve plot data from result cache
   ├─ logic.render_plot() → PNG bytes
   └─ Return to client
             ↓
8. Client ← {plot_base64: "iVBORw0KGgoAAAANS..."}
```

**Timeline**:
- Steps 1-4: ~1-3 seconds (synchronous processing)
- Steps 5-6: ~5-15 seconds (GPU inference + calculation)
- **Total**: ~6-18 seconds for first request

### Scenario 2: Quick Plot Request (Cached)

```
1. Client → POST /quick_plot {csv_base64}
             ↓
2. core.process_and_cache()
   ├─ Compute handle
   ├─ Check cache → HIT
   └─ Return handle immediately
             ↓
3. core.generate_plot_from_handle(handle, index=0)
   ├─ Check result cache → HIT (already calculated)
   ├─ logic.render_plot() → PNG bytes
   └─ Return to client
             ↓
4. Client ← {plot_base64: "..."}
```

**Timeline**: ~100-500ms (cache hit, rendering only)

### Scenario 3: Superset Matching

```
User uploads 2 hours of data, but 8 hours already cached:

1. Compute handle_new
2. DataCache.contains(handle_new) → False
3. Extract time range: [T-2h, T]
4. DataCache.find_superset(T-2h, T)
   ├─ Search metadata index
   └─ Find: handle_old with range [T-8h, T]
5. Return handle_old (reuse existing)
6. Client requests plot → uses cached results from larger dataset
```

**Benefit**: Avoid reprocessing subset of already processed data

---

## Design Patterns

### 1. Singleton Pattern
**Usage**: All manager classes (ModelManager, BackgroundProcessor, DataCache, TaskRegistry, StateManager, APIKeyManager)

**Rationale**:
- Single source of truth for shared state
- Avoid resource duplication (models, cache)
- Simplified dependency injection

**Implementation**: Custom `SingletonMeta` metaclass

### 2. Object Pool Pattern
**Usage**: `ModelManager` with asyncio.Queue

**Rationale**:
- Reuse expensive resources (loaded ML models)
- Limit concurrent access (prevent OOM)
- Fair scheduling (FIFO queue)

### 3. Producer-Consumer Pattern
**Usage**: Background processing (Inference Queue → Workers → Calc Queue)

**Rationale**:
- Decouple request handling from computation
- Enable async, non-blocking responses
- Prioritize interactive requests

### 4. Priority Queue Pattern
**Usage**: Both inference and calculation queues

**Rationale**:
- Interactive requests (priority 0) processed first
- Background tasks (priority 1) fill idle time
- Timestamp tiebreaker (FIFO within priority)

### 5. Future/Promise Pattern
**Usage**: `TaskRegistry` for async coordination

**Rationale**:
- Multiple requests can wait for same result
- Avoid duplicate computation
- Clean async/await syntax

### 6. Two-Tier Caching
**Usage**: `DataCache` (memory + disk)

**Rationale**:
- Hot data in memory (fast access)
- Cold data on disk (persistence)
- Automatic promotion/demotion

### 7. Content-Addressable Storage
**Usage**: Handle computation via SHA-256

**Rationale**:
- Identical inputs → identical results
- Cache key is deterministic
- Natural deduplication

### 8. Versioning for Optimistic Locking
**Usage**: UUID version IDs in cache entries

**Rationale**:
- Detect race conditions (concurrent updates)
- Allow "larger-wins" strategy for dataset conflicts
- Invalidate stale tasks

---

## Scalability & Performance

### Horizontal Scaling Considerations

**Current Architecture**: Single-instance (vertical scaling)

**Scalable Components**:
1. **Stateless API Layer**: FastAPI app can be replicated behind load balancer
2. **Shared Cache**: Requires external store (Redis, Memcached)
3. **Model Serving**: Separate inference service (Triton, TorchServe)

**Bottlenecks**:
- In-memory cache (shared state)
- Local disk persistence
- Background workers (single instance)

**Recommended Multi-Instance Architecture**:
```
Load Balancer
     ↓
[FastAPI 1] [FastAPI 2] [FastAPI 3]
     ↓           ↓           ↓
     └───────────┴───────────┘
                 ↓
       Shared Redis Cache
                 ↓
     ┌───────────┴───────────┐
     ↓                       ↓
Celery Workers      GPU Inference Cluster
(Calculation)       (Triton Server)
```

### Performance Optimizations

1. **Batch Inference**: Process 32 samples simultaneously
2. **Lazy Loading**: Models loaded on first use
3. **Negative Indexing**: O(1) access to last sample (index 0)
4. **Polars DataFrames**: Faster than Pandas for large datasets
5. **Async I/O**: Non-blocking disk operations
6. **Background Processing**: Offload computation from request path

### Resource Management

**Memory**:
- Cache size limit: `MAX_CACHE_SIZE * dataset_size`
- Model memory: `NUM_COPIES * model_size (~500MB per copy)`
- Total estimate: ~10-50GB for typical workload

**GPU**:
- Model allocation per GPU: `NUM_COPIES_PER_DEVICE` copies
- Inference batch size: 32 (configurable)
- Memory per GPU: ~2-4GB

**Disk**:
- Parquet compression: ~10MB per cached dataset
- Total: `cached_datasets * 10MB`

---

## Security

### Authentication
- **API Key Authentication**: Optional (disabled by default)
- **Key Storage**: Plain text file (`api_keys_list`)
- **Protected Endpoints**: All except `/health` and `/convert_to_unified`
- **Header**: `X-API-Key: <key>`

### Input Validation
- **Pydantic Models**: Automatic type/structure validation
- **File Format Validation**: `cgm_format` library handles parsing safely
- **Size Limits**: FastAPI default (100MB body limit)

### Security Recommendations
1. **Enable API Keys** in production (`ENABLE_API_KEYS=true`)
2. **Use HTTPS** (terminate TLS at reverse proxy)
3. **Rate Limiting**: Add per-key limits (Redis-based)
4. **Input Sanitization**: Already handled by Pydantic + Polars
5. **Secret Management**: Move to environment variables or vault
6. **CORS Configuration**: Restrict origins if serving browser clients

### Potential Vulnerabilities
- **No rate limiting**: DoS risk (expensive computations)
- **No key rotation**: Static keys in file
- **No audit logging**: Track API key usage
- **No input size validation**: Large files could cause OOM

---

## Extension Points

### Adding New Endpoints
1. Define schema in `schemas.py`
2. Create action handler in `core.py`
3. Add endpoint in `app.py`

### Adding New Models
1. Extend `GluformerModelConfig` with new parameters
2. Update `InferenceWrapper.load_if_needed()` for new model type
3. Implement model-specific inference logic in `logic.py`

### Adding New Data Sources
1. Extend `cgm_format` library with new parser
2. Update `FormatParser.parse_file()` to detect format
3. No changes needed in GluRPC (auto-detected)

### Custom Cache Strategies
1. Extend `DataCache` class
2. Override `get()`, `set()`, `find_superset()`
3. Implement custom eviction policy

---

## Deployment

### Environment Variables
```bash
# Cache
export MAX_CACHE_SIZE=128
export ENABLE_CACHE_PERSISTENCE=True

# Security
export ENABLE_API_KEYS=True

# Performance
export NUM_COPIES_PER_DEVICE=2
export BACKGROUND_WORKERS_COUNT=4
export BATCH_SIZE=32
export NUM_SAMPLES=10

# Data Processing
export MINIMUM_DURATION_MINUTES=540
export MAXIMUM_WANTED_DURATION=1080
```

### Startup Command
```bash
python -m glurpc.app
# or
uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

### Docker Considerations
- **Base Image**: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`
- **Volume Mounts**:
  - `/app/cache_storage`: Persistent cache
  - `/app/logs`: Application logs
- **GPU Support**: `--gpus all` flag

---

## Monitoring & Observability

### Health Check Endpoint
```json
GET /health
{
    "status": "ok",
    "cache_size": 42,
    "models_initialized": true,
    "queue_length": 0,
    "avg_fulfillment_time_ms": 123.45,
    "vmem_usage_mb": 3584.2,
    "device": "cuda",
    "total_requests_processed": 1234,
    "total_errors": 5
}
```

### Logging
- **File**: `logs/glurpc_YYYYMMDD_HHMMSS.log`
- **Console**: Simultaneous output
- **Levels**:
  - INFO: Request/response logging
  - DEBUG: Detailed execution traces
  - ERROR: Exceptions and failures

### Key Metrics to Monitor
1. **Request Latency**: P50, P95, P99
2. **Cache Hit Rate**: `cache_hits / total_requests`
3. **Queue Depth**: Inference + calc queue sizes
4. **GPU Utilization**: Per-device usage
5. **Error Rate**: `errors / total_requests`
6. **Memory Usage**: Cache size, model memory

---

## Conclusion

GluRPC is a production-ready glucose prediction service with:
- ✅ **Robust architecture**: Layered, modular design
- ✅ **High performance**: Async I/O, batch processing, intelligent caching
- ✅ **Scalability**: Vertical scaling, horizontal scaling with modifications
- ✅ **Reliability**: Graceful shutdown, error handling, version control
- ✅ **Extensibility**: Clean abstractions, dependency injection

The architecture prioritizes **low latency for interactive requests** while maximizing **throughput for background processing** through priority-based scheduling and two-tier caching.

