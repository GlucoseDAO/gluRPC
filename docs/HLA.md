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

**GluRPC** is a high-performance glucose prediction service that processes continuous glucose monitoring (CGM) data and generates probabilistic forecasts using deep learning models (Gluformer - Transformer-based architecture). The service provides RESTful APIs for data conversion, processing, caching, and visualization with interactive plot generation.

### Key Features
- **Multi-format CGM data ingestion** with automatic format detection (`cgm_format` library)
- **ML-powered glucose prediction** using Gluformer Transformer models
- **Uncertainty quantification** via Monte Carlo Dropout (10 stochastic samples)
- **Intelligent two-tier caching** (hot LRU memory + persistent disk with `diskcache`)
- **Priority-based background processing** with concurrent request deduplication
- **Real-time plot generation** with fan charts for uncertainty visualization (Plotly)
- **Client disconnect detection** and graceful request cancellation
- **Queue overload protection** with automatic service degradation
- **Request deduplication** with last-write-wins semantics

### Technology Stack
- **Framework**: FastAPI (async)
- **ML**: PyTorch, Darts TimeSeries
- **Data Processing**: Polars (primary), Pandas (Darts compatibility)
- **Plotting**: Plotly
- **Caching**: `cachetools.LRUCache` (hot) + `diskcache.Index` (persistent)
- **Concurrency**: `asyncio`, `aiorwlock` (async RW locks), `threading.Lock` (model safety)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                           │
│                        (app.py)                                  │
│                                                                   │
│  Middleware: DisconnectMiddleware | RequestCounterMiddleware    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ├─► API Endpoints (REST)
                 │   ├─ /convert_to_unified (public, multipart)
                 │   ├─ /process_unified (auth, JSON)
                 │   ├─ /draw_a_plot (auth, JSON, returns Plotly JSON)
                 │   ├─ /quick_plot (auth, JSON, returns Plotly JSON)
                 │   ├─ /cache_management (auth, query params)
                 │   └─ /health (public, JSON)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Action Layer                            │
│                         (core.py)                                │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ convert_to_    │  │ parse_and_       │  │ generate_plot_  │ │
│  │ unified_action │  │ schedule         │  │ from_handle     │ │
│  └────────────────┘  └──────────────────┘  └─────────────────┘ │
│  ┌────────────────┐                                             │
│  │ quick_plot_    │                                             │
│  │ action         │                                             │
│  └────────────────┘                                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┬─────────────┐
    ▼             ▼             ▼              ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│ Logic   │  │ Engine  │  │  State   │  │ Schemas │  │  Config  │
│ (logic. │  │(engine. │  │ (state.  │  │(schemas │  │ (config. │
│  py)    │  │  py)    │  │   py)    │  │  .py)   │  │   py)    │
│         │  │         │  │          │  │         │  │          │
│Data Proc│  │Models & │  │Caches &  │  │Pydantic │  │Env Vars  │
│ML Infer │  │Workers  │  │Tracking  │  │Models   │  │Defaults  │
│Plot Calc│  │Priority │  │Disconnect│  │         │  │          │
└─────────┘  └─────────┘  └──────────┘  └─────────┘  └──────────┘
                  │
                  ├─► cache.py: HybridLRUCache (Hot + Disk)
                  ├─► middleware.py: Disconnect tracking, metrics
                  └─► data_classes.py: Domain models (PredictionsData, etc.)
```

### Architecture Layers

1. **Presentation Layer** (app.py)
   - FastAPI endpoints with async handlers
   - Request validation (Pydantic schemas)
   - Optional API key authentication (`X-API-Key` header)
   - Response serialization (JSON)
   - Middleware: disconnect detection, request counting

2. **Business Logic Layer** (core.py)
   - Action orchestration
   - Cache coordination (InferenceCache, PlotCache)
   - Background task scheduling (BackgroundProcessor)
   - Request lifecycle management

3. **Service Layer**
   - **Processing Service** (logic.py): 
     - Data transformation (`cgm_format` integration)
     - ML inference (Gluformer via Darts)
     - Plot calculation (KDE-based fan charts)
   - **Engine Service** (engine.py): 
     - Model lifecycle management (InferenceWrapper, ModelManager)
     - Background worker pools (inference + calculation)
     - Priority queues (0=interactive, 1=background, 3=prefetch)
   - **State Service** (state.py): 
     - Caching (InferenceCache, PlotCache)
     - Task coordination (TaskRegistry)
     - Disconnect tracking (DisconnectTracker)

4. **Data Layer**
   - **Hot cache**: In-memory LRU (cachetools) - fast access, configurable size
   - **Cold cache**: Disk persistence (diskcache Index) - survives restarts
   - **Hybrid strategy**: Automatic promotion/demotion with RW locks

---

## Module Structure

### 1. app.py - FastAPI Application

**Purpose**: HTTP API layer and application lifecycle management

**Key Components**:

#### `lifespan()` Context Manager
- **Startup**:
  - Initializes `ModelManager` (downloads from HuggingFace, loads models)
  - Starts `BackgroundProcessor` (spawns worker pools)
  - Configures logging levels from environment
- **Shutdown**:
  - Cancels all background tasks
  - Stops workers gracefully (5s timeout)
  - Critical model failures → app termination

#### Middleware Stack
1. **DisconnectMiddleware**: 
   - Polls `request.is_disconnected()` (50ms interval)
   - Attaches `disconnect_event` to `Request.state`
   - Enables graceful request cancellation
2. **RequestCounterMiddleware**: 
   - Tracks HTTP metrics (total requests, errors, timing)
   - Stores in `app.state.request_metrics`

#### API Endpoints

| Endpoint | Method | Auth | Input | Output | Purpose |
|----------|--------|------|-------|--------|---------|
| `/convert_to_unified` | POST | No | `multipart/form-data` (file) | JSON (`csv_content` string) | Convert proprietary CGM formats to Unified CSV |
| `/process_unified` | POST | Yes | JSON (`csv_base64`, `force_calculate`) | JSON (`handle`, `total_samples`, `warnings`) | Parse CSV, schedule inference, return cache handle |
| `/draw_a_plot` | POST | Yes | JSON (`handle`, `index`, `force_calculate`) | JSON (Plotly figure dict) | Generate plot for specific sample (Gradio compatible) |
| `/quick_plot` | POST | Yes | JSON (`csv_base64`, `force_calculate`) | JSON (`plot_data`, `warnings`) | One-shot: process + plot last sample |
| `/cache_management` | POST | Yes | Query params (`action`, `handle`) | JSON (operation result) | Cache operations (flush/info/delete) |
| `/health` | GET | No | None | JSON (health metrics) | Service status and statistics |

#### Overload Protection
- Checks queue utilization before processing (`check_queue_overload()`)
- Returns `503 Service Unavailable` when queues >99% full
- Load statuses: idle → lightly loaded → loaded → heavily loaded → overloaded → full

#### Disconnect Handling
- Links middleware `disconnect_event` to request-specific `disconnect_future`
- Graceful cancellation via `TaskRegistry.cancel_request()`
- Proper cleanup with `DisconnectTracker.unregister_request()`

---

### 2. core.py - Core Business Logic

**Purpose**: Orchestrates actions by coordinating services (logic, engine, state)

**Key Functions**:

#### `convert_to_unified_action(content_base64: str) → ConvertResponse`
- Parses arbitrary CGM file formats (via `cgm_format.FormatParser`)
- Supports: Dexcom, Freestyle Libre, Abbott, Clarity exports, etc.
- Returns: Unified CSV format (`sequence_id`, `datetime`, `glucose`)
- Error handling: Format detection failures, malformed data

#### `parse_and_schedule(content_base64: str, ...) → UnifiedResponse`
**Workflow**:
1. **Parse & Hash**:
   - Decode base64 → parse CSV → compute SHA-256 handle (content-addressable)
2. **Data Preparation**:
   - Detect/assign sequences (`FormatProcessor.detect_and_assign_sequences`)
   - Interpolate gaps (max 15 min, via `cgm_format`)
   - Synchronize timestamps to 5-minute intervals
   - Quality validation (duration, out-of-range, duplicates)
   - Calculate expected dataset length (based on `input_chunk_length` + `output_chunk_length`)
3. **Cache Check**:
   - **Hit**: Return handle immediately if cached length ≥ requested length
   - **Pending**: Reuse if already queued with sufficient length
   - **Miss**: Proceed to scheduling
4. **Inference Scheduling**:
   - Enqueue to `BackgroundProcessor` with priority (0=interactive, 1=background, 3=prefetch)
   - Store pending status to prevent duplicate enqueueing
   - Return handle + warnings (non-blocking)

**Key Optimization**: Calculates expected length from **actual prepared data** (not theoretical max duration) to account for quality filtering/gaps

#### `generate_plot_from_handle(handle: str, index: int, ...) → Dict[str, Any]`
**Workflow**:
1. **Validate Request**:
   - Check `InferenceCache` for handle existence
   - Validate index range (0 = last sample, negative indexing)
   - Handle pending inference (wait with timeout if not ready)
2. **Plot Cache Check**:
   - Check `PlotCache` by `(version, index)` key
   - **Hit**: Return Plotly JSON immediately
   - **Miss**: Proceed to inline calculation (skip if `force_calculate=True`)
3. **Ensure Predictions Available**:
   - Wait for inference completion if `predictions` is `None`
   - Handle version mismatches (pending larger dataset)
   - Timeout: `INFERENCE_TIMEOUT` (10 min GPU / 120 min CPU)
4. **Calculate Inline**:
   - Call `logic.calculate_plot_data()` in thread pool
   - Store result in `PlotCache`
   - Return Plotly figure as JSON dict

**Disconnect Handling**: Races `disconnect_future` vs computation, raises `asyncio.CancelledError` on disconnect

#### `quick_plot_action(content_base64: str, ...) → QuickPlotResponse`
**Workflow**:
1. Call `parse_and_schedule()` with `minimum_duration` (9 hours) and priority 0 (interactive)
2. Schedule delayed background full calculation (18 hours, priority 3, 5s delay)
3. Call `generate_plot_from_handle()` for index 0 (last sample)
4. Return Plotly JSON + warnings

**Rationale**: Quick response with minimal data, background prefetch for full history

---

### 3. logic.py - Data Processing & ML Inference

**Purpose**: Pure business logic for data transformation, ML execution, and plot generation

#### Data Processing Functions

##### `parse_csv_content(content_base64: str) → pl.DataFrame`
- Decode base64 → write temp file → auto-detect format via `FormatParser.parse_file()`
- Returns: Unified Polars DataFrame (`sequence_id`, `datetime`, `glucose`)
- Error handling: Unknown format, malformed data, validation errors

##### `compute_handle(unified_df: pl.DataFrame) → str`
- Serialize to CSV → compute SHA-256 hash
- **Content-addressable**: identical data → identical handle (enables deduplication)

##### `analyse_and_prepare_df(unified_df, min_duration, max_duration) → (pl.DataFrame, ProcessingWarning, int)`
**Processing Pipeline**:
1. Detect/assign sequences (continuous segments)
2. Interpolate gaps (max 15 min)
3. Synchronize timestamps (5 min intervals)
4. Quality validation (via `FormatProcessor.prepare_for_inference`)
   - Duration check (min 9 hours for model requirements)
   - Out-of-range detection (glucose values)
   - Duplicate timestamps
   - Calibration events
   - Imputation flags
5. Convert to glucose-only DataFrame for length calculation

**Returns**: `(inference_df, warning_flags, actual_input_samples)`

##### `create_dataset_from_df(inference_df, warning_flags) → DatasetCreationResult`
**Pipeline**:
1. Convert to glucose-only DataFrame (drop service columns, deduplicate)
2. Create inference dataset via `create_inference_dataset_fast_local()`:
   - Rename columns (`sequence_id`→`id`, `datetime`→`time`, `glucose`→`gl`)
   - Interpolate gaps (legacy `glucobench` method, 45 min threshold)
   - Encode temporal features (`day`, `month`, `year`, `hour`, `minute`, `second`)
   - Segment by `id_segment` (continuous sequences)
   - Create Darts `TimeSeries` objects per segment
   - Scale target and covariates (`ScalerCustom`)
   - Create `SamplingDatasetInferenceDual` (input: 96 steps, output: 12 steps)
   - Infer feature dimensions from first sample
3. Generate `GluformerModelConfig` (architecture parameters)

**Returns**: `DatasetCreationResult` (TypedDict with `success`, `dataset`, `scalers`, `model_config`, `warning_flags`)

#### ML Inference Functions

##### `load_model(model_config, model_path, device) → ModelState`
- Instantiate `Gluformer` with config parameters
- Load state dict from file (HuggingFace download)
- Move to device (CPU/CUDA)
- **Set to train mode** (critical for MC Dropout)
- Returns: `(model_config, model)` tuple

##### `run_inference_full(dataset, model_config, model_state, ...) → (PredictionsArray, LogVarsArray)`
- Validate model config matches dataset requirements
- Run `model.predict()` with MC Dropout:
  - `num_samples=10`: 10 stochastic forward passes per input
  - `batch_size=32`: process 32 samples simultaneously
- Returns: `(N, 12, 10)` predictions array + `(N, 1, 10)` logvars

#### Plot Calculation Functions

##### `calculate_plot_data(predictions: PredictionsData, index: int) → (str, PlotData)`
**Steps**:
1. **Extract Forecast**: Get `(12, 10)` forecast for specified index
2. **Denormalization**: 
   - Apply `(x - scaler.min) / scaler.scale` to bring back to mg/dL
   - **Validated approach** (see code comments referencing `tests/debug_scaling.py`)
   - Same postprocessing for forecasts AND historical data (ensures alignment)
3. **Retrieve Ground Truth**: 
   - Use `dataset.get_sample_data(ds_index)` to get `(past_target, true_future)`
   - Denormalize with same formula
4. **Compute Median Forecast**: 50th percentile across 10 MC samples
5. **Generate Fan Charts** (KDE-based):
   - For each forecast time step (12 steps):
     - Fit Gaussian KDE to 10 MC samples
     - Generate probability density curve (100 points)
     - Normalize to [0, 1]
     - Assign color with opacity gradient (closer forecast = darker)
   - Store as `FanChartData` structs
6. **Extract Timestamps** (if available):
   - Use `dataset.get_sample_timestamps()` for datetime labels
   - Converts to ISO format strings

**Returns**: `(plot_json_str, PlotData)` where `plot_json_str` is serialized Plotly figure

##### `render_plot(plot_data: PlotData) → bytes`
- Create Plotly figure with:
  - **Fan charts**: Filled probability density fans (scatter mode='none', fill='toself')
  - **True values**: Blue line (past + future)
  - **Median forecast**: Red line
  - Glucose range shading (54-180 mg/dL target zone)
  - Annotations for time ranges
- Export to PNG (1000x600px) or return as Plotly JSON dict

---

### 4. engine.py - Model & Worker Management

**Purpose**: ML model lifecycle, background task execution, priority queues

#### Class: `InferenceWrapper`
**Responsibility**: Thread-safe model loading with configuration validation

**Key Methods**:
- `load_if_needed(required_config)`: 
  - Lazy loading with config matching
  - If current config ≠ required config → reload model
  - Thread-safe (uses `threading.Lock`)
- `run_inference(dataset, required_config, ...)`: 
  - Validates config before execution
  - Holds lock during inference to prevent concurrent model swaps
  - Returns predictions + logvars

**Rationale**: Different datasets may require different feature dimensions → dynamic model reloading

#### Class: `ModelManager` (Singleton)
**Responsibility**: Pool of model instances with queue-based access control

**Initialization**:
1. Download model from HuggingFace Hub (`Livia-Zaharia/gluformer_models`)
2. Calculate total copies: `NUM_COPIES = max(2, GPU_count * NUM_COPIES_PER_DEVICE)`
3. Distribute copies across GPUs (round-robin: model 0→cuda:0, model 1→cuda:1, ...)
4. Warm up each copy with default config
5. **Split into queues**:
   - **Priority Queue**: Contains model #0 only (reserved for priority 0 requests)
   - **General Queue**: Contains models #1+ (available for all requests)

**Usage Pattern** (Context Manager):
```python
async with model_manager.acquire(1, priority=0) as [wrapper]:
    predictions, logvars = wrapper.run_inference(...)
```

**Priority Logic**:
- **Priority 0** (interactive): Try general queue first → fallback to model #0
- **Priority > 0** (background): Only use general queue (never touch model #0)

**Statistics Tracking**:
- Available models (priority + general queues)
- Average fulfillment time (queue wait duration)
- VRAM usage per GPU / RSS memory for CPU
- Total requests by priority
- Total inference errors

#### Class: `BackgroundProcessor` (Singleton)
**Responsibility**: Background task execution via priority-based worker pools

**Architecture**:
```
HTTP Request → parse_and_schedule() → Enqueue Inference
                                            ↓
                                 [Inference Queue (Priority)]
                                            ↓
                           ┌────────────────┴────────────────┐
                           ↓                                  ↓
                  [Inference Workers × NUM_COPIES]    (Dedup: _pending_inference)
                           ↓
                   1. Acquire Model
                   2. Check Staleness (request_id)
                   3. Create Dataset
                   4. Run Inference
                   5. Store in InferenceCache
                           ↓
                  [Calculation Queue (Priority)]
                           ↓
                  [Calc Workers × 4]
                           ↓
                   1. Validate Version
                   2. Check Staleness
                   3. Denormalize + KDE
                   4. Store in PlotCache
                   5. Notify TaskRegistry
```

**Two Queue Types**:

##### 1. Inference Queue
- **Item**: `(priority, neg_timestamp, handle, indices, inference_df, warning_flags, expected_dataset_len, inference_config, force_calculate, request_id)`
- **Priority**:
  - `0` = Interactive (user waiting)
  - `1` = Background (prefetch)
  - `3` = Low priority prefetch (quick_plot delayed task)
- **Deduplication**: `_pending_inference` dict prevents redundant enqueueing for same handle

##### 2. Calculation Queue
- **Item**: `(priority, neg_timestamp, handle, index, forecasts, version, request_id)`
- **Priority**:
  - `0` = Interactive (specific indices)
  - `1` = Background (index 0 first, then newest→oldest)

**Workers**:

##### Inference Workers (Count: NUM_COPIES)
**Workflow**:
1. Dequeue item from priority queue
2. **Staleness Check**: Skip if `request_id < latest_request_id` (last-write-wins)
3. **Dataset Creation**: Call `logic.create_dataset_from_df()` (thread pool)
4. **Acquire Model**: `async with model_manager.acquire(1, priority):`
5. **Run Inference**: Call `wrapper.run_inference()` (thread pool)
6. **Store Results**: Update `InferenceCache` via transaction:
   - Create `PredictionsData` with predictions array, dataset DTO, scalers
   - Atomic cache update with version check
7. **Notify Waiters**: Call `TaskRegistry.notify_success(handle, index)`
8. **Remove Pending**: Clear from `_pending_inference`

**Error Handling**: On failure, notifies all waiters via `TaskRegistry.notify_error()`

##### Calculation Workers (Count: 4)
**Workflow**:
1. Dequeue item from priority queue
2. **Staleness Check**: Skip if `request_id < latest_request_id`
3. **Version Validation**: Ensure cache version hasn't changed (race condition protection)
4. **Calculate Plot**: Call `logic.calculate_plot_data()` (thread pool)
5. **Store Plot**: Update `PlotCache.update_plot(version, index, plot_json)`
6. **Notify Waiters**: Call `TaskRegistry.notify_success(handle, index, request_id)`

**Graceful Shutdown**:
- Set `StateManager().shutdown_started = True`
- Cancel all workers
- Wait with 5s timeout

---

### 5. state.py - State Management

**Purpose**: Centralized state with caching, task coordination, disconnect tracking

#### Class: `APIKeyManager` (Singleton)
**Responsibility**: API key authentication

- Loads keys from `api_keys_list` file (one key per line, `#` for comments)
- `verify_api_key(key)`: Check if key exists in loaded set
- `is_restricted(endpoint)`: Determine if endpoint requires auth (all except `/health`, `/convert_to_unified`)

#### Class: `StateManager` (Singleton)
**Responsibility**: Application-wide flags

- `shutdown_started`: Boolean flag for graceful shutdown coordination

#### Class: `HybridLRUCache[K, V]` (Generic, Singleton Base)
**Responsibility**: Two-tier cache with async RW locks

**Architecture**:
```
Memory Cache (LRUCache)  ←→  Disk Storage (diskcache.Index)
       ↓                              ↓
  Fast Access                  Persistence Layer
  (Hot Data)                    (Survives Restarts)
       ↓                              ↓
   LRU Eviction              Async RW Lock Protection
```

**Key Methods**:
- `get(key)`: 
  - Try hot cache (read lock) → return if hit
  - Load from disk (thread pool, no lock) → promote to hot (write lock)
- `set(key, value)`: 
  - Update hot cache (write lock)
  - Persist to disk (thread pool, async)
- `delete(key)`: 
  - Mark as tombstone in hot (write lock)
  - Remove from disk (thread pool)
  - Clean tombstone (write lock)
- `transaction(key)`: 
  - Atomic read-modify-write (write lock held throughout)
  - Usage: `async with cache.transaction(key) as txn: txn.set(new_value)`

**Tombstone Pattern**: Prevents concurrent reads from re-promoting deleted keys during deletion

#### Class: `InferenceCache(HybridLRUCache[str, PredictionsData])`
**Responsibility**: Cache for inference results

- **Key**: handle (SHA-256 hash)
- **Value**: `PredictionsData` (predictions array, dataset DTO, scalers, model config, warnings, version)
- **Auto-initialization**: On `set()`, creates corresponding `PlotCache` entry
- **Cascade Deletion**: On `delete()`, removes associated plots from `PlotCache`

**Directory**: `cache_storage/<params_hash>/inference/`

#### Class: `PlotCache(HybridLRUCache[str, PlotCacheEntry])`
**Responsibility**: Cache for plot data

- **Key**: version (from `PredictionsData.version`, UUID)
- **Value**: `PlotCacheEntry` (arrays of plot JSONs and PlotData objects)
- **Per-Index Access**: `get_plot(version, index)` retrieves single plot JSON
- **Atomic Update**: `update_plot(version, index, json_str)` uses updater callback

**Directory**: `cache_storage/<params_hash>/plots/`

#### Class: `DisconnectTracker` (Singleton)
**Responsibility**: Manage request disconnect counters per (handle, index)

**Architecture Implementation** (THREADING_ARCHITECTURE.md):
- Request ID Assignment: Monotonic sequence per (handle, index)
- Per-Request Disconnect Futures: Individual disconnect detection
- Shared Disconnect Future: Resolves when ALL requests disconnect
- Request Counter: Tracks active request count
- Last-Write-Wins: Via `BackgroundProcessor._latest_request_id`
- Unified Cancellation Hook: `cancel_request()` for per-request cancellation

**Data Structures**:
```python
# Shared tracking: { (handle, index): {"seq": int, "count": int, "disconnect_future": Future} }
# Per-request futures: { (handle, index, request_id): Future }
```

**Flow**:
1. Request arrives → `register_request()` → increment count, assign request_id
2. Get per-request disconnect future via `get_disconnect_future()`
3. Request processes → race disconnect_future vs work completion
4. Request completes/disconnects → `unregister_request()` → decrement count
5. Counter reaches 0 → shared disconnect_future resolves
6. Individual disconnect → per-request future resolves immediately

**Duplicate Handling**:
- Multiple concurrent requests for same (handle, index) share computation
- Each has unique request_id and per-request disconnect future
- Workers check `is_request_stale()` before expensive operations
- Stale requests (request_id < latest) are skipped

#### Class: `TaskRegistry` (Singleton)
**Responsibility**: Async coordination for plot generation (Future-based notifications)

**Problem**: Multiple requests for same plot may arrive while calculation in progress

**Solution**: Future-based notification system with request_id granularity

**Data Structure**:
```python
{ (handle, index, request_id): Future }
```

**Methods**:
- `wait_for_result(handle, index, request_id, disconnect_future, timeout)`: 
  - Register future in registry
  - Race: `await asyncio.wait([future, disconnect_future], timeout=timeout)`
  - Return: success/error/timeout/disconnect
- `notify_success(handle, index, request_id)`: Wake up specific request (or all if `request_id=None`)
- `notify_error(handle, index, error, request_id)`: Notify error
- `cancel_request(handle, index, request_id, reason)`: Unified cancellation hook
- `cancel_all_for_handle(handle)`: Cancel all requests for handle (on inference failure)

---

### 6. cache.py - Generic Hybrid Cache

**Purpose**: Generic two-tier cache implementation with async RW locks

#### Class: `HybridLRUCache[K, V]` (Generic)
**Design**: Hot LRU (cachetools) + Persistent Disk (diskcache) with async RW locks (aiorwlock)

**Key Features**:
- **Concurrency**: Async read/write locks for safe concurrent access
- **Hot/Cold Split**: Fast memory access with disk fallback
- **Automatic Promotion**: Disk hits → promoted to hot cache
- **Tombstone Pattern**: Safe deletion with concurrent read protection
- **Transaction Support**: Atomic read-modify-write operations

**Lock Logging**: All lock operations logged via `glurpc.locks` logger (DEBUG level, defaults to ERROR to reduce noise)

---

### 7. middleware.py - Request Middleware

**Purpose**: Request lifecycle management and metrics collection

#### `RequestCounterMiddleware`
- Tracks total HTTP requests, errors (4xx/5xx), request times
- Stores metrics in `app.state.request_metrics` (RequestMetrics Pydantic model)
- Thread-safe (uses asyncio.Lock for request_times list)

#### `DisconnectMiddleware`
- Polls `request.is_disconnected()` every 50ms (with 100ms timeout)
- Attaches `disconnect_event` to `Request.state`
- Watcher task runs in background, cancelled on request completion
- Enables graceful cancellation via `asyncio.wait()` racing

---

## Data Flow & Lifecycle

### Scenario 1: Quick Plot Request (First Time - Cache Miss)

```
1. Client → POST /quick_plot {csv_base64}
             ↓
2. app.quick_plot()
   ├─ Check overload (queue utilization)
   ├─ Assign request_id
   ├─ Get disconnect_event from middleware
   └─ Call core.quick_plot_action()
             ↓
3. core.quick_plot_action(csv_base64, request_id, disconnect_future)
   ├─ Call parse_and_schedule(max_duration=9h, priority=0)
   │  ├─ Parse CSV → compute handle (SHA-256)
   │  ├─ Analyse & prepare → inference_df, warnings, actual_samples
   │  ├─ Check InferenceCache → MISS
   │  ├─ Enqueue inference (priority=0, request_id)
   │  └─ Return UnifiedResponse(handle, total_samples, warnings)
   │
   ├─ Schedule delayed background task (max_duration=18h, priority=3, delay=5s)
   └─ Call generate_plot_from_handle(handle, index=0, request_id)
      ├─ Check PlotCache → MISS
      ├─ Check InferenceCache → predictions=None (pending)
      ├─ Register in DisconnectTracker
      ├─ Call TaskRegistry.wait_for_result(handle, 0, request_id)
      └─ Wait (race: result vs disconnect vs timeout)
             ↓
4. [Background] Inference Worker
   ├─ Dequeue (priority=0, handle, request_id)
   ├─ Check staleness (request_id < latest?) → NO
   ├─ Create dataset (logic.create_dataset_from_df) → dataset, scalers, model_config
   ├─ Acquire model (priority=0) → try general queue → fallback to model #0
   ├─ Run inference (batch_size=32, num_samples=10) → predictions (N, 12, 10)
   ├─ Create PredictionsData (with dataset DTO, scalers)
   ├─ Store in InferenceCache (atomic transaction)
   ├─ Notify TaskRegistry.notify_success(handle, 0)
   └─ Enqueue calc (priority=0) → for index 0
             ↓
5. [Background] Calculation Worker
   ├─ Dequeue (priority=0, handle, index=0, request_id)
   ├─ Check staleness → NO
   ├─ Get PredictionsData from InferenceCache
   ├─ Calculate plot (logic.calculate_plot_data)
   │  ├─ Extract forecast (12, 10)
   │  ├─ Denormalize (forecasts, past, true_future)
   │  ├─ Compute median
   │  ├─ Generate KDE fan charts
   │  └─ Render Plotly figure → JSON string
   ├─ Store in PlotCache.update_plot(version, 0, plot_json)
   └─ Notify TaskRegistry.notify_success(handle, 0, request_id)
             ↓
6. core.generate_plot_from_handle (resumed from wait)
   ├─ TaskRegistry future resolved
   ├─ Get plot from PlotCache (now available)
   ├─ Return Plotly JSON dict
             ↓
7. Client ← QuickPlotResponse{plot_data: {...}, warnings: {...}}
```

**Timeline**:
- Steps 1-3: ~1-3 seconds (parsing, scheduling)
- Steps 4-5: ~5-15 seconds (GPU inference + KDE calculation)
- **Total**: ~6-18 seconds for first request

### Scenario 2: Quick Plot Request (Cache Hit)

```
1. Client → POST /quick_plot {csv_base64}
             ↓
2. core.quick_plot_action()
   ├─ parse_and_schedule()
   │  ├─ Compute handle
   │  ├─ Check InferenceCache → HIT (predictions exist)
   │  └─ Return handle immediately
   │
   └─ generate_plot_from_handle(handle, 0)
      ├─ Check PlotCache → HIT (plot exists)
      ├─ Parse JSON → Plotly dict
      └─ Return immediately
             ↓
3. Client ← QuickPlotResponse{plot_data: {...}, warnings: {...}}
```

**Timeline**: ~100-500ms (cache hit, JSON parsing only)

### Scenario 3: Concurrent Duplicate Requests

```
Client A → /draw_a_plot {handle=H1, index=-10}  (req_id=1)
Client B → /draw_a_plot {handle=H1, index=-10}  (req_id=2)  [arrives 50ms later]
Client C → /draw_a_plot {handle=H1, index=-10}  (req_id=3)  [arrives 100ms later]

1. Client A arrives:
   ├─ DisconnectTracker.register_request(H1, -10) → req_id=1, count=1
   ├─ BackgroundProcessor.update_latest_request_id(H1, -10, 1)
   ├─ TaskRegistry.wait_for_result(H1, -10, 1, disconnect_future_A)
   └─ [Waits for result]

2. Client B arrives (50ms later):
   ├─ DisconnectTracker.register_request(H1, -10) → req_id=2, count=2
   ├─ BackgroundProcessor.update_latest_request_id(H1, -10, 2)  [NOW latest=2]
   ├─ TaskRegistry.wait_for_result(H1, -10, 2, disconnect_future_B)
   └─ [Waits for result]

3. Client C arrives (100ms later):
   ├─ DisconnectTracker.register_request(H1, -10) → req_id=3, count=3
   ├─ BackgroundProcessor.update_latest_request_id(H1, -10, 3)  [NOW latest=3]
   ├─ TaskRegistry.wait_for_result(H1, -10, 3, disconnect_future_C)
   └─ [Waits for result]

4. Inference Worker (if needed):
   ├─ Dequeue inference task (handle=H1, req_id=1)
   ├─ Check staleness: is_request_stale(H1, -10, 1) → YES (latest=3)
   ├─ Skip inference (stale)
   ├─ cleanup_stale_jobs(H1, -10, 1)
   └─ Continue to next task

5. Client B Disconnects (user cancels):
   ├─ Middleware detects disconnect → disconnect_event_B.set()
   ├─ Link task resolves disconnect_future_B
   ├─ TaskRegistry future cancelled via race condition
   ├─ DisconnectTracker.unregister_request(H1, -10, 2) → count=2
   └─ Shared disconnect_future NOT resolved (count > 0)

6. Calculation Worker (fresh inference with req_id=3):
   ├─ Dequeue calc task (handle=H1, index=-10, req_id=3)
   ├─ Check staleness: is_request_stale(H1, -10, 3) → NO (latest=3)
   ├─ Calculate plot
   ├─ Store in PlotCache
   └─ TaskRegistry.notify_success(H1, -10, None)  [Notify ALL for this handle/index]

7. Results:
   ├─ Client A: Gets result (req_id=1 was waiting, notification works for all)
   ├─ Client B: Already disconnected (got 499 Client Closed Request)
   └─ Client C: Gets result (req_id=3, latest request)
```

**Key Insights**:
- Requests 1 & 2 are marked stale (last-write-wins)
- Only request 3 proceeds to computation
- Disconnected requests (B) don't block others
- All waiting requests notified on completion (even stale ones receive result)

### Scenario 4: Client Disconnect During Inference

```
1. Client → /draw_a_plot {handle=H1, index=-5}
             ↓
2. Inference Worker starts (long GPU computation ~10s)
             ↓
3. Client closes connection (5s into computation)
   ├─ Middleware detects disconnect
   ├─ disconnect_future resolves
   ├─ TaskRegistry future races: disconnect wins
   ├─ Raises asyncio.CancelledError
   └─ app.draw_a_plot() catches, calls TaskRegistry.cancel_request()
             ↓
4. Worker continues (already in progress, cannot be interrupted mid-inference)
   ├─ Stores result in cache (work not wasted)
   └─ Notifies TaskRegistry (no waiters left)
             ↓
5. Client receives: 499 Client Closed Request
6. Next request for same (handle, index) gets cached result immediately
```

**Rationale**: Inference work is expensive, don't waste it. Result benefits future requests.

---

## Design Patterns

### 1. Singleton Pattern
**Usage**: All manager classes (ModelManager, BackgroundProcessor, InferenceCache, PlotCache, TaskRegistry, DisconnectTracker, StateManager, APIKeyManager)

**Implementation**: Custom `SingletonMeta` metaclass with `_instances` dict

**Rationale**:
- Single source of truth for shared state
- Avoid resource duplication (models, cache, workers)
- Simplified dependency injection (call constructor anywhere)

### 2. Object Pool Pattern
**Usage**: `ModelManager` with dual asyncio.Queue (priority + general)

**Rationale**:
- Reuse expensive resources (loaded ML models ~500MB each)
- Limit concurrent access (prevent OOM)
- Fair scheduling (FIFO within priority)
- Priority isolation (model #0 reserved for interactive requests)

### 3. Producer-Consumer Pattern
**Usage**: Background processing (Inference Queue → Workers → Calc Queue)

**Rationale**:
- Decouple request handling from computation
- Enable async, non-blocking responses
- Priority-based scheduling (interactive first)

### 4. Priority Queue Pattern
**Usage**: Both inference and calculation queues (asyncio.PriorityQueue)

**Priority Levels**:
- `0` = Interactive (user waiting)
- `1` = Background (prefetch)
- `3` = Low priority background (delayed tasks)

**Tie-breaker**: Negative timestamp (FIFO within priority)

**Rationale**:
- Interactive requests processed first
- Background tasks fill idle time
- Prevents starvation via priority aging

### 5. Future/Promise Pattern
**Usage**: `TaskRegistry` for async coordination, disconnect futures

**Rationale**:
- Multiple requests wait for same result (deduplication)
- Avoid duplicate computation
- Clean async/await syntax
- Graceful cancellation via future resolution

### 6. Two-Tier Caching
**Usage**: `HybridLRUCache` (hot LRU + persistent disk)

**Rationale**:
- Hot data in memory (fast access, O(1) lookup)
- Cold data on disk (persistence, survives restarts)
- Automatic promotion/demotion (LRU eviction)
- RW locks for concurrent safety

### 7. Content-Addressable Storage
**Usage**: Handle computation via SHA-256 of CSV content

**Rationale**:
- Identical inputs → identical handles → cache hit
- Deterministic cache keys
- Natural deduplication across requests

### 8. Last-Write-Wins (Request ID Versioning)
**Usage**: `BackgroundProcessor._latest_request_id` + `DisconnectTracker` seq counter

**Rationale**:
- Concurrent duplicate requests → only newest proceeds
- Stale jobs skipped before expensive operations
- Per-request disconnect futures (individual cancellation)
- Shared computation benefits all waiters

### 9. Tombstone Pattern
**Usage**: `HybridLRUCache` deletion with `_TOMBSTONE` sentinel

**Rationale**:
- Concurrent reads during deletion don't re-promote deleted keys
- Two-phase deletion (mark → disk delete → cleanup)
- Prevents race conditions

### 10. Transaction Pattern
**Usage**: `HybridLRUCache.transaction()` context manager

**Rationale**:
- Atomic read-modify-write operations
- Prevents partial updates during concurrent access
- Critical section with write lock held throughout

---

## Scalability & Performance

### Current Architecture: Single-Instance (Vertical Scaling)

**Scalable Components**:
1. **Model Copies**: Horizontal within instance (multiple GPUs)
   - Default: 2 copies per GPU
   - Load balancing via queue fairness
2. **Worker Pools**: Configurable parallelism
   - Inference workers: 1 per model copy
   - Calculation workers: 4 (configurable)

**Bottlenecks**:
1. **In-memory cache**: Single-instance only
   - Max size: 128 entries (configurable)
   - No cross-instance sharing
2. **Disk cache**: Local filesystem
   - Not shared across instances
   - No distributed locking
3. **Background workers**: Single-instance
   - Queues not distributed
   - No task stealing

### Recommended Multi-Instance Architecture

```
                     Load Balancer (Round-Robin + Sticky Sessions)
                                    ↓
        ┌───────────────┬───────────────────┬───────────────────┐
        ↓               ↓                   ↓                   ↓
   [Instance 1]    [Instance 2]        [Instance 3]        [Instance 4]
   FastAPI + GPU   FastAPI + GPU       FastAPI + GPU       FastAPI + GPU
        ↓               ↓                   ↓                   ↓
        └───────────────┴───────────────────┴───────────────────┘
                                    ↓
                          Shared Redis Cluster
                          (Cache + Task Queue)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Celery Workers                  Shared S3/NFS Storage
            (Plot Calculation)              (Disk Cache Backend)
```

**Changes Required**:
1. **Shared Cache**: Replace `HybridLRUCache` with Redis
   - InferenceCache → Redis Hash per handle
   - PlotCache → Redis Hash per version
   - Use Redis TTL for automatic eviction
2. **Distributed Queues**: Replace `asyncio.PriorityQueue` with Celery/Redis
   - Inference tasks → Celery queue (priority-based routing)
   - Calculation tasks → Celery queue
3. **Shared Disk**: Replace local `diskcache` with S3/NFS
   - Parquet files → S3 bucket
   - Use object versioning for consistency
4. **Session Affinity**: Load balancer sticky sessions (handle-based)
   - Routes duplicate requests to same instance
   - Reduces inter-instance coordination
5. **Model Serving**: Optional separate Triton/TorchServe cluster
   - Decouple model inference from web layer
   - Dedicated GPU scaling

**Cost/Complexity Trade-offs**:
- Adds: Redis cluster, S3, Celery workers
- Benefit: Horizontal scaling, fault tolerance
- Complexity: Distributed system challenges (consistency, latency)

### Performance Optimizations

1. **Batch Inference**: Process 32 samples simultaneously (GPU throughput)
2. **Lazy Loading**: Models loaded on first use (fast startup)
3. **Negative Indexing**: O(1) access to last sample (index 0)
4. **Polars DataFrames**: ~10x faster than Pandas for large datasets
5. **Async I/O**: Non-blocking disk operations (via `asyncio.to_thread`)
6. **Background Processing**: Offload computation from request path
7. **Priority Queues**: Interactive requests bypass background tasks
8. **Content-Addressable Cache**: Deduplication across time and users
9. **Two-Tier Cache**: Hot path avoids disk I/O
10. **RW Locks**: Concurrent reads don't block each other

### Resource Management

**Memory**:
- Hot cache: `MAX_CACHE_SIZE * avg_entry_size` (~128 * 50MB = 6.4GB)
- Model memory: `NUM_COPIES * model_size` (~4 models * 500MB = 2GB)
- Worker memory: ~2GB per process
- **Total estimate**: ~10-15GB for typical workload

**GPU**:
- Model allocation: 2 copies per GPU (default)
- Inference batch size: 32 (configurable)
- VRAM per model copy: ~500MB (Gluformer base)
- **Total per GPU**: ~1-2GB VRAM

**Disk**:
- Parquet files: ~10-50MB per cached dataset (compressed)
- Plots: ~100KB per plot JSON
- **Total**: `cached_datasets * 50MB` (~6.4GB for 128 entries)

**CPU**:
- FastAPI workers: 1 (async event loop)
- Inference workers: `NUM_COPIES` threads
- Calculation workers: 4 threads
- Background tasks: Spawned as needed

---

## Security

### Authentication
- **API Key Authentication**: Optional (disabled by default, `ENABLE_API_KEYS=true`)
- **Key Storage**: Plain text file (`api_keys_list`, one key per line)
- **Protected Endpoints**: All except `/health` and `/convert_to_unified`
- **Header**: `X-API-Key: <key>`
- **Verification**: In-memory set lookup (O(1))

### Input Validation
- **Pydantic Models**: Automatic type/structure validation on all endpoints
- **File Format Validation**: `cgm_format` library handles parsing safely
  - Unknown format → 400 error
  - Malformed data → 400 error with details
- **Size Limits**: FastAPI default (100MB body limit, configurable)
- **Index Validation**: Range checks against dataset length

### Security Recommendations

**Production Hardening**:
1. **Enable API Keys**: `ENABLE_API_KEYS=true` in environment
2. **Use HTTPS**: Terminate TLS at reverse proxy (nginx, ALB)
3. **Rate Limiting**: Add per-key limits (Redis-based, e.g., `slowapi`)
4. **Secret Management**: Move keys to environment variables or vault (AWS Secrets Manager, HashiCorp Vault)
5. **CORS Configuration**: Restrict origins if serving browser clients (via FastAPI middleware)
6. **Input Sanitization**: Already handled by Pydantic + Polars, but validate business logic
7. **Audit Logging**: Track API key usage, errors, suspicious patterns
8. **Queue Limits**: Already implemented (`MAX_INFERENCE_QUEUE_SIZE`, `MAX_CALC_QUEUE_SIZE`)

### Potential Vulnerabilities

**Current Risks**:
1. **No rate limiting**: DoS risk (expensive GPU computations)
   - **Mitigation**: Implemented queue overload protection (503 on >99% full)
2. **No key rotation**: Static keys in file
   - **Mitigation**: Restart required to reload keys
3. **No audit logging**: Can't track API key usage patterns
4. **No input size validation**: Large files could cause OOM
   - **Partial mitigation**: FastAPI body size limits
5. **Plain text keys**: Keys visible in filesystem
   - **Mitigation**: Use environment variables + secrets manager

**Not Vulnerable To**:
- SQL Injection: No SQL database
- XSS: JSON API only (no HTML rendering)
- Path Traversal: No user-controlled file paths
- Command Injection: No shell execution with user input

---

## Extension Points

### Adding New Endpoints

**Steps**:
1. Define request/response schemas in `schemas.py` (Pydantic models)
2. Create action handler in `core.py` (orchestration logic)
3. Implement business logic in `logic.py` (pure functions)
4. Add endpoint in `app.py` with dependencies (`require_api_key` if needed)

**Example**: Adding a batch processing endpoint:
```python
# schemas.py
class BatchProcessRequest(BaseModel):
    csv_files: List[str] = Field(..., description="List of base64 CSV files")

# core.py
async def batch_process_action(csv_files: List[str]) -> List[UnifiedResponse]:
    tasks = [parse_and_schedule(csv) for csv in csv_files]
    return await asyncio.gather(*tasks)

# app.py
@app.post("/batch_process", response_model=List[UnifiedResponse])
async def batch_process(request: BatchProcessRequest, api_key: str = Depends(require_api_key)):
    return await batch_process_action(request.csv_files)
```

### Adding New Models

**Steps**:
1. Extend `GluformerModelConfig` in `data_classes.py` with new parameters
2. Update `InferenceWrapper.load_if_needed()` for new model type (if architecture differs)
3. Implement model-specific inference logic in `logic.py` (if needed)
4. Upload model weights to HuggingFace Hub
5. Update default model name in `config.py` or pass via environment

**Example**: Adding a larger Gluformer variant:
```python
# config.py
MODEL_NAME = os.getenv("MODEL_NAME", "gluformer_large_weights.pth")

# data_classes.py
class GluformerLargeConfig(GluformerModelConfig):
    d_model: int = 1024  # Override default
    n_heads: int = 16
```

### Adding New Data Sources

**Steps**:
1. Extend `cgm_format` library with new parser (separate repo)
2. Update `FormatParser.parse_file()` to detect new format
3. No changes needed in GluRPC (auto-detected via `cgm_format`)

**Example**: Adding support for Medtronic format:
```python
# In cgm_format library
class MedtronicParser(BaseParser):
    def parse(self, file_path: str) -> pl.DataFrame:
        # Custom parsing logic
        return unified_df

# FormatParser will automatically try new parser
```

### Custom Cache Strategies

**Steps**:
1. Subclass `HybridLRUCache` in `cache.py`
2. Override `get()`, `set()`, `delete()` methods
3. Implement custom eviction policy (e.g., LFU, TTL-based)
4. Update `InferenceCache`/`PlotCache` to use new base class

**Example**: Adding TTL-based eviction:
```python
class TTLCache(HybridLRUCache[K, V]):
    def __init__(self, directory: str, max_hot: int, ttl_seconds: int):
        super().__init__(directory, max_hot)
        self._ttl = ttl_seconds
        self._timestamps: Dict[K, float] = {}
    
    async def get(self, key: K) -> Optional[V]:
        if key in self._timestamps and time.time() - self._timestamps[key] > self._ttl:
            await self.delete(key)
            return None
        return await super().get(key)
```

---

## Deployment

### Environment Variables

**Cache Configuration**:
```bash
export MAX_CACHE_SIZE=128                  # Number of entries in hot cache
export ENABLE_CACHE_PERSISTENCE=True       # Enable disk persistence
```

**Security**:
```bash
export ENABLE_API_KEYS=True                # Enable API key auth
```

**Performance**:
```bash
export NUM_COPIES_PER_DEVICE=2             # Models per GPU
export BACKGROUND_WORKERS_COUNT=4          # Calculation workers
export BATCH_SIZE=32                       # Inference batch size
export NUM_SAMPLES=10                      # MC Dropout samples
export MAX_INFERENCE_QUEUE_SIZE=64         # Inference queue limit
export MAX_CALC_QUEUE_SIZE=8192            # Calc queue limit
```

**Data Processing**:
```bash
export MINIMUM_DURATION_MINUTES=540        # 9 hours min (model requirement)
export MAXIMUM_WANTED_DURATION=1080        # 18 hours max (default)
export STEP_SIZE_MINUTES=5                 # Time step (fixed by model)
```

**Timeouts**:
```bash
export INFERENCE_TIMEOUT_GPU=600.0         # 10 minutes for GPU
export INFERENCE_TIMEOUT_CPU=7200.0        # 120 minutes for CPU
```

**Logging** (DEBUG=10, INFO=20, WARNING=30, ERROR=40):
```bash
export LOG_LEVEL_ROOT=INFO                 # glurpc.*
export LOG_LEVEL_LOGIC=INFO                # glurpc.logic.*
export LOG_LEVEL_ENGINE=INFO               # glurpc.engine.*
export LOG_LEVEL_CORE=INFO                 # glurpc.core
export LOG_LEVEL_APP=INFO                  # glurpc.app
export LOG_LEVEL_STATE=INFO                # glurpc.state
export LOG_LEVEL_CACHE=INFO                # glurpc.cache
export LOG_LEVEL_LOCKS=ERROR               # glurpc.locks (app-wide)
```

### Startup Command

**Direct Python**:
```bash
python -m glurpc.app
```

**Uvicorn** (recommended):
```bash
uvicorn glurpc.app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use `--workers 1` due to Singleton state (ModelManager, BackgroundProcessor). Multi-worker support requires shared state refactoring.

### Docker Deployment

**Base Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`

**Volume Mounts**:
- `/app/cache_storage`: Persistent cache (survives container restarts)
- `/app/logs`: Application logs

**GPU Support**:
```bash
docker run --gpus all -v $(pwd)/cache_storage:/app/cache_storage -p 8000:8000 glurpc:latest
```

**Environment File**:
```dockerfile
ENV MAX_CACHE_SIZE=128
ENV ENABLE_CACHE_PERSISTENCE=True
ENV ENABLE_API_KEYS=False
ENV NUM_COPIES_PER_DEVICE=2
ENV BACKGROUND_WORKERS_COUNT=4
```

### Kubernetes Deployment

**Pod Spec** (see `deployment/kubernetes/`):
- **Requests**: 4 CPU, 16GB RAM, 1 GPU
- **Limits**: 8 CPU, 32GB RAM, 1 GPU
- **Health Probes**: 
  - Liveness: `/health` (30s interval)
  - Readiness: `/health` + `models_initialized=true`
- **Persistent Volume**: For cache storage
- **GPU Node Selector**: `nvidia.com/gpu.product=A10` (or similar)

---

## Monitoring & Observability

### Health Check Endpoint

**GET /health** returns comprehensive metrics:
```json
{
  "status": "ok",                          // "ok" | "degraded" | "error"
  "load_status": "loaded",                 // idle → loaded → overloaded → full
  "cache_size": 42,                        // Entries in InferenceCache
  "models_initialized": true,              // Models loaded and ready
  "available_priority_models": 1,          // Model #0 availability (0 or 1)
  "available_general_models": 3,           // Models #1+ availability
  "avg_fulfillment_time_ms": 123.45,       // Model acquisition latency
  "vmem_usage_mb": 3584.2,                 // VRAM (GPU) or RSS (CPU)
  "device": "cuda:0",                      // Inference device
  "total_http_requests": 1234,             // All HTTP requests
  "total_http_errors": 5,                  // 4xx/5xx errors
  "avg_request_time_ms": 250.5,            // Mean request duration
  "median_request_time_ms": 180.0,         // Median request duration
  "min_request_time_ms": 50.0,
  "max_request_time_ms": 5000.0,
  "inference_requests_by_priority": {      // Requests by priority level
    "0": 450,                              // Interactive
    "1": 780                               // Background
  },
  "total_inference_errors": 2,             // ML inference failures
  "total_calc_runs": 1200,                 // Plot calculations completed
  "total_calc_errors": 3,                  // Plot calculation failures
  "inference_queue_size": 2,               // Current queue depth
  "inference_queue_capacity": 64,
  "calc_queue_size": 15,
  "calc_queue_capacity": 8192
}
```

### Logging

**File**: `logs/glurpc_YYYYMMDD_HHMMSS.log`
**Format**: `%(asctime)s - %(name)-19s - %(levelname)-8s - %(message)s`

**Loggers**:
- `glurpc`: Root logger (inherits to all modules)
- `glurpc.app`: Endpoint handlers, lifecycle
- `glurpc.core`: Action orchestration
- `glurpc.logic`: Data processing, inference, plot calculation
  - `glurpc.logic.data`: Preprocessing (FormatProcessor, dataset creation)
  - `glurpc.logic.infer`: ML inference (model loading, prediction)
  - `glurpc.logic.calc`: Plot calculation (KDE, rendering)
- `glurpc.engine`: Model management, workers
  - `glurpc.engine.infer`: Inference worker loop
  - `glurpc.engine.calc`: Calculation worker loop
  - `glurpc.engine.data`: Worker-side dataset operations
- `glurpc.state`: Cache, task registry, disconnect tracking
- `glurpc.cache`: Hybrid cache operations
- `glurpc.locks`: **App-wide lock logger** (all lock acquire/release, defaults to ERROR)

**Log Levels**:
- INFO: Request/response, cache hits/misses, queue operations
- DEBUG: Detailed execution traces, lock operations, worker loops
- WARNING: Overload conditions, stale jobs, cache evictions
- ERROR: Exceptions, failures, timeouts

### Key Metrics to Monitor

**Application Metrics**:
1. **Request Latency**: P50, P95, P99 (from `/health` endpoint)
2. **Cache Hit Rate**: `cache_hits / total_requests` (derive from logs)
3. **Error Rate**: `total_http_errors / total_http_requests`
4. **Queue Depth**: `inference_queue_size + calc_queue_size` (overload indicator)

**Resource Metrics**:
5. **GPU Utilization**: VRAM usage, compute usage (from `/health` or `nvidia-smi`)
6. **Memory Usage**: Cache size, RSS memory (from `/health`)
7. **Disk Usage**: Cache storage directory size

**Business Metrics**:
8. **Inference Throughput**: `inference_requests_by_priority` changes over time
9. **Plot Generation Rate**: `total_calc_runs` changes over time
10. **Model Fulfillment Time**: `avg_fulfillment_time_ms` (queue wait, not inference)

**Alerting Thresholds**:
- Queue depth > 80% capacity → Warning
- Queue depth > 95% capacity → Critical (service returns 503)
- Inference errors > 5% → Critical
- Cache hit rate < 30% → Warning (may need larger cache)
- VRAM usage > 90% → Warning (risk of OOM)

### Recommended Monitoring Stack

**Metrics Collection**:
- Prometheus: Scrape `/health` endpoint (convert JSON to metrics)
- Node Exporter: System metrics (CPU, memory, disk, network)
- NVIDIA DCGM Exporter: GPU metrics

**Visualization**:
- Grafana: Dashboards for all metrics
- Example panels:
  - Request latency histogram
  - Queue depth over time
  - Cache hit rate gauge
  - GPU utilization heatmap

**Alerting**:
- Alertmanager: Route alerts to Slack/PagerDuty
- Example rules:
  - `rate(total_http_errors[5m]) > 0.05` → Page
  - `queue_depth / queue_capacity > 0.95` → Page
  - `available_general_models == 0 for 1m` → Page

---

## Conclusion

GluRPC is a production-ready glucose prediction service with:
- ✅ **Robust architecture**: Layered, modular design with clear separation of concerns
- ✅ **High performance**: Async I/O, batch processing, intelligent two-tier caching
- ✅ **Concurrency safety**: RW locks, atomic transactions, tombstone pattern
- ✅ **Scalability**: Vertical (multi-GPU, configurable workers), horizontal (with modifications)
- ✅ **Reliability**: Graceful shutdown, disconnect detection, request deduplication, stale job cleanup
- ✅ **Observability**: Comprehensive health endpoint, structured logging, metrics tracking
- ✅ **Extensibility**: Clean abstractions, generic caching, dependency injection via Singletons

**Key Innovations**:
1. **Priority-Based Dual Model Queues**: Model #0 reserved for interactive requests, load balancing across general pool
2. **Request Deduplication with Last-Write-Wins**: Concurrent duplicate requests share computation, newest wins
3. **Per-Request Disconnect Futures**: Individual cancellation without affecting other waiters
4. **Two-Tier Hybrid Cache**: Hot LRU + persistent disk with async RW locks
5. **Queue Overload Protection**: Automatic 503 responses when system is saturated
6. **Graceful Cancellation**: Client disconnect → skip expensive operations, but don't waste in-progress work

The architecture prioritizes **low latency for interactive requests** while maximizing **throughput for background processing** through priority-based scheduling, intelligent caching, and request deduplication.
