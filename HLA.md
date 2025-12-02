# High Level Architecture (HLA)

## 1. Current Architecture (As Is)

The current implementation attempts to use an async worker pattern but suffers from race conditions, lockups, and suboptimal queue management.

### Lifecycle
1.  **Startup**: `app.lifespan` triggers `MODEL_MANAGER.initialize()` and `BACKGROUND_PROCESSOR.start()`.
2.  **Model Loading**: Models are loaded into memory (RAM/VRAM).
3.  **Background Processing**: Two sets of workers (`inference_workers` and `calc_workers`) consume from `inference_queue` and `calc_queue`.

### Components
-   **ModelManager**: Handles model loading and "acquiring" models for inference threads.
-   **BackgroundProcessor**: Manages `asyncio.PriorityQueue`s and worker tasks.
-   **CACHE**: A global `Dict` protected by `asyncio.Lock`, storing datasets, scalers, intermediate predictions, and final plot results.

### Data Flows
-   **Process & Cache**:
    -   Parses CSV.
    -   Stores dataset in `CACHE`.
    -   Enqueues **ALL** indices into `inference_queue` immediately (flooding the queue).
-   **Draw Plot**:
    -   Checks `CACHE` for result.
    -   If missing, enqueues a high-priority task and waits on an `asyncio.Future`.
    -   *Issue*: The waiting mechanism combined with complex lock usage and potential queue flooding leads to deadlocks or timeouts.
-   **Quick Plot**:
    -   Calls `Process & Cache` (floods queue).
    -   Immediately requests plot for the last index (high priority).
    -   Deletes cache entry after completion.
    -   *Issue*: Extremely inefficient resource usage; processes entire dataset logic for a single point, and fills queue with zombie tasks (tasks for a handle that is deleted shortly after).

---

## 2. Target Architecture (To Be) - Reactive Pattern

The new architecture focuses on a strict reactive lifecycle, separation of concerns, and efficient queue management with LIFO prioritization for user requests.

### Lifecycle
1.  **App Spin-up**: FastAPI starts.
2.  **Queue System**: Request queues and worker queues initialize.
3.  **Model Loading (Critical)**: Models are loaded. If this fails, the application terminates.
4.  **Cache Initialization**:
    -   **Data Cache**: Stores input datasets (parsed CSVs) and scalers. Key: SHA256 Hash.
    -   **Result Cache**: Stores computed plot data. Key: SHA256 Hash + Index.
5.  **Worker Activation**:
    -   **Inference Workers**: Consume `(Hash, Index)` from Inference Queue.
    -   **Calculation Workers**: Consume `(Hash, Index, Forecast)` from Calculation Queue.

### Queues & Prioritization
-   **Inference Queue**: `PriorityQueue`. Items: `(Priority, Timestamp, Hash, Index)`.
    -   **Priority 0 (High/User)**: Interactive requests (Draw/QuickDraw). Uses LIFO (Last-In-First-Out) via timestamp to prioritize most recent user actions.
    -   **Priority 1 (Low/Background)**: Background pre-calculation of entire datasets.
-   **Calculation Queue**: `PriorityQueue`. Items: `(Priority, Timestamp, Hash, Index, Forecast)`. Matches Inference Queue priorities.

### Workers Logic

#### Inference Worker
1.  Pop `(Hash, Index)` from Inference Queue.
2.  Check `Data Cache` for dataset. If missing (expired/deleted), discard task.
3.  Run Inference (GPU/CPU).
4.  Push result to `Calculation Queue`.

#### Calculation Worker
1.  Pop `(Hash, Index, Forecast)` from Calculation Queue.
2.  Perform CPU-heavy math (unscaling, KDE).
3.  Store result in `Result Cache`.
4.  Notify waiting request (if any) via `Event` or `Future`.

#### Draw Worker (Request Handler)
Logic for handling `draw_a_plot` requests:
1.  **Check Hash**:
    -   Absent in `Data Cache` -> **Error 404** (Stale request or invalid handle).
2.  **Check Result**:
    -   Present in `Result Cache` -> **Return Result** (Cache Hit).
    -   Absent -> **Enqueue Request**:
        -   Push `(Hash, Index)` to Inference Queue with **Priority 0 (High)**.
        -   Wait for result (subscribe to completion event).

### QuickDraw Logic
Optimized flow for single-shot requests:
1.  **Parse & Store**: Parse CSV, store in `Data Cache` (transient).
2.  **Enqueue One**: Push **only** the last index to Inference Queue (Priority 0).
3.  **Wait**: Wait for result.
4.  **Return**: Return plot.
5.  **Cleanup**: Remove Hash from `Data Cache` and `Result Cache`.
    -   *Note*: Background workers encountering this Hash later (if any leaked) will simply discard tasks as "Hash absent".

### Shutdown
0.  Write health status/stats snapshot to log.
1.  Stop accepting new requests.
2.  Terminate Workers.
3.  Unload Models.
4.  Flush Inference Queue.
5.  Flush Calculation Queue.
6.  Flush Caches.

