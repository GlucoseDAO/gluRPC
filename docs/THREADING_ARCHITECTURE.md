# Threading & Async Architecture - Complete Code Analysis

## Overview

The gluRPC application implements a **hybrid async + thread pool architecture**:
- **Main Event Loop**: All I/O, coordination, state management, and synchronization
- **Thread Pool**: CPU-intensive operations offloaded via `asyncio.to_thread()` (Python 3.9+)
- **Synchronization**: `AsyncRWLock` for cache, `asyncio.Lock` for registries and state
- **Thread-Safe Model Access**: `threading.Lock` in `InferenceWrapper` for model protection

**Critical Architecture Principle**: Thread pool work is **isolated computation** - threads read inputs, compute, and return results. ALL **state mutations** happen on the event loop after `await` completes. This eliminates most threading complexity.

---

## 1. Execution Contexts

### 1.1 Main Event Loop (Single Thread)

**What runs here:**
- FastAPI/Uvicorn HTTP server
- All async coroutines (`async def` functions)
- BackgroundProcessor worker loops (as asyncio.Task coroutines)
- ALL state mutations (caches, registries, queues, dictionaries)
- ALL I/O operations

**Key components on event loop:**
- `app.py`: FastAPI request handlers (L159-497)
- `engine.py`: `BackgroundProcessor._inference_worker_loop()` (L634-822)
- `engine.py`: `BackgroundProcessor._calc_worker_loop()` (L847-918)
- `state.py`: `TaskRegistry` methods (`notify_success`, `notify_error`, etc.)
- `cache.py`: `HybridLRUCache` - all public async methods
- `core.py`: Action handlers orchestrating the workflow

### 1.2 Thread Pool (Multiple OS Threads)

**What runs here:**
- Heavy CPU-bound computations offloaded via `asyncio.to_thread()`
- NO direct state mutation - read-only or isolated work
- Model inference operations (thread-safe via `threading.Lock`)

**Operations running in threads:**

1. **CSV Parsing** (`logic.py:273-309: parse_csv_content`)
   - Called from: `core.py` L102, L127
   - File I/O and pandas/polars operations
   ```python
   # core.py L102
   result = await asyncio.to_thread(logic.convert_logic, content_base64)
   
   # core.py L127
   handle, unified_df = await asyncio.to_thread(logic.get_handle_and_df, content_base64)
   ```

2. **Data Analysis & Preparation** (`logic.py:338-390: analyse_and_prepare_df`)
   - Called from: `core.py` L131-136
   - Heavy dataframe operations, gap interpolation, timestamp synchronization
   ```python
   # core.py L131-136
   inference_df, warning_flags, actual_input_samples = await asyncio.to_thread(
       logic.analyse_and_prepare_df,
       unified_df,
       MINIMUM_DURATION_MINUTES,
       maximum_wanted_duration
   )
   ```

3. **Dataset Creation** (`logic.py:392-443: create_dataset_from_df`)
   - Called from: `engine.py` L686-690
   - Creates Darts dataset with scaling, interpolation, encoding
   ```python
   # engine.py L686-690
   result = await asyncio.to_thread(
       create_dataset_from_df,
       inference_df,
       warning_flags,
   )
   ```

4. **Model Inference** (`InferenceWrapper.run_inference`)
   - Called from: `engine.py` L726-732
   - Heavy ML inference with PyTorch (GIL released in native code)
   - Protected by `threading.Lock` in wrapper (L113-153)
   ```python
   # engine.py L724-732
   async with ModelManager().acquire(1, priority=priority) as wrappers:
       wrapper = wrappers[0]
       full_forecasts_array, logvars = await asyncio.to_thread(
           wrapper.run_inference, 
           dataset, 
           required_config,
           BATCH_SIZE, 
           NUM_SAMPLES 
       )
   ```

5. **Plot Calculation** (`logic.py:511-807: calculate_plot_data`)
   - Called from: `engine.py` L889-894 and `core.py` L322
   - NumPy operations, KDE calculations, Plotly rendering
   ```python
   # engine.py L889-894
   plot_json_str, plot_data = await asyncio.to_thread(
       calculate_plot_data,
       data, 
       index,
   )
   
   # core.py L322
   plot_json_str, _plot_data = await asyncio.to_thread(logic.calculate_plot_data, data, index)
   ```

6. **Disk I/O** (via `diskcache` in `HybridLRUCache`)
   - Called from: `cache.py` L156, L176, L187, L221, L236, L257, L272
   - All disk operations offloaded to avoid blocking event loop
   - **Thread-safe**: `diskcache.Index` has internal locks for concurrent access
   ```python
   # cache.py L156: Read from disk
   value: Optional[V] = await asyncio.to_thread(self._disk_get, key, default)
   # Thread-safe: diskcache.Index internal lock protects concurrent reads
   
   # cache.py L176: Write to disk
   await asyncio.to_thread(self._disk_set, key, value)
   # Thread-safe: diskcache.Index internal lock protects concurrent writes
   
   # cache.py L187: Delete from disk
   await asyncio.to_thread(self._disk_pop, key, None)
   
   # cache.py L221: Clear all
   await asyncio.to_thread(self._disk_clear)
   ```

---

## 2. Worker Architecture

### 2.1 Background Processor Initialization

**File**: `engine.py` L423-437

```python
async def start(self, num_inference_workers: int = NUM_COPIES, num_calc_workers: int = BACKGROUND_WORKERS_COUNT):
    if self.running:
        return
        
    self.running = True
    StateManager().reset_shutdown()
    logger.info(f"Starting {num_inference_workers} inference workers and {num_calc_workers} calculation workers...")
    
    # L431-433: Create inference worker tasks
    for i in range(num_inference_workers):
        task = asyncio.create_task(self._inference_worker_loop(i))
        self.inference_workers.append(task)
        
    # L435-437: Create calc worker tasks
    for i in range(num_calc_workers):
        task = asyncio.create_task(self._calc_worker_loop(i))
        self.calc_workers.append(task)
```

**Key Point**: These are **asyncio Tasks**, NOT OS threads. They run as coroutines on the main event loop, cooperatively multitasked.

### 2.2 Inference Worker Loop

**File**: `engine.py` L634-822

**Execution Context**: Event loop (async coroutine)

**Flow with Context Switches**:

```
[Event Loop] Inference Worker Loop (worker_id)
    │
    ├─ L642: await self.inference_queue.get()  
    │   └─ [Event Loop - Async wait for queue item]
    │
    ├─ L645-661: Check staleness (if indices provided + request_id)
    │   ├─ L650: await self.is_request_stale(handle, idx, request_id)
    │   │   └─ [Event Loop - Async lock in L584-589]
    │   └─ L652: await self.cleanup_stale_jobs(handle, idx, request_id)
    │       └─ [Event Loop]
    │
    ├─ L666: cached_data = await inf_cache.get(handle)  
    │   └─ [Event Loop - Async read lock in cache.py L144-153]
    │   └─ If cache miss: L156: await asyncio.to_thread(self._disk_get, ...)
    │       └─ [SWITCHES TO THREAD POOL] → Disk read operation
    │       └─ [RETURNS TO EVENT LOOP] → Data available
    │
    ├─ L686-690: await asyncio.to_thread(create_dataset_from_df, ...)
    │   └─ [SWITCHES TO THREAD POOL] → Heavy dataset creation
    │       └─ logic.py:392-443: Polars/Pandas ops, interpolation, scaling
    │   └─ [RETURNS TO EVENT LOOP] → Result dict available
    │
    ├─ L695: TaskRegistry().cancel_all_for_handle(handle)
    │   └─ [Event Loop - Sync method, dict.pop() is atomic via GIL]
    │
    ├─ L724-732: async with ModelManager().acquire(1, priority=priority):
    │   ├─ L300 or L305: await self.general_queue.get() or await self.priority_queue.get()
    │   │   └─ [Event Loop - Async wait for model availability]
    │   │
    │   ├─ L726-732: await asyncio.to_thread(wrapper.run_inference, ...)
    │   │   └─ [SWITCHES TO THREAD POOL] → Heavy ML inference
    │   │       ├─ L121-133: threading.Lock acquired/released (model protection)
    │   │       ├─ L148-153: threading.Lock acquired/released (inference protection)
    │   │       └─ logic.py:464-509: PyTorch operations (GIL released)
    │   │   └─ [RETURNS TO EVENT LOOP] → Forecasts & logvars ready
    │   │
    │   ├─ L755-780: async with inf_cache.transaction(handle):
    │   │   └─ [Event Loop - Holds write lock for cache update]
    │   │       ├─ cache.py L90-124: Transaction context
    │   │       └─ L122: self._hot[key] = value (under write lock)
    │   │       └─ L124: self._backend[key] = value (sync dict op, no await)
    │   │
    │   └─ L317-322: Model returned to queue
    │       └─ [Event Loop - Queue.put_nowait()]
    │
    └─ L783-793: self._enqueue_calculations(...)
        └─ [Event Loop - Queue.put_nowait() for calc queue]
    
    └─ L807-812: async with self._inference_lock:
        └─ [Event Loop - Update _pending_inference dict under lock]
```

**Critical Observations**: 
- State mutations (L695, L777, L810) happen on **event loop**
- Heavy work (L686-690, L726-732) happens in **thread pool**
- After `await asyncio.to_thread()` completes, execution returns to **event loop**
- Model access protected by `threading.Lock` (L113-153 in engine.py)
- Cache write protected by `AsyncRWLock.writer` (L755-780)

### 2.3 Calculation Worker Loop

**File**: `engine.py` L847-918

**Execution Context**: Event loop (async coroutine)

**Flow with Context Switches**:

```
[Event Loop] Calc Worker Loop (worker_id)
    │
    ├─ L856: await self.calc_queue.get()  
    │   └─ [Event Loop - Async wait]
    │
    ├─ L860-866: Check staleness and notify error if stale
    │   ├─ L860: await self.is_request_stale(handle, index, request_id)
    │   │   └─ [Event Loop - Async lock in L584-589]
    │   └─ L865: task_registry.notify_error(handle, index, e, request_id)
    │       └─ [Event Loop - Sync method, state.py L444-463]
    │
    ├─ L870: existing_plot = await plot_cache.get_plot(task_version, index)
    │   └─ [Event Loop]
    │       ├─ L106: entry = await self.get(version)  
    │       │   └─ [Event Loop - cache.py L139-165]
    │       │       ├─ L144-152: async with self._lock.reader (read lock)
    │       │       └─ If miss: L156: await asyncio.to_thread(self._disk_get, ...)
    │       │           └─ [SWITCHES TO THREAD POOL] → Disk read
    │       │           └─ [RETURNS TO EVENT LOOP]
    │       └─ L110: return entry.plots_jsons[array_index]
    │
    ├─ L872: task_registry.notify_success(handle, index, request_id)
    │   └─ [Event Loop - Sync method, state.py L423-442]
    │
    ├─ L876: data = await inf_cache.get(handle)
    │   └─ [Event Loop - Same as L666 above]
    │
    ├─ L884: task_registry.notify_error(...) if version mismatch
    │   └─ [Event Loop]
    │
    ├─ L889-894: await asyncio.to_thread(calculate_plot_data, ...)
    │   └─ [SWITCHES TO THREAD POOL] → Heavy plot calculation
    │       └─ logic.py:511-807: NumPy ops, KDE, Plotly rendering
    │   └─ [RETURNS TO EVENT LOOP] → plot_json_str & plot_data ready
    │
    ├─ L897: await self.increment_calc_runs()
    │   └─ [Event Loop - L607-613: async with self._calc_lock]
    │
    ├─ L900: await plot_cache.update_plot(task_version, index, plot_json_str, plot_data)
    │   └─ [Event Loop - state.py L113-135]
    │       └─ L135: await self.update_entry(version, updater)
    │           └─ cache.py L240-272: async with self._write_lock_logged(...)
    │
    ├─ L903: task_registry.notify_success(handle, index, request_id)
    │   └─ [Event Loop]
    │
    └─ L907: task_registry.notify_error(handle, index, e, request_id)
        └─ [Event Loop]
```

**Critical Observations**: 
- ALL `task_registry.notify_*` calls happen **after** `await asyncio.to_thread()` returns
- This means they execute on the **event loop thread**, not in thread pool
- The only thread pool work is plot calculation (L889-894)
- All cache operations use async locks (AsyncRWLock)

---

## 3. Synchronization Mechanisms

### 3.1 InferenceWrapper (Thread-Safe Model Access)

**File**: `engine.py` L104-163

**Lock Type**: `threading.Lock` - for protecting model state during multi-threaded access

**Architecture**:
```python
class InferenceWrapper:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model_state: Optional[ModelState] = None
        self._lock = threading.Lock()  # L113: Threading lock for model protection
```

**Operations**:

1. **load_if_needed** (L115-133)
```python
def load_if_needed(self, required_config: GluformerModelConfig):
    # L121: locks_logger.debug("Acquiring threading lock")
    with self._lock:  # L122: THREADING LOCK ACQUIRED
        # L124: locks_logger.debug("Acquired threading lock")
        if self.model_state is not None:
            current_config, _ = self.model_state
            if current_config == required_config:
                # L127: locks_logger.debug("Releasing threading lock (no reload)")
                return  # L128: THREADING LOCK RELEASED

        # L131: Reload model (inside lock)
        self.model_state = load_model(required_config, self.model_path, self.device)
        # L133: locks_logger.debug("Releasing threading lock (reloaded)")
        # L133: THREADING LOCK RELEASED
```

2. **run_inference** (L135-163)
```python
def run_inference(self, dataset, required_config, batch_size, num_samples):
    self.load_if_needed(required_config)  # May acquire/release lock
    
    # L148: locks_logger.debug("Acquiring threading lock for run_inference")
    with self._lock:  # L149: THREADING LOCK ACQUIRED
        # L150: locks_logger.debug("Acquired threading lock for run_inference")
        current_state = self.model_state
        # L153: locks_logger.debug("Releasing threading lock for run_inference")
        # L153: THREADING LOCK RELEASED
    
    # L155-162: Run inference WITHOUT lock (model state is immutable)
    predictions, logvars = run_inference_full(...)
    return predictions, logvars
```

**Why threading.Lock here?**
- `InferenceWrapper.run_inference()` is called from thread pool (via `asyncio.to_thread()`)
- Multiple threads can access the same wrapper concurrently
- `threading.Lock` ensures model loading/swapping is thread-safe
- Only holds lock during model state read/write, NOT during inference (which is long)

### 3.2 TaskRegistry (Cross-Task Notification)

**File**: `state.py` L375-607

**Lock Type**: `asyncio.Lock` - for protecting registry dict from concurrent coroutines

**Purpose**: Allow async tasks to wait for completion of background work

**Architecture**:
```python
class TaskRegistry(Singleton):
    def __init__(self):
        # L401: Key is (handle, index, request_id) triple
        self._registry: DefaultDict[Tuple[str, int, int], asyncio.Future] = {}
        self._lock = asyncio.Lock()  # L402: ASYNC LOCK
```

**Operations**:

1. **wait_for_result** (L534-606)
```python
async def wait_for_result(self, handle, index, request_id, disconnect_future, timeout):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    key = (handle, index, request_id)
    # L564: locks_logger.debug("Acquiring lock for wait_for_result")
    async with self._lock:  # L565: ASYNC LOCK ACQUIRED
        # L566: locks_logger.debug("Acquired lock for wait_for_result")
        self._registry[key] = future  # L567: Register future
        # L568: locks_logger.debug("Releasing lock for wait_for_result")
        # L568: ASYNC LOCK RELEASED
    
    # L570-596: Wait for future with timeout/disconnect racing (NO LOCK)
    # L600: locks_logger.debug("Acquiring lock for wait_for_result cleanup")
    async with self._lock:  # L601: ASYNC LOCK ACQUIRED
        # L602: locks_logger.debug("Acquired lock for wait_for_result cleanup")
        if key in self._registry:
            del self._registry[key]  # L604: Cleanup on timeout/cancel
        # L605: locks_logger.debug("Releasing lock for wait_for_result cleanup")
        # L605: ASYNC LOCK RELEASED
```

2. **notify_success** (L423-442)
```python
def notify_success(self, handle: str, index: int, request_id: Optional[int] = None):
    # Called from event loop (NOT from threads)
    # NO LOCK - dict.pop() is atomic via GIL
    if request_id is not None:
        key = (handle, index, request_id)
        if key in self._registry:
            f = self._registry.pop(key)  # L433: Atomic pop
            if not f.done():
                f.set_result(True)  # L435: Thread-safe future operation
    else:
        # L438-442: Notify all requests for (handle, index)
        keys_to_remove = [k for k in list(self._registry.keys()) if k[0] == handle and k[1] == index]
        for key in keys_to_remove:
            f = self._registry.pop(key)
            if not f.done():
                f.set_result(True)
```

3. **notify_error** (L444-463) - Similar to notify_success

4. **cancel_request** (L465-500)
```python
async def cancel_request(self, handle, index, request_id, reason="Request cancelled"):
    key = (handle, index, request_id)
    
    # L485-489: Cancel TaskRegistry future (NO LOCK - called from event loop)
    if key in self._registry:
        f = self._registry.pop(key)  # Atomic pop
        if not f.done():
            f.set_exception(Exception(f"{reason}: {handle[:8]}:{index}:{request_id}"))
    
    # L492-494: Trigger per-request disconnect
    disconnect_tracker = DisconnectTracker()
    await disconnect_tracker.cancel_request(handle, index, request_id)
```

**Thread-Safety Pattern**:
- All `notify_*` and `cancel_*` methods are called from event loop only (not threads)
- `dict.pop()` is atomic in CPython (GIL protection)
- `future.set_result()` / `future.set_exception()` are thread-safe
- Registration (adding futures) is protected by `asyncio.Lock`
- No thread→event-loop callbacks needed in current architecture

**Why asyncio.Lock here?**
- `TaskRegistry` is accessed from multiple event loop coroutines
- `asyncio.Lock` provides async-aware mutual exclusion
- Using `threading.Lock` would BLOCK the event loop (BAD)

### 3.3 DisconnectTracker (Request Lifecycle Management)

**File**: `state.py` L207-372

**Lock Type**: `asyncio.Lock` - for protecting tracking state

**Architecture**:
```python
class DisconnectTracker(Singleton):
    def __init__(self):
        # L241: Per (handle, index): {"seq": int, "count": int, "disconnect_future": Future}
        self._tracking: DefaultDict[Tuple[str, int], Dict] = defaultdict(...)
        
        # L244: Per-request futures: (handle, index, request_id) -> Future
        self._per_request_futures: Dict[Tuple[str, int, int], asyncio.Future] = {}
        
        self._lock = asyncio.Lock()  # L246: ASYNC LOCK
```

**Operations**:

1. **register_request** (L248-275)
```python
async def register_request(self, handle: str, index: int) -> int:
    key = (handle, index)
    # L255: locks_logger.debug("Acquiring lock for register_request")
    async with self._lock:  # L256: ASYNC LOCK ACQUIRED
        # L257: locks_logger.debug("Acquired lock for register_request")
        entry = self._tracking[key]
        entry["seq"] += 1  # L259: Increment sequence
        entry["count"] += 1  # L260: Increment active count
        request_id = entry["seq"]
        
        # L264-266: Create shared disconnect future if needed
        if entry["disconnect_future"] is None or entry["disconnect_future"].done():
            loop = asyncio.get_running_loop()
            entry["disconnect_future"] = loop.create_future()
        
        # L269-271: Create per-request disconnect future
        req_key = (handle, index, request_id)
        loop = asyncio.get_running_loop()
        self._per_request_futures[req_key] = loop.create_future()
        
        # L274: locks_logger.debug("Releasing lock for register_request")
        return request_id  # L275: ASYNC LOCK RELEASED
```

2. **unregister_request** (L277-304)
```python
async def unregister_request(self, handle: str, index: int, request_id: int):
    key = (handle, index)
    req_key = (handle, index, request_id)
    # L285: locks_logger.debug("Acquiring lock for unregister_request")
    async with self._lock:  # L286: ASYNC LOCK ACQUIRED
        # L287: locks_logger.debug("Acquired lock for unregister_request")
        entry = self._tracking[key]
        entry["count"] = max(0, entry["count"] - 1)  # L289: Decrement count
        
        # L294-297: Clean up per-request future
        if req_key in self._per_request_futures:
            per_req_future = self._per_request_futures.pop(req_key)
            if not per_req_future.done():
                per_req_future.set_result(True)
        
        # L300-302: Resolve shared future when count reaches 0
        if entry["count"] == 0 and entry["disconnect_future"] and not entry["disconnect_future"].done():
            entry["disconnect_future"].set_result(True)
        
        # L304: locks_logger.debug("Releasing lock for unregister_request")
        # L304: ASYNC LOCK RELEASED
```

3. **get_disconnect_future** (L306-333) - Returns per-request or shared future under lock

### 3.4 HybridLRUCache (Inference & Plot Caches)

**File**: `cache.py` L48-273

**Lock Type**: `aiorwlock.RWLock` - async reader-writer lock

**Architecture**:
```python
class HybridLRUCache(Generic[K, V], Singleton):
    def __init__(self, directory: str, max_hot: int = 8):
        self._backend = DiskIndex(directory)  # L62: Disk backend (has internal thread-safe locks)
        self._hot: LRUMemCache[K, V] = LRUMemCache(maxsize=max_hot)  # L64: Hot cache
        self._lock = AsyncRWLock()  # L65: ASYNC RW LOCK (protects hot cache)
```

**Two-Level Locking Design**:
- `self._lock` (AsyncRWLock): Coordinates event loop coroutines accessing `_hot` cache
- `diskcache.Index` internal locks: Thread-safe synchronization for `_backend` disk operations
- This separation allows disk I/O to run in thread pool without holding AsyncRWLock

**Operations**:

1. **get** (L139-165)
```python
async def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
    # 1) Try hot LRU under read lock
    # L144: locks_logger.debug("Acquiring read lock for key=...")
    async with self._lock.reader:  # L145: READ LOCK ACQUIRED
        # L146: locks_logger.debug("Acquired read lock for key=...")
        if key in self._hot:
            val = self._hot[key]
            # L149: locks_logger.debug("Releasing read lock (cache hit)")
            if val is _TOMBSTONE:
                return default
            return val  # L152: READ LOCK RELEASED
        # L153: locks_logger.debug("Releasing read lock (cache miss)")
        # L153: READ LOCK RELEASED

    # 2) Miss in hot: load from disk in thread (NO LOCK)
    value: Optional[V] = await asyncio.to_thread(self._disk_get, key, default)  # L156
    # [SWITCHES TO THREAD POOL] → Disk read
    # [RETURNS TO EVENT LOOP] → value ready

    # 3) If found on disk, promote to hot under write lock
    if value is not None:
        async with self._write_lock_logged(key, "get (from disk)"):  # L160
            # L160: WRITE LOCK ACQUIRED
            # L162-164: Promote to hot if not already there
            if key not in self._hot:
                self._hot[key] = value
            # WRITE LOCK RELEASED
    return value
```

2. **set** (L167-176)
```python
async def set(self, key: K, value: V) -> None:
    # Update hot LRU under write lock
    async with self._write_lock_logged(key, "set"):  # L172
        # L70-76: locks_logger.debug("Acquiring/Acquired/Releasing write lock")
        self._hot[key] = value  # L173: Fast update under lock
        # WRITE LOCK RELEASED

    # Persist to disk in background thread (NO LOCK)
    await asyncio.to_thread(self._disk_set, key, value)  # L176
    # [SWITCHES TO THREAD POOL] → Disk write
    # [RETURNS TO EVENT LOOP]
```

3. **delete** (L178-192)
```python
async def delete(self, key: K) -> None:
    # Remove from hot first, marking as deleted
    async with self._write_lock_logged(key, "delete"):  # L183
        self._hot[key] = _TOMBSTONE  # L184: Mark as deleted
        # WRITE LOCK RELEASED

    # Remove from disk (ignore if missing) - NO LOCK
    await asyncio.to_thread(self._disk_pop, key, None)  # L187
    # [SWITCHES TO THREAD POOL] → Disk delete
    # [RETURNS TO EVENT LOOP]

    # Remove tombstone
    async with self._write_lock_logged(key, "delete (after tombstone)"):  # L190
        if self._hot.get(key) is _TOMBSTONE:
            del self._hot[key]  # L192: Clean up tombstone
        # WRITE LOCK RELEASED
```

4. **transaction** (L79-124)
```python
@asynccontextmanager
async def transaction(self, key: K):
    async with self._write_lock_logged(key, "transaction"):  # L90
        # L90: WRITE LOCK ACQUIRED - HELD ENTIRE TRANSACTION
        
        # 1. Fetch current value (Hot -> Disk)
        current_value = None
        if key in self._hot:
            val = self._hot[key]
            if val is not _TOMBSTONE:
                current_value = val
        else:
            # L102: Read from disk (NO AWAIT - sync operation)
            current_value = self._backend.get(key, None)

        # 2. Setup context & yield to user code
        class TxnWrapper:
            def set(self, value: V):
                self._new_value = value
                self._should_update = True
        
        ctx = TxnWrapper(current_value)
        yield ctx  # L117: User code executes under lock
        
        # 3. Apply changes if set() was called
        if ctx._should_update:
            self._hot[key] = ctx._new_value  # L122: Update hot
            self._backend[key] = ctx._new_value  # L124: Update disk (sync)
        # WRITE LOCK RELEASED
```

**Thread-Safety Pattern**:
- **_hot cache** (in-memory): Protected by AsyncRWLock - only mutated under write lock
- **_backend** (disk): `diskcache.Index` has its own internal thread-safe locking
  - Can be safely accessed from thread pool via `asyncio.to_thread()`
  - Internal locks handle concurrent access from multiple threads
  - No need to hold AsyncRWLock during disk operations
- **TOMBSTONE pattern**: Prevents race where concurrent read tries to promote deleted key

**Critical Architecture Detail - Two-Level Locking**:
1. **AsyncRWLock** (`self._lock`): Protects hot cache and coordinates event loop coroutines
2. **diskcache internal locks**: Protect disk backend from concurrent thread access

This dual-locking strategy allows:
- Fast hot cache updates under AsyncRWLock (event loop)
- Slow disk I/O offloaded to thread pool without holding AsyncRWLock
- Thread-safe disk access via diskcache's internal synchronization

**Why disk I/O doesn't hold AsyncRWLock**:
```python
# BAD: Would block all cache ops during slow disk I/O
async with self._lock.writer:
    await asyncio.to_thread(self._disk_set, key, value)  # Lock held during I/O!

# GOOD: Update hot first (fast), then disk without lock
async with self._lock.writer:
    self._hot[key] = value  # Fast update under lock
# AsyncRWLock released here ↑
await asyncio.to_thread(self._disk_set, key, value)  # Slow I/O without AsyncRWLock
                                                       # diskcache.Index has its own lock
```

**Why AsyncRWLock here?**
- Multiple coroutines can read cache concurrently (readers)
- Only one coroutine can write at a time (writer)
- Readers and writer are mutually exclusive
- This is event loop concurrency, not thread parallelism

### 3.5 BackgroundProcessor State

**File**: `engine.py` L393-632

**Shared State**:
```python
# L406: Track pending inference to avoid redundant enqueueing
self._pending_inference: Dict[str, Tuple[int, int, int]] = {}
self._inference_lock = asyncio.Lock()  # L407: ASYNC LOCK

# L411: Track latest request_id per (handle, index) for stale detection
self._latest_request_id: Dict[Tuple[str, int], int] = {}
self._request_id_lock = asyncio.Lock()  # L412: ASYNC LOCK

# L421: Stats counters
self._total_calc_runs = 0
self._total_calc_errors = 0
self._calc_lock = asyncio.Lock()  # L421: ASYNC LOCK
```

**Protection Pattern**:

1. **enqueue_inference** (L489-540)
```python
async def enqueue_inference(self, handle, inference_df, warning_flags, expected_dataset_len, 
                            inference_config, priority, indices, force_calculate, request_id):
    # L510: locks_logger.debug("Acquiring asyncio lock for enqueue_inference")
    async with self._inference_lock:  # L511: ASYNC LOCK ACQUIRED
        # L512: locks_logger.debug("Acquired asyncio lock for enqueue_inference")
        
        # L514-525: Check if inference already pending
        if handle in self._pending_inference:
            pending_prio, pending_len, pending_req_id = self._pending_inference[handle]
            # Skip if already covered
            if pending_len >= expected_dataset_len:
                return
        
        # L533: Update pending state
        self._pending_inference[handle] = (priority, expected_dataset_len, request_id or 0)
        # L534: locks_logger.debug("Releasing asyncio lock for enqueue_inference")
        # L534: ASYNC LOCK RELEASED
    
    # L539: Enqueue to priority queue (without lock)
    self.inference_queue.put_nowait(item)
```

2. **is_request_stale** (L567-594)
```python
async def is_request_stale(self, handle: str, index: int, request_id: Optional[int]) -> bool:
    if request_id is None:
        return False  # Background jobs never stale
    
    key = (handle, index)
    # L584: locks_logger.debug("Acquiring lock for is_request_stale")
    async with self._request_id_lock:  # L585: ASYNC LOCK ACQUIRED
        # L586: locks_logger.debug("Acquired lock for is_request_stale")
        latest = self._latest_request_id.get(key, 0)
        is_stale = request_id < latest
        # L589: locks_logger.debug("Releasing lock for is_request_stale")
        # L589: ASYNC LOCK RELEASED
        
        if is_stale:
            logger.info(f"Request {handle[:8]}:{index}:{request_id} is stale (latest={latest})")
        
        return is_stale
```

3. **Cleanup in worker** (L807-812)
```python
# In _inference_worker_loop finally block
finally:
    # Clear the pending flag for this handle
    # L807: locks_logger.debug("Acquiring asyncio lock for pending_inference cleanup")
    async with self._inference_lock:  # L808: ASYNC LOCK ACQUIRED
        # L809: locks_logger.debug("Acquired asyncio lock for pending_inference cleanup")
        if handle in self._pending_inference:
            del self._pending_inference[handle]
        # L812: locks_logger.debug("Releasing asyncio lock for pending_inference cleanup")
        # L812: ASYNC LOCK RELEASED
```

**Context**: All accesses from event loop (worker coroutines), so `asyncio.Lock` is appropriate.

### 3.6 ModelManager (Model Pool Management)

**File**: `engine.py` L167-367

**Lock Type**: `asyncio.Lock` for initialization

**Architecture**:
```python
class ModelManager(Singleton):
    def __init__(self):
        self.models: List[InferenceWrapper] = []
        self.priority_queue = asyncio.Queue()  # L176: Model #0 (priority 0 only)
        self.general_queue = asyncio.Queue()   # L177: Models #1+ (all requests)
        self.initialized = False
        self._init_lock = asyncio.Lock()  # L179: ASYNC LOCK for initialization
```

**Operations**:

1. **initialize** (L195-220)
```python
async def initialize(self, model_name: str = "..."):
    if self.initialized:
        return

    # L199: locks_logger.debug("Acquiring asyncio lock for initialize")
    async with self._init_lock:  # L200: ASYNC LOCK ACQUIRED
        # L201: locks_logger.debug("Acquired asyncio lock for initialize")
        if self.initialized:
            return  # Double-check
            
        logger.info(f"Initializing ModelManager with model: {model_name}")
        
        # L207: Load models in thread pool
        await asyncio.to_thread(self._load_models_sync, model_name)
        # [SWITCHES TO THREAD POOL] → Model loading (PyTorch)
        # [RETURNS TO EVENT LOOP] → Models ready
        
        # L210-216: Distribute models to queues
        if len(self.models) > 0:
            self.priority_queue.put_nowait(self.models[0])
        for model in self.models[1:]:
            self.general_queue.put_nowait(model)
        
        self.initialized = True
        # L220: locks_logger.debug("Releasing asyncio lock for initialize")
        # L220: ASYNC LOCK RELEASED
```

2. **acquire** (L267-327)
```python
@asynccontextmanager
async def acquire(self, requested_copies: int = 1, priority: int = 1):
    if not self.initialized:
        raise RuntimeError("Models not initialized")
        
    start_time = time.time()
    self.increment_inference_request(priority)  # L284: No lock (atomic int increment)
    
    acquired_models = []
    try:
        if priority == 0:
            # High priority: Try general pool first, fall back to priority model
            try:
                model = self.general_queue.get_nowait()  # L295: NO AWAIT
                acquired_models.append(model)
            except asyncio.QueueEmpty:
                model = await self.priority_queue.get()  # L300: AWAIT queue
                acquired_models.append(model)
        else:
            # Background: Only use general pool
            model = await self.general_queue.get()  # L305: AWAIT queue
            acquired_models.append(model)
        
        yield acquired_models  # L309: User code runs with model
    finally:
        # L316-322: Return models to queues
        for model in acquired_models:
            if model == self.models[0]:
                self.priority_queue.put_nowait(model)
            else:
                self.general_queue.put_nowait(model)
```

**No explicit lock needed for acquire**:
- `asyncio.Queue.get()` and `put_nowait()` are thread-safe and async-safe
- Model instances are immutable once loaded
- Model internal state is protected by `threading.Lock` in `InferenceWrapper`

---

## 4. Complete Request Flow Examples

### 4.1 HTTP Request → Prediction → Plot Response

**Files**: `app.py` L201-273 (`draw_a_plot` endpoint), `core.py` L222-329 (`generate_plot_from_handle`)

```
[1] HTTP Request arrives at /draw_a_plot
    └─ [Event Loop - FastAPI handler app.py L202]

[2] L214: Check for overload
    └─ check_queue_overload()
    └─ [Event Loop - Check queue sizes]

[3] L224-225: Register request
    └─ disconnect_tracker.register_request(handle, index)
    └─ [Event Loop - async with lock L256, returns request_id]

[4] L228-229: Update latest request_id
    └─ bg_processor.update_latest_request_id(handle, index, request_id)
    └─ [Event Loop - async with lock L562-565]

[5] L233: Get disconnect future
    └─ disconnect_tracker.get_disconnect_future(handle, index, request_id)
    └─ [Event Loop - async with lock L314-325]

[6] L242-248: Call generate_plot_from_handle()
    └─ [Event Loop - core.py L222]
    
    [6a] L235: Check inference cache
         └─ data = await inf_cache.get(handle)
         └─ [Event Loop - cache.py L139-165]
             ├─ L145: async with self._lock.reader (read lock)
             └─ If miss: L156: await asyncio.to_thread(self._disk_get, ...)
                 └─ [SWITCHES TO THREAD POOL] → Disk read
                 └─ [RETURNS TO EVENT LOOP] → data ready
    
    [6b] L238-243: Check if inference pending
         └─ pending_status = bg_processor.get_pending_status(handle)
         └─ [Event Loop - dict access, no lock]
    
    [6c] L246-260: If not in cache and processing, wait
         └─ await TaskRegistry().wait_for_result(handle, index, request_id, ...)
         └─ [Event Loop - state.py L534-606]
             ├─ L565: async with self._lock (register future)
             ├─ L571-596: await asyncio.wait([work_task, disconnect_future])
             └─ Waiting for notification from worker...
    
    [6d] L287-293: Check plot cache
         └─ plot_json_str = await plot_cache.get_plot(data.version, index)
         └─ [Event Loop - state.py L104-111]
             └─ L106: entry = await self.get(version)
                 └─ [Event Loop - cache.py L139-165, read lock]
    
    [6e] L297-316: If predictions pending, wait again
         └─ await TaskRegistry().wait_for_result(...)
         └─ [Event Loop - same as 6c]
    
    [6f] L322: Calculate plot inline if not cached
         └─ plot_json_str, _plot_data = await asyncio.to_thread(logic.calculate_plot_data, data, index)
         └─ [SWITCHES TO THREAD POOL] → Plot calculation
             └─ logic.py:511-807: NumPy, KDE, Plotly
         └─ [RETURNS TO EVENT LOOP] → plot_json_str ready
    
    [6g] L325: Store in plot cache
         └─ await plot_cache.update_plot(data.version, index, plot_json_str)
         └─ [Event Loop - state.py L113-135]
             └─ L135: await self.update_entry(version, updater)
                 └─ cache.py L240-272: async with write lock

[7] L249: Return plot_dict to client
    └─ [Event Loop - JSON serialization]

[8] L273: Finally block - unregister request
    └─ disconnect_tracker.unregister_request(handle, index, request_id)
    └─ [Event Loop - async with lock L286, decrement count]
```

### 4.2 Background Inference Flow (Process Unified)

**Files**: `app.py` L176-199 (`process_unified`), `core.py` L112-220 (`parse_and_schedule`), `engine.py` L634-822 (worker)

```
[1] HTTP Request arrives at /process_unified
    └─ [Event Loop - FastAPI handler app.py L177]

[2] L185: Check for overload
    └─ check_queue_overload()
    └─ [Event Loop]

[3] L194: Call parse_and_schedule()
    └─ [Event Loop - core.py L112]
    
    [3a] L127: Get handle and dataframe
         └─ handle, unified_df = await asyncio.to_thread(logic.get_handle_and_df, content_base64)
         └─ [SWITCHES TO THREAD POOL] → CSV parsing
             └─ logic.py:322-336: Base64 decode, file I/O, FormatParser
         └─ [RETURNS TO EVENT LOOP] → handle, unified_df ready
    
    [3b] L131-136: Analyze and prepare
         └─ inference_df, warning_flags, actual_input_samples = await asyncio.to_thread(
                logic.analyse_and_prepare_df, unified_df, ...)
         └─ [SWITCHES TO THREAD POOL] → Heavy dataframe ops
             └─ logic.py:338-390: Gap interpolation, timestamp sync
         └─ [RETURNS TO EVENT LOOP] → inference_df ready
    
    [3c] L142-146: Calculate expected dataset length
         └─ expected_dataset_len = logic.calculate_dataset_length_from_input(...)
         └─ [Event Loop - Simple math]
    
    [3d] L158: Check inference cache
         └─ data = await inf_cache.get(handle)
         └─ [Event Loop - cache.py L139-165, read lock]
    
    [3e] L179: Check pending status
         └─ pending_status = bg_processor.get_pending_status(handle)
         └─ [Event Loop - dict access]
    
    [3f] L194-203: Enqueue inference
         └─ await bg_processor.enqueue_inference(handle, inference_df, ...)
         └─ [Event Loop - engine.py L489-540]
             ├─ L511: async with self._inference_lock
             ├─ L539: self.inference_queue.put_nowait(item)
             └─ Returns immediately (fire-and-forget)

[4] Inference Worker picks up job
    └─ [Event Loop - Inference Worker Loop engine.py L634]
    
    [4a] L642: await self.inference_queue.get()
         └─ [Event Loop - Async wait]
    
    [4b] L645-661: Check staleness (if request_id)
         └─ await self.is_request_stale(handle, idx, request_id)
         └─ [Event Loop - async with lock L585]
    
    [4c] L666: Check cache again (avoid redundant work)
         └─ cached_data = await inf_cache.get(handle)
         └─ [Event Loop - read lock]
    
    [4d] L686-690: Create dataset
         └─ result = await asyncio.to_thread(create_dataset_from_df, inference_df, warning_flags)
         └─ [SWITCHES TO THREAD POOL] → Dataset creation
             └─ logic.py:392-443: Polars ops, Darts TimeSeries, scaling
         └─ [RETURNS TO EVENT LOOP] → dataset ready
    
    [4e] L724-732: Acquire model and run inference
         └─ async with ModelManager().acquire(1, priority=priority) as wrappers:
             └─ [Event Loop - L300 or L305: await queue.get()]
             └─ wrapper = wrappers[0]
             └─ full_forecasts_array, logvars = await asyncio.to_thread(
                    wrapper.run_inference, dataset, required_config, ...)
             └─ [SWITCHES TO THREAD POOL] → ML inference
                 ├─ engine.py L115-133: threading.Lock for model protection
                 ├─ engine.py L148-153: threading.Lock for inference call
                 └─ logic.py:464-509: PyTorch operations (GIL released)
             └─ [RETURNS TO EVENT LOOP] → predictions ready
    
    [4f] L755-780: Save to cache with transaction
         └─ async with inf_cache.transaction(handle) as txn:
             └─ [Event Loop - Holds write lock]
                 ├─ cache.py L90-124: Transaction context
                 ├─ L122: self._hot[key] = value (under lock)
                 └─ L124: self._backend[key] = value (sync, under lock)
    
    [4g] L783-793: Enqueue calculations
         └─ self._enqueue_calculations(handle, forecasts, version, indices, ...)
         └─ [Event Loop - calc_queue.put_nowait()]
    
    [4h] L807-812: Clear pending flag
         └─ async with self._inference_lock:
             └─ del self._pending_inference[handle]

[5] Calc Worker picks up calculation
    └─ [Event Loop - Calc Worker Loop engine.py L847]
    
    [5a] L856: await self.calc_queue.get()
         └─ [Event Loop - Async wait]
    
    [5b] L860-866: Check staleness
         └─ await self.is_request_stale(handle, index, request_id)
         └─ [Event Loop - async with lock]
    
    [5c] L870: Check plot cache
         └─ existing_plot = await plot_cache.get_plot(task_version, index)
         └─ [Event Loop - read lock]
    
    [5d] L876: Get inference data
         └─ data = await inf_cache.get(handle)
         └─ [Event Loop - read lock]
    
    [5e] L889-894: Calculate plot
         └─ plot_json_str, plot_data = await asyncio.to_thread(calculate_plot_data, data, index)
         └─ [SWITCHES TO THREAD POOL] → Plot calculation
             └─ logic.py:511-807: NumPy ops, KDE, Plotly rendering
         └─ [RETURNS TO EVENT LOOP] → plot_json_str ready
    
    [5f] L897: Increment stats
         └─ await self.increment_calc_runs()
         └─ [Event Loop - async with lock L610]
    
    [5g] L900: Store plot
         └─ await plot_cache.update_plot(task_version, index, plot_json_str, plot_data)
         └─ [Event Loop - write lock]
    
    [5h] L903: Notify waiting requests
         └─ task_registry.notify_success(handle, index, request_id)
         └─ [Event Loop - sync method]
             └─ state.py L423-442: dict.pop(), future.set_result()

[6] Original request (if waiting) resumes
    └─ [Event Loop - TaskRegistry future resolved]
    └─ Returns to caller with handle and warnings
```

**Key Observations**:
- **NO direct thread→event-loop state mutation** - all mutations happen on event loop
- **Thread work is isolated** - pure computation with inputs/outputs
- **Coordination via await** - thread work returns to event loop before any state change
- **Futures for async wait** - TaskRegistry lets one coroutine wait for another's work

---

## 5. Lock Type Summary

| Component | Lock Type | Purpose | Context | Location |
|-----------|-----------|---------|---------|----------|
| **InferenceWrapper._lock** | `threading.Lock` | Protect model state during load/access | Thread pool (multi-threaded) | engine.py L113 |
| **TaskRegistry._lock** | `asyncio.Lock` | Protect _registry dict mutations | Event loop (async) | state.py L402 |
| **DisconnectTracker._lock** | `asyncio.Lock` | Protect tracking state | Event loop (async) | state.py L246 |
| **BackgroundProcessor._inference_lock** | `asyncio.Lock` | Protect _pending_inference dict | Event loop (async) | engine.py L407 |
| **BackgroundProcessor._request_id_lock** | `asyncio.Lock` | Protect _latest_request_id dict | Event loop (async) | engine.py L412 |
| **BackgroundProcessor._calc_lock** | `asyncio.Lock` | Protect stats counters | Event loop (async) | engine.py L421 |
| **ModelManager._init_lock** | `asyncio.Lock` | Protect initialization | Event loop (async) | engine.py L179 |
| **HybridLRUCache._lock** | `AsyncRWLock` | Protect hot cache and coordinate disk I/O | Event loop (async) | cache.py L65 |
| **diskcache.Index** (internal) | Thread-safe locks | Protect disk backend from concurrent threads | Thread pool + Event loop | External library |
| **InferenceCache** (inherits) | `AsyncRWLock` | Cache consistency | Event loop (async) | state.py L142 |
| **PlotCache** (inherits) | `AsyncRWLock` | Cache consistency | Event loop (async) | state.py L94 |

**Lock Type Matching**:
- ✅ **Threading locks ONLY in thread pool context** (`InferenceWrapper._lock`)
- ✅ **Async locks ONLY in event loop context** (all other locks)
- ✅ **NO async locks used in thread pool code** (would cause runtime error)
- ✅ **NO threading locks used in event loop code** (would block event loop)

---

## 6. Context Switch Summary

### All `asyncio.to_thread()` Invocations

| File | Line | Function Call | Context Switch | Purpose |
|------|------|---------------|----------------|---------|
| **core.py** | L102 | `logic.convert_logic` | Event Loop → Thread Pool → Event Loop | CSV parsing and conversion |
| **core.py** | L127 | `logic.get_handle_and_df` | Event Loop → Thread Pool → Event Loop | CSV parsing, handle computation |
| **core.py** | L131-136 | `logic.analyse_and_prepare_df` | Event Loop → Thread Pool → Event Loop | Heavy dataframe processing |
| **core.py** | L322 | `logic.calculate_plot_data` | Event Loop → Thread Pool → Event Loop | Plot calculation (inline) |
| **engine.py** | L207 | `self._load_models_sync` | Event Loop → Thread Pool → Event Loop | Model loading on startup |
| **engine.py** | L686-690 | `create_dataset_from_df` | Event Loop → Thread Pool → Event Loop | Dataset creation from dataframe |
| **engine.py** | L726-732 | `wrapper.run_inference` | Event Loop → Thread Pool → Event Loop | ML model inference (PyTorch) |
| **engine.py** | L889-894 | `calculate_plot_data` | Event Loop → Thread Pool → Event Loop | Plot calculation in worker |
| **cache.py** | L156 | `self._disk_get` | Event Loop → Thread Pool → Event Loop | Disk read (cache miss) |
| **cache.py** | L176 | `self._disk_set` | Event Loop → Thread Pool → Event Loop | Disk write (async persist) |
| **cache.py** | L187 | `self._disk_pop` | Event Loop → Thread Pool → Event Loop | Disk delete |
| **cache.py** | L221 | `self._disk_clear` | Event Loop → Thread Pool → Event Loop | Clear all disk cache |
| **cache.py** | L236 | `list, self._backend` | Event Loop → Thread Pool → Event Loop | Iterate disk keys |
| **cache.py** | L257 | `self._disk_get` | Event Loop → Thread Pool → Event Loop | Read during update_entry |
| **cache.py** | L272 | `self._disk_set` | Event Loop → Thread Pool → Event Loop | Write during update_entry |

**Total Context Switches**: 15 distinct `asyncio.to_thread()` call sites

**Pattern**: 
```
[Event Loop] → await asyncio.to_thread(func, args)
    ├─ [Context Switch OUT]
    ├─ [Thread Pool] → func(args) executes
    ├─ [Context Switch IN]
    └─ [Event Loop] → result available
```

---

## 7. Safety Guarantees

### 7.1 What Makes This Safe?

1. **Single-threaded state mutation**:
   - All dict/cache/registry mutations happen on event loop thread
   - No concurrent writes to shared state from threads

2. **Async locks for event loop concurrency**:
   - `AsyncRWLock` for caches (multiple reader coroutines, exclusive writer)
   - `asyncio.Lock` for registries and state dicts
   - Proper async context managers (`async with`)

3. **Threading locks for thread pool operations**:
   - `threading.Lock` in `InferenceWrapper` for model state protection
   - Ensures model loading/swapping is thread-safe

4. **Thread isolation**:
   - Thread pool work is pure computation
   - No shared mutable state accessed from threads (except model state with lock)
   - Results returned to event loop via `await`

5. **Atomic operations**:
   - `dict.pop()` is atomic in CPython (GIL protection)
   - `future.set_result()` / `future.set_exception()` are thread-safe

### 7.2 Common Pitfalls (Avoided)

❌ **BAD**: Direct state mutation from thread
```python
def thread_work():
    cache[key] = value  # RACE CONDITION!
await asyncio.to_thread(thread_work)
```

✅ **GOOD**: Return value, mutate on event loop
```python
def thread_work():
    return computed_value  # Read-only work
result = await asyncio.to_thread(thread_work)
cache[key] = result  # Mutation on event loop
```

❌ **BAD**: Threading.Lock in async code
```python
lock = threading.Lock()
async def handler():
    lock.acquire()  # BLOCKS EVENT LOOP!
```

✅ **GOOD**: Asyncio.Lock for event loop
```python
lock = asyncio.Lock()
async def handler():
    async with lock:  # Cooperatively yields
        ...
```

❌ **BAD**: Async lock in thread
```python
def thread_work():
    async with cache._lock:  # CAN'T AWAIT IN THREAD!
        ...
```

✅ **GOOD**: Separate thread-safe sync calls or return to loop
```python
def thread_work():
    return value
result = await asyncio.to_thread(thread_work)
async with cache._lock:
    cache._hot[key] = result
```

❌ **BAD**: AsyncRWLock in thread pool
```python
def thread_work():
    async with self._lock.writer:  # RUNTIME ERROR - no event loop
        self._hot[key] = value
```

✅ **GOOD**: Threading.Lock if needed in thread, or return to event loop
```python
# In InferenceWrapper (thread pool context)
def run_inference(self, dataset, config, batch_size, num_samples):
    with self._lock:  # threading.Lock
        current_state = self.model_state
    return run_inference_full(dataset, config, current_state, ...)
```

---

## 8. Performance Characteristics

### 8.1 Why This Architecture?

**Problem**: ML inference is CPU-intensive and blocks the event loop
```python
# This would freeze all HTTP requests during inference:
async def bad_handler():
    result = expensive_ml_inference()  # Blocks event loop for seconds!
    return result
```

**Solution**: Offload CPU work to threads with `asyncio.to_thread()` (Python 3.9+)
```python
async def good_handler():
    result = await asyncio.to_thread(expensive_ml_inference)  # Event loop free!
    return result
```

### 8.2 Concurrency Model

**Event Loop Concurrency**: Multiple coroutines (I/O-bound)
- ~10-100 concurrent HTTP requests
- ~NUM_COPIES inference worker coroutines (default: 2+)
- ~BACKGROUND_WORKERS_COUNT calc worker coroutines (default: 4)
- All cooperatively scheduled (no GIL contention)

**Thread Pool Parallelism**: True parallel execution (CPU-bound)
- Default ThreadPoolExecutor size (typically 5x CPU cores, min 32 threads)
- Heavy ML inference runs in parallel across threads
- GIL released during NumPy/PyTorch operations (native code)
- Model state protected by `threading.Lock` to prevent corruption

### 8.3 Bottlenecks

1. **Cache write lock**: Serializes cache updates
   - Mitigated by: Short critical sections, disk I/O outside lock

2. **Model availability**: Limited by NUM_COPIES models
   - Mitigated by: Priority queue (model #0 for interactive requests)

3. **GIL for pure Python code**: Limits thread parallelism
   - Mitigated by: Most heavy work in NumPy/PyTorch (releases GIL)

4. **Queue sizes**: Bounded to prevent memory exhaustion
   - Inference queue: MAX_INFERENCE_QUEUE_SIZE = 32
   - Calc queue: MAX_CALC_QUEUE_SIZE = 256
   - Overload returns HTTP 503 when >75% full

---

## 9. Implementation Status (from Original Doc)

### ✅ Completed Steps

**Step 1: Instrumentation (Partial)**
- ✅ Track per-queue depth (inference, calc) and emit in `/health` endpoint
- ✅ Track queue capacities (`MAX_INFERENCE_QUEUE_SIZE=32`, `MAX_CALC_QUEUE_SIZE=256`)
- ✅ Calculate load status: `loaded` (<50%), `overloaded` (50-75%), `full` (>75%)
- ✅ Expose `load_status`, queue sizes, and capacities in `/health` response
- ⏳ TODO: Track cancellation events, enqueue latency, worker busy counts

**Step 3: Unified Cancellation Hook (Partial)**
- ✅ Request ID assignment: `DisconnectTracker.register_request()` (state.py L248)
- ✅ Per-request tracking: `_latest_request_id` dict (engine.py L411)
- ✅ Stale detection: `is_request_stale()` (engine.py L567)
- ✅ Worker checks: Before dataset creation (L650) and inference (L860)
- ✅ Cancellation hook: `TaskRegistry.cancel_request()` (state.py L465)
- ⏳ TODO: Queue removal (not possible with asyncio.PriorityQueue)

**Step 4: Overload Controls**
- ✅ Bounded in-memory queues with configurable capacities
- ✅ On queue full (>99% utilization), return 503 Service Unavailable with `Retry-After: 30` header
- ✅ Processing endpoints check for overload (app.py L185, L214, L286)
- ✅ Health endpoint exposes queue status (app.py L443)
- ⏳ TODO: Drain mode, per-priority/per-tenant limits

### ⏳ Remaining Steps
- Step 2: Request lifecycle hooks (disconnect detection) - ✅ PARTIALLY DONE (DisconnectMiddleware)
- Step 5: Cooperative cancellation in workers - ✅ PARTIALLY DONE (staleness checks)
- Step 6: Per-stage timeouts
- Step 7: Graceful shutdown with drain mode
- Step 8: Enhanced observability (metrics, alerts)
- Step 9: Comprehensive test coverage

---

## Summary

**Architecture**: Async event loop + thread pool hybrid

**Execution Contexts**:
- Event loop: All coordination, state, I/O, synchronization
- Thread pool: CPU-bound ML work only (isolated computation)

**Safety Mechanisms**:
- `AsyncRWLock` for caches (event loop concurrency)
- `asyncio.Lock` for registries and state (event loop concurrency)
- `threading.Lock` for model state (thread pool safety)
- NO async locks in thread pool code
- NO threading locks blocking event loop

**Key Insight**: Thread pool work is **isolated computation** - it reads inputs, computes, returns. All **state mutations** happen back on the event loop after `await` completes. This eliminates most threading complexity.

**Thread-Safety**: Achieved by **separation** (threads don't touch shared state except model with lock) rather than **locks everywhere** (which would be error-prone).

**Lock Hygiene**: ✅ Perfect match - async locks in async context, threading locks in thread context.

**Context Switches**: 15 `asyncio.to_thread()` call sites, all properly returning to event loop before state mutation.
