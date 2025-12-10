# Threading & Async Architecture - Complete Dissection

## Overview

The gluRPC application uses a **hybrid async + thread pool architecture**:
- **Main Event Loop**: All I/O, coordination, and state management
- **Thread Pool**: CPU-intensive operations (dataset creation, model inference, plot calculations)
- **Synchronization**: AsyncRWLock for cache, asyncio.Lock for registries, loop.call_soon_threadsafe for cross-thread signaling

---

## 1. Execution Contexts

### 1.1 Main Event Loop (Single Thread)

**What runs here:**
- FastAPI/Uvicorn HTTP server
- All async coroutines (`async def` functions)
- BackgroundProcessor worker loops (as asyncio.Task coroutines)
- All state mutations (caches, registries, queues)
- All I/O operations

**Key components on event loop:**
- `app.py`: FastAPI request handlers
- `engine.py`: `BackgroundProcessor._inference_worker_loop()` (L445-601)
- `engine.py`: `BackgroundProcessor._calc_worker_loop()` (L626-688)
- `state.py`: `TaskRegistry` internal methods (`_notify_success_internal`, etc.)
- `cache.py`: `HybridLRUCache` - all public async methods

### 1.2 Thread Pool (Multiple OS Threads)

**What runs here:**
- Heavy CPU-bound computations offloaded via `asyncio.to_thread()` (Python 3.9+)
- NO direct state mutation - read-only or isolated work

**Operations running in threads:**
1. **Dataset Creation** (`logic.py`: `create_dataset_from_df`)
   - Called from: `engine.py` L479-483
   ```python
   result = await asyncio.to_thread(
       create_dataset_from_df,
       inference_df,
       warning_flags
   )
   ```

2. **Model Inference** (`InferenceWrapper.run_inference`)
   - Called from: `engine.py` L512-518
   ```python
   full_forecasts_array, logvars = await asyncio.to_thread(
       wrapper.run_inference,
       dataset,
       required_config,
       BATCH_SIZE,
       NUM_SAMPLES
   )
   ```

3. **Plot Calculation** (`logic.py`: `calculate_plot_data`)
   - Called from: `engine.py` L656-661
   ```python
   plot_json_str, plot_data = await asyncio.to_thread(
       calculate_plot_data,
       data,
       index
   )
   ```

4. **Disk I/O** (via `diskcache`)
   - Called from: `cache.py` L135, L155, L166, L193, L229, L244
   ```python
   await asyncio.to_thread(self._disk_get, key, default)
   await asyncio.to_thread(self._disk_set, key, value)
   await asyncio.to_thread(self._disk_pop, key, None)
   await asyncio.to_thread(self._disk_clear)
   ```

---

## 2. Worker Architecture

### 2.1 Background Processor Initialization

**File**: `engine.py` L325-340

```python
async def start(self, num_inference_workers, num_calc_workers):
    for i in range(num_inference_workers):
        task = asyncio.create_task(self._inference_worker_loop(i))  # L334
        self.inference_workers.append(task)
    
    for i in range(num_calc_workers):
        task = asyncio.create_task(self._calc_worker_loop(i))  # L338
        self.calc_workers.append(task)
```

**Key Point**: These are **asyncio Tasks**, NOT threads. They run as coroutines on the main event loop, cooperatively multitasked.

### 2.2 Inference Worker Loop

**File**: `engine.py` L445-601

**Execution Context**: Event loop (async coroutine)

**Flow**:
```
[Event Loop] Inference Worker Loop
    │
    ├─ L453: await self.inference_queue.get()  [Event Loop - async wait]
    │
    ├─ L459: cached_data = await inf_cache.get(handle)  [Event Loop]
    │
    ├─ L479-483: await asyncio.to_thread(create_dataset_from_df, ...)
    │   └─ [SWITCHES TO THREAD POOL] → Heavy dataset creation
    │   └─ [RETURNS TO EVENT LOOP] → Result available
    │
    ├─ L489, L503, L586: TaskRegistry().cancel_all_for_handle(handle)
    │   └─ [Event Loop] → Calls loop.call_soon_threadsafe internally
    │
    ├─ L511-521: async with ModelManager().acquire(...):
    │   ├─ L512-518: await asyncio.to_thread(wrapper.run_inference, ...)
    │   │   └─ [SWITCHES TO THREAD POOL] → Heavy ML inference
    │   │   └─ [RETURNS TO EVENT LOOP] → Forecasts ready
    │   │
    │   └─ L544-569: async with inf_cache.transaction(handle):
    │       └─ [Event Loop - Holds write lock for cache update]
    │
    └─ L589-591: async with self._inference_lock:
        └─ [Event Loop - Updates _pending_inference dict]
```

**Critical Observation**: 
- State mutations (L489, L503, L566, L589-591) happen on **event loop**
- Heavy work (L479-483, L512-518) happens in **thread pool**
- After `await asyncio.to_thread()` completes, execution returns to **event loop**

### 2.3 Calculation Worker Loop

**File**: `engine.py` L626-688

**Execution Context**: Event loop (async coroutine)

**Flow**:
```
[Event Loop] Calc Worker Loop
    │
    ├─ L635: await self.calc_queue.get()  [Event Loop]
    │
    ├─ L640: existing_plot = await plot_cache.get_plot(...)  [Event Loop]
    │   └─ If found: L642: task_registry.notify_success(...)  [Event Loop]
    │
    ├─ L646: data = await inf_cache.get(handle)  [Event Loop]
    │
    ├─ L654: task_registry.notify_error(...) if version mismatch  [Event Loop]
    │
    ├─ L656-661: await asyncio.to_thread(calculate_plot_data, ...)
    │   └─ [SWITCHES TO THREAD POOL] → Heavy plot calculation
    │   └─ [RETURNS TO EVENT LOOP] → JSON string ready
    │
    ├─ L667: await self.increment_calc_runs()  [Event Loop]
    │
    ├─ L670: await plot_cache.update_plot(...)  [Event Loop]
    │
    ├─ L673: task_registry.notify_success(handle, index)  [Event Loop]
    │
    └─ L677: task_registry.notify_error(handle, index, e)  [Event Loop]
```

**Critical Observation**: 
- ALL `task_registry.notify_*` calls happen **after** `await asyncio.to_thread()` returns
- This means they execute on the **event loop thread**, not in thread pool
- The thread-safety in TaskRegistry (loop.call_soon_threadsafe) is defensive/future-proof

---

## 3. Synchronization Mechanisms

### 3.1 TaskRegistry (Cross-Task Notification)

**File**: `state.py` L154-260

**Purpose**: Allow async tasks to wait for completion of background work

**Architecture**:
```
Public Sync API (called from event loop only):
├─ notify_success(handle, index)  [L229-240]
│  └─ Directly pops futures from _registry and sets results
│
├─ notify_error(handle, index, error)  [L242-252]
│  └─ Directly pops futures from _registry and sets exceptions
│
└─ cancel_all_for_handle(handle)  [L254-265]
   └─ Directly pops and cancels all futures for handle

Async API (for waiting):
└─ wait_for_result(handle, index, timeout)  [L276-302]
   ├─ L289: async with self._lock: [Protects _registry mutations]
   └─ L292: await asyncio.wait_for(future, timeout)

Async Maintenance:
└─ reset()  [L216-225]
   └─ async with self._lock: cancels all futures and clears registry
```

**Thread-Safety Pattern**:
```python
# Called from event loop coroutines (NOT from threads):
def notify_success(self, handle: str, index: int) -> None:
    futures = self._registry.pop((handle, index), [])  # Atomic operation
    for f in futures:
        f.set_result(True)  # future.set_result() is thread-safe

# Registration happens under async lock:
async def wait_for_result(self, handle: str, index: int, timeout: float):
    async with self._lock:
        self._registry[key].append(future)  # Protected by lock
    await asyncio.wait_for(future, timeout)
```

**Safety Guarantees**:
- All `notify_*` and `cancel_*` methods are called from event loop coroutines only
- `dict.pop()` is atomic in CPython (GIL protection)
- `future.set_result()` / `future.set_exception()` are thread-safe
- Registration (appending futures) is protected by `asyncio.Lock`
- No thread→event-loop callbacks needed in current architecture

### 3.2 HybridLRUCache (Inference & Plot Caches)

**File**: `cache.py` L31-247

**Lock Type**: `aiorwlock.RWLock` - async reader-writer lock

**Architecture**:
```
Data Stores:
├─ self._hot: LRUCache (in-memory, fast)  [L47]
└─ self._backend: DiskIndex (persistent)  [L45]

Lock: self._lock = AsyncRWLock()  [L48]

Operations:
├─ get(key)  [L122-144]
│  ├─ async with self._lock.reader: check _hot  [L127]
│  ├─ await asyncio.to_thread(self._disk_get, ...)  [L135 - NO LOCK]
│  └─ async with self._lock.writer: promote to hot  [L139]
│
├─ set(key, value)  [L146-155]
│  ├─ async with self._lock.writer: update _hot  [L151]
│  └─ await asyncio.to_thread(self._disk_set, ...)  [L155 - NO LOCK]
│
├─ delete(key)  [L157-171]
│  ├─ async with self._lock.writer: mark TOMBSTONE  [L162]
│  ├─ await asyncio.to_thread(self._disk_pop, ...)  [L166 - NO LOCK]
│  └─ async with self._lock.writer: remove tombstone  [L169]
│
└─ transaction(key)  [L62-107]
   └─ async with self._lock.writer: [HOLD LOCK ENTIRE TIME]
      ├─ Read current value
      ├─ Yield to user code
      └─ Write new value if txn.set() was called
```

**Thread-Safety Pattern**:
- **_hot cache** (in-memory): Protected by AsyncRWLock - only mutated under write lock
- **_backend** (disk): Offloaded to thread pool - no lock during disk I/O (disk index is thread-safe)
- **TOMBSTONE pattern**: Prevents race where concurrent read tries to promote deleted key

**Why disk I/O doesn't hold lock**:
```python
# BAD: Would block all cache ops during slow disk I/O
async with self._lock.writer:
    await asyncio.to_thread(self._disk_set, key, value)  # Lock held during I/O!

# GOOD: Update hot first (fast), then disk without lock
async with self._lock.writer:
    self._hot[key] = value  # Fast update under lock
await asyncio.to_thread(self._disk_set, key, value)  # Slow I/O without lock
```

### 3.3 BackgroundProcessor State

**File**: `engine.py` L304-323

**Shared State**:
```python
self._pending_inference: Dict[str, Tuple[int, int]]  [L317]
self._inference_lock = asyncio.Lock()  [L318]
```

**Protection Pattern**:
```python
# When checking/updating _pending_inference:
async with self._inference_lock:  # L384, L589
    if handle in self._pending_inference:
        # ... check/update ...
    self._pending_inference[handle] = (priority, length)
```

**Context**: All accesses from event loop (inference worker coroutines), so `asyncio.Lock` is appropriate.

---

## 4. Request Flow Example

### HTTP Request → Prediction → Response

**File**: `app.py` L166-245 (`quick_plot` endpoint)

```
[1] HTTP Request arrives
    └─ [Event Loop - FastAPI handler]

[2] L186-187: Check plot cache
    └─ await plot_cache.get_plot(cached_data.version, index)
    └─ [Event Loop] → Async cache read with RWLock

[3] L195-201: If not cached, register future and enqueue work
    ├─ wait_task = asyncio.create_task(wait_for_result(handle, index))
    │  └─ [Event Loop - Task waiting for notification]
    │
    └─ await processor.enqueue_inference(...)
       └─ [Event Loop - Adds to priority queue]

[4] Inference Worker picks up request
    └─ [Event Loop - Inference Worker Loop]
    
    [4a] L479-483: Create dataset
         └─ [SWITCH TO THREAD] → create_dataset_from_df()
         └─ [RETURN TO EVENT LOOP] → dataset ready
    
    [4b] L512-518: Run inference
         └─ [SWITCH TO THREAD] → wrapper.run_inference()
         └─ [RETURN TO EVENT LOOP] → predictions ready
    
    [4c] L544-569: Save to cache
         └─ [Event Loop - Transaction with write lock]
    
    [4d] L573-581: Enqueue calculations
         └─ [Event Loop - Add to calc queue]

[5] Calc Worker picks up calculation
    └─ [Event Loop - Calc Worker Loop]
    
    [5a] L656-661: Calculate plot
         └─ [SWITCH TO THREAD] → calculate_plot_data()
         └─ [RETURN TO EVENT LOOP] → plot JSON ready
    
    [5b] L670: Store plot
         └─ await plot_cache.update_plot(version, index, plot_json)
         └─ [Event Loop - Async cache write with write lock]
    
    [5c] L673: Notify waiting request
         └─ task_registry.notify_success(handle, index)
         └─ [Event Loop] → Uses loop.call_soon_threadsafe (schedules immediately)
         └─ [Event Loop] → _notify_success_internal() executes
         └─ [Event Loop] → future.set_result(True) wakes up wait_task

[6] Original request handler resumes
    └─ [Event Loop - wait_task completes]
    └─ L225-226: Fetch completed plot from cache
    └─ L238: Return JSON response
```

**Key Observations**:
- **NO direct thread→event-loop state mutation** - all mutations happen on event loop
- **Thread work is isolated** - pure computation with inputs/outputs
- **Coordination via await** - thread work returns to event loop before any state change
- **Futures for async wait** - TaskRegistry lets one coroutine wait for another's work

---

## 5. Safety Guarantees

### 5.1 What Makes This Safe?

1. **Single-threaded state mutation**:
   - All dict/cache/registry mutations happen on event loop thread
   - No concurrent writes to shared state

2. **Async locks for event loop concurrency**:
   - `AsyncRWLock` for caches (multiple reader coroutines, exclusive writer)
   - `asyncio.Lock` for _pending_inference dict

3. **Thread isolation**:
   - Thread pool work is pure computation
   - No shared mutable state accessed from threads
   - Results returned to event loop via `await`

4. **Cross-thread signaling**:
   - `loop.call_soon_threadsafe()` for potential thread→event-loop callbacks
   - Currently unused (all calls from event loop) but safe if needed

### 5.2 Common Pitfalls (Avoided)

❌ **BAD**: Direct state mutation from thread
```python
def thread_work():
    cache[key] = value  # RACE CONDITION!
await run_in_executor(None, thread_work)
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
```

❌ **BAD**: Async lock in thread
```python
def thread_work():
    async with cache._lock:  # CAN'T AWAIT IN THREAD!
```

✅ **GOOD**: Separate thread-safe sync calls or return to loop
```python
def thread_work():
    return value
result = await asyncio.to_thread(thread_work)
async with cache._lock:
    cache._hot[key] = result
```

---

## 6. Performance Characteristics

### 6.1 Why This Architecture?

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

### 6.2 Concurrency Model

**Event Loop Concurrency**: Multiple coroutines (I/O-bound)
- ~10-100 concurrent HTTP requests
- ~NUM_COPIES inference worker coroutines
- ~BACKGROUND_WORKERS_COUNT calc worker coroutines
- All cooperatively scheduled (no GIL contention)

**Thread Pool Parallelism**: True parallel execution (CPU-bound)
- Default ThreadPoolExecutor size (typically 5x CPU cores)
- Heavy ML inference runs in parallel across threads
- GIL released during NumPy/Torch operations (native code)

### 6.3 Bottlenecks

1. **Cache write lock**: Serializes cache updates
   - Mitigated by: Short critical sections, disk I/O outside lock

2. **Model availability**: Limited by NUM_COPIES models
   - Mitigated by: Priority queue (model #0 for interactive requests)

3. **GIL for pure Python code**: Limits thread parallelism
   - Mitigated by: Most heavy work in NumPy/Torch (releases GIL)

---

## 7. Testing Considerations

### 7.1 Thread-Safe Components

**TaskRegistry** (`tests/test_task_registry_threadsafe.py`):
- Tests `notify_*` calls from actual threads
- Verifies `loop.call_soon_threadsafe` works correctly
- Uses `registry.reset()` for clean test isolation

### 7.2 Integration Tests

**Process Flow** (`tests/test_integration.py`):
- End-to-end: HTTP → inference → calc → response
- Validates async coordination across workers
- Cache state consistency checks

---

## 8. Future Improvements

### 8.1 Potential Optimizations

1. **Fine-grained cache locking**:
   - Per-key locks instead of global RWLock
   - Would allow concurrent updates to different keys

2. **Lock-free hot cache**:
   - Use `asyncio.Queue` for cache updates
   - Single writer task to serialize updates

3. **Process pool for inference**:
   - Replace threads with processes for true parallelism
   - Avoid GIL entirely (at cost of IPC overhead)

### 8.2 Monitoring Additions

Track contention metrics:
- Time spent waiting for locks
- Queue depths over time
- Thread pool utilization

---

## Summary

**Architecture**: Async event loop + thread pool hybrid

**Execution Contexts**:
- Event loop: All coordination, state, I/O
- Thread pool: CPU-bound ML work only

**Safety Mechanisms**:
- AsyncRWLock for caches (event loop concurrency)
- asyncio.Lock for registries (event loop concurrency)
- loop.call_soon_threadsafe for thread→loop callbacks (defensive)

**Key Insight**: Thread pool work is **isolated computation** - it reads inputs, computes, returns. All **state mutations** happen back on the event loop after `await` completes. This eliminates most threading complexity.

**Thread-Safety**: Achieved by **separation** (threads don't touch shared state) rather than **locks everywhere** (which would be error-prone).

---

## Cancellation & Overload Roadmap

Goal: make request cancellation and overload handling predictable, visible, and bounded without changing the single-loop + threadpool architecture.

### 1) Instrumentation (prerequisite)
✅ **COMPLETED:**
- ✅ Track per-queue depth (inference, calc) and emit in `/health` endpoint
- ✅ Track queue capacities (`MAX_INFERENCE_QUEUE_SIZE`, `MAX_CALC_QUEUE_SIZE`)
- ✅ Calculate load status: `loaded` (<50%), `overloaded` (50-75%), `full` (>75%)
- ✅ Expose `load_status`, queue sizes, and capacities in `/health` response

**TODO:**
- Track cancellation events (initiated, skipped, succeeded) and dropped responses (broken pipe) by endpoint.
- Add histogram for time spent waiting in queues vs. executing.
- Track worker busy counts and enqueue latency.

### 2) Request lifecycle hooks
- In handlers, create a disconnect future (`request.is_disconnected`) and race it with the work future using `asyncio.wait(return_when=FIRST_COMPLETED)`.
- On disconnect: cancel the waiting task, then invoke a unified cancellation hook (see step 3) to attempt to drop queued work.
- If sending the response raises a broken pipe, log and stop writing; do not treat as server error.

### 3) Unified cancellation hook (per-request, not global)
- Use a per-request id/sequence (monotonic per handle/index) and store it with the queue entry and TaskRegistry future.
- Add a function (e.g., `cancel_request(handle, index, req_id)`) that:
  - Cancels only the matching waiter (req_id) rather than all duplicates.
  - Attempts to remove the matching queued job; if removal is not possible, mark the job as stale so the worker discards it when it sees a newer seq.
  - Marks a cancellation outcome metric.
- Last-write-wins: when a newer seq arrives for the same handle/index, older ones become stale but the newest must not be cancelled by earlier disconnects.
- Make handlers call this hook on disconnect or timeout; workers should check the seq/token before expensive work (see step 5).

### 4) Overload controls (backpressure)
- Bound in-memory queues. On enqueue rejection, return 429/503 with a clear retry-after hint.
- Add a simple “drain” mode: stop accepting new enqueues when health indicates overload (queue depth or latency threshold).
- Consider per-priority or per-tenant limits if applicable.

### 5) Cooperative cancellation in workers
- Before starting heavy work in worker loops, check whether the job has been cancelled (a per-job cancellation flag/token set by the hook).
- For threadpool work (dataset creation, inference, plot calc), cancellation is cooperative: check the token before dispatch and after return; if cancelled, skip cache writes and notifications.
- Ensure TaskRegistry notifications only fire for non-cancelled work; cancelled jobs should notify waiters with a cancellation error.

### 6) Timeouts
- Per-stage timeouts: enqueue wait timeout, inference timeout, calc timeout. Surface stage and cause in errors.
- Handler-level timeout to avoid hanging responses; on expiry, call the cancellation hook.

### 7) Graceful shutdown
- On shutdown, stop accepting new requests, then wait for queues to drain with a deadline; cancel remaining jobs after the deadline.
- Expose shutdown status in health (e.g., `draining=true`) so load balancers can stop sending traffic.

### 8) Observability and alerts
- Emit structured logs for: enqueue rejected, cancelled, timed out, broken pipe, worker exception.
- Add counters/gauges for cancellations, queue drops, and overload rejections; alert on sustained queue depth or high rejection rate.

### 9) Validation & tests
- Unit: TaskRegistry cancellation path; dequeue/remove semantics; cancellation token checked in worker loops.
- Integration: client disconnect while waiting; queue-full returns 503; timeout triggers cancellation and no cache write; shutdown drains then cancels.

---

## Implementation Status Summary

### ✅ Completed Steps

**Step 1: Instrumentation (Partial)**
- ✅ Track per-queue depth (inference, calc) and emit in `/health` endpoint
- ✅ Track queue capacities (`MAX_INFERENCE_QUEUE_SIZE=32`, `MAX_CALC_QUEUE_SIZE=256`)
- ✅ Calculate load status: `loaded` (<50%), `overloaded` (50-75%), `full` (>75%)
- ✅ Expose `load_status`, queue sizes, and capacities in `/health` response
- ⏳ TODO: Track cancellation events, enqueue latency, worker busy counts, queue wait histograms

**Step 4: Overload Controls**
- ✅ Bounded in-memory queues with configurable capacities
- ✅ On queue full (>75% utilization), return 503 Service Unavailable with `Retry-After: 30` header
- ✅ Processing endpoints (`/process_unified`, `/draw_a_plot`, `/quick_plot`) check for overload
- ✅ Health and cache management endpoints exempt from overload rejection
- ✅ Load status calculation based on max utilization across both queues
- ⏳ TODO: Drain mode, per-priority/per-tenant limits

### ⏳ Remaining Steps
- Step 2: Request lifecycle hooks (disconnect detection)
- Step 3: Unified cancellation hook (per-request cancellation)
- Step 5: Cooperative cancellation in workers
- Step 6: Per-stage timeouts
- Step 7: Graceful shutdown with drain mode
- Step 8: Enhanced observability (metrics, alerts)
- Step 9: Comprehensive test coverage
