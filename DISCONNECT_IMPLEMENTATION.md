# Request Lifecycle Hooks with Disconnect Detection - Implementation Summary

## Overview
This implementation adds request lifecycle management with disconnect detection and duplication-aware handling to the gluRPC system, as specified in THREADING_ARCHITECTURE.md sections 565-577.

## Key Features Implemented

### 1. Request ID Assignment and Tracking
- **DisconnectTracker** singleton class manages per-request tracking
- Each request receives a unique monotonically increasing `request_id` 
- Request IDs enable "last-write-wins" semantics for duplicate requests
- Tracks active request counters per (handle, index) pair

### 2. Disconnect Future with Request Counter
- Disconnect future only resolves when ALL requests for a (handle, index) have disconnected
- Counter-based approach allows multiple concurrent requests for same data
- Individual requests await either: (disconnect_future, work_completion)
- When a request disconnects/completes, counter is decremented

### 3. Unified Cancellation Hook
- `TaskRegistry.cancel_request(handle, index, request_id)` - cancels specific request
- `BackgroundProcessor.update_latest_request_id()` - marks latest request for stale detection
- `BackgroundProcessor.is_request_stale()` - checks if job should be discarded
- Workers check staleness before expensive operations

### 4. Stale Job Detection
- Latest request_id tracked per (handle, index)
- Jobs with older request_id are marked stale and discarded
- Prevents wasted computation on superseded requests
- Background jobs (request_id=None) are never considered stale

### 5. Disconnect Detection in Handlers
- `/draw_a_plot` endpoint: Registers request, monitors disconnect, unregisters on completion
- `/quick_plot` endpoint: Similar disconnect monitoring with temporary request_id
- Uses `request.is_disconnected()` to poll for client disconnection
- Disconnect task runs concurrently with work

### 6. Worker Stale Check Integration
- **Inference Worker**: Checks staleness before dataset creation and inference
- **Calculation Worker**: Checks staleness before plot calculation
- Stale jobs notify their requestor with cancellation error
- Prevents duplicate work when newer requests arrive

## Data Structure Changes

### state.py
- Added `DisconnectTracker` class with:
  - `register_request()` - assigns request_id, increments counter
  - `unregister_request()` - decrements counter, resolves disconnect future at 0
  - `get_disconnect_future()` - returns future that resolves when all requests disconnect
  - `get_current_seq()` - returns current sequence number

- Modified `TaskRegistry`:
  - Changed key from `(handle, index)` to `(handle, index, request_id)`
  - Updated `notify_success()`, `notify_error()` to handle optional request_id
  - Added `cancel_request()` method for specific request cancellation
  - Modified `wait_for_result()` to race disconnect_future with work completion

### engine.py
- Modified `BackgroundProcessor`:
  - Added `_latest_request_id` tracking dict
  - Updated queue items to include `request_id` field
  - Added `update_latest_request_id()` method
  - Added `is_request_stale()` method
  - Updated `enqueue_inference()` to accept request_id
  - Updated `enqueue_calc()` to accept request_id

- Modified `_inference_worker_loop()`:
  - Checks staleness before dataset creation
  - Passes request_id through to calculation enqueuing

- Modified `_calc_worker_loop()`:
  - Checks staleness before plot calculation
  - Notifies specific request_id on success/error

### core.py
- Updated `parse_and_schedule()` to accept `request_id` parameter
- Updated `generate_plot_from_handle()` to accept `request_id` and `disconnect_future`
- Updated `quick_plot_action()` to accept `request_id` and `disconnect_future`
- All functions pass disconnect_future to TaskRegistry.wait_for_result()

### app.py
- Added `asyncio` import
- Imported `DisconnectTracker`
- Modified `/draw_a_plot` endpoint:
  - Registers request with DisconnectTracker
  - Updates latest request_id in BackgroundProcessor
  - Creates disconnect detection task
  - Passes request_id and disconnect_future to core functions
  - Unregisters request in finally block

- Modified `/quick_plot` endpoint:
  - Generates temporary request_id
  - Creates disconnect future and detection task
  - Passes to quick_plot_action
  - Handles CancelledError gracefully

## Request Flow Example

### Scenario: User requests plot, then requests again while first is pending

1. **Request 1 arrives** (t=0)
   - DisconnectTracker assigns request_id=1
   - Counter for (handle, index) = 1
   - BackgroundProcessor.update_latest_request_id(handle, index, 1)
   - Work queued with request_id=1

2. **Request 2 arrives** (t=1) 
   - DisconnectTracker assigns request_id=2
   - Counter for (handle, index) = 2
   - BackgroundProcessor.update_latest_request_id(handle, index, 2)
   - Work queued with request_id=2

3. **Worker picks up request 1** (t=2)
   - Checks is_request_stale(handle, index, 1)
   - Returns True (latest is 2)
   - Worker discards job, notifies request_id=1 with cancellation error
   - Request 1 client gets "Request superseded" response

4. **Worker picks up request 2** (t=3)
   - Checks is_request_stale(handle, index, 2)
   - Returns False (latest is 2)
   - Proceeds with work
   - On completion, notifies request_id=2
   - Request 2 client gets result

5. **Both requests complete/disconnect**
   - Counters decremented
   - When counter reaches 0, disconnect_future resolves
   - Resources cleaned up

## Testing Recommendations

1. **Single request flow**: Verify request_id assignment and completion
2. **Duplicate requests**: Test that older requests are cancelled appropriately
3. **Disconnect during work**: Test counter decrements and disconnect_future resolution
4. **Multiple concurrent requests**: Verify counter tracking is accurate
5. **Background vs Interactive**: Ensure priority 0 requests don't get marked stale
6. **Staleness edge cases**: Request arrives just as work completes

## Metrics to Monitor

- Request cancellation rate (stale job detection)
- Average request counter per (handle, index)
- Disconnect future resolution time
- Worker stale check hit rate

## Future Improvements

1. Add metrics collection for cancellation/staleness rates
2. Add request_id to structured logging for better traceability
3. Consider TTL for stale request_id entries in BackgroundProcessor
4. Add health check endpoint showing active request counters
