# Load Test Refactoring

## Overview

The `test_integration_load.py` file has been refactored from a single monolithic test into multiple parametrized tests for better progress tracking and debugging.

## Key Changes

### Before
- One massive `run_load_test()` function with 6 sequential phases
- 20+ minutes with no progress indicators
- Difficult to debug which phase or file caused issues

### After
- **6 separate parametrized test phases** that show progress dots
- Each phase can be run independently
- Clear tracking of which file/iteration fails

## Test Configuration

Located at the top of `test_integration_load.py`:

```python
HAMMERING_ITERATIONS = 10  # Number of hammering cycles
PARALLEL_REQUESTS_PER_CYCLE = 100  # Total requests per cycle
TRACKED_REQUESTS_PER_CYCLE = 20  # Requests we wait for
FIRE_AND_FORGET_REQUESTS_PER_CYCLE = 80  # Fire-and-forget requests
```

## Test Phases

### Phase 1: Process CSV Files
**Test:** `test_process_single_file`
- **Parametrized by:** Each CSV file + `force_calculate` (forced/cached)
- **What it does:** Tests `/process_unified` endpoint with each input file
- **Progress tracking:** One dot per file (shows exactly which file is being processed)
- **Stores:** Handles in `pytest.handle_storage` for later phases

### Phase 2: Sanity Check
**Test:** `test_sanity_check_first_plot`
- **Parametrized by:** `force_calculate` (forced/cached)
- **What it does:** Waits for first plot to ensure inference system is warmed up
- **Stores:** Success flag in `pytest.sanity_check_passed`

### Phase 3: Hammering Iterations
**Test:** `test_hammering_iteration`
- **Parametrized by:** `force_calculate` × 10 iterations
- **What it does:** 
  - Fires 100 parallel requests per iteration
  - **Tracks 20 requests** (waits for completion)
  - **Fire-and-forget 80 requests** (don't wait)
- **Progress tracking:** One dot per iteration (20 dots instead of 20-minute silence)
- **Key feature:** Test advances when 20 tracked requests complete, not when all 100 finish

### Phase 4: Health Endpoint Hammering
**Test:** `test_health_endpoint_hammering`
- **Parametrized by:** `force_calculate` (forced/cached)
- **What it does:** Hammers `/health` with 200 concurrent requests
- **Purpose:** Ensure health endpoint stays responsive under load

### Phase 5: Random Plot Sampling
**Test:** `test_random_plot_sampling`
- **Parametrized by:** `force_calculate` (forced/cached)
- **What it does:** Samples random plots and saves to disk
- **Output:** HTML and SVG files in `test_outputs/`

### Phase 6: Mixed Index Hammering
**Test:** `test_mixed_index_hammering`
- **Parametrized by:** `force_calculate` (forced/cached)
- **What it does:** Tests with valid (-20 to 0) and invalid (-200 to -100) indices
- **Purpose:** Validate error handling under load

### Phase 7: ULTRAKILL - Chaotic Mixed Endpoint Hammering
**Test:** `test_ultrakill_mixed_endpoints`
- **Parametrized by:** `force_calculate` × 10 iterations
- **What it does:** 
  - Randomly hammers **ALL** endpoints in the same iteration
  - **Endpoint probabilities (relative weights):**
    - Health: 5x (most common - ~62%)
    - Draw plot (valid): 1x (~12%)
    - Draw plot (invalid): 1x (~12%)
    - Process forced: 0.2x (5x less - ~2.5%)
  - **Wait strategy:**
    - 5% chance: Wait for response (tracked)
    - 95% chance: Fire-and-forget (don't wait)
- **Purpose:** Maximum chaos stress test - all endpoints hammered simultaneously
- **Key feature:** Creates realistic production load with mixed traffic patterns

## Running the Tests

### Run all load tests
```bash
uv run pytest tests/test_integration_load.py -v
```

### Run specific phase
```bash
# Phase 1: Processing files
uv run pytest tests/test_integration_load.py::test_process_single_file -v

# Phase 3: Hammering iterations
uv run pytest tests/test_integration_load.py::test_hammering_iteration -v

# Phase 4: Health checks
uv run pytest tests/test_integration_load.py::test_health_endpoint_hammering -v

# Phase 7: ULTRAKILL chaos test
uv run pytest tests/test_integration_load.py::test_ultrakill_mixed_endpoints -v
```

### Run with specific parameters
```bash
# Only forced calculation tests
uv run pytest tests/test_integration_load.py -v -k "forced"

# Only cached tests
uv run pytest tests/test_integration_load.py -v -k "cached"

# Specific iteration
uv run pytest tests/test_integration_load.py -v -k "iter5"

# Only ULTRAKILL tests
uv run pytest tests/test_integration_load.py -v -k "ultrakill"
```

### Parallel execution (if using pytest-xdist)
```bash
# Run file processing in parallel
uv run pytest tests/test_integration_load.py::test_process_single_file -n auto
```

## Benefits

1. **Progress Visibility:** See dots/progress for each test iteration
2. **Faster Debugging:** Pinpoint exactly which file or iteration fails
3. **Selective Testing:** Run specific phases or parameters
4. **Better Logs:** Each parametrized test has clear context in logs
5. **Fire-and-Forget Pattern:** 100 requests per cycle, but only wait for 20
6. **Realistic Load:** ULTRAKILL phase simulates production traffic patterns with mixed endpoints

## Expected Output

```
test_integration_load.py::test_process_single_file[forced-file1.csv] PASSED [1%]
test_integration_load.py::test_process_single_file[forced-file2.csv] PASSED [2%]
...
test_integration_load.py::test_sanity_check_first_plot[forced] PASSED [15%]
test_integration_load.py::test_hammering_iteration[forced-iter0] PASSED [16%]
test_integration_load.py::test_hammering_iteration[forced-iter1] PASSED [17%]
test_integration_load.py::test_hammering_iteration[forced-iter2] PASSED [18%]
...
test_integration_load.py::test_health_endpoint_hammering[forced] PASSED [30%]
test_integration_load.py::test_random_plot_sampling[forced] PASSED [31%]
test_integration_load.py::test_mixed_index_hammering[forced] PASSED [32%]
test_integration_load.py::test_ultrakill_mixed_endpoints[forced-ultrakill0] PASSED [33%]
test_integration_load.py::test_ultrakill_mixed_endpoints[forced-ultrakill1] PASSED [34%]
...
```

## Notes

- **State Sharing:** Tests share data via `pytest` module attributes (e.g., `pytest.handle_storage`)
- **Test Ordering:** Tests run in file order (phases 1-7)
- **Skipping:** Later phases skip if prerequisites fail (e.g., no valid handles)
- **Cleanup:** Each test manages its own async context (no shared state leakage)

## ULTRAKILL Phase Details

The ULTRAKILL phase (Phase 7) is designed to create maximum chaos and stress test the entire system under realistic production conditions:

### Endpoint Selection Strategy
Endpoints are randomly selected with weighted probabilities to simulate real-world usage:
- **Health checks (62%):** Most frequent, simulating UI polling
- **Valid plot requests (12%):** Normal user requests
- **Invalid plot requests (12%):** Error cases and edge cases
- **Forced processing (2.5%):** Expensive operations (5x less likely)

### Fire-and-Forget Strategy
- **95% of requests:** Fire-and-forget (don't wait for completion)
- **5% of requests:** Tracked (wait for completion to verify success)

This creates a realistic scenario where:
- The system is under constant load (100 requests/cycle)
- Only a small subset is monitored for success
- Most requests complete in the background
- Simulates real production traffic patterns

### Why ULTRAKILL?
1. **Realistic Load:** Real applications hit multiple endpoints simultaneously
2. **Race Condition Detection:** Concurrent access to different resources
3. **Resource Contention:** Tests how system handles mixed workloads
4. **Worst-Case Scenario:** All endpoints stressed at once
5. **Production Readiness:** If it survives ULTRAKILL, it's production-ready
