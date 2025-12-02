# Future Optimizations & Architectural Improvements

## 1. Cache Structure Optimization (High Priority)
Current implementation uses a `dict` for `FORECAST_CACHE` and `RESULT_CACHE`, which incurs high memory overhead and Python object creation costs.

**Proposal:**
- Transition to a **Polars DataFrame** based cache.
- Store forecasts and plot data directly as additional columns in a "Result DataFrame" associated with the dataset handle.
- This would eliminate the need for a separate `RESULT_CACHE` dict.
- **Consumer Logic Change**: Update `calculate_plot_data` and `render_plot` to read directly from this Polars structure (e.g., selecting a row/slice) rather than dict lookups.
- **Benefits**: 
    - Significant memory reduction (contiguous memory vs dict overhead).
    - Faster access using optimized C/Rust paths in Polars/Numpy.
    - Simplified state management (one source of truth).

## 2. Cache Persistence & Subset Matching
Currently, cache is in-memory and strict on handle matching (full dataset hash).

**Persistence:**
Serialize the Polars cache structure to disk (Parquet/Arrow) to survive restarts.

**Subset Matching Strategy:**
- When a new request arrives, before declaring a "Cache Miss":
    - Check if the new dataset's time range and content is a **subset** of an existing cached dataset.
    - Since inputs are hashed and during processing only older data may be discarded, subsets will match
    - Use timestamp alignment to find the corresponding indices in the cached superset.
- **Implementation**:
    - Store `start_time`, and `content_signature` in cache metadata.
    - **Index Realignment**: Align indices such that the **last** data point is always index **0**. All previous points have negative indices (e.g., -1, -2, ...).
        - This makes the index relative to the "current moment" (end of data).
        - A subset (shorter history) would simply map to a different range of negative indices in the superset.
- **Colision**: Cache for a hash is empty. A long inference (full dataset) runs in background. A short inference request comes (quickdraw) and 


## 3. Batched Calculation Workers
Current `CalcWorker` processes one index at a time, acquiring `RESULT_CACHE_LOCK` for every single write. This can lead to high lock contention when processing thousands of points.

**Proposal:**
- Modify `CalcWorker` to process **batches** of indices (e.g., 16 or 32 at a time).
- Accumulate results in a local buffer.
- Acquire `RESULT_CACHE_LOCK` once per batch to write all results.
- **Benefit**: Drastically reduces lock contention and context switching overhead.

## 4. Inference Output Optimization
- `run_inference_full` currently converts the efficient numpy array output `(N, 12, 10)` into a dictionary `{idx: array}`. This is O(N) in Python and wasteful.
- **Refactor**: Keep the forecast as a raw numpy array or tensor in the cache. Access indices directly from this array during calculation.
