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
- Serialize the Polars cache structure to disk (Parquet) to survive restarts.
- add cache_management endpoint (return cache state, allow granular invalidation, disk save/load and total flush)
- add force_calculate flag to respective requests alowing to not use cached values, in this query.

**Subset Matching Strategy:**
- When a new request arrives, before declaring a "Cache Miss":
    - Check if the new dataset's time range and content is a **subset** of an existing cached dataset.
    - Since inputs are hashed and during processing only older data may be discarded, subsets with same hash should normally align by end.
    - Use timestamp alignment to find the corresponding indices in the cached superset.
    - **Index Realignment**: Align indices such that the **last** data point is always index **0**. All previous points have negative indices (e.g., -1, -2, ...).
        - This makes the index relative to the "current moment" (end of data).
        - A subset (shorter history) would simply map to a different range of negative indices in the superset.
        - Indices of smaller and larger results are auto-aligned
    - cache load: for each entry check which calcs are done, rerun the missing tasks as idle (store nan for missing values to distinguish)

- **Implementation**:
    - Store `start_time`, and `content_signature` in cache metadata. end time should match by design, check during replacement but don't store.
    - data processing def should emit max_index (or rather min_index, negative, or index_len positive) for a given input
    - quickdraw request should accept indices other than 0, set maximum_wanted_duration accordingly to cover this specific index or return an error before attempt if the index is out ou bounds for this dataset  

- **Colision**: Cache for a hash is empty. A long inference tsk A (full dataset) runs in background. A short inference request B comes AFTER (quickdraw) and: 
    - executes BEFORE, filling the cache. When A finishes it finds the cache entry, validates tat its result is BIGGER and matches by end and replaces it, force flag = same behavior
    - executes AFTER, comming to a cache filled by A already.  It validates that its result is smaller than the one existing. Force flag: replaces with the smaller cache, no force = only return result, don't cache. 

    - calc tasks should distingush which result they were started for (some uniq id value of hash entry) and account for races:
        - task finished, but cache entry alredy changed, drop result.
        - worker gets to task but caheid and task's cacheid mismatches, 
            - idle = just drop the task 
            - highprio = restart the task if the index is within range for new cache.
            - request waits for some calc task for index calc (highprio) but it index no loger available for the cache entry (cache flushed for example). Schedule a new inference task, sleep until index is valid, reschedule, restart the task 


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
