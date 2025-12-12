# REFERENCE_LOGIC.md

## Summary

This document provides comprehensive reference documentation for `logic.py`, the core business logic module of gluRPC. It covers:

- **13 functions** for data processing, inference, and visualization
- **Type-safe data structures** (Pydantic models, numpy arrays)
- **Multi-stage data pipeline** with independent caching points
- **Monte Carlo Dropout inference** for uncertainty quantification
- **Configuration parameters** and environment variables
- **Common pitfalls** and troubleshooting guidance

**Quick Links**:
- [Data Flow Architecture](#data-flow-architecture)
- [Function Reference](#function-reference)
- [Important Implementation Notes](#important-implementation-notes)
- [Configuration Values](#appendix-configuration-values)

**Function Quick Reference**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `get_time_range()` | Extract time bounds | DataFrame | (start, end) timestamps |
| `calculate_expected_dataset_length()` | Predict dataset size | Duration, config | Expected length |
| `parse_csv_content()` | Decode and parse CSV | Base64 string | Unified DataFrame |
| `compute_handle()` | Generate content hash | DataFrame | SHA256 hash |
| `get_handle_and_df()` | Parse + hash | Base64 string | (handle, DataFrame) |
| `analyse_and_prepare_df()` | Preprocess data | Unified DataFrame | (Inference DF, warnings) |
| `create_dataset_from_df()` | Create inference dataset | Inference DataFrame | Dataset + scalers + config |
| `create_inference_dataset_fast_local()` | Build Darts dataset | DataFrame + config | Dataset + model config + scalers |
| `load_model()` | Load model weights | Config + path | ModelState |
| `run_inference_full()` | Execute inference | Dataset + model | (Predictions, logvars) |
| `calculate_plot_data()` | Generate visualization | Predictions + index | (Plot JSON, PlotData) |
| `convert_logic()` | Convert CSV format | Base64 string | Unified CSV string |
| `reconstruct_dataset()` | Deserialize dataset | DartsDataset DTO | SamplingDatasetInferenceDual |

---

## Overview

The `logic.py` module contains the core business logic for the gluRPC service. It provides functions for:
- Data parsing and validation (CSV → unified format)
- Dataset preparation for inference (unified format → Darts dataset)
- Model loading and configuration
- Inference execution (Monte Carlo Dropout)
- Visualization and plotting (forecasts → interactive charts)
- Data format conversion (utility operations)

This module bridges the gap between the API layer (`app.py`), state management (`state.py`), and the execution engine (`engine.py`).

**Key Design Principles**:
1. **Functional Design**: Most functions are pure/stateless (easier to test/cache)
2. **Structured Logging**: Four specialized loggers for different operation types
3. **Type Safety**: Full type hints using Pydantic models and numpy typing
4. **Error Handling**: Graceful error returns (dicts) or explicit exceptions
5. **Performance**: Direct array operations, optional scaler reuse, batched inference

## Dependencies

### External Libraries
- **pandas/polars**: Data manipulation (polars preferred per project rules)
- **numpy**: Numerical operations
- **torch**: Deep learning framework for model inference
- **plotly**: Visualization
- **scipy.stats**: Statistical functions (KDE for uncertainty visualization)
- **darts**: Time series library (TimeSeries, ScalerCustom)

### Internal Dependencies
- **glucobench**: Model architecture, data formatting, preprocessing utilities
- **cgm_format**: CGM data parsing and processing (FormatParser, FormatProcessor)
- **glurpc**: Internal data classes and schemas

## Data Flow Architecture

The data flows through multiple transformation stages:

```
CSV (base64) 
  → parse_csv_content() 
  → unified_df (polars)
  
unified_df
  → compute_handle() → SHA256 hash (for caching)
  
unified_df
  → analyse_and_prepare_df()
  → FormatProcessor pipeline (interpolation, sync, validation)
  → inference_df + warning_flags
  
inference_df + warning_flags
  → create_dataset_from_df()
  → FormatProcessor.to_data_only_df() (glucose extraction)
  → create_inference_dataset_fast_local()
  → SamplingDatasetInferenceDual + scalers + config
  
dataset + model_config + model_state
  → run_inference_full()
  → forecasts (numpy array)
  
forecasts + dataset + scalers
  → calculate_plot_data()
  → PlotData + plot_json (Plotly figure)
```

**Key Separation Points**:
1. **Parsing Stage**: CSV → unified DataFrame (format-agnostic)
2. **Preprocessing Stage**: unified → inference-ready (quality checks, interpolation)
3. **Dataset Creation**: inference-ready → Darts dataset (feature engineering)
4. **Inference Stage**: dataset → predictions (model execution)
5. **Visualization Stage**: predictions → plot (post-processing, KDE)

This separation enables:
- Independent caching at each stage
- Parallel processing (preprocessing, inference, plotting)
- Error isolation and recovery
- Flexible pipeline composition

## Dataset & Inference Data Structures

### 1. `DartsDataset` DTO (Pydantic)

To allow efficient caching and serialization without pickling entire Python objects, we use `DartsDataset` (defined in `data_classes.py`) as a Data Transfer Object.

**Fields:**
- `target_series`: List of numpy arrays (float32/64)
- `covariates`: List of numpy arrays (optional)
- `static_covariates`: List of numpy arrays (optional)
- `input_chunk_length`, `output_chunk_length`: Model dimensions

**Key Capability: Logic Encapsulation**
The DTO encapsulates critical logic previously scattered in `logic.py`:
- `total_samples`: Computes total valid inference samples across all series segments.
- `get_sample_location(index)`: Maps a flat dataset index (0..N) to a specific `(series_index, offset)` tuple. This allows O(1) access to the correct data segment without reconstructing Darts objects.

### 2. `PredictionsData` Structure

The `PredictionsData` class acts as the primary container for inference results and slicing logic. It merges previously separate concepts (`NegIndexSlice`) into a single source of truth.

**Indexing Philosophy:**
- **User Index (Negative)**: Users request predictions relative to the "end of data" (e.g., `0` = end, `-1` = one step back).
- **Dataset Index (Positive)**: The internal 0-based index into the dataset arrays.
- **Array Index**: The index into the local `predictions` array.

**Index Conversion Methods:**
- `get_dataset_sample_index(user_index)`: Converts `-1` → `total_samples - 2`.
- `get_dataset_index(user_index)`: Same as above, used for array access.

### 3. Model Output (Forecasts)

The `run_inference_full` function returns a tuple of `(predictions, logvars)`.

**Predictions Shape**: `(N, output_chunk_length, num_samples)`
- **N**: Number of samples in the dataset (`dataset.total_samples`)
- **output_chunk_length**: Forecast horizon (default: 12 steps = 1 hour)
- **num_samples**: Number of Monte Carlo Dropout samples (default: 10)

**Logvars Shape**: `(N, 1, num_samples)`
- Log-variance estimates for uncertainty quantification
- Same N as predictions
- Shape includes extra dimension for compatibility

**Content**:
- Contains **only** the predicted future values
- Does **not** include the input history
- Values are **scaled** (must be inverse-transformed for plotting)

**Type Annotations**:
```python
PredictionsArray = NDArray[Shape["* s, * x, * y"], Union[float64, float32]]
LogVarsArray = NDArray[Shape["* s, * v, * y"], Union[float64, float32]]
```

### 4. Plot Data Calculation (`calculate_plot_data`)

This function generates the visualization data. **Key architectural decision**: The rendering logic is embedded within this function rather than separated.

**Rationale for Embedded Rendering**:
- Enables caching of both plot data AND rendered visualization together
- Reduces function call overhead
- Simplifies cache management (single cache entry per plot)
- Previous separate `render_plot()` function is now integrated (see line 636 comment in code)

**Refactoring Benefits**:
1. **Direct DTO Access**: Uses `dataset_dto` directly instead of reconstructing `SamplingDatasetInferenceDual`
2. **Encapsulated Indexing**: Calls `predictions.get_dataset_sample_index()` to find the correct row
3. **Efficient Slicing**: Uses `dataset_dto.get_sample_location()` to find the exact source array and slice `past_target` and `true_future` directly from numpy buffers

*Post-processing Note*: We use the same inverse transform logic `(x - min) / scale` for both forecasts and historical values to ensure alignment. This was extensively validated (see lines 535-552 in code).

---

## Function Reference

### 1. `get_time_range(unified_df: pl.DataFrame) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]`

**Purpose**: Extract start and end timestamps from a unified DataFrame.

**Input**: 
- `unified_df`: Polars DataFrame with 'datetime' column

**Output**:
- `(start_time, end_time)`: Tuple of datetime objects, or `(None, None)` on error

**Process**:
1. Checks if 'datetime' column exists
2. Extracts min and max timestamps
3. Returns as Python datetime objects

**Usage**: Used to determine temporal range of input data for validation and logging.

---

### 2. `calculate_expected_dataset_length(maximum_wanted_duration_minutes: int, time_step: int, input_chunk_length: int, output_chunk_length: int) -> int`

**Purpose**: Calculate expected dataset length based on duration and model parameters.

**Formula**:
```python
expected_input_samples = (maximum_wanted_duration_minutes // time_step) + 1
expected_dataset_len = expected_input_samples - (input_chunk_length + output_chunk_length - 1)
```

**Returns**: Expected number of samples in the dataset.

---

### 3. `create_inference_dataset_fast_local()`

**Signature**:
```python
def create_inference_dataset_fast_local(
    data: pl.DataFrame,
    config: GluformerInferenceConfig,
    scaler_target: Optional[ScalerCustom] = None,
    scaler_covs: Optional[ScalerCustom] = None
) -> Tuple[SamplingDatasetInferenceDual, GluformerModelConfig, ScalerCustom, ScalerCustom]
```

**Purpose**: Transforms a polars DataFrame into a Darts-compatible inference dataset with proper scaling and feature engineering.

**Process**:
1. **Column Mapping**: Maps standard column names to internal format
   - `sequence_id` → `id`
   - `datetime` → `time`
   - `glucose` → `gl`

2. **Type Conversion**: Converts to pandas with proper dtypes
   - `time`: datetime64
   - `gl`: float32

3. **Data Formatting**: Uses glucobench formatters
   - Interpolates gaps with configurable thresholds
   - Encodes temporal features (day, month, year, hour, minute, second)

4. **TimeSeries Creation**: Creates Darts TimeSeries objects
   - Target series: glucose values
   - Future covariates: temporal features
   - Static covariates: sequence ID

5. **Scaling**: Applies or fits scalers
   - Separate scalers for targets and covariates
   - Uses `ScalerCustom` from glucobench

6. **Dataset Creation**: Builds `SamplingDatasetInferenceDual`
   - Input chunk length: historical window
   - Output chunk length: prediction horizon
   - Supports static covariates

7. **Feature Dimension Inference**: Automatically detects feature dimensions from first sample
   - `num_dynamic_features`: From future covariates shape
   - `num_static_features`: From static covariates shape

**Output**:
- `dataset`: SamplingDatasetInferenceDual ready for model.predict()
- `model_config`: GluformerModelConfig with inferred dimensions
- `scaler_target`: Fitted target scaler (can be reused)
- `scaler_covs`: Fitted covariates scaler (can be reused)

**Key Configuration Parameters** (from `GluformerInferenceConfig`):
- `input_chunk_length`: 96 (8 hours at 5-min intervals)
- `output_chunk_length`: 12 (1 hour at 5-min intervals)
- `gap_threshold`: 45 minutes
- `min_drop_length`: 12 samples
- `interval_length`: 5 minutes (derived from time_step)

---

### 4. `parse_csv_content(content_base64: str) -> pl.DataFrame`

**Purpose**: Decodes base64-encoded CSV content and parses it using cgm_format's FormatParser.

**Process**:
1. Base64 decode
2. Write to temporary file
3. Parse with `FormatParser.parse_file()`
4. Clean up temporary file
5. Return unified polars DataFrame

**Error Handling**:
- `UnknownFormatError`: Unsupported file format
- `MalformedDataError`, `ColumnOrderError`, `ColumnTypeError`: Data validation errors
- `MissingColumnError`, `ExtraColumnError`: Column structure errors
- `ZeroValidInputError`: No valid data after processing
- All exceptions converted to `ValueError` with descriptive messages
- Always cleans up temporary files (even on error)

**Output Format**: Unified DataFrame with standardized columns:
- `sequence_id`: Identifier for data segments
- `datetime`: Timestamp
- `glucose`: Blood glucose value in mg/dL
- Service columns (e.g., flags, quality indicators)

---

### 5. `compute_handle(unified_df: pl.DataFrame) -> str`

**Purpose**: Generates a content-addressable SHA256 hash for caching/deduplication.

**Process**:
1. Serialize DataFrame to CSV (in-memory)
2. Compute SHA256 hash of CSV bytes
3. Return hex digest

**Properties**:
- Deterministic: Same data → same hash
- Content-based: Different data → different hash
- Used for cache keys in state management

---

### 6. `get_handle_and_df(content_base64: str) -> Tuple[str, pl.DataFrame]`

**Purpose**: Convenience function combining parsing and hashing.

**Output**: `(handle, unified_df)`

**Usage**: Primary entry point for new data ingestion in the API.

---

### 7. `analyse_and_prepare_df()`

**Signature**:
```python
def analyse_and_prepare_df(
    unified_df: pl.DataFrame,
    minimum_duration_minutes: int = MINIMUM_DURATION_MINUTES_MODEL,
    maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION_DEFAULT
) -> Tuple[pl.DataFrame, ProcessingWarning]
```

**Purpose**: Processes unified DataFrame through the FormatProcessor pipeline to prepare for inference.

**Process**:
1. **Initialize FormatProcessor**:
   - `expected_interval_minutes=5`
   - `small_gap_max_minutes=15`

2. **Gap Interpolation**: Fills small gaps in data

3. **Timestamp Synchronization**: Aligns to 5-minute intervals

4. **Inference Preparation**:
   - `minimum_duration_minutes`: Minimum usable segment (default: 540 min = 9 hours)
   - `maximum_wanted_duration`: Maximum duration (default: 1080 min = 18 hours)
   - Returns processed data + warning flags

**Output**:
- `inference_df`: Processed DataFrame ready for dataset creation
- `warning_flags`: ProcessingWarning bitwise flags

**Error Handling**:
- `MalformedDataError`, `ZeroValidInputError`: Raised as `ValueError`
- All errors include context about the processing stage

---

### 8. `create_dataset_from_df()`

**Signature**:
```python
def create_dataset_from_df(
    inference_df: pl.DataFrame,
    warning_flags: ProcessingWarning,
) -> DatasetCreationResult
```

**Purpose**: Complete dataset creation pipeline from inference-ready DataFrame to Darts dataset.

**Process**:
1. **Data Quality Check**: Validates sufficient duration using warning flags

2. **Glucose Extraction**: Converts to glucose-only DataFrame
   - Calls `FormatProcessor.to_data_only_df()`
   - `drop_service_columns=False`
   - `drop_duplicates=True`
   - `glucose_only=True`

3. **Dataset Creation**: Calls `create_inference_dataset_fast_local()`
   - Uses default `GluformerInferenceConfig()`

**Output** (Success):
```python
{
    'success': True,
    'dataset': SamplingDatasetInferenceDual,
    'scaler_target': ScalerCustom,
    'scaler_covs': ScalerCustom,
    'model_config': GluformerModelConfig,
    'warning_flags': ProcessingWarning
}
```

**Output** (Failure):
```python
{
    'success': False,
    'error': str  # Error message
}
```

**Error Handling**:
- Returns error dict (not exceptions) for graceful handling
- Catches `MalformedDataError`, `ZeroValidInputError`, `ValueError`
- Unexpected errors also caught and returned as processing error

---

## Common Pitfalls and Troubleshooting

### 1. Model Not in Train Mode
**Symptom**: All predictions identical, no uncertainty
**Cause**: Model loaded with `.eval()` or not explicitly set to `.train()`
**Solution**: Ensure `model.train()` called after loading (line 438)

### 2. Incorrect Scaling Post-Processing
**Symptom**: Impossible glucose values (negative, > 500 mg/dL)
**Cause**: Using wrong inverse transform formula
**Solution**: Use `(x - min) / scale` for both forecasts and historical data

### 3. Index Out of Bounds
**Symptom**: `ValueError` when accessing predictions with negative index
**Cause**: Index outside available prediction range
**Solution**: Check `first_index` and `last_index` in PredictionsData

### 4. KDE Failures
**Symptom**: Missing fan charts in visualization
**Cause**: Low variance (`std < 1e-6`) or degenerate distributions
**Solution**: This is expected behavior, logged as debug message

### 5. Dataset Length Mismatch
**Symptom**: `ValueError` during PredictionsData validation
**Cause**: Predictions array size doesn't match dataset
**Solution**: Ensure full inference completed before creating PredictionsData

### 6. Memory Issues with Large Datasets
**Symptom**: OOM errors during dataset creation
**Cause**: Very long time series (> 10,000 points)
**Solution**: Use `maximum_wanted_duration` parameter to limit duration

### 7. Temporary File Cleanup Failures
**Symptom**: Files accumulating in `/tmp`
**Cause**: Exception before cleanup in finally block
**Solution**: Uses tempfile with automatic cleanup; check disk space

---

### 9. `load_model(model_config: GluformerModelConfig, model_path: str, device: str) -> ModelState`

**Purpose**: Instantiates and loads a Gluformer model from checkpoint.

**Process**:
1. Instantiate `Gluformer` with config parameters
2. Load state dict from file
3. Move to device (CPU/CUDA)
4. **CRITICAL**: Set to `.train()` mode for MC Dropout

**MC Dropout**:
- Model must be in training mode during inference
- Dropout layers remain active
- Enables uncertainty quantification
- Multiple forward passes produce different predictions

**Output**: `ModelState` = `Tuple[GluformerModelConfig, Gluformer]`
- Config included for validation in inference

**Error Handling**: Raises `RuntimeError` on load failure

---

### 10. `run_inference_full()`

**Signature**:
```python
def run_inference_full(
    dataset: SamplingDatasetInferenceDual, 
    model_config: GluformerModelConfig,
    model_state: ModelState,
    batch_size: int = BATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = "cpu"
) -> Tuple[PredictionsArray, LogVarsArray]
```

**Purpose**: Executes Monte Carlo Dropout inference over entire dataset.

**Process**:
1. **Config Validation**: Ensures model_config matches loaded model
2. **Prediction**: Calls `model.predict()` with MC sampling
   - `num_samples`: Number of stochastic forward passes
   - `batch_size`: Batch size for inference
3. **Result Formatting**: Returns tuple of `(forecasts, logvars)`.

**MC Dropout Details**:
- Each sample produces a different forecast (dropout randomness)
- Aggregating samples provides uncertainty estimates
- 10 samples is typical (configurable)

**Output**: 
```python
(predictions, logvars)
# predictions: PredictionsArray shape (N, output_chunk_length, num_samples)
# logvars: LogVarsArray shape (N, 1, num_samples)
```

**Default Parameters** (from config.py):
- `batch_size`: 32 (configurable via BATCH_SIZE env var)
- `num_samples`: 10 (configurable via NUM_SAMPLES env var)

**Error Handling**: Raises `RuntimeError` on:
- Config mismatch
- Prediction failure

---

### 11. `calculate_plot_data()`

**Signature**:
```python
def calculate_plot_data(
    predictions: PredictionsData,
    index: int,
    time_step: int = STEP_SIZE_MINUTES
) -> Tuple[str, PlotData]
```

**Purpose**: Transforms raw forecasts into visualization-ready data structure.

**Input**:
- `predictions`: PredictionsData containing forecasts, dataset, and scalers
- `index`: Negative index relative to dataset end (e.g., -1 = last sample)
- `time_step`: Time step in minutes (default: 5)

**Process**:
1. **Index Resolution**: Converts user index to dataset array index
   - Uses `predictions.get_dataset_index(index)`

2. **Extract Forecast**: Gets specific forecast for this index
   - Shape: `(len_pred, num_samples)` → `(12, 10)`

3. **Inverse Scaling**: Converts scaled values back to mg/dL
   - Formula: `(value - scaler.min) / scaler.scale`
   - **CRITICAL**: Same formula applied to forecasts AND historical data
   - This was empirically validated (see comments in code)

4. **Extract Past Context**: Gets input history from dataset
   - Uses `predictions.get_dataset_sample_index(index)`
   - Calls `dataset_dto.get_sample_data(ds_index)`
   - Returns `past_target` and `true_future`

5. **Calculate Median**: 50th percentile across MC samples

6. **Generate Fan Charts**: Uncertainty visualization
   - For each time point, create Gaussian KDE from MC samples
   - Normalize density to [0, 1]
   - Skip points with very low variance (`std < 1e-6`)
   - Color intensity increases with time (alpha ∝ time_index)

**Fan Chart Details**:
- Uses `scipy.stats.gaussian_kde` for smooth density estimation
- Grid: 200 points between `0.8*min` and `1.2*max`
- Color: `rgba(53, 138, 217, alpha)` where alpha = `(point + 1) / num_points`
- Handles edge cases: equal min/max, KDE failures

**Output**: 
```python
(plot_json_str, plot_data)
# plot_json_str: Plotly figure as JSON string
# plot_data: PlotData object with visualization data
```

**PlotData Structure**:
- `index`: Input index (negative)
- `true_values_x`: Time coordinates for history+future ([-len_pred, ..., len_pred-1])
- `true_values_y`: Actual glucose values
- `median_x`: Time coordinates for forecast ([-time_step, 0, time_step, ...])
- `median_y`: Median forecast with anchor point (last true value)
- `fan_charts`: List of FanChartData objects

**Visualization Design**:
- Anchors median forecast to last true value for visual continuity
- Shows `len_pred` points of history for context
- Fan charts positioned at forecast time coordinates
- Horizontal fan width represents probability density

**Error Handling**:
- KDE failures caught and logged (chart skipped for that time point)
- Returns error if dataset or scaler not provided in PredictionsData
- Index validation via `get_dataset_index()`

**Implementation Notes**:
The function returns both the plot JSON string AND the PlotData object. The JSON is ready for immediate use, while PlotData enables alternative rendering formats.

The calculation is embedded in the function rather than separated, allowing it to be cached alongside the plot visualization.

**Rendering Process** (embedded in function):
1. **Create Plotly Figure**: Initialize `go.Figure()`

2. **Add Fan Chart Traces**: For each FanChartData
   - Convert density values to x-offsets
   - Formula: `x = [time_coord, time_coord - density*0.8*time_step]`
   - Creates horizontal "violin" effect
   - Use `fill='tonexty'` for area fill
   - Color from `fan.fillcolor` (rgba with increasing alpha)

3. **Add True Values Trace**: Blue line with markers
   - Historical + future context

4. **Add Median Forecast Trace**: Red line with markers
   - Anchored to last true value

5. **Layout Configuration**:
   - Title: "Gluformer Prediction"
   - X-axis: "Time in minutes"
   - Y-axis: "Glucose (mg/dL)"
   - Size: 1000x600
   - Template: "plotly_white"

6. **Serialize to JSON**: Convert figure to JSON string
   - Uses `fig.to_json()`
   - Parse back to dict for validation
   - Ensures numpy arrays converted to lists

---

### 12. `convert_logic(content_base64: str) -> ConvertResponse`

**Purpose**: Simple conversion endpoint: arbitrary CSV → unified CSV format.

**Process**:
1. Parse CSV content using `parse_csv_content()`
2. Write unified DataFrame to in-memory buffer
3. Return CSV string in response

**Output**: `ConvertResponse` with:
- `csv_content`: Unified CSV as string (on success)
- `error`: Error message (on failure)

**Use Case**: Allows users to convert their CGM data to standardized format without inference.

**Implementation Notes**:
- Uses `io.StringIO()` for in-memory CSV generation (not temporary files)
- No handle computation needed (stateless operation)

---

### 13. `reconstruct_dataset(dataset_data: DartsDataset) -> SamplingDatasetInferenceDual`

**Purpose**: Reconstructs SamplingDatasetInferenceDual from a DartsDataset DTO.

**Process**:
1. **Reconstruct Target Series**: Convert numpy arrays to TimeSeries
   - `TimeSeries.from_values(np.array(ts))`

2. **Reconstruct Covariates**: Convert if present
   - Same process as target series

3. **Attach Static Covariates**: Add to each target series
   - Convert to pandas DataFrame
   - Attach using `ts.with_static_covariates(static_df)`

4. **Create Dataset**: Instantiate SamplingDatasetInferenceDual
   - Uses configuration from DartsDataset DTO

**Use Case**: 
- Reconstructing datasets from cache
- Restoring serialized state
- Enables plot calculation without re-running preprocessing

**Implementation Notes**:
- Handles `None` values for optional fields (covariates, static_covariates)
- Maintains `array_output_only=True` for performance
- Preserves all dataset configuration parameters

---

## Type Definitions

### ModelState
```python
ModelState = Tuple[GluformerModelConfig, Gluformer]
```
A pair of config and loaded model, ensuring config validation during inference.

---

## Important Implementation Notes

### 1. MC Dropout Requirement

**CRITICAL**: The model MUST be set to `.train()` mode during inference (line 438 in `load_model()`).

```python
model.train()  # Enable dropout for uncertainty estimation
```

This is counterintuitive but essential:
- Dropout layers remain active during inference
- Multiple forward passes produce different predictions (stochastic)
- Enables uncertainty quantification via Monte Carlo sampling
- Without this, all predictions would be identical (deterministic)

### 2. Scaling Post-Processing

**VALIDATED**: The inverse scaling formula is `(x - scaler.min) / scaler.scale` for BOTH forecasts and historical data.

This was extensively investigated (see lines 535-552 in code):
- Initial hypothesis: `(x * scale + min)` → produced implausible values (mean -0.5 mg/dL)
- Correct formula: `(x - min) / scale` → produced plausible values (mean 157.5 mg/dL)
- Empirically validated via `tests/debug_scaling.py`
- Same pattern used in `glucosedao/tools.py`

**Key Insight**: Using the same formula for forecasts and historical data ensures proper alignment on plots.

### 3. Embedded Rendering Decision

The `render_plot()` function was originally separate but is now embedded in `calculate_plot_data()` (see line 636 comment).

**Rationale**:
- Simplifies cache management (single entry per plot)
- Reduces overhead (no separate function call)
- Enables atomic caching of data + visualization
- Returns both JSON string and PlotData for flexibility

### 4. Dataset Index Conversion

The indexing system uses negative indices relative to dataset end:
- User Index: `-1` = last sample, `0` = end of predictions
- Dataset Index: Positive 0-based index into arrays
- Conversion handled by `PredictionsData.get_dataset_index()`

This enables intuitive "sliding window" access to recent predictions.

### 5. Feature Dimension Inference

Feature dimensions (`num_dynamic_features`, `num_static_features`) are inferred from the first sample of the dataset (lines 218-236):

```python
if len(dataset) > 0:
    sample = dataset[0]
    num_dynamic = sample[2].shape[1]  # Future covariates
    num_static = sample[3].shape[1]   # Static covariates
```

This allows automatic adaptation to different data formats without manual configuration.

---

## Design Decisions

### 1. **Why Polars?**
- Faster than pandas for large datasets
- Better memory efficiency
- Project rule: prefer polars over pandas
- Converted to pandas only for Darts compatibility

### 2. **Why MC Dropout?**
- Provides uncertainty quantification
- No need for ensemble of models
- Computationally efficient
- Requires model in `.train()` mode

### 3. **Why Content-Addressable Hashing?**
- Deduplication: Same data → same handle → cache hit
- Idempotency: Re-uploading same data doesn't create duplicates
- Transparency: Hash reveals if data changed

### 4. **Why Temporary Files?**
- FormatParser expects file paths
- Secure: tempfile handles cleanup automatically
- Simple: avoids in-memory file-like object complexity

### 5. **Why Two Scalers?**
- Targets and covariates have different distributions
- Separate scaling improves model performance
- Allows independent scaling strategies

### 6. **Why Fan Charts?**
- Intuitive uncertainty visualization
- Shows full probability distribution
- Better than simple confidence intervals
- Increasing opacity shows time progression

---

## Error Handling Philosophy

Following project rules:
- **Avoid nested try-catch**: Most errors propagate naturally
- **Let eliot handle logging**: With `start_action()` context
- **Fail fast**: Invalid data/config raises immediately
- **Informative errors**: Always include context (e.g., shape, columns)

Functions either:
1. **Return success dict** with results (e.g., `create_dataset_from_df`)
2. **Raise exceptions** for unrecoverable errors (e.g., `load_model`)
3. **Return error response** for user-facing endpoints (e.g., `convert_logic`)

---

## Performance Considerations

### 1. **Batch Inference**
- Process all dataset samples in single `model.predict()` call
- GPU batching when available
- Typical: 32 samples/batch

### 2. **Parallel KDE**
- Could be parallelized with ThreadPoolExecutor
- Currently sequential (simple, sufficient for 12 time points)

### 3. **Scaler Reuse**
- Scalers can be cached and reused
- Avoids re-fitting on similar data
- Currently fitted per request (stateless design)

### 4. **Memory Management**
- Temporary files cleaned immediately
- Large arrays not kept in memory longer than needed
- Polars → pandas conversion only when required

---

## Integration Points

### Called By
- `app.py`: API endpoints
- `engine.py`: Async execution wrappers
- `core.py`: High-level workflows

### Calls
- `cgm_format`: Data parsing and preprocessing
- `glucobench`: Model, formatters, dataset utilities
- `torch`: Model inference
- `plotly`: Visualization

### State Interactions
- Reads from `SessionState` (via engine/core)
- Does not directly modify state (functional design)

---

## Testing Considerations

### Unit Testing
- Mock cgm_format parsers for isolation
- Use synthetic polars DataFrames
- Test each function independently
- Focus on edge cases (empty data, single point, gaps)

### Integration Testing
- Use real CSV files from various CGM devices
- Test full pipeline: CSV → plot
- Verify model predictions shape and range
- Check warning flag handling
- Validate scaling post-processing

### Current Tests
- `tests/test_integration.py`: Full API flow with real data
- `tests/test_integration_load.py`: Load/stress testing
- `tests/test_data_classes.py`: DTO and indexing logic
- `tests/debug_scaling.py`: Scaling validation (referenced in code comments)

### Key Test Scenarios
1. **Minimal Data**: Exactly minimum duration (15 minutes)
2. **Gaps**: Small gaps (< 15 min) vs large gaps (> 45 min)
3. **Multiple Segments**: Data with multiple sequence IDs
4. **Edge Cases**: Single segment, no valid data, extreme glucose values
5. **Scaling**: Verify inverse transform produces valid mg/dL values (50-400 range)

---

## Logging Strategy

The module uses structured logging with multiple specialized loggers:

```python
logger = logging.getLogger("glurpc.logic")              # General logic operations
calc_logger = logging.getLogger("glurpc.logic.calc")    # Plot calculations
inference_logger = logging.getLogger("glurpc.logic.infer")  # ML inference
preprocessing_logger = logging.getLogger("glurpc.logic.data")  # Data preprocessing
```

**Log Level Configuration** (via environment variables):
- `LOG_LEVEL_LOGIC`: Controls all logic.* loggers (default: INFO)
- Individual loggers inherit from parent but can be overridden
- See `config.py` for full configuration options

**Debug logs include**:
- Data shapes at each step
- Column names after transformations
- Sample counts
- Feature dimensions
- Execution time for expensive operations

**Info logs include**:
- Pipeline milestones
- Success/failure summaries
- Result counts

**Error logs include**:
- Exception details
- Input context (shapes, types)
- Stack traces (via `exc_info=True`)

---

## Future Enhancements

### Potential Improvements
1. **Parallel KDE calculation** for fan charts
2. **Scaler persistence** in state for reuse
3. **Streaming inference** for very long sequences
4. **Alternative uncertainty visualizations** (quantile bands, HDI)
5. **Caching of processed datasets** (not just handles)
6. **GPU acceleration** for preprocessing
7. **Adaptive num_samples** based on uncertainty convergence

### Backward Compatibility
- Config classes are dataclasses (easy serialization)
- Type hints throughout (easier refactoring)
- Functional design (easy to test/modify)

---

## Appendix: Configuration Values

### GluformerInferenceConfig (defaults)
```python
# Architecture parameters
d_model: 512                # Model dimension
n_heads: 10                 # Number of attention heads
d_fcn: 1024                 # Fully connected layer dimension
num_enc_layers: 2           # Number of encoder layers
num_dec_layers: 2           # Number of decoder layers
r_drop: 0.2                 # Dropout rate
activ: "gelu"               # Activation function
distil: True                # Use distillation

# Sequence lengths
input_chunk_length: 96      # 8 hours at 5-min intervals
output_chunk_length: 12     # 1 hour at 5-min intervals
time_step: 5                # minutes

# Feature dimensions (inferred from data)
num_dynamic_features: 6     # Date features (day, month, year, hour, minute, second)
num_static_features: 1      # Sequence ID

# Data processing
gap_threshold: 45           # Max gap to interpolate (minutes)
min_drop_length: 12         # Min segment length (samples)
```

### FormatProcessor Parameters
```python
expected_interval_minutes: 5
small_gap_max_minutes: 15
minimum_duration_minutes: 540     # 9 hours (default from MINIMUM_DURATION_MINUTES_MODEL)
maximum_wanted_duration: 1080     # 18 hours (default from MAXIMUM_WANTED_DURATION_DEFAULT)
```

### Inference Parameters
```python
batch_size: 32              # Default from BATCH_SIZE env var
num_samples: 10             # MC Dropout samples from NUM_SAMPLES env var
device: "cpu"               # or "cuda" (auto-detected)

# Timeouts (from config.py)
INFERENCE_TIMEOUT_GPU: 600.0    # 10 minutes
INFERENCE_TIMEOUT_CPU: 7200.0   # 120 minutes
```

### Duration Limits (configurable via env vars)
```python
MINIMUM_DURATION_MINUTES_MODEL: 540  # 9 hours (108 samples × 5 min)
                                      # Formula: time_step × (input_chunk + output_chunk)
                                      # = 5 × (96 + 12) = 540 minutes
MAXIMUM_WANTED_DURATION_DEFAULT: 1080  # 18 hours (2× minimum)

# Runtime overridable
MINIMUM_DURATION_MINUTES: env var or model minimum
MAXIMUM_WANTED_DURATION: env var or default
```

### Queue Configuration
```python
MAX_INFERENCE_QUEUE_SIZE: 64    # Max inference tasks queued
MAX_CALC_QUEUE_SIZE: 8192       # Max calculation tasks queued
```

### Model Management
```python
NUM_COPIES_PER_DEVICE: 2        # Model copies per GPU
BACKGROUND_WORKERS_COUNT: 4     # Background calc workers
```

---

**Last Updated**: 2025-12-12  
**Module Version**: See `pyproject.toml`  
**Dependencies**: See `pyproject.toml` and `uv.lock`
