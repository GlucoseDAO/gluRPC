# REFERENCE_LOGIC.md

## Overview

The `logic.py` module contains the core business logic for the gluRPC service. It provides functions for:
- Data parsing and validation
- Dataset preparation for inference
- Model loading and inference execution
- Visualization and plotting
- Data format conversion

This module bridges the gap between the API layer (`app.py`), state management (`state.py`), and the execution engine (`engine.py`).

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

```
CSV (base64) 
  → parse_csv_content() 
  → unified_df (polars)
  → compute_handle() → SHA256 hash
  → create_dataset_from_df()
  → FormatProcessor pipeline
  → create_inference_dataset_fast_local()
  → SamplingDatasetInferenceDual
  → run_inference_full()
  → forecasts (numpy)
  → calculate_plot_data()
  → render_plot()
  → PNG image
```

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

**Shape**: `(N, output_chunk_length, num_samples)`
* **N**: Number of samples in the dataset (`dataset.total_samples`).
* **output_chunk_length**: Forecast horizon (default: 12 steps = 1 hour).
* **num_samples**: Number of Monte Carlo Dropout samples (default: 10).

**Content**:
* Contains **only** the predicted future values.
* Does **not** include the input history.
* Values are **scaled** (must be inverse-transformed for plotting).

### 4. Plot Data Calculation (`calculate_plot_data`)

This function generates the visualization data. Refactored for efficiency:
1. **Direct DTO Access**: Uses `dataset_dto` directly instead of reconstructing `SamplingDatasetInferenceDual`.
2. **Encapsulated Indexing**: Calls `predictions.get_dataset_sample_index()` to find the correct row.
3. **Efficient Slicing**: Uses `dataset_dto.get_sample_location()` to find the exact source array and slice `past_target` and `true_future` directly from numpy buffers.

*Post-processing Note*: We use the same inverse transform logic `(x - min) / scale` for both forecasts and historical values to ensure alignment.

---

## Function Reference

### 1. `format_warnings(warning_flags: ProcessingWarning) -> Dict[str, Any]`

**Purpose**: Converts bitwise warning flags from cgm_format into human-readable messages.

**Input**: 
- `warning_flags`: Bitwise flags from ProcessingWarning enum

**Output**:
```python
{
    'flags': int,              # Raw flag value
    'has_warnings': bool,      # Any warnings present
    'messages': List[str]      # Human-readable messages
}
```

**Warning Types**:
- `TOO_SHORT`: Insufficient data duration
- `CALIBRATION`: Calibration issues detected
- `QUALITY`: Data quality concerns
- `IMPUTATION`: Values were imputed
- `OUT_OF_RANGE`: Values outside expected range
- `TIME_DUPLICATES`: Duplicate timestamps found

**Usage**: Called after data preprocessing to inform users about data quality issues.

---

### 2. `create_inference_dataset_fast_local()`

**Signature**:
```python
def create_inference_dataset_fast_local(
    data: pl.DataFrame,
    config: GluformerInferenceConfig,
    scaler_target: Optional[ScalerCustom] = None,
    scaler_covs: Optional[ScalerCustom] = None
) -> Tuple[SamplingDatasetInferenceDual, ScalerCustom, GluformerModelConfig]
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
- `dataset`: Ready for model.predict()
- `scaler_target`: Fitted scaler (can be reused)
- `model_config`: GluformerModelConfig with inferred dimensions

**Key Configuration Parameters** (from `GluformerInferenceConfig`):
- `input_chunk_length`: 96 (8 hours at 5-min intervals)
- `output_chunk_length`: 12 (1 hour at 5-min intervals)
- `gap_threshold`: 15 minutes
- `min_drop_length`: 60 minutes
- `interval_length`: 5 minutes

---

### 3. `parse_csv_content(content_base64: str) -> pl.DataFrame`

**Purpose**: Decodes base64-encoded CSV content and parses it using cgm_format's FormatParser.

**Process**:
1. Base64 decode
2. Write to temporary file
3. Parse with `FormatParser.parse_file()`
4. Clean up temporary file
5. Return unified polars DataFrame

**Error Handling**:
- Raises `ValueError` on base64 decode failure
- Raises `ValueError` on parsing failure
- Always cleans up temporary files

**Output Format**: Unified DataFrame with standardized columns:
- `sequence_id`: Identifier for data segments
- `datetime`: Timestamp
- `glucose`: Blood glucose value in mg/dL
- Service columns (e.g., flags, quality indicators)

---

### 4. `compute_handle(unified_df: pl.DataFrame) -> str`

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

### 5. `get_handle_and_df(content_base64: str) -> Tuple[str, pl.DataFrame]`

**Purpose**: Convenience function combining parsing and hashing.

**Output**: `(handle, unified_df)`

**Usage**: Primary entry point for new data ingestion in the API.

---

### 6. `create_dataset_from_df(unified_df: pl.DataFrame) -> Dict[str, Any]`

**Purpose**: Complete preprocessing pipeline from unified DataFrame to inference-ready dataset.

**Process**:
1. **Initialize FormatProcessor**:
   - `expected_interval_minutes=5`
   - `small_gap_max_minutes=15`

2. **Gap Interpolation**: Fills small gaps in data

3. **Timestamp Synchronization**: Aligns to 5-minute intervals

4. **Inference Preparation**:
   - `minimum_duration_minutes=15` (minimum usable segment)
   - `maximum_wanted_duration=480` (8 hours max)
   - Returns processed data + warning flags

5. **Data Quality Check**: Returns error if data insufficient

6. **Glucose Extraction**: Converts to glucose-only DataFrame

7. **Dataset Creation**: Calls `create_inference_dataset_fast_local()`

**Output** (Success):
```python
{
    'success': True,
    'dataset': SamplingDatasetInferenceDual,
    'scaler_target': ScalerCustom,
    'model_config': GluformerModelConfig,
    'warning_flags': ProcessingWarning
}
```

**Output** (Failure):
```python
{'error': str}  # Error message
```

---

### 7. `load_model(model_config: GluformerModelConfig, model_path: str, device: str) -> ModelState`

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

### 8. `run_inference_full()`

**Signature**:
```python
def run_inference_full(
    dataset: SamplingDatasetInferenceDual, 
    model_config: GluformerModelConfig,
    model_state: ModelState,
    batch_size: int = 32,
    num_samples: int = 10,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]
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
# predictions shape: (N, output_chunk_length, num_samples)
```

**Error Handling**: Raises `RuntimeError` on:
- Config mismatch
- Prediction failure

---

### 9. `calculate_plot_data(predictions: PredictionsData, index: int) -> PlotData`

**Purpose**: Transforms raw forecasts into visualization-ready data structure.

**Process**:
1. **Inverse Scaling**: Converts scaled predictions back to mg/dL
   - Formula: `(value - min_) / scale_`
   - Applied to forecasts and true values

2. **Extract Past Context**: Last 12 points (1 hour) for continuity.
   - Extracted using `dataset_dto.get_sample_location(index)`.

3. **Extract True Future**: Ground truth for comparison (if available).

4. **Calculate Median**: 50th percentile across MC samples.

5. **Generate Fan Charts**: Uncertainty visualization
   - For each time point, create KDE from MC samples
   - Normalize density to [0, 1]
   - Skip points with very low variance
   - Color intensity increases with time
   - Each fan chart shows probability distribution

**Fan Chart Details**:
- Uses Gaussian KDE for smooth density estimation
- Grid: 200 points between 0.8*min and 1.2*max
- Color: `rgba(53, 138, 217, alpha)` where alpha ∝ time index
- Skip condition: `std < 1e-6` (deterministic/collapsed distribution)

**Output**: `PlotData` dataclass with:
- `true_values_x`: [-12, -11, ..., 10, 11] (5-min intervals)
- `true_values_y`: Glucose values
- `median_x`: [-1, 0, 1, ..., 11]
- `median_y`: Median forecast with anchor point
- `fan_charts`: List of `FanChartData` objects

**Error Handling**: KDE failures are caught and logged, chart skipped

---

### 10. `render_plot(plot_data: PlotData) -> bytes`

**Purpose**: Renders a plotly figure as PNG image bytes.

**Process**:
1. **Create Figure**: Initialize plotly Figure

2. **Add Fan Charts**: For each FanChartData
   - Create filled polygon using density as x-offset
   - Formula: `x = [point, point, ..., point-density*0.9, ...]`
   - Creates a "violin plot" effect horizontally
   - Fill with semi-transparent color

3. **Add True Values**: Blue line with markers

4. **Add Median Forecast**: Red line with markers
   - Anchored to last true value for continuity

5. **Layout Configuration**:
   - Title: "Gluformer Prediction"
   - X-axis: "Time (in 5 minute intervals)"
   - Y-axis: "Glucose (mg/dL)"
   - Size: 1000x600
   - Template: "plotly_white"

6. **Export to PNG**: Using `fig.to_image(format="png")`

**Output**: PNG image as bytes (ready for base64 encoding in API)

**Visualization Design**:
- Fan charts show uncertainty increasing over time
- Median provides point estimate
- True values enable visual accuracy assessment
- Horizontal fan charts avoid overlapping the time series

---

### 11. `convert_logic(content_base64: str) -> ConvertResponse`

**Purpose**: Simple conversion endpoint: arbitrary CSV → unified CSV format.

**Process**:
1. Parse CSV content
2. Write unified DataFrame to temporary CSV
3. Read CSV content as string
4. Clean up temporary file
5. Return CSV string in response

**Output**: `ConvertResponse` with:
- `csv_content`: Unified CSV as string (on success)
- `error`: Error message (on failure)

**Use Case**: Allows users to convert their CGM data to standardized format without inference.

---

## Type Definitions

### ModelState
```python
ModelState = Tuple[GluformerModelConfig, Gluformer]
```
A pair of config and loaded model, ensuring config validation during inference.

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
- Mock cgm_format parsers
- Use synthetic polars DataFrames
- Test each function in isolation

### Integration Testing
- Use real CSV files
- Test full pipeline: CSV → plot
- Verify model predictions shape
- Check warning flag handling

### Current Tests
- `tests/test_integration.py`: Full API flow
- `tests/test_integration_load.py`: Load testing
- `tests/test_data_classes.py`: DTO and indexing logic (NEW)

---

## Logging Strategy

All functions use structured logging:
```python
logger.info("High-level milestones")
logger.debug("Detailed state (shapes, columns, counts)")
logger.error("Failures with context")
```

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
input_chunk_length: 96      # 8 hours
output_chunk_length: 12     # 1 hour
gap_threshold: 15           # minutes
min_drop_length: 60         # minutes
interval_length: 5          # minutes
d_model: 64
n_heads: 4
d_fcn: 256
num_enc_layers: 2
num_dec_layers: 2
r_drop: 0.1
activ: "gelu"
distil: True
```

### FormatProcessor Parameters
```python
expected_interval_minutes: 5
small_gap_max_minutes: 15
minimum_duration_minutes: 15
maximum_wanted_duration: 480  # 8 hours
```

### Inference Parameters
```python
batch_size: 32
num_samples: 10  # MC Dropout samples
device: "cpu"    # or "cuda"
```

---

**Last Updated**: 2025-12-07
**Module Version**: See `pyproject.toml`  
**Dependencies**: See `pyproject.toml` and `uv.lock`
