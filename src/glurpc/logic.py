import base64
import hashlib
import io
import logging
import os
import tempfile
import datetime

import json
from typing import Dict, Optional, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import torch
from darts import TimeSeries
from scipy import stats

# Dependencies from glucobench
from glucobench.data_formatter import types as formatter_types
from glucobench.data_formatter import utils as formatter_utils
from glucobench.lib.gluformer.model import Gluformer
from glucobench.utils.darts_dataset import SamplingDatasetInferenceDual
from glucobench.utils.darts_processing import ScalerCustom

# Dependencies from cgm_format
from cgm_format import FormatParser, FormatProcessor
from cgm_format.interface import ProcessingWarning, WarningDescription
from cgm_format.interface.cgm_interface import (
    UnknownFormatError,
    MalformedDataError,
    ColumnOrderError,
    ColumnTypeError,
    ExtraColumnError,
    MissingColumnError,
    ZeroValidInputError,
    ValidationMethod
)

# Dependencies from glurpc
from glurpc.config import (
    STEP_SIZE_MINUTES,
    MINIMUM_DURATION_MINUTES_MODEL, 
    MAXIMUM_WANTED_DURATION_DEFAULT, 
    BATCH_SIZE, 
    NUM_SAMPLES
)
from glurpc.data_classes import (
    GluformerModelConfig,
    GluformerInferenceConfig, 
    PlotData, 
    PredictionsData, 
    FanChartData, 
    PredictionsArray, 
    LogVarsArray,
    DartsDataset,
    DartsScaler,
    DatasetCreationResult,
    FormattedWarnings
)
from glurpc.schemas import ConvertResponse

logger = logging.getLogger("glurpc.logic")
calc_logger = logging.getLogger("glurpc.logic.calc")
inference_logger = logging.getLogger("glurpc.logic.infer")
preprocessing_logger = logging.getLogger("glurpc.logic.data")

# Model state, a pair of the model i config dict and model class
ModelState = Tuple[GluformerModelConfig, Gluformer]


# --- Helper Functions (Logic) ---

def get_time_range(unified_df: pl.DataFrame) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
    """
    Extract start and end timestamps from a unified DataFrame.
    Assumes 'datetime' column exists (standard unified format).
    """
    if 'datetime' not in unified_df.columns:
        return None, None
    
    try:
        # Polars datetime column
        times = unified_df['datetime']
        if times.len() == 0:
            return None, None
        
        start_time = times.min()
        end_time = times.max()
        
        # Ensure python datetime objects
        if isinstance(start_time, (int, float)):
             # If timestamp, convert? Usually unified is datetime type.
             pass
             
        return start_time, end_time
    except Exception as e:
        logger.error(f"Failed to extract time range: {e}")
        return None, None

def calculate_dataset_length_from_input(
    input_samples: int,
    input_chunk_length: int,
    output_chunk_length: int
) -> int:
    """
    Calculate the expected dataset length based on the maximum wanted duration, time step, input length, input chunk length, and output chunk length.
    """
    expected_dataset_len = input_samples - (input_chunk_length + output_chunk_length - 1)
    return expected_dataset_len

def calculate_expected_dataset_length(
    maximum_wanted_duration_minutes: int,
    time_step: int,
    input_chunk_length: int,
    output_chunk_length: int
) -> int:
    """
    Calculate the expected dataset length based on the maximum wanted duration, time step, input length, input chunk length, and output chunk length.
    """
    expected_input_samples = (maximum_wanted_duration_minutes // time_step) + 1
    return calculate_dataset_length_from_input(expected_input_samples, input_chunk_length, output_chunk_length)

def create_inference_dataset_fast_local(data: pl.DataFrame, config: GluformerInferenceConfig, scaler_target: Optional[ScalerCustom] = None, scaler_covs: Optional[ScalerCustom] = None) -> Tuple[SamplingDatasetInferenceDual, GluformerModelConfig, ScalerCustom, ScalerCustom]:
    preprocessing_logger.info("=== Creating Inference Dataset ===")
    preprocessing_logger.debug(f"Input data shape: {data.shape}, columns: {data.columns}")
    preprocessing_logger.debug(f"Config: input_chunk={config.input_chunk_length}, output_chunk={config.output_chunk_length}")
    mapping = {}
    if 'sequence_id' in data.columns: mapping['sequence_id'] = 'id'
    if 'datetime' in data.columns: mapping['datetime'] = 'time'
    if 'glucose' in data.columns: mapping['glucose'] = 'gl'
    
    if mapping:
        preprocessing_logger.debug(f"Applying column mapping: {mapping}")
        data = data.rename(mapping)
    
    preprocessing_logger.debug("Converting to pandas and fixing dtypes")
    df = data.to_pandas()
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'])
    if 'gl' in df.columns: df['gl'] = df['gl'].astype(np.float32)
    
    DataTypes = formatter_types.DataTypes
    InputTypes = formatter_types.InputTypes
    
    column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('time', DataTypes.DATE, InputTypes.TIME),
        ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET)
    ]
    preprocessing_logger.debug(f"Before interpolation, shape: {df.shape}")
    preprocessing_logger.debug(f"Interpolating with gap_threshold={config.gap_threshold}, min_drop_length={config.min_drop_length}")
    df_interp, updated_col_def = formatter_utils.interpolate(
        df, 
        column_definition, 
        gap_threshold=config.gap_threshold,
        min_drop_length=config.min_drop_length,
        interval_length=config.interval_length
    )
    preprocessing_logger.debug(f"After interpolation, shape: {df_interp.shape}")
    if not df.shape == df_interp.shape:
        preprocessing_logger.warning(f"Interpolation changed shape from {df.shape} to {df_interp.shape}")
        preprocessing_logger.warning("Investigate on cgm_format package to see if this is expected behavior.")
    date_features = ['day', 'month', 'year', 'hour', 'minute', 'second']
    preprocessing_logger.debug(f"Encoding with date features: {date_features}")
    df_encoded, final_col_def, _ = formatter_utils.encode(
        df_interp,
        updated_col_def,
        date=date_features
    )
    preprocessing_logger.debug(f"After encoding, shape: {df_encoded.shape}, columns: {list(df_encoded.columns)}")
    
    target_series_list = []
    future_covariates_list = []
    
    target_col = 'gl'
    future_cols = [c for c in df_encoded.columns if any(f in c for f in date_features) and c not in ['id', 'time', 'gl', 'id_segment']]
    preprocessing_logger.debug(f"Target column: {target_col}, Future covariates: {future_cols}")
    
    groups = df_encoded.groupby('id_segment')
    preprocessing_logger.debug(f"Creating TimeSeries for {len(groups)} segments")
    
    for i, (seg_id, group) in enumerate(groups):
        group = group.sort_values('time')
        # logger.debug(f"Segment {i} (id={seg_id}): {len(group)} samples")
        
        ts_target = TimeSeries.from_dataframe(
            group, time_col='time', value_cols=[target_col], fill_missing_dates=False
        )
        
        ts_future = TimeSeries.from_dataframe(
            group, time_col='time', value_cols=future_cols, fill_missing_dates=False
        )
        
        original_id = group['id'].iloc[0]
        static_cov_df = pd.DataFrame({'id': [original_id]})
        ts_target = ts_target.with_static_covariates(static_cov_df)
        
        target_series_list.append(ts_target)
        future_covariates_list.append(ts_future)

    preprocessing_logger.debug(f"Created {len(target_series_list)} target series")
    if scaler_target is None:
        preprocessing_logger.debug("Fitting new target scaler")
        scaler_target = ScalerCustom()
        target_series_scaled = scaler_target.fit_transform(target_series_list)
    else:
        preprocessing_logger.debug("Using provided target scaler")
        target_series_scaled = scaler_target.transform(target_series_list)
        
    if scaler_covs is None:
        preprocessing_logger.debug("Fitting new covariates scaler")
        scaler_covs = ScalerCustom()
        future_covariates_scaled = scaler_covs.fit_transform(future_covariates_list)
    else:
        preprocessing_logger.debug("Using provided covariates scaler")
        future_covariates_scaled = scaler_covs.transform(future_covariates_list)
        
    preprocessing_logger.debug(f"Creating dataset with input_chunk_length={config.input_chunk_length}, output_chunk_length={config.output_chunk_length}")
    dataset = SamplingDatasetInferenceDual(
        target_series=target_series_scaled,
        covariates=future_covariates_scaled,
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        use_static_covariates=True,
        array_output_only=True
    )
    preprocessing_logger.info(f"Dataset created successfully with {len(dataset)} samples")

    # Infer feature dimensions from the first sample of the dataset
    if len(dataset) > 0:
        sample = dataset[0]
        # sample is likely (past_target, future_target, future_covariates, static_covariates)
        # Check future covariates (index 2)
        if len(sample) > 2 and sample[2] is not None:
             num_dynamic = sample[2].shape[1]
        else:
             num_dynamic = config.num_dynamic_features # fallback

        # Check static covariates (index 3)
        if len(sample) > 3 and sample[3] is not None:
             num_static = sample[3].shape[1]
        else:
             num_static = config.num_static_features # fallback
    else:
        num_dynamic = config.num_dynamic_features
        num_static = config.num_static_features
        
    preprocessing_logger.debug(f"Inferred features: dynamic={num_dynamic}, static={num_static}")

    # Create Model Config
    model_config = GluformerModelConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_fcn=config.d_fcn,
        num_enc_layers=config.num_enc_layers,
        num_dec_layers=config.num_dec_layers,
        
        len_seq=config.input_chunk_length,
        label_len=config.input_chunk_length // 3,
        len_pred=config.output_chunk_length,
        
        num_dynamic_features=num_dynamic,
        num_static_features=num_static,
        
        r_drop=config.r_drop,
        activ=config.activ,
        distil=config.distil
    )
    
    return dataset, model_config, scaler_target, scaler_covs

def parse_csv_content(content_base64: str) -> pl.DataFrame:
    preprocessing_logger.debug("Starting CSV content parsing")
    try:
        preprocessing_logger.debug(f"Parsing base64 content (length: {len(content_base64)} chars)")
        unified_df = FormatParser.parse_base64(content_base64)
        preprocessing_logger.info(f"Successfully parsed CSV: shape={unified_df.shape}, columns={unified_df.columns}")
        return unified_df
    except UnknownFormatError as e:
        # Expected error - invalid format
        preprocessing_logger.warning(f"Unknown file format: {e}")
        raise e
    except (MalformedDataError, ColumnOrderError, ColumnTypeError, 
            MissingColumnError, ExtraColumnError, ZeroValidInputError) as e:
        # Expected errors - known data quality/structure issues
        error_name = type(e).__name__.replace("Error", "")
        preprocessing_logger.warning(f"Data validation error ({error_name}): {e}")
        
        raise e
    except ValueError as e:
        # Expected errors during parsing or decoding
        preprocessing_logger.warning(f"Error decoding content: {e}")
        raise e
    except Exception as e:
        # Unexpected errors - log with traceback
        preprocessing_logger.exception(f"Unexpected parsing error: {e}")
        raise e


def compute_handle(unified_df: pl.DataFrame) -> str:
    # Create canonical hash from unified dataframe
    # We serialize to CSV and hash that to ensure content-based addressability
    preprocessing_logger.debug(f"Computing handle for dataframe: shape={unified_df.shape}")
    buffer = io.BytesIO()
    unified_df.write_csv(buffer)
    content = buffer.getvalue()
    handle = hashlib.sha256(content).hexdigest()
    preprocessing_logger.debug(f"Computed handle: {handle[:16]}... (length: {len(content)} bytes)")
    return handle

def get_handle_and_df(content_base64: str) -> Tuple[str, pl.DataFrame]:
    """
    Parses CSV and computes canonical hash (handle).
    Returns (handle, unified_df).
    """
    try:
        unified_df = parse_csv_content(content_base64)
        handle = compute_handle(unified_df)
        preprocessing_logger.info(f"Parsed CSV. Unified Shape: {unified_df.shape} Handle: {handle[:8]}")
        return handle, unified_df
    except ValueError as e:
        # Re-raise ValueError as-is (already formatted)
        raise
    except Exception as e:
        raise ValueError(f"Failed to process file: {str(e)}")

def analyse_and_prepare_df(
    unified_df: pl.DataFrame, 
    minimum_duration_minutes: int = MINIMUM_DURATION_MINUTES_MODEL, 
    maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION_DEFAULT
    ) -> Tuple[pl.DataFrame, ProcessingWarning, int]:
    """
    Creates dataset from unified dataframe.
    
    Returns:
        Tuple of (inference_df, warning_flags, dataset_length)
    Raises:
        ValueError: For data quality/format issues
    """
    preprocessing_logger.info("=== Analysing and preparing DataFrame for inference ===")
    preprocessing_logger.debug(f"Input unified_df shape: {unified_df.shape}")
    
    try:
        preprocessing_logger.debug("Initializing FormatProcessor")
        unified_df = FormatProcessor.detect_and_assign_sequences(unified_df)
        
        preprocessing_logger.debug("Interpolating gaps")
        unified_df = FormatProcessor.interpolate_gaps(unified_df)
        preprocessing_logger.debug(f"After gap interpolation: shape={unified_df.shape}")
        
        preprocessing_logger.debug("Synchronizing timestamps")
        unified_df = FormatProcessor.synchronize_timestamps(unified_df)
        preprocessing_logger.debug(f"After timestamp sync: shape={unified_df.shape}")
        
        preprocessing_logger.debug("Preparing for inference (minimum_duration=15min, max_duration=8h)")
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=minimum_duration_minutes,
            maximum_wanted_duration=maximum_wanted_duration,
            validation_mode=ValidationMethod.INPUT | ValidationMethod.OUTPUT
        )
        preprocessing_logger.debug("Converting to glucose-only dataframe to measure dataset length")
        data_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
 
        return inference_df, warning_flags, len(data_only_df)
    except (MalformedDataError, ZeroValidInputError) as e:
        # Known data quality issues during processing
        error_name = type(e).__name__.replace("Error", "")
        preprocessing_logger.error(f"Data processing error ({error_name}): {e}")
        raise ValueError(f"Data quality issue: {str(e)}")
    except Exception as e:
        # Unexpected processing errors
        preprocessing_logger.exception(f"Unexpected error during data preparation: {e}")
        raise ValueError(f"Failed to prepare data: {str(e)}")
    
def create_dataset_from_df(
    inference_df: pl.DataFrame, 
    warning_flags: ProcessingWarning,
    ) -> DatasetCreationResult:
    """
    Creates dataset from unified dataframe.
    
    Returns:
        DatasetCreationResult: Dict with 'success' bool and either dataset components or 'error' message
    """    
    try:    
        preprocessing_logger.info("=== Creating dataset from inference-ready dataframe ===")
        if inference_df is None or len(inference_df) == 0 or warning_flags & ProcessingWarning.TOO_SHORT:
            preprocessing_logger.warning("Data quality insufficient for inference")
            return {'success': False, 'error': "Data quality insufficient for inference (duration too short)"}

        preprocessing_logger.debug(f"Inference-ready data: shape={inference_df.shape}")
        preprocessing_logger.debug("Converting to glucose-only dataframe")
        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        preprocessing_logger.debug(f"Glucose-only data: shape={glucose_only_df.shape}")
        
        preprocessing_logger.debug("Creating inference config and dataset")
        config = GluformerInferenceConfig()
        dataset, model_config, scaler_target, scaler_covs = create_inference_dataset_fast_local(glucose_only_df, config)
        
        preprocessing_logger.info(f"Dataset creation successful: {len(dataset)} samples, warnings={warning_flags.value}")
        return {
            'success': True,
            'dataset': dataset,
            'scaler_target': scaler_target,
            'scaler_covs': scaler_covs,
            'model_config': model_config,
            'warning_flags': warning_flags
        }
    except (MalformedDataError, ZeroValidInputError) as e:
        # Known data quality issues
        error_name = type(e).__name__.replace("Error", "")
        preprocessing_logger.error(f"Dataset creation failed ({error_name}): {e}")
        return {'success': False, 'error': f"Data quality issue: {str(e)}"}
    except ValueError as e:
        # Expected validation errors (e.g., from upstream)
        preprocessing_logger.error(f"Dataset creation failed (validation): {e}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        # Unexpected errors
        preprocessing_logger.exception("Dataset creation failed with unexpected error")
        return {'success': False, 'error': f"Processing error: {str(e)}"}


def load_model(model_config: GluformerModelConfig, model_path: str, device: str) -> ModelState:
    """
    Instantiates and loads a Gluformer model from a state dictionary.
    Sets the model to train mode for MC Dropout inference.
    """
    inference_logger.info(f"Loading model from {model_path} on {device}")
    try:
        model = Gluformer(**model_config.model_dump())
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.train() # CRITICAL Enable dropout for uncertainty estimation
        inference_logger.debug(f"Model loaded successfully, dynamic features: {model_config.num_dynamic_features}, static features: {model_config.num_static_features}")
        return (model_config, model)
    except Exception as e:
        inference_logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

def run_inference_full(
    dataset: SamplingDatasetInferenceDual, 
    model_config: GluformerModelConfig,
    model_state: ModelState,
    batch_size: int = BATCH_SIZE,
    num_samples: int = NUM_SAMPLES, # number of stochastic samples
    device: str = "cpu"
) -> Tuple[PredictionsArray, LogVarsArray]:
    """
    Run inference for the entire dataset using MC Dropout aggregation.
    Requires a pre-loaded model instance.
    Validates model against config before execution.
    Returns a tuple of (predictions, logvars).
    -------
    Predictions
        The predicted future target series in shape n x len_pred x num_samples, where
        n is total number of predictions.
    Logvar
        The logvariance of the predicted future target series in shape n x len_pred.
    """
    inference_logger.info("=== Running Full Inference ===")
    inference_logger.debug(f"Dataset size: {len(dataset)}")
    
    # Validate model config
    (loaded_config, model) = model_state
    if loaded_config != model_config:
        raise RuntimeError("Model config mismatch detected during inference preparation")
    
    inference_logger.debug(f"Running prediction with num_samples={num_samples}, batch_size={batch_size}")
    
    try:
        forecasts, logvars = model.predict(
            dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device
        )
    except Exception as e:
        inference_logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")

    inference_logger.debug(f"Prediction complete. Forecasts shape: {forecasts.shape}")
    
    # Optimization: Return raw array (N, 12, 10) directly
    inference_logger.info(f"Inference complete: {len(forecasts)} forecasts generated")
    return forecasts, logvars

def calculate_plot_data(
    predictions: PredictionsData, 
    index: int, time_step: int = STEP_SIZE_MINUTES
) -> Tuple[str, PlotData]:
    """
    Calculates plot data for a given index.
    Args:
        forecasts: PredictionsData containing predictions array and metadata
        dataset: SamplingDatasetInferenceDual
        scalers: dict of scalers
        index: int (negative index relative to end of dataset)
    Returns:
        PlotData
    """
    calc_logger.debug(f"=== Calculating Plot Data for index {index} ===")
    
    # Calculate relative index in the predictions array
    # index is typically negative (e.g. -1, -2). first_index is also negative (e.g. -100).
    # If first_index is -100, and we want index -100, relative is 0.
    # If we want index -1, relative is 99.
    forecasts = predictions.predictions
    dataset_dto: Optional[DartsDataset] = predictions.dataset
    target_scaler: Optional[DartsScaler] = predictions.target_scaler
    
    if dataset_dto is None or target_scaler is None:
        calc_logger.error(f"Dataset and target scaler are required in PredictionsData for plot calculation: {predictions.model_dump_json()}")
        raise ValueError("Dataset and target scaler are required in PredictionsData for plot calculation")

    len_pred = predictions.len_pred
    
    aligned_index = predictions.get_dataset_index(index)
    
    # Extract specific forecast for this index: (12, 10)
    if forecasts is not None:
        current_forecast = forecasts[aligned_index]
    else:
        calc_logger.error(f"No forecasts supplied in {predictions.model_dump_json()}")
        raise ValueError(f"No forecasts supplied in PredictionsData")
    
    calc_logger.debug(f"Extracted forecast shape: {current_forecast.shape} for index {index}")
    

    
    # NOTE: This postprocessing logic was extensively investigated and validated.
    # Initial hypothesis suggested using inverse transform (x * scale + min), but
    # empirical validation via tests/debug_scaling.py demonstrated that the correct
    # approach is to use (x - min) / scale for BOTH forecasts and historical data.
    # 
    # Key insight: The scaler's min_ and scale_ parameters work such that this operation
    # correctly brings scaled values back to the original glucose units (mg/dL).
    # Using the same postprocessing for forecasts and historical ensures they align
    # properly on the plot, which was the original issue.
    #
    # Reference: glucosedao/tools.py uses the same pattern for inputs and forecasts.
    
    # VALIDATED: The correct postprocessing for both forecasts and historical data
    # is (x - scaler.min_) / scaler.scale_ to bring scaled values back to original units.
    # This was empirically validated via tests/debug_scaling.py which showed:
    # - Method (x * scale + min) produced implausible values (mean -0.5 mg/dL)
    # - Method (x - min) / scale produced plausible values (mean 157.5 mg/dL)
    # The key insight: use the SAME postprocessing for forecasts and historical data.
    calc_logger.debug("Postprocessing forecast to Real Values (mg/dL)")
    current_forecast = (current_forecast - target_scaler.min) / target_scaler.scale
    
    calc_logger.debug("Getting true future values")
    
    
    
    # Convert custom negative indexing (0 is last) to dataset positive index
    ds_index = predictions.get_dataset_sample_index(index)
    # Find sample in the list of series
    try:
        past_target, true_future = dataset_dto.get_sample_data(ds_index)
        past_timestamps, future_timestamps = dataset_dto.get_sample_timestamps(ds_index)
    except ValueError as e:
        raise ValueError(f"Could not find sample for index {ds_index}: {e}")

    true_future = true_future.flatten()
    # Apply SAME postprocessing as forecasts (validated approach)
    true_future = (true_future - target_scaler.min) / target_scaler.scale
    
    calc_logger.debug("Getting past target values")
    # past_target is already extracted
    # Apply SAME postprocessing as forecasts (validated approach)
    past_target = (past_target - target_scaler.min) / target_scaler.scale
    past_target = past_target.flatten()
    
    calc_logger.debug(f"Past target length: {len(past_target)}, True future length: {len(true_future)}")
    
    samples = current_forecast.T
    calc_logger.debug(f"Samples shape: {samples.shape} (MC samples x time points)")
    fan_charts = []
    
    calc_logger.debug("Creating fan charts (KDE distributions)")
    for point in range(samples.shape[1]):
        # samples is (10, 12). point iterates 0..11.
        pts = samples[:, point]
        if np.std(pts) < 1e-6:
            # calc_logger.debug(f"Point {point}: skipping (low variance)")
            continue
            
        try:
            kde = stats.gaussian_kde(pts)
            maxi, mini = 1.2 * np.max(pts), 0.8 * np.min(pts)
            if maxi == mini:
                maxi += 1
                mini -= 1
            y_grid = np.linspace(mini, maxi, 200)
            x = kde(y_grid)
            
            x = x / np.max(x) if np.max(x) > 0 else x
            
            color = f'rgba(53, 138, 217, {(point + 1) / samples.shape[1]})'
            
            fan_charts.append(FanChartData(
                x=x.tolist(),
                y=y_grid.tolist(),
                fillcolor=color,
                time_index=point
            ))
            # calc_logger.debug(f"Point {point}: fan chart created")
        except Exception as e:
            calc_logger.debug(f"Point {point}: KDE failed ({e})")
            pass

    calc_logger.debug(f"Created {len(fan_charts)} fan charts")
    
    true_values = np.concatenate([past_target[-len_pred:], true_future])
    true_values_x = [x*time_step for x in range(-len_pred, len_pred)]
    
    median = np.quantile(samples, 0.5, axis=0)
    last_true_value = past_target[-1]
    median_with_anchor = [last_true_value] + median.tolist()
    median_x = [-1*time_step] + [x*time_step for x in range(len_pred)]
    
    # Extract actual timestamps if available
    true_values_timestamps = None
    median_timestamps = None
    
    if past_timestamps is not None and future_timestamps is not None:
        calc_logger.debug("Extracting actual timestamps for plot")
        # For true values: last len_pred of past + all future
        past_times_for_plot = past_timestamps[-len_pred:]
        true_values_timestamps = past_times_for_plot + future_timestamps
        
        # For median: last past timestamp + all future timestamps
        median_timestamps = [past_timestamps[-1]] + future_timestamps
        
        calc_logger.debug(f"Timestamps extracted: true_values={len(true_values_timestamps)}, median={len(median_timestamps)}")
    
    calc_logger.info(f"Completed calculation for index {index}: {len(fan_charts)} fan charts")
    plot_data = PlotData(
        index=index,
        true_values_x=true_values_x,
        true_values_y=true_values.tolist(),
        median_x=median_x,
        median_y=median_with_anchor,
        fan_charts=fan_charts,
        true_values_timestamps=true_values_timestamps,
        median_timestamps=median_timestamps
    )
    #why have a separate function, we shall store the final plot data in the cache entry
#def render_plot(plot_data: PlotData) -> Dict[str, Any]:
    calc_logger.debug("=== Pre-rendering Plot ===")
    
    fig = go.Figure() #not a pydantic type, serializing as dict
    
    # Determine if we should use datetime labels
    use_datetime_labels = (plot_data.true_values_timestamps is not None and 
                          plot_data.median_timestamps is not None)
    
    if use_datetime_labels:
        calc_logger.debug("Using actual datetime labels for x-axis")
        # Use timestamps for x-axis
        true_x_values = plot_data.true_values_timestamps
        median_x_values = plot_data.median_timestamps
        x_axis_title = 'Time'
    else:
        calc_logger.debug("Using relative time in minutes for x-axis")
        # Fallback to relative time in minutes
        true_x_values = plot_data.true_values_x
        median_x_values = plot_data.median_x
        x_axis_title = 'Time in minutes'
    
    calc_logger.debug("Adding fan chart traces")
    for i, fan in enumerate(plot_data.fan_charts):
        point = fan.time_index
        y_grid = np.array(fan.y)
        x_density = np.array(fan.x)
        fillcolor = fan.fillcolor

        if use_datetime_labels:
            # For datetime labels, use the actual timestamp at this forecast point
            # median_x_values[0] is the anchor point (last observed), median_x_values[1:] are forecasts
            # So forecast point 0 corresponds to median_x_values[1]
            if point + 1 < len(median_x_values):
                from datetime import datetime, timedelta
                
                time_coord_str = median_x_values[point + 1]
                time_coord = datetime.fromisoformat(time_coord_str)
                
                # Calculate time offsets for the fan width
                # Use density values to create offsets (scaled to minutes)
                max_offset_minutes = 0.9 * time_step
                time_offsets = x_density * max_offset_minutes
                
                # Create left edge (center time) and right edge (center - offset)
                x_trace_left = [time_coord.isoformat()] * len(y_grid)
                x_trace_right = [(time_coord - timedelta(minutes=offset)).isoformat() 
                                for offset in time_offsets][::-1]
                
                x_trace = x_trace_left + x_trace_right
            else:
                continue
        else:
            # Original behavior for relative time
            time_coord = point * time_step
            x_trace = np.concatenate([
                np.full_like(y_grid, time_coord), 
                np.full_like(y_grid, time_coord - x_density * 0.9*time_step)[::-1]
            ])
        
        y_trace = np.concatenate([y_grid, y_grid[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_trace,
            y=y_trace,
            fill='tonexty',
            fillcolor=fillcolor,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    calc_logger.debug("Adding true values trace")
    fig.add_trace(go.Scatter(
        x=true_x_values,
        y=plot_data.true_values_y,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        name='True Values'
    ))
    
    calc_logger.debug("Adding median forecast trace")
    fig.add_trace(go.Scatter(
        x=median_x_values,
        y=plot_data.median_y,
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        name='Median Forecast'
    ))
    
    # Configure layout with appropriate grid settings
    layout_config = {
        'title': 'Gluformer Prediction',
        'xaxis_title': x_axis_title,
        'yaxis_title': 'Glucose (mg/dL)',
        'width': 1000,
        'height': 600,
        'template': "plotly_white"
    }
    
    if use_datetime_labels:
        # For datetime x-axis, align grid to the divergence point (where forecast starts)
        # This is median_x_values[0] - the last observed point
        divergence_point = median_x_values[0]
        
        # dtick in milliseconds: 10 minutes = 10 * 60 * 1000 = 600000
        layout_config['xaxis'] = {
            'title': x_axis_title,
            'tick0': divergence_point,  # Align grid to divergence point
            'dtick': 600000,  # 10 minutes in milliseconds
            'tickformat': '%H:%M\n%b %d',  # Hour:Minute on first line, Month Day on second
            'gridcolor': 'lightgray',
            'showgrid': True
        }
    else:
        # For relative time, align grid to the divergence point (0)
        tick_interval = 10 if time_step == 1 else (10 // time_step) * time_step * 2
        layout_config['xaxis'] = {
            'title': x_axis_title,
            'tick0': 0,  # Align grid to the divergence point
            'dtick': tick_interval,
            'gridcolor': 'lightgray',
            'showgrid': True
        }
    
    fig.update_layout(**layout_config)
    
    calc_logger.debug("Converting plot to Plotly JSON dict")
    # Use to_json() then parse back to ensure numpy arrays are converted to lists
    plot_json_str = fig.to_json()
    plot_dict = json.loads(plot_json_str)
    calc_logger.info(f"Plot rendered successfully as JSON (data keys: {list(plot_dict.keys())})")
    return plot_json_str, plot_data #intermediary value for alternative renders?


def convert_logic(content_base64: str) -> ConvertResponse:
    calc_logger.info("=== Convert Logic Called ===")
    calc_logger.debug(f"Input base64 length: {len(content_base64)} chars")
    try:
        calc_logger.debug("Parsing CSV content")
        unified_df = parse_csv_content(content_base64)
        calc_logger.debug(f"Parsed unified_df: shape={unified_df.shape}")
        
        calc_logger.debug("Writing unified dataframe to CSV in memory")
        buffer = io.StringIO()
        unified_df.write_csv(buffer)
        csv_content = buffer.getvalue()
        calc_logger.debug(f"CSV content size: {len(csv_content)} chars")     
        calc_logger.info("Convert logic completed successfully")
        return ConvertResponse(csv_content=csv_content)
        
    except ValueError as e:
        # Expected errors from parse_csv_content (invalid format, malformed data, etc.)
        calc_logger.info(f"Reporting handled error to client: {e}")
        return ConvertResponse(error=str(e))
    except Exception as e:
        # Unexpected errors - log with traceback
        calc_logger.error(f"Convert logic crashed (unexpected): {e}", exc_info=True)
        return ConvertResponse(error=str(e))

def reconstruct_dataset(dataset_data: DartsDataset) -> SamplingDatasetInferenceDual:
    """
    Reconstructs SamplingDatasetInferenceDual from a DartsDataset DTO.
    This handles the complexity of recreating TimeSeries objects from raw arrays.
    """
    # Reconstructing dataset for inference
    target_series = [TimeSeries.from_values(np.array(ts)) for ts in dataset_data.target_series]
    covariates = None
    if dataset_data.covariates:
        covariates = [TimeSeries.from_values(np.array(ts)) for ts in dataset_data.covariates]
    
    # Static covariates
    if dataset_data.static_covariates:
         # Need to attach to target_series
         new_target_series = []
         for ts, static in zip(target_series, dataset_data.static_covariates):
             # Convert to pandas DataFrame for Darts
             # static is numpy array. Convert to dict/df.
             static_df = pd.DataFrame(np.array(static))
             ts = ts.with_static_covariates(static_df)
             new_target_series.append(ts)
         target_series = new_target_series
             
    return SamplingDatasetInferenceDual(
        target_series=target_series,
        covariates=covariates,
        input_chunk_length=dataset_data.input_chunk_length,
        output_chunk_length=dataset_data.output_chunk_length,
        use_static_covariates=dataset_data.use_static_covariates,
        array_output_only=True # We want arrays back
    )
