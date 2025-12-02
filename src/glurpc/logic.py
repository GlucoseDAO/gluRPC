import os
import io
import base64
import hashlib
import tempfile
import logging
from typing import Dict, Optional, Any, List, Tuple
import threading

import pandas as pd
import polars as pl
import numpy as np
import torch
import plotly.graph_objects as go
from scipy import stats
from darts import TimeSeries
import concurrent.futures

# Dependencies from glucobench
from glucobench.utils.darts_processing import ScalerCustom
from glucobench.utils.darts_dataset import SamplingDatasetInferenceDual
from glucobench.data_formatter import types as formatter_types
from glucobench.data_formatter import utils as formatter_utils

# Dependencies from cgm_format
from cgm_format import FormatParser, FormatProcessor
from cgm_format.interface import ProcessingWarning, WarningDescription

# Dependencies from glurpc
from glurpc.data_classes import GluformerInferenceConfig, GluformerModelConfig, PlotData, FanChartData
from glurpc.schemas import ConvertResponse

# Model dependencies
from glucobench.lib.gluformer.model import Gluformer

logger = logging.getLogger("glurpc.logic")

# Model state, a pair of the model i config dict and model class
ModelState = Tuple[GluformerModelConfig, Gluformer]


# --- Helper Functions (Logic) ---

def format_warnings(warning_flags: ProcessingWarning) -> Dict[str, Any]:
    logger.debug(f"Formatting warning flags: {warning_flags.value}")
    warnings_list = []
    if warning_flags & ProcessingWarning.TOO_SHORT:
        warnings_list.append(f"TOO_SHORT: {WarningDescription.TOO_SHORT.value}")
    if warning_flags & ProcessingWarning.CALIBRATION:
        warnings_list.append(f"CALIBRATION: {WarningDescription.CALIBRATION.value}")
    if warning_flags & ProcessingWarning.QUALITY:
        warnings_list.append(f"QUALITY: {WarningDescription.QUALITY.value}")
    if warning_flags & ProcessingWarning.IMPUTATION:
        warnings_list.append(f"IMPUTATION: {WarningDescription.IMPUTATION.value}")
    if warning_flags & ProcessingWarning.OUT_OF_RANGE:
        warnings_list.append(f"OUT_OF_RANGE: {WarningDescription.OUT_OF_RANGE.value}")
    if warning_flags & ProcessingWarning.TIME_DUPLICATES:
        warnings_list.append(f"TIME_DUPLICATES: {WarningDescription.TIME_DUPLICATES.value}")
    
    logger.debug(f"Parsed {len(warnings_list)} warnings from flags")
    return {
        'flags': warning_flags.value,
        'has_warnings': len(warnings_list) > 0,
        'messages': warnings_list
    }

def create_inference_dataset_fast_local(
    data: pl.DataFrame,
    config: GluformerInferenceConfig,
    scaler_target: Optional[ScalerCustom] = None,
    scaler_covs: Optional[ScalerCustom] = None
) -> Tuple[SamplingDatasetInferenceDual, ScalerCustom, GluformerModelConfig]:
    logger.info("=== Creating Inference Dataset ===")
    logger.debug(f"Input data shape: {data.shape}, columns: {data.columns}")
    logger.debug(f"Config: input_chunk={config.input_chunk_length}, output_chunk={config.output_chunk_length}")
    
    mapping = {}
    if 'sequence_id' in data.columns: mapping['sequence_id'] = 'id'
    if 'datetime' in data.columns: mapping['datetime'] = 'time'
    if 'glucose' in data.columns: mapping['glucose'] = 'gl'
    
    if mapping:
        logger.debug(f"Applying column mapping: {mapping}")
        data = data.rename(mapping)
    
    logger.debug("Converting to pandas and fixing dtypes")
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
    
    logger.debug(f"Interpolating with gap_threshold={config.gap_threshold}, min_drop_length={config.min_drop_length}")
    df_interp, updated_col_def = formatter_utils.interpolate(
        df, 
        column_definition, 
        gap_threshold=config.gap_threshold,
        min_drop_length=config.min_drop_length,
        interval_length=config.interval_length
    )
    logger.debug(f"After interpolation, shape: {df_interp.shape}")
    
    date_features = ['day', 'month', 'year', 'hour', 'minute', 'second']
    logger.debug(f"Encoding with date features: {date_features}")
    df_encoded, final_col_def, _ = formatter_utils.encode(
        df_interp,
        updated_col_def,
        date=date_features
    )
    logger.debug(f"After encoding, shape: {df_encoded.shape}, columns: {list(df_encoded.columns)}")
    
    target_series_list = []
    future_covariates_list = []
    
    target_col = 'gl'
    future_cols = [c for c in df_encoded.columns if any(f in c for f in date_features) and c not in ['id', 'time', 'gl', 'id_segment']]
    logger.debug(f"Target column: {target_col}, Future covariates: {future_cols}")
    
    groups = df_encoded.groupby('id_segment')
    logger.debug(f"Creating TimeSeries for {len(groups)} segments")
    
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

    logger.debug(f"Created {len(target_series_list)} target series")
    if scaler_target is None:
        logger.debug("Fitting new target scaler")
        scaler_target = ScalerCustom()
        target_series_scaled = scaler_target.fit_transform(target_series_list)
    else:
        logger.debug("Using provided target scaler")
        target_series_scaled = scaler_target.transform(target_series_list)
        
    if scaler_covs is None:
        logger.debug("Fitting new covariates scaler")
        scaler_covs = ScalerCustom()
        future_covariates_scaled = scaler_covs.fit_transform(future_covariates_list)
    else:
        logger.debug("Using provided covariates scaler")
        future_covariates_scaled = scaler_covs.transform(future_covariates_list)
        
    logger.debug(f"Creating dataset with input_chunk_length={config.input_chunk_length}, output_chunk_length={config.output_chunk_length}")
    dataset = SamplingDatasetInferenceDual(
        target_series=target_series_scaled,
        covariates=future_covariates_scaled,
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        use_static_covariates=True,
        array_output_only=True
    )
    logger.info(f"Dataset created successfully with {len(dataset)} samples")

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
        
    logger.debug(f"Inferred features: dynamic={num_dynamic}, static={num_static}")

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
    
    return dataset, scaler_target, model_config

def parse_csv_content(content_base64: str) -> pl.DataFrame:
    logger.debug("Starting CSV content parsing")
    try:
        logger.debug(f"Decoding base64 content (length: {len(content_base64)} chars)")
        content = base64.b64decode(content_base64)
        logger.debug(f"Decoded to {len(content)} bytes")
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        raise ValueError(f"Invalid base64: {e}")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    logger.debug(f"Wrote content to temporary file: {tmp_path}")
    
    try:
        logger.debug("Parsing file with FormatParser")
        unified_df = FormatParser.parse_file(tmp_path)
        logger.info(f"Successfully parsed CSV: shape={unified_df.shape}, columns={unified_df.columns}")
        return unified_df
    except Exception as e:
        logger.error(f"Parsing failed: {e}", exc_info=True)
        raise ValueError(f"Parsing failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Cleaned up temporary file: {tmp_path}")

def compute_handle(unified_df: pl.DataFrame) -> str:
    # Create canonical hash from unified dataframe
    # We serialize to CSV and hash that to ensure content-based addressability
    logger.debug(f"Computing handle for dataframe: shape={unified_df.shape}")
    buffer = io.BytesIO()
    unified_df.write_csv(buffer)
    content = buffer.getvalue()
    handle = hashlib.sha256(content).hexdigest()
    logger.debug(f"Computed handle: {handle[:16]}... (length: {len(content)} bytes)")
    return handle

def get_handle_and_df(content_base64: str) -> Tuple[str, pl.DataFrame]:
    """
    Parses CSV and computes canonical hash (handle).
    Returns (handle, unified_df).
    """
    try:
        unified_df = parse_csv_content(content_base64)
        handle = compute_handle(unified_df)
        logger.info(f"Parsed CSV. Unified Shape: {unified_df.shape} Handle: {handle[:8]}")
        return handle, unified_df
    except Exception as e:
        raise ValueError(f"Failed to parse or hash: {str(e)}")

def create_dataset_from_df(unified_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Creates dataset from unified dataframe.
    """
    logger.info("=== Creating Dataset from DataFrame ===")
    logger.debug(f"Input unified_df shape: {unified_df.shape}")
    try:
        logger.debug("Initializing FormatProcessor")
        processor = FormatProcessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15
        )
        
        logger.debug("Interpolating gaps")
        unified_df = processor.interpolate_gaps(unified_df)
        logger.debug(f"After gap interpolation: shape={unified_df.shape}")
        
        logger.debug("Synchronizing timestamps")
        unified_df = processor.synchronize_timestamps(unified_df)
        logger.debug(f"After timestamp sync: shape={unified_df.shape}")
        
        logger.debug("Preparing for inference (minimum_duration=15min, max_duration=8h)")
        inference_df, warning_flags = processor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=15,
            maximum_wanted_duration=9 * 60
        )
        
        if inference_df is None or len(inference_df) == 0:
            logger.warning("Data quality insufficient for inference")
            return {'error': "Data quality insufficient for inference"}

        logger.debug(f"Inference-ready data: shape={inference_df.shape}")
        logger.debug("Converting to glucose-only dataframe")
        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        logger.debug(f"Glucose-only data: shape={glucose_only_df.shape}")
        
        logger.debug("Creating inference config and dataset")
        config = GluformerInferenceConfig()
        dataset, scaler_target, model_config = create_inference_dataset_fast_local(glucose_only_df, config)
        
        logger.info(f"Dataset creation successful: {len(dataset)} samples, warnings={warning_flags.value}")
        return {
            'success': True,
            'dataset': dataset,
            'scaler_target': scaler_target,
            'model_config': model_config,
            'warning_flags': warning_flags
        }
    except Exception as e:
        logger.exception("Dataset creation failed")
        return {'error': f"Processing error: {str(e)}"}


def load_model(model_config: GluformerModelConfig, model_path: str, device: str) -> ModelState:
    """
    Instantiates and loads a Gluformer model from a state dictionary.
    Sets the model to train mode for MC Dropout inference.
    """
    logger.info(f"Loading model from {model_path} on {device}")
    try:
        model = Gluformer(**model_config.model_dump())
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.train() # CRITICAL Enable dropout for uncertainty estimation
        logger.debug(f"Model loaded successfully, dynamic features: {model_config.num_dynamic_features}, static features: {model_config.num_static_features}")
        return (model_config, model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

def run_inference_full(
    dataset: SamplingDatasetInferenceDual, 
    model_config: GluformerModelConfig,
    model_state: ModelState,
    batch_size: int = 32,
    num_samples: int = 10, # number of stochastic samples
    device: str = "cpu"
) -> Dict[int, np.ndarray]:
    """
    Run inference for the entire dataset using MC Dropout aggregation.
    Requires a pre-loaded model instance.
    Validates model against config before execution.
    Returns a dictionary of index -> forecasts (np.ndarray).
    """
    logger.info("=== Running Full Inference ===")
    logger.debug(f"Dataset size: {len(dataset)}")
    
    # Validate model config
    (loaded_config, model) = model_state
    if loaded_config != model_config:
        raise RuntimeError("Model config mismatch detected during inference preparation")
    
    logger.debug(f"Running prediction with num_samples={num_samples}, batch_size={batch_size}")
    
    try:
        forecasts, _ = model.predict(
            dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")

    logger.debug(f"Prediction complete. Forecasts shape: {forecasts.shape}")
    
    final_forecasts = {}
    dataset_len = len(dataset)
    
    for idx in range(dataset_len):
        # forecasts[idx] is the prediction for sample idx (12, 10)
        # We add the batch dim back
        final_forecasts[idx] = forecasts[idx][np.newaxis, ...]

    logger.info(f"Inference complete: {len(final_forecasts)} forecasts generated")
    return final_forecasts

def calculate_plot_data(forecasts: np.ndarray, dataset, scalers, index: int) -> PlotData:
    logger.debug(f"=== Calculating Plot Data for index {index} ===")
    logger.debug(f"Forecasts shape: {forecasts.shape}")
    
    current_forecast = forecasts[0]
    target_scaler = scalers['target']
    
    logger.debug("Inverse transforming forecast")
    current_forecast = (current_forecast - target_scaler.min_) / target_scaler.scale_
    
    logger.debug("Getting true future values")
    true_future = dataset.evalsample(index).values().flatten()
    true_future = (true_future - target_scaler.min_) / target_scaler.scale_
    
    logger.debug("Getting past target values")
    past_target_scaled = dataset[index][0]
    past_target = (past_target_scaled - target_scaler.min_) / target_scaler.scale_
    past_target = past_target.flatten()
    
    logger.debug(f"Past target length: {len(past_target)}, True future length: {len(true_future)}")
    
    samples = current_forecast.T
    logger.debug(f"Samples shape: {samples.shape} (MC samples x time points)")
    fan_charts = []
    
    logger.debug("Creating fan charts (KDE distributions)")
    for point in range(samples.shape[1]):
        pts = samples[:, point]
        if np.std(pts) < 1e-6:
            logger.debug(f"Point {point}: skipping (low variance)")
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
            # logger.debug(f"Point {point}: fan chart created")
        except Exception as e:
            logger.debug(f"Point {point}: KDE failed ({e})")
            pass

    logger.debug(f"Created {len(fan_charts)} fan charts")
    
    true_values = np.concatenate([past_target[-12:], true_future])
    true_values_x = list(range(-12, 12))
    
    median = np.quantile(samples, 0.5, axis=0)
    last_true_value = past_target[-1]
    median_with_anchor = [last_true_value] + median.tolist()
    median_x = [-1] + list(range(12))
    
    logger.debug("Plot data calculation complete")
    return PlotData(
        true_values_x=true_values_x,
        true_values_y=true_values.tolist(),
        median_x=median_x,
        median_y=median_with_anchor,
        fan_charts=fan_charts
    )

def render_plot(plot_data: PlotData) -> bytes:
    logger.debug("=== Rendering Plot ===")
    logger.debug(f"Fan charts: {len(plot_data.fan_charts)}")
    logger.debug(f"True values points: {len(plot_data.true_values_x)}")
    logger.debug(f"Median forecast points: {len(plot_data.median_x)}")
    
    fig = go.Figure()
    
    logger.debug("Adding fan chart traces")
    for i, fan in enumerate(plot_data.fan_charts):
        point = fan.time_index
        y_grid = np.array(fan.y)
        x_density = np.array(fan.x)
        
        x_trace = np.concatenate([
            np.full_like(y_grid, point), 
            np.full_like(y_grid, point - x_density * 0.9)[::-1]
        ])
        y_trace = np.concatenate([y_grid, y_grid[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_trace,
            y=y_trace,
            fill='tonexty',
            fillcolor=fan.fillcolor,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
    
    logger.debug("Adding true values trace")
    fig.add_trace(go.Scatter(
        x=plot_data.true_values_x,
        y=plot_data.true_values_y,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        name='True Values'
    ))
    
    logger.debug("Adding median forecast trace")
    fig.add_trace(go.Scatter(
        x=plot_data.median_x,
        y=plot_data.median_y,
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        name='Median Forecast'
    ))
    
    fig.update_layout(
        title='Gluformer Prediction',
        xaxis_title='Time (in 5 minute intervals)',
        yaxis_title='Glucose (mg/dL)',
        width=1000,
        height=600,
        template="plotly_white"
    )
    
    logger.debug("Converting plot to PNG image")
    image_bytes = fig.to_image(format="png")
    logger.info(f"Plot rendered successfully: {len(image_bytes)} bytes")
    return image_bytes

def convert_logic(content_base64: str) -> ConvertResponse:
    logger.info("=== Convert Logic Called ===")
    logger.debug(f"Input base64 length: {len(content_base64)} chars")
    try:
        logger.debug("Parsing CSV content")
        unified_df = parse_csv_content(content_base64)
        logger.debug(f"Parsed unified_df: shape={unified_df.shape}")
        
        logger.debug("Writing unified dataframe to temporary CSV")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            unified_df.write_csv(tmp.name)
            tmp_path = tmp.name
            
        logger.debug(f"Reading CSV from {tmp_path}")
        with open(tmp_path, 'r') as f:
            csv_content = f.read()
            
        logger.debug(f"CSV content size: {len(csv_content)} chars")
        os.remove(tmp_path)
        logger.debug("Temporary file cleaned up")
        
        logger.info("Convert logic completed successfully")
        return ConvertResponse(csv_content=csv_content)
        
    except Exception as e:
        logger.error(f"Convert logic failed: {e}", exc_info=True)
        return ConvertResponse(error=str(e))


