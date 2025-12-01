import logging
import os
import base64
import hashlib
import uuid
import tempfile
from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
import polars as pl
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
from scipy import stats

# Dependencies
from lib.gluformer.model import Gluformer
from utils.darts_processing import ScalerCustom
from utils.darts_dataset import SamplingDatasetInferenceDual
from cgm_format import FormatParser, FormatProcessor
from cgm_format.interface import ProcessingWarning, WarningDescription
import data_formatter.utils as formatter_utils
from data_formatter import types as formatter_types
from darts import TimeSeries
from glurpc.data_classes import GluformerModelConfig, GluformerInferenceConfig
from glurpc.schemas import UnifiedResponse, PlotRequest, QuickPlotResponse, ConvertResponse

import datetime

# Setup logging
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"glurpc_{timestamp}.log"
log_path = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("glurpc")
logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplication if reloaded
if logger.hasHandlers():
    logger.handlers.clear()
    
file_handler = logging.FileHandler(log_path, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

logger.info(f"Logging initialized to {log_path}")

CACHE: Dict[str, Any] = {}
MAX_CACHE_SIZE = 128
MODEL: Optional[Gluformer] = None
MODEL_CONFIG: Optional[GluformerInferenceConfig] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Management ---

def get_model():
    global MODEL
    if MODEL is None:
        load_model()
    return MODEL

def load_model(model_name: str = "gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth"):
    global MODEL, MODEL_CONFIG
    logger.info(f"Loading model: {model_name}")
    
    try:
        config = GluformerInferenceConfig()
        
        repo_id = "Livia-Zaharia/gluformer_models"
        model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
        
        model_params = GluformerModelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_fcn=config.d_fcn,
            num_enc_layers=config.num_enc_layers,
            num_dec_layers=config.num_dec_layers,
            len_seq=config.input_chunk_length,
            label_len=config.input_chunk_length // 3,
            len_pred=config.output_chunk_length,
            num_dynamic_features=6,
            num_static_features=1,
            r_drop=config.r_drop,
            activ=config.activ,
            distil=config.distil
        )
        
        MODEL = Gluformer(**model_params.model_dump())
        MODEL.load_state_dict(torch.load(str(model_path), map_location=torch.device(DEVICE)))
        MODEL.to(DEVICE)
        MODEL.eval()
        
        MODEL_CONFIG = config
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# --- Helpers ---

def _format_warnings(warning_flags: ProcessingWarning) -> Dict[str, Any]:
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
    
    return {
        'flags': warning_flags.value, # Int value for serialization
        'has_warnings': len(warnings_list) > 0,
        'messages': warnings_list
    }

def create_inference_dataset_fast_local(
    data: pl.DataFrame,
    config: GluformerInferenceConfig,
    scaler_target: Optional[ScalerCustom] = None,
    scaler_covs: Optional[ScalerCustom] = None
):
    logger.info("=== Creating Inference Dataset ===")
    # 1. Polars to Pandas and rename
    mapping = {}
    if 'sequence_id' in data.columns: mapping['sequence_id'] = 'id'
    if 'datetime' in data.columns: mapping['datetime'] = 'time'
    if 'glucose' in data.columns: mapping['glucose'] = 'gl'
    
    if mapping:
        data = data.rename(mapping)
    
    df = data.to_pandas()
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'])
    if 'gl' in df.columns: df['gl'] = df['gl'].astype(np.float32)
    
    logger.info(f"Input DF Shape: {df.shape}")
    
    # 2. Column definition
    DataTypes = formatter_types.DataTypes
    InputTypes = formatter_types.InputTypes
    
    column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('time', DataTypes.DATE, InputTypes.TIME),
        ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET)
    ]
    
    # 3. Interpolate
    df_interp, updated_col_def = formatter_utils.interpolate(
        df, 
        column_definition, 
        gap_threshold=config.gap_threshold,
        min_drop_length=config.min_drop_length,
        interval_length=config.interval_length
    )
    logger.info(f"Interpolated DF Shape: {df_interp.shape}")
    
    # 4. Encode Datetime
    date_features = ['day', 'month', 'year', 'hour', 'minute', 'second']
    df_encoded, final_col_def, _ = formatter_utils.encode(
        df_interp,
        updated_col_def,
        date=date_features
    )
    
    # 5. Darts Series
    target_series_list = []
    future_covariates_list = []
    
    target_col = 'gl'
    future_cols = [c for c in df_encoded.columns if any(f in c for f in date_features) and c not in ['id', 'time', 'gl', 'id_segment']]
    
    groups = df_encoded.groupby('id_segment')
    logger.info(f"Number of segments found: {len(groups)}")
    
    for i, (seg_id, group) in enumerate(groups):
        group = group.sort_values('time')
        
        # Log segment stats
        logger.info(f"Segment {i} (ID {seg_id}): Length {len(group)}, Start {group['time'].min()}, End {group['time'].max()}")
        
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

    # 6. Scaling
    if scaler_target is None:
        logger.info("Fitting new Target Scaler")
        scaler_target = ScalerCustom()
        target_series_scaled = scaler_target.fit_transform(target_series_list)
        logger.info(f"Target Scaler Fitted: Min={scaler_target.min_}, Scale={scaler_target.scale_}")
    else:
        logger.info(f"Using Existing Target Scaler: Min={scaler_target.min_}, Scale={scaler_target.scale_}")
        target_series_scaled = scaler_target.transform(target_series_list)
        
    if scaler_covs is None:
        logger.info("Fitting new Covariates Scaler")
        scaler_covs = ScalerCustom()
        future_covariates_scaled = scaler_covs.fit_transform(future_covariates_list)
    else:
        future_covariates_scaled = scaler_covs.transform(future_covariates_list)
        
    # Debug Scaled Values
    if len(target_series_scaled) > 0:
         logger.info(f"First scaled series mean: {target_series_scaled[0].values().mean()}")
         logger.info(f"First scaled series sample values: {target_series_scaled[0].values().flatten()[:10]}")
    
    # 7. Dataset
    dataset = SamplingDatasetInferenceDual(
        target_series=target_series_scaled,
        covariates=future_covariates_scaled,
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        use_static_covariates=True,
        array_output_only=True
    )
    logger.info(f"Created Dataset with {len(dataset)} samples")
    
    return dataset, scaler_target

# --- Main Functions ---

def parse_csv_content(content_base64: str) -> pl.DataFrame:
    try:
        content = base64.b64decode(content_base64)
    except Exception as e:
        raise ValueError(f"Invalid base64: {e}")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        unified_df = FormatParser.parse_file(tmp_path)
        return unified_df
    except Exception as e:
        raise ValueError(f"Parsing failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_and_cache(content_base64: str) -> UnifiedResponse:
    # Clean cache if full
    if len(CACHE) >= MAX_CACHE_SIZE:
        key_to_remove = next(iter(CACHE))
        del CACHE[key_to_remove]

    try:
        unified_df = parse_csv_content(content_base64)
        logger.info(f"Parsed CSV. Unified Shape: {unified_df.shape}")
    except ValueError as e:
        return UnifiedResponse(error=str(e))

    # Process
    try:
        processor = FormatProcessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15
        )
        
        unified_df = processor.interpolate_gaps(unified_df)
        unified_df = processor.synchronize_timestamps(unified_df)
        
        inference_df, warning_flags = processor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=15,
            maximum_wanted_duration=24 * 60
        )
        logger.info(f"Prepare for Inference Result: {inference_df.shape if inference_df is not None else 'None'}")
        
        if inference_df is None or len(inference_df) == 0:
            return UnifiedResponse(error="Data quality insufficient for inference")

        # Create Dataset
        glucose_only_df = FormatProcessor.to_data_only_df(
            unified_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        logger.info(f"Glucose Only DF Shape: {glucose_only_df.shape}")
        
        config = GluformerInferenceConfig()
        dataset, scaler_target = create_inference_dataset_fast_local(glucose_only_df, config)
        
        # Cache
        handle = hashlib.sha256(content_base64.encode()).hexdigest()
        CACHE[handle] = {
            'dataset': dataset,
            'scalers': {'target': scaler_target},
            'timestamp': pd.Timestamp.now()
        }
        logger.info(f"Cached dataset with handle {handle}")
        
        return UnifiedResponse(
            handle=handle,
            warnings=_format_warnings(warning_flags)
        )
        
    except Exception as e:
        logger.exception("Processing failed")
        return UnifiedResponse(error=f"Processing error: {str(e)}")

def generate_plot_from_handle(handle: str, index: int) -> bytes:
    if handle not in CACHE:
        raise ValueError("Handle not found or expired")
    
    data = CACHE[handle]
    dataset = data['dataset']
    scalers = data['scalers']
    
    logger.info(f"Generate Plot: Index={index}, Dataset Size={len(dataset)}")
    
    if index < 0 or index >= len(dataset):
        raise ValueError(f"Index {index} out of range (0-{len(dataset)-1})")
        
    # Run inference
    model = get_model()
    from torch.utils.data import Subset
    subset = Subset(dataset, [index])
    
    # Debug Input
    sample = dataset[index]
    logger.info(f"Input Sample [0] (Past Target Scaled) Mean: {sample[0].mean()}")
    logger.info(f"Input Sample [0] (Past Target Scaled) First 5: {sample[0][:5].flatten()}")
    
    forecasts, _ = model.predict(subset, batch_size=1, num_samples=24, device=DEVICE)
    logger.info(f"Forecast Shape: {forecasts.shape}")
    logger.info(f"Forecast Mean: {forecasts.mean()}")
    
    return _create_plot_image(forecasts, dataset, scalers, index, is_subset=True)

def quick_plot_action(content_base64: str) -> QuickPlotResponse:
    logger.info("Quick Plot Action Triggered")
    res = process_and_cache(content_base64)
    if res.error:
        return QuickPlotResponse(plot_base64="", warnings={}, error=res.error)
    
    handle = res.handle
    data = CACHE[handle]
    dataset = data['dataset']
    
    if len(dataset) == 0:
         return QuickPlotResponse(plot_base64="", warnings=res.warnings, error="No valid samples found")
         
    last_index = len(dataset) - 1
    logger.info(f"Quick Plot using last index: {last_index}")
    
    try:
        png_bytes = generate_plot_from_handle(handle, last_index)
        del CACHE[handle]
        
        return QuickPlotResponse(
            plot_base64=base64.b64encode(png_bytes).decode(),
            warnings=res.warnings
        )
    except Exception as e:
        logger.exception("Quick Plot Failed")
        return QuickPlotResponse(plot_base64="", warnings=res.warnings, error=f"Plotting failed: {e}")

def convert_to_unified_action(content_base64: str) -> ConvertResponse:
    try:
        unified_df = parse_csv_content(content_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            unified_df.write_csv(tmp.name)
            tmp_path = tmp.name
            
        with open(tmp_path, 'r') as f:
            csv_content = f.read()
            
        os.remove(tmp_path)
        return ConvertResponse(csv_content=csv_content)
        
    except Exception as e:
        return ConvertResponse(error=str(e))

def _create_plot_image(forecasts: np.ndarray, dataset, scalers, index: int, is_subset: bool = False) -> bytes:
    forecast_idx = 0 if is_subset else index
    current_forecast = forecasts[forecast_idx] # [12, 24]
    
    target_scaler = scalers['target']
    logger.info(f"Unscaling with: Min={target_scaler.min_}, Scale={target_scaler.scale_}")
    
    # Unscale forecast
    current_forecast = (current_forecast - target_scaler.min_) / target_scaler.scale_
    logger.info(f"Unscaled Forecast Mean: {current_forecast.mean()}")
    
    # Get True values (Future)
    # dataset.evalsample(index) returns SCALED target part for prediction window
    true_future_scaled = dataset.evalsample(index)
    true_future = dataset.evalsample(index).values().flatten()
    true_future = (true_future - target_scaler.min_) / target_scaler.scale_
    logger.info(f"True Future (Unscaled): {true_future}")
    
    # Get Past values
    # dataset[index] -> (past_target, ...) where past_target is scaled
    past_target_scaled = dataset[index][0]
    past_target = (past_target_scaled - target_scaler.min_) / target_scaler.scale_
    past_target = past_target.flatten()
    logger.info(f"Past Target Last 5: {past_target[-5:]}")
    
    # Plotting
    samples = current_forecast.T # [24, 12]
    
    fig = go.Figure()
    
    for point in range(samples.shape[1]):
        pts = samples[:, point]
        if np.std(pts) < 1e-6:
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
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([np.full_like(y_grid, point), np.full_like(y_grid, point - x * 0.4)[::-1]]),
                y=np.concatenate([y_grid, y_grid[::-1]]),
                fill='tonexty',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
        except:
            pass

    true_values = np.concatenate([past_target[-12:], true_future])
    
    fig.add_trace(go.Scatter(
        x=list(range(-12, 12)),
        y=true_values.tolist(),
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        name='True Values'
    ))
    
    median = np.quantile(samples, 0.5, axis=0)
    last_true_value = past_target[-1]
    median_with_anchor = [last_true_value] + median.tolist()
    median_x = [-1] + list(range(12))
    
    fig.add_trace(go.Scatter(
        x=median_x,
        y=median_with_anchor,
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
    
    return fig.to_image(format="png")
