import asyncio
import base64
import datetime
import logging
import json
import os
from typing import Dict, Optional, Any, Set

import polars as pl

# Dependencies from glurpc
import glurpc.logic as logic
from glurpc.state import (
    APIKeyManager, InferenceCache, PlotCache, TaskRegistry, DisconnectTracker,
    MINIMUM_DURATION_MINUTES, MAXIMUM_WANTED_DURATION, STEP_SIZE_MINUTES, 
    DEFAULT_INPUT_CHUNK_LENGTH, DEFAULT_OUTPUT_CHUNK_LENGTH
)
from glurpc.config import (
    MAX_CACHE_SIZE,
    ENABLE_API_KEYS,
    NUM_SAMPLES,
    DEFAULT_CONFIG
)
from glurpc.data_classes import PredictionsData, DartsDataset, DartsScaler, GluformerInferenceConfig
from glurpc.engine import ModelManager, BackgroundProcessor, INFERENCE_TIMEOUT
from glurpc.schemas import UnifiedResponse, QuickPlotResponse, ConvertResponse, FormattedWarnings

# --- Configuration & Logging ---

# Setup logging
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"glurpc_{timestamp}.log"
log_path = os.path.join(logs_dir, log_filename)

# Configure root logger ONCE
root_logger = logging.getLogger()
# Pad logger name to 19 chars for aligned output (longest is glurpc.engine.data = 17 chars)
formatter = logging.Formatter('%(asctime)s - %(name)-19s - %(levelname)-8s - %(message)s')

# Ensure FileHandler is present (even if other handlers exist, e.g. from pytest)
file_handler_exists = any(
    isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(log_path)
    for h in root_logger.handlers
)

if not file_handler_exists:
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

# Ensure StreamHandler is present if no handlers exist
if not root_logger.hasHandlers():
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

root_logger.setLevel(logging.DEBUG)

# Get glurpc.core logger (inherits from root, so no need to add handlers again)
logger = logging.getLogger("glurpc.core")
logger.setLevel(logging.DEBUG)

logger.info(f"Logging initialized to {log_path}")




# Initialize API key manager if enabled
if ENABLE_API_KEYS:
    api_key_manager = APIKeyManager()
    api_key_manager.load_api_keys()
    logger.info(f"API key authentication enabled with {api_key_manager.key_count} keys")
else:
    logger.info("API key authentication disabled")


# Legacy compatibility functions
def load_api_keys() -> None:
    """Legacy function for backward compatibility."""
    APIKeyManager().load_api_keys()


def verify_api_key(api_key: Optional[str]) -> bool:
    """Legacy function for backward compatibility."""
    return APIKeyManager().verify_api_key(api_key)


def is_restricted(endpoint_path: str) -> bool:
    """Legacy function for backward compatibility."""
    return APIKeyManager.is_restricted(endpoint_path)

# --- Action Handlers ---

async def convert_to_unified_action(content_base64: str) -> ConvertResponse:
    logger.info("Action: convert_to_unified_action started")
    
    try:
        result = await asyncio.to_thread(logic.convert_logic, content_base64)
        if result.error:
             logger.info(f"Action: convert_to_unified_action - error={result.error}")
        else:
             logger.info(f"Action: convert_to_unified_action completed successfully - csv_length={len(result.csv_content) if result.csv_content else 0}")
        return result
    except Exception as e:
        logger.error(f"Action: convert_to_unified_action - exception: {e}")
        raise
    
async def parse_and_schedule(
    content_base64: str, 
    maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION,
    inference_config: GluformerInferenceConfig = DEFAULT_CONFIG,
    force_calculate: bool = False,
    priority: int = 1,
    request_id: Optional[int] = None
) -> UnifiedResponse:
    logger.info(f"Action: parse_and_schedule started (force={force_calculate}, req_id={request_id})")
    inf_cache = InferenceCache()
    bg_processor = BackgroundProcessor()
    
    try:
        # 1. Convert to unified format (Parse Only)
        # We need handle and unified_df to proceed.
        handle, unified_df = await asyncio.to_thread(logic.get_handle_and_df, content_base64)
        logger.info(f"Action: parse_and_schedule - generated handle={handle[:8]}..., df_shape={unified_df.shape}")
        
        # 2. Analyze and Prepare (Get Warnings)
        inference_df, warning_flags, actual_input_samples = await asyncio.to_thread(
            logic.analyse_and_prepare_df,
            unified_df,
            MINIMUM_DURATION_MINUTES,
            maximum_wanted_duration
        )
        
        # 3. Calculate Expected Length using ACTUAL prepared data length
        # Use the real inference_df length instead of theoretical maximum_wanted_duration
        # This accounts for data quality issues, gaps, and filtering during preparation

        expected_dataset_len = logic.calculate_dataset_length_from_input(
            actual_input_samples,
            inference_config.input_chunk_length,
            inference_config.output_chunk_length
        )
        
        logger.debug(f"Expected dataset length: {expected_dataset_len} (glucose_only_samples={actual_input_samples}, max_duration={maximum_wanted_duration}m)")
        
        if expected_dataset_len <= 0:
             msg = f"Calculated expected length {expected_dataset_len} is non-positive. Duration too short?"
             logger.error(msg)
             return UnifiedResponse(error=msg)

        # 4. Check Cache & Pending
        if not force_calculate:
            # Check Cache
            data = await inf_cache.get(handle)
            if data and data.predictions is not None:
                cached_len = data.dataset_length
                if cached_len >= expected_dataset_len:
                    logger.info(f"Cache Hit for handle {handle[:8]} (len={cached_len} >= {expected_dataset_len})")
                    
                    # Check for warnings in cached data (if stored)
                    if data.warning_flags:
                        cached_warnings = logic.ProcessingWarning(data.warning_flags)
                    else:
                        cached_warnings = logic.ProcessingWarning(0)
                    
                    return UnifiedResponse(
                        handle=handle, 
                        total_samples=cached_len,
                        warnings=FormattedWarnings.from_flags(cached_warnings)
                    )
                else:
                    logger.info(f"Cache Hit but insufficient length: {cached_len} < {expected_dataset_len}. Recalculating.")
            
            # Check Pending
            pending_status = bg_processor.get_pending_status(handle)
            if pending_status:
                _, pending_len, _ = pending_status
                if pending_len >= expected_dataset_len:
                     logger.info(f"Pending task found for handle {handle[:8]} (len={pending_len} >= {expected_dataset_len}). Reusing.")
                     # Use currently calculated warnings for the response
                     return UnifiedResponse(
                        handle=handle,
                        total_samples=pending_len, 
                        warnings=FormattedWarnings.from_flags(warning_flags)
                     )

        # 5. Schedule Inference (Miss or Insufficient)
        logger.info(f"Scheduling inference for handle {handle[:8]} (expected_len={expected_dataset_len}, priority={priority}, req_id={request_id})")
        
        await bg_processor.enqueue_inference(
            handle, 
            inference_df=inference_df,
            warning_flags=warning_flags,
            expected_dataset_len=expected_dataset_len,
            inference_config=inference_config,
            priority=priority, 
            force_calculate=force_calculate,
            request_id=request_id
        )
            
        # Return response with expected length and computed warnings
        return UnifiedResponse(
            handle=handle,
            total_samples=expected_dataset_len,
            warnings=FormattedWarnings.from_flags(warning_flags)
        )
        
    except ValueError as e:
        # Expected validation errors (parsing failures, format issues, etc.)
        error_msg = str(e)
        logger.error(f"Parse and schedule failed: {error_msg}")
        return UnifiedResponse(error=error_msg)
    except Exception as e:
        # Unexpected errors - log with full traceback
        logger.exception(f"Parse and schedule failed with unexpected error: {e}")
        return UnifiedResponse(error=f"Internal error: {str(e)}")

async def generate_plot_from_handle(
    handle: str, 
    index: int, 
    force_calculate: bool = False,
    request_id: Optional[int] = None,
    disconnect_future: Optional[asyncio.Future] = None
) -> Dict[str, Any]:
    logger.info(f"Action: generate_plot_from_handle - handle={handle[:8]}..., index={index}, force={force_calculate}, req_id={request_id}")
    inf_cache = InferenceCache()
    plot_cache = PlotCache()
    bg_processor = BackgroundProcessor()
    
    # 1. Check Inference Cache
    data = await inf_cache.get(handle)
    
    # Check if there's a pending inference with potentially larger dataset
    pending_status = bg_processor.get_pending_status(handle)
    expected_len = None
    if pending_status:
        _, expected_len, _ = pending_status
        logger.debug(f"Pending inference for {handle[:8]} with expected_len={expected_len}")
    
    # If not in cache, it might be pending in background
    if not data:
        if bg_processor.is_processing(handle):
             logger.info(f"Action: generate_plot_from_handle - handle {handle[:8]} processing, waiting for result...")
             # Wait for calculation to complete
             try:
                 if request_id is not None:
                     await TaskRegistry().wait_for_result(handle, index, request_id, disconnect_future, timeout=INFERENCE_TIMEOUT)
                 else:
                     # Legacy fallback for background tasks without request_id
                     await TaskRegistry().wait_for_result(handle, index, 0, disconnect_future, timeout=INFERENCE_TIMEOUT)
             except asyncio.TimeoutError:
                 logger.error(f"Timeout waiting for result handle={handle[:8]} index={index}")
                 raise RuntimeError("Timeout waiting for inference result")
             except asyncio.CancelledError:
                 logger.info(f"Request cancelled for handle={handle[:8]} index={index} req_id={request_id}")
                 raise
             
             # Retry cache get
             data = await inf_cache.get(handle)
        else:
             logger.info(f"Action: generate_plot_from_handle - handle {handle[:8]}... not found in cache and not pending")
             # If we are here, handle is invalid or expired
             raise ValueError("Handle not found or expired")

    if not data:
        raise RuntimeError("Inference processing completed but data not found in cache")
    
    # Use dataset_length from pending inference if larger than cached data
    # This handles the case where old cache exists but new larger inference is pending
    dataset_len = data.dataset_length
    if expected_len is not None and expected_len > dataset_len:
        logger.info(f"Using pending expected_len={expected_len} instead of cached dataset_len={dataset_len} for validation")
        dataset_len = expected_len
    
    # Validate index against the effective dataset length
    if index > 0:
        raise ValueError(f"Positive indices not supported. Use negative indexing: 0 (last) to -{dataset_len - 1} (first)")
    if index < -(dataset_len - 1):
        raise ValueError(f"Index {index} out of range. Valid range: -{dataset_len - 1} to 0")

    # 2. Check Plot Cache (skip if force_calculate=True)
    if not force_calculate:
        plot_json_str = await plot_cache.get_plot(data.version, index)
        
        if plot_json_str:
            logger.info(f"Action: generate_plot_from_handle - plot cache hit")
            return json.loads(plot_json_str)
    else:
        logger.info(f"Action: generate_plot_from_handle - skipping plot cache due to force_calculate=True")
    
    # 3. Ensure Predictions Available and Sufficient
    # If cached data is too small (old cache) and new inference is pending, wait for it
    if data.predictions is None or (expected_len is not None and data.dataset_length < expected_len):
         if data.predictions is None:
             logger.warning(f"Action: generate_plot_from_handle - predictions pending for {handle[:8]} (cached), waiting...")
         else:
             logger.info(f"Action: generate_plot_from_handle - cached data insufficient (len={data.dataset_length} < expected={expected_len}), waiting for new inference...")
         
         # Wait for calculation to complete
         if request_id is not None:
             await TaskRegistry().wait_for_result(handle, index, request_id, disconnect_future, timeout=INFERENCE_TIMEOUT)
         else:
             await TaskRegistry().wait_for_result(handle, index, 0, disconnect_future, timeout=INFERENCE_TIMEOUT)
         
         # Refresh data
         data = await inf_cache.get(handle)
         if not data or data.predictions is None:
              raise RuntimeError("Inference failed or timed out")
         
         # Verify we now have sufficient data
         if index < -(data.dataset_length - 1):
             raise ValueError(f"Index {index} out of range even after new inference. Valid range: -{data.dataset_length - 1} to 0")

    # 4. Calculate Inline (Cache Miss)
    logger.info(f"Action: generate_plot_from_handle - plot cache miss, calculating inline")
    
    # Pass PredictionsData directly to calculate_plot_data
    plot_json_str, _plot_data = await asyncio.to_thread(logic.calculate_plot_data, data, index)
    
    # Store
    await plot_cache.update_plot(data.version, index, plot_json_str)
    
    plot_dict = json.loads(plot_json_str)
    logger.info(f"Action: generate_plot_from_handle completed")
    return plot_dict

async def quick_plot_action(
    content_base64: str, 
    force_calculate: bool = False,
    request_id: Optional[int] = None,
    disconnect_future: Optional[asyncio.Future] = None
) -> QuickPlotResponse:
    logger.info(f"Action: quick_plot_action started (force={force_calculate}, req_id={request_id})")
    warnings = {}
    handle = None
    last_index = 0 # 0 is LAST in new Negative Indexing Scheme
   
    try:        
        # 1. Process and cache data with minimum duration (for quick plot)
        logger.info(f"Action: quick_plot_action - starting processing and caching data...")
        response = await parse_and_schedule(
            content_base64, 
            maximum_wanted_duration=MINIMUM_DURATION_MINUTES,
            force_calculate=force_calculate,
            priority=0,  # High priority for interactive quick plots
            request_id=request_id
        )
        if response.error:
            logger.info(f"Action: quick_plot_action - error during parse_and_schedule: {response.error}")
            return QuickPlotResponse(plot_data={}, warnings=warnings, error=response.error)
        
        handle = response.handle
        warnings = response.warnings
        logger.info(f"Action: quick_plot_action - data processed for handle {handle[:8]}...")
        
        # 2. Proactively schedule full calculation in background (fire-and-forget)
        # Delayed by 5 seconds to avoid stealing models from the priority 0 request
        bg_processor = BackgroundProcessor()
        
        async def delayed_background_task():
            await asyncio.sleep(5)  # Let priority 0 acquire model first
            await parse_and_schedule(
                content_base64,
                maximum_wanted_duration=MAXIMUM_WANTED_DURATION,
                force_calculate=False,
                priority=3  # Low priority background task
            )
        
        task = asyncio.create_task(delayed_background_task())
        await bg_processor.register_background_task(task)
        logger.info(f"Action: quick_plot_action - scheduled background full calculation (priority 3, delayed 5s)")
 
        # 3. Generate plot using generate_plot_from_handle
        logger.info(f"Action: quick_plot_action - generating plot for last index={last_index}")
        plot_dict = await generate_plot_from_handle(handle, last_index, request_id=request_id, disconnect_future=disconnect_future)
        logger.info(f"Action: quick_plot_action completed successfully - plot_keys={list(plot_dict.keys())}")
        
        return QuickPlotResponse(
            plot_data=plot_dict,
            warnings=warnings
        )
        
    except ValueError as e:
        # Expected validation errors
        error_msg = str(e)
        logger.error(f"Quick Plot Failed: {error_msg}")
        return QuickPlotResponse(plot_data={}, warnings=warnings, error=error_msg)
    except asyncio.CancelledError:
        # Request was cancelled (disconnect)
        logger.info(f"Quick Plot cancelled for req_id={request_id}")
        return QuickPlotResponse(plot_data={}, warnings=warnings, error="Request cancelled")
    except Exception as e:
        # Unexpected errors - log with full traceback
        logger.exception(f"Quick Plot Failed with unexpected error: {e}")
        return QuickPlotResponse(plot_data={}, warnings=warnings, error=f"Internal error: {str(e)}")


async def get_model_manager():
    model_manager = ModelManager()
    if not model_manager.initialized:
        await model_manager.initialize()
    return model_manager
