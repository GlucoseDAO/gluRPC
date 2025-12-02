import logging
import os
import base64
import datetime
import asyncio
from typing import Dict, Optional, Any

import pandas as pd

# Dependencies from glurpc
from glurpc.schemas import UnifiedResponse, QuickPlotResponse, ConvertResponse
import glurpc.logic as logic
import glurpc.state as state
from glurpc.engine import MODEL_MANAGER, BACKGROUND_PROCESSOR

# --- Configuration & Logging ---

# Setup logging
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"glurpc_{timestamp}.log"
log_path = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("glurpc")
logger.setLevel(logging.DEBUG) 

# Clear existing handlers to avoid duplication if reloaded
if logger.hasHandlers():
    logger.handlers.clear()
    
file_handler = logging.FileHandler(log_path, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

logger.info(f"Logging initialized to {log_path}")

# Config
MAX_CACHE_SIZE = state.MAX_CACHE_SIZE

# --- Action Handlers ---

async def process_and_cache(content_base64: str) -> UnifiedResponse:
    logger.info("Action: process_and_cache started")
    try:
        loop = asyncio.get_running_loop()
        
        # 1. Parse and compute canonical Handle
        handle, unified_df = await loop.run_in_executor(None, logic.get_handle_and_df, content_base64)
        logger.info(f"Action: process_and_cache - generated handle={handle[:8]}..., df_shape={unified_df.shape}")
        
        # 2. Check Cache (Hit)
        async with state.DATA_CACHE_LOCK:
            cached = handle in state.DATA_CACHE
            
        if cached:
            logger.info(f"Cache Hit for handle {handle[:8]}")
            MODEL_MANAGER.increment_requests()
            return UnifiedResponse(handle=handle, warnings={'flags': 0, 'has_warnings': False, 'messages': []})
        
        # 3. Process Data (Miss)
        logger.info(f"Cache Miss for handle {handle[:8]}, processing data")
        result = await loop.run_in_executor(None, logic.create_dataset_from_df, unified_df)
        
        if 'error' in result:
            MODEL_MANAGER.increment_errors()
            logger.info(f"Action: process_and_cache - error={result['error']}")
            return UnifiedResponse(error=result['error'])
            
        dataset = result['dataset']
        logger.info(f"Action: process_and_cache - dataset created with {len(dataset)} samples")
        
        # 4. Store in DATA_CACHE
        async with state.DATA_CACHE_LOCK:
            if len(state.DATA_CACHE) >= MAX_CACHE_SIZE:
                # FIFO Eviction
                key_to_remove = next(iter(state.DATA_CACHE))
                logger.info(f"Cache eviction: removing handle {key_to_remove[:8]}...")
                del state.DATA_CACHE[key_to_remove]
                # Clean other caches too
                async with state.RESULT_CACHE_LOCK:
                    if key_to_remove in state.RESULT_CACHE:
                        del state.RESULT_CACHE[key_to_remove]
                async with state.FORECAST_CACHE_LOCK:
                    if key_to_remove in state.FORECAST_CACHE:
                        del state.FORECAST_CACHE[key_to_remove]

            state.DATA_CACHE[handle] = {
                'dataset': dataset,
                'scalers': {'target': result['scaler_target']},
                'model_config': result['model_config'],
                'warning_flags': result['warning_flags'],
                'timestamp': pd.Timestamp.now()
            }
            
        # 5. Enqueue Full Inference (Low Priority)
        BACKGROUND_PROCESSOR.enqueue_inference(handle, priority=1)
        logger.info(f"Action: process_and_cache - enqueued inference for handle {handle[:8]}...")
            
        MODEL_MANAGER.increment_requests()
        logger.info(f"Action: process_and_cache completed successfully - handle={handle[:8]}...")
        return UnifiedResponse(
            handle=handle,
            warnings=logic.format_warnings(result['warning_flags'])
        )
    except Exception as e:
        MODEL_MANAGER.increment_errors()
        logger.exception(f"Process and cache failed: {e}")
        raise e

async def generate_plot_from_handle(handle: str, index: int) -> bytes:
    logger.info(f"Action: generate_plot_from_handle - handle={handle[:8]}..., index={index}")
    # 1. Check Data Cache Existence
    async with state.DATA_CACHE_LOCK:
        if handle not in state.DATA_CACHE:
             logger.info(f"Action: generate_plot_from_handle - handle {handle[:8]}... not found in cache")
             raise ValueError("Handle not found or expired")
        dataset_len = len(state.DATA_CACHE[handle]['dataset'])

    if index < 0 or index >= dataset_len:
        logger.info(f"Action: generate_plot_from_handle - index {index} out of range (0-{dataset_len-1})")
        raise ValueError(f"Index {index} out of range")

    # 2. Check Result Cache
    async with state.RESULT_CACHE_LOCK:
        if handle in state.RESULT_CACHE and index in state.RESULT_CACHE[handle]:
             plot_data = state.RESULT_CACHE[handle][index]
             logger.info(f"Action: generate_plot_from_handle - plot data found in result cache")
        else:
             plot_data = None

    # 3. If missing, request and wait
    if not plot_data:
        logger.info(f"Cache miss for {handle[:8]} idx {index}, waiting for result...")
        
        # Enqueue High Priority Inference if not already running/done?
        # Background process manages full inference.
        # If plot is missing, it means either:
        # a) Inference hasn't run yet -> Enqueue High Prio Inference
        # b) Inference ran but Calc hasn't run -> Wait
        
        # Check Forecast Cache
        async with state.FORECAST_CACHE_LOCK:
            forecast_ready = handle in state.FORECAST_CACHE and index in state.FORECAST_CACHE[handle]
            
        if not forecast_ready:
             # If forecasts missing, we trigger Inference (High Prio)
             logger.info(f"Action: generate_plot_from_handle - enqueuing high-priority inference for handle {handle[:8]}...")
             BACKGROUND_PROCESSOR.enqueue_inference(handle, priority=0, indices=[index])
        
        # Wait for completion (Calc creates result)
        await state.wait_for_result(handle, index)
        logger.info(f"Action: generate_plot_from_handle - result ready after wait")
        
        # Fetch result
        async with state.RESULT_CACHE_LOCK:
             plot_data = state.RESULT_CACHE.get(handle, {}).get(index)
             
        if not plot_data:
            logger.info(f"Action: generate_plot_from_handle - calculation failed or returned empty")
            raise RuntimeError("Calculation failed or returned empty")

    # 4. Render
    logger.debug(f"Rendering plot for {handle[:8]} idx {index}")
    loop = asyncio.get_running_loop()
    png_bytes = await loop.run_in_executor(None, logic.render_plot, plot_data)
    logger.info(f"Action: generate_plot_from_handle completed - png_size={len(png_bytes)} bytes")
    return png_bytes

async def quick_plot_action(content_base64: str) -> QuickPlotResponse:
    logger.info("Action: quick_plot_action started")
    
    try:
        loop = asyncio.get_running_loop()
        
        # 1. Parse & Handle
        handle, unified_df = await loop.run_in_executor(None, logic.get_handle_and_df, content_base64)
        logger.info(f"Action: quick_plot_action - generated handle={handle[:8]}..., df_shape={unified_df.shape}")
        
        # 2. Process Data (Transient/Cache check?)
        async with state.DATA_CACHE_LOCK:
             cached = handle in state.DATA_CACHE
        
        dataset = None
        warnings = {}
        
        if not cached:
            logger.info(f"Action: quick_plot_action - processing new data for handle {handle[:8]}...")
            result = await loop.run_in_executor(None, logic.create_dataset_from_df, unified_df)
            if 'error' in result:
                logger.info(f"Action: quick_plot_action - error during dataset creation: {result['error']}")
                return QuickPlotResponse(plot_base64="", warnings={}, error=result['error'])
            dataset = result['dataset']
            warnings = logic.format_warnings(result['warning_flags'])
            logger.info(f"Action: quick_plot_action - dataset created with {len(dataset)} samples")
            
            # Store Transient
            async with state.DATA_CACHE_LOCK:
                state.DATA_CACHE[handle] = {
                    'dataset': dataset,
                    'scalers': {'target': result['scaler_target']},
                    'model_config': result['model_config'],
                    'warning_flags': result['warning_flags'],
                    'timestamp': pd.Timestamp.now()
                }
        else:
             logger.info(f"Action: quick_plot_action - using cached data for handle {handle[:8]}...")
             async with state.DATA_CACHE_LOCK:
                 dataset = state.DATA_CACHE[handle]['dataset']
        
        last_index = len(dataset) - 1
        logger.info(f"Action: quick_plot_action - generating plot for last index={last_index}")
        
        # 3. Enqueue & Wait (High Prio)
        BACKGROUND_PROCESSOR.enqueue_inference(handle, priority=0, indices=[last_index])
        await state.wait_for_result(handle, last_index)
        logger.info(f"Action: quick_plot_action - inference completed")
        
        # 4. Render
        async with state.RESULT_CACHE_LOCK:
             plot_data = state.RESULT_CACHE.get(handle, {}).get(last_index)
             
        if not plot_data:
            logger.info(f"Action: quick_plot_action - calculation failed, no plot data")
            raise RuntimeError("Calculation failed")
            
        png_bytes = await loop.run_in_executor(None, logic.render_plot, plot_data)
        logger.info(f"Action: quick_plot_action completed successfully - png_size={len(png_bytes)} bytes")
        
        return QuickPlotResponse(
            plot_base64=base64.b64encode(png_bytes).decode(),
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Quick Plot Failed: {e}")
        MODEL_MANAGER.increment_errors()
        return QuickPlotResponse(plot_base64="", warnings={}, error=str(e))
        
    finally:
        # 6. Cleanup if it was transient
        if not cached:
             logger.info(f"Action: quick_plot_action - cleaning up transient cache for handle {handle[:8]}...")
             async with state.DATA_CACHE_LOCK:
                if handle in state.DATA_CACHE:
                    del state.DATA_CACHE[handle]
             async with state.RESULT_CACHE_LOCK:
                if handle in state.RESULT_CACHE:
                    del state.RESULT_CACHE[handle]
             async with state.FORECAST_CACHE_LOCK:
                if handle in state.FORECAST_CACHE:
                    del state.FORECAST_CACHE[handle]

async def convert_to_unified_action(content_base64: str) -> ConvertResponse:
    logger.info("Action: convert_to_unified_action started")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, logic.convert_logic, content_base64)
        MODEL_MANAGER.increment_requests()
        if result.error:
             MODEL_MANAGER.increment_errors()
             logger.info(f"Action: convert_to_unified_action - error={result.error}")
        else:
             logger.info(f"Action: convert_to_unified_action completed successfully - csv_length={len(result.csv_content) if result.csv_content else 0}")
        return result
    except Exception as e:
        MODEL_MANAGER.increment_errors()
        logger.error(f"Action: convert_to_unified_action - exception: {e}")
        raise

async def get_model_manager():
    if not MODEL_MANAGER.initialized:
        await MODEL_MANAGER.initialize()
    return MODEL_MANAGER
