import asyncio
import base64
import datetime
import logging
import os
from typing import Dict, Optional, Any, Set

import pandas as pd

# Dependencies from glurpc
import glurpc.logic as logic
import glurpc.state as state
from glurpc.engine import MODEL_MANAGER, BACKGROUND_PROCESSOR
from glurpc.schemas import UnifiedResponse, QuickPlotResponse, ConvertResponse

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

# --- API Key Management ---

# Enable/disable API key authentication
ENABLE_API_KEYS = os.getenv("ENABLE_API_KEYS", "False").lower() in ("true", "1", "yes")

API_KEYS: Set[str] = set()

def load_api_keys() -> None:
    """Load API keys from api_keys_list file."""
    global API_KEYS
    api_keys_file = os.path.join(os.getcwd(), "api_keys_list")
    
    if not os.path.exists(api_keys_file):
        logger.warning(f"API keys file not found at {api_keys_file}, no keys loaded")
        return
    
    try:
        with open(api_keys_file, 'r') as f:
            keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            API_KEYS = set(keys)
            logger.info(f"Loaded {len(API_KEYS)} API keys from {api_keys_file}")
    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
        API_KEYS = set()

def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify if the provided API key is valid."""
    if not api_key:
        return False
    return api_key in API_KEYS

def is_restricted(endpoint_path: str) -> bool:
    """Determine if an endpoint requires API key authentication."""
    unrestricted_endpoints = {"/health", "/convert_to_unified"}
    return endpoint_path not in unrestricted_endpoints

# Load API keys on module initialization if enabled
if ENABLE_API_KEYS:
    load_api_keys()
    logger.info(f"API key authentication enabled with {len(API_KEYS)} keys")
else:
    logger.info("API key authentication disabled")

# --- Action Handlers ---

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
    
async def process_and_cache(content_base64: str, maximum_wanted_duration: int = state.MAXIMUM_WANTED_DURATION) -> UnifiedResponse:
    logger.info("Action: process_and_cache started")
    try:
        loop = asyncio.get_running_loop()
        
        # 1. Convert to unified format (reusing convert_to_unified_action logic)
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
        result = await loop.run_in_executor(
            None, 
            logic.create_dataset_from_df, 
            unified_df, 
            state.MINIMUM_DURATION_MINUTES, 
            maximum_wanted_duration) #min + 1 step
        
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

            state.DATA_CACHE[handle] = {
                'dataset': dataset,
                'scalers': {'target': result['scaler_target']},
                'model_config': result['model_config'],
                'warning_flags': result['warning_flags'],
                'timestamp': pd.Timestamp.now(),
                'forecasts': None,  # Initialize as None, will be (N, 12, 10) array
                'results': [None] * len(dataset) # Initialize list for results
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
        
        data_entry = state.DATA_CACHE[handle]
        dataset_len = len(data_entry['dataset'])
        results_list = data_entry['results']
        forecasts_array = data_entry['forecasts']

    if index < 0 or index >= dataset_len:
        logger.info(f"Action: generate_plot_from_handle - index {index} out of range (0-{dataset_len-1})")
        raise ValueError(f"Index {index} out of range")

    # 2. Check Result Cache (List in Data Cache)
    # We trust list atomic read/write for existence check?
    # To be safe, we can use DATA_CACHE_LOCK or just rely on GIL atomic assignment for single item.
    # But since we have a list, random access is safe.
    plot_data = results_list[index]

    if plot_data:
        logger.info(f"Action: generate_plot_from_handle - plot data found in result cache")
    else:
        logger.info(f"Cache miss for {handle[:8]} idx {index}, waiting for result...")
        
        # Check if forecasts are ready
        forecast_ready = forecasts_array is not None
            
        if not forecast_ready:
             # If forecasts missing, we trigger Inference (High Prio)
             logger.info(f"Action: generate_plot_from_handle - enqueuing high-priority inference for handle {handle[:8]}...")
             BACKGROUND_PROCESSOR.enqueue_inference(handle, priority=0, indices=[index])
        
        # Wait for completion (Calc creates result)
        await state.wait_for_result(handle, index)
        logger.info(f"Action: generate_plot_from_handle - result ready after wait")
        
        # Fetch result
        # Re-acquire to be safe against eviction
        async with state.DATA_CACHE_LOCK:
             if handle in state.DATA_CACHE:
                 plot_data = state.DATA_CACHE[handle]['results'][index]
             else:
                 plot_data = None
             
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
    warnings = {}
    base64_plot = ""
    handle = None
    last_index = 0 # TODO: process_and_cache should return the last index and cache_hit
   
    try:        
        # 1. Process and cache data with minimum duration (for quick plot)
        logger.info(f"Action: quick_plot_action - starting processing and caching data...")
        response = await process_and_cache(
            content_base64, 
            maximum_wanted_duration=state.MINIMUM_DURATION_MINUTES + state.STEP_SIZE_MINUTES
        )
        if response.error:
            logger.info(f"Action: quick_plot_action - error during process_and_cache: {response.error}")
            return QuickPlotResponse(plot_base64=base64_plot, warnings=warnings, error=response.error)
        
        handle = response.handle
        warnings = response.warnings
        logger.info(f"Action: quick_plot_action - data processed for handle {handle[:8]}...")
        
 
        logger.info(f"Action: quick_plot_action - generating plot for last index={last_index}")
        
        # 4. Generate plot using generate_plot_from_handle
        png_bytes = await generate_plot_from_handle(handle, last_index)
        logger.info(f"Action: quick_plot_action completed successfully - png_size={len(png_bytes)} bytes")
        
        return QuickPlotResponse(
            plot_base64=base64.b64encode(png_bytes).decode(),
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Quick Plot Failed: {e}")
        MODEL_MANAGER.increment_errors()
        return QuickPlotResponse(plot_base64="", warnings=warnings, error=str(e))


async def get_model_manager():
    if not MODEL_MANAGER.initialized:
        await MODEL_MANAGER.initialize()
    return MODEL_MANAGER
