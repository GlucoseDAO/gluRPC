import asyncio
import base64
import datetime
import logging
import os
from typing import Dict, Optional, Any, Set

import polars as pl

# Dependencies from glurpc
import glurpc.logic as logic
from glurpc.state import (
    SingletonMeta, StateManager, DataCache, TaskRegistry,
    MINIMUM_DURATION_MINUTES, MAXIMUM_WANTED_DURATION, STEP_SIZE_MINUTES,
    RESULT_SCHEMA
)
from glurpc.config import (
    MAX_CACHE_SIZE,
    ENABLE_API_KEYS
)
from glurpc.engine import ModelManager, BackgroundProcessor
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

# Configure calc logger to be less verbose (INFO level only)
calc_logger = logging.getLogger("glurpc.logic.calc")
calc_logger.setLevel(logging.INFO)

# --- API Key Management ---

class APIKeyManager(metaclass=SingletonMeta):
    """
    Singleton manager for API key authentication.
    """
    def __init__(self):
        self._keys: Set[str] = set()
        self._loaded = False
    
    def load_api_keys(self) -> None:
        """Load API keys from api_keys_list file."""
        if self._loaded:
            return
            
        api_keys_file = os.path.join(os.getcwd(), "api_keys_list")
        
        if not os.path.exists(api_keys_file):
            logger.warning(f"API keys file not found at {api_keys_file}, no keys loaded")
            return
        
        try:
            with open(api_keys_file, 'r') as f:
                keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                self._keys = set(keys)
                self._loaded = True
                logger.info(f"Loaded {len(self._keys)} API keys from {api_keys_file}")
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            self._keys = set()
    
    def verify_api_key(self, api_key: Optional[str]) -> bool:
        """Verify if the provided API key is valid."""
        if not api_key:
            return False
        return api_key in self._keys
    
    @staticmethod
    def is_restricted(endpoint_path: str) -> bool:
        """Determine if an endpoint requires API key authentication."""
        unrestricted_endpoints = {"/health", "/convert_to_unified"}
        return endpoint_path not in unrestricted_endpoints
    
    @property
    def key_count(self) -> int:
        """Return the number of loaded API keys."""
        return len(self._keys)


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
    loop = asyncio.get_running_loop()
    model_manager = ModelManager()
    
    try:
        result = await loop.run_in_executor(None, logic.convert_logic, content_base64)
        model_manager.increment_requests()
        if result.error:
             model_manager.increment_errors()
             logger.info(f"Action: convert_to_unified_action - error={result.error}")
        else:
             logger.info(f"Action: convert_to_unified_action completed successfully - csv_length={len(result.csv_content) if result.csv_content else 0}")
        return result
    except Exception as e:
        model_manager.increment_errors()
        logger.error(f"Action: convert_to_unified_action - exception: {e}")
        raise
    
async def process_and_cache(
    content_base64: str, 
    maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION,
    force_calculate: bool = False
) -> UnifiedResponse:
    logger.info(f"Action: process_and_cache started (force={force_calculate})")
    model_manager = ModelManager()
    data_cache = DataCache()
    bg_processor = BackgroundProcessor()
    
    try:
        loop = asyncio.get_running_loop()
        
        # 1. Convert to unified format (reusing convert_to_unified_action logic)
        handle, unified_df = await loop.run_in_executor(None, logic.get_handle_and_df, content_base64)
        logger.info(f"Action: process_and_cache - generated handle={handle[:8]}..., df_shape={unified_df.shape}")
        
        # Extract time range (needed for cache storage regardless of force_calculate)
        start_time, end_time = logic.get_time_range(unified_df)
        
        if not force_calculate:
            # 2. Check Cache (Hit)
            cached = await data_cache.contains(handle)
                
            if cached:
                logger.info(f"Cache Hit for handle {handle[:8]}")
                # Retrieve dataset length from cache
                cached_data = await data_cache.get(handle)
                dataset_len = len(cached_data['dataset']) if cached_data and 'dataset' in cached_data else None
                model_manager.increment_requests()
                return UnifiedResponse(
                    handle=handle, 
                    total_samples=dataset_len,
                    warnings={'flags': 0, 'has_warnings': False, 'messages': []}
                )

            # 2.1 Check Subset Match (Superset)
            if start_time and end_time:
                superset_handle = await data_cache.find_superset(start_time, end_time)
                if superset_handle:
                    logger.info(f"Subset Match found! Using superset {superset_handle[:8]} for {handle[:8]}")
                    # Retrieve dataset length from superset cache
                    superset_data = await data_cache.get(superset_handle)
                    superset_len = len(superset_data['dataset']) if superset_data and 'dataset' in superset_data else None
                    model_manager.increment_requests()
                    return UnifiedResponse(
                        handle=superset_handle,
                        total_samples=superset_len,
                        warnings={'flags': 0, 'has_warnings': False, 'messages': [f"Used cached superset {superset_handle[:8]}"]}
                    )
        else:
             logger.info(f"Force calculate enabled for {handle[:8]}, skipping cache/subset check")
        
        # 3. Process Data (Miss)
        logger.info(f"Cache Miss for handle {handle[:8]}, processing data")
        result = await loop.run_in_executor(
            None, 
            logic.create_dataset_from_df, 
            unified_df, 
            MINIMUM_DURATION_MINUTES, 
            maximum_wanted_duration) #min + 1 step
        
        if 'error' in result:
            model_manager.increment_errors()
            logger.info(f"Action: process_and_cache - error={result['error']}")
            return UnifiedResponse(error=result['error'])
            
        dataset = result['dataset']
        logger.info(f"Action: process_and_cache - dataset created with {len(dataset)} samples")
        
        # 4. Store in DATA_CACHE
        dataset_len = len(dataset)
        # Negative Indexing: 0 is last.
        # Range: [-(dataset_len - 1), ..., 0]
        indices = list(range(-(dataset_len - 1), 1))

        await data_cache.set(handle, {
            'dataset': dataset,
            'scalers': {'target': result['scaler_target']},
            'model_config': result['model_config'],
            'warning_flags': result['warning_flags'],
            'timestamp': datetime.datetime.now(),
            'start_time': start_time,
            'end_time': end_time,
            'data_df': pl.DataFrame(
                {
                    "index": indices,
                    "forecast": [None] * len(dataset),
                    "true_values_x": [None] * len(dataset),
                    "true_values_y": [None] * len(dataset),
                    "median_x": [None] * len(dataset),
                    "median_y": [None] * len(dataset),
                    "fan_charts": [None] * len(dataset),
                    "is_calculated": [False] * len(dataset)
                },
                schema=RESULT_SCHEMA
            )
        })
            
        # 5. Enqueue Full Inference (Low Priority)
        bg_processor.enqueue_inference(handle, priority=1)
        logger.info(f"Action: process_and_cache - enqueued inference for handle {handle[:8]}...")
            
        model_manager.increment_requests()
        logger.info(f"Action: process_and_cache completed successfully - handle={handle[:8]}, total_samples={dataset_len}")
        return UnifiedResponse(
            handle=handle,
            total_samples=dataset_len,
            warnings=logic.format_warnings(result['warning_flags'])
        )
    except Exception as e:
        model_manager.increment_errors()
        logger.exception(f"Process and cache failed: {e}")
        return UnifiedResponse(error=str(e))

async def generate_plot_from_handle(handle: str, index: int) -> bytes:
    logger.info(f"Action: generate_plot_from_handle - handle={handle[:8]}..., index={index}")
    data_cache = DataCache()
    bg_processor = BackgroundProcessor()
    
    # 1. Check Data Cache Existence
    data_entry = await data_cache.get(handle)
    if not data_entry:
        logger.info(f"Action: generate_plot_from_handle - handle {handle[:8]}... not found in cache")
        raise ValueError("Handle not found or expired")
    
    dataset_len = len(data_entry['dataset'])
    result_df = data_entry['data_df']

    # Support Legacy Positive Indices (convert to negative)
    # 0 is last (New Scheme). Positive i means i-th from start (Legacy).
    # i-th from start = i - (dataset_len - 1)
    if index > 0:
         logger.debug(f"Mapping legacy positive index {index} to negative index")
         index = index - (dataset_len - 1)

    # Valid range: [-(dataset_len - 1), 0]
    if index > 0 or index < -(dataset_len - 1):
        logger.info(f"Action: generate_plot_from_handle - index {index} out of range (dataset_len={dataset_len})")
        raise ValueError(f"Index {index} out of range")

    # 2. Check Result Cache (Polars DataFrame)
    # Access row to check status
    # Positional 0 is index -(N-1). Positional N-1 is index 0.
    # pos = index + (N - 1)
    pos_index = index + (dataset_len - 1)
    row_data = result_df.row(pos_index, named=True)
    is_calculated = row_data['is_calculated']

    if is_calculated:
        logger.info(f"Action: generate_plot_from_handle - plot data found in result cache")
        plot_data = row_data
    else:
        logger.info(f"Cache miss for {handle[:8]} idx {index}, waiting for result...")
        
        # Check if forecasts are ready
        forecast_ready = row_data['forecast'] is not None
            
        if not forecast_ready:
             # If forecasts missing, we trigger Inference (High Prio)
             logger.info(f"Action: generate_plot_from_handle - enqueuing high-priority inference for handle {handle[:8]}...")
             bg_processor.enqueue_inference(handle, priority=0, indices=[index])
        
        # Wait for completion (Calc creates result)
        await TaskRegistry().wait_for_result(handle, index)
        logger.info(f"Action: generate_plot_from_handle - result ready after wait")
        
        # Fetch result
        # Re-acquire to be safe against eviction
        data_entry = await data_cache.get(handle)
        if data_entry:
            result_df = data_entry['data_df']
            # Recalculate positional index after re-acquiring
            dataset_len = len(data_entry['dataset'])
            pos_index = index + (dataset_len - 1)
            logger.debug(f"Fetching row: index={index}, dataset_len={dataset_len}, pos_index={pos_index}")
            row_data = result_df.row(pos_index, named=True)
            if row_data['is_calculated']:
                logger.debug(f"Row data is_calculated=True")
                plot_data = row_data
            else:
                logger.debug(f"Row data is_calculated=False")
                plot_data = None
        else:
            logger.debug("Data entry not found after re-acquisition")
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

async def quick_plot_action(content_base64: str, force_calculate: bool = False) -> QuickPlotResponse:
    logger.info(f"Action: quick_plot_action started (force={force_calculate})")
    warnings = {}
    base64_plot = ""
    handle = None
    last_index = 0 # 0 is LAST in new Negative Indexing Scheme
    model_manager = ModelManager()
   
    try:        
        # 1. Process and cache data with minimum duration (for quick plot)
        logger.info(f"Action: quick_plot_action - starting processing and caching data...")
        response = await process_and_cache(
            content_base64, 
            maximum_wanted_duration=MINIMUM_DURATION_MINUTES + STEP_SIZE_MINUTES,
            force_calculate=force_calculate
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
        model_manager.increment_errors()
        return QuickPlotResponse(plot_base64="", warnings=warnings, error=str(e))


async def get_model_manager():
    model_manager = ModelManager()
    if not model_manager.initialized:
        await model_manager.initialize()
    return model_manager
