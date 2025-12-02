import logging
import asyncio
import time
import threading
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import torch
import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download


# Dependencies from glurpc
from glurpc.data_classes import GluformerModelConfig, GluformerInferenceConfig
from glurpc.config import NUM_COPIES_PER_DEVICE, BACKGROUND_WORKERS_COUNT, BATCH_SIZE, NUM_SAMPLES
from glurpc.logic import ModelState, load_model, run_inference_full, calculate_plot_data, SamplingDatasetInferenceDual
from glurpc.state import SingletonMeta, StateManager, DataCache, TaskRegistry


logger = logging.getLogger("glurpc")

# --- Engine-specific Dynamic Configuration ---
# These are determined at runtime based on available hardware

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
"""Device to use for inference (detected dynamically)."""


def get_total_copies() -> int:
    """Calculate total number of model copies based on device."""
    if DEVICE == "cpu":
        return 1
    return torch.cuda.device_count() * NUM_COPIES_PER_DEVICE

NUM_COPIES: int = get_total_copies()
"""Total number of model copies across all devices."""


class InferenceWrapper:
    """
    Wrapper for inference that handles model loading and validation.
    Ensures the loaded model matches the required configuration for the dataset.
    """
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model_state: Optional[ModelState] = None
        self._lock = threading.Lock()

    def load_if_needed(self, required_config: GluformerModelConfig):
        """
        Checks if the current loaded model matches the required configuration.
        If not, reloads the model.
        """
        # Use a lock to ensure thread safety during reload
        with self._lock:
            if self.model_state is not None:
                current_config, _ = self.model_state
                if current_config == required_config:
                    return

            # Reload needed
            logger.info("Model config mismatch or not loaded. Reloading model...")
            self.model_state = load_model(required_config, self.model_path, self.device)

    def run_inference(
        self, 
        dataset: SamplingDatasetInferenceDual, 
        required_config: GluformerModelConfig,
        batch_size: int,
        num_samples: int
    ) -> Dict[int, np.ndarray]:
        """
        Runs inference, ensuring model is loaded with correct config.
        """
        self.load_if_needed(required_config)
        
        with self._lock:
            # We hold the lock to ensure the model isn't swapped out from under us
            current_state = self.model_state
        
        return run_inference_full(
            dataset=dataset,
            model_config=required_config,
            model_state=current_state,
            batch_size=batch_size,
            num_samples=num_samples,
            device=self.device
        )



class ModelManager(metaclass=SingletonMeta):
    """
    Singleton manager for ML model instances and inference requests.
    """
    def __init__(self):
        self.models: List[InferenceWrapper] = []
        self.queue = asyncio.Queue()
        self.initialized = False
        self._init_lock = asyncio.Lock()
        
        # Stats
        self._fulfillment_times: List[float] = []
        self._max_stats_history = 1000
        self._total_requests = 0
        self._total_errors = 0
        
    def increment_requests(self):
        self._total_requests += 1
        
    def increment_errors(self):
        self._total_errors += 1
    
    async def initialize(self, model_name: str = "gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth"):
        if self.initialized:
            return

        async with self._init_lock:
            if self.initialized:
                return
                
            logger.info(f"Initializing ModelManager with model: {model_name}")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_models_sync, model_name)
            
            for model in self.models:
                self.queue.put_nowait(model)
            
            self.initialized = True
            logger.info(f"ModelManager initialized with {len(self.models)} models")

    def _load_models_sync(self, model_name: str):
        try:
            config = GluformerInferenceConfig()
            repo_id = "Livia-Zaharia/gluformer_models"
            model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
            
            # Initial config for warm-up (uses defaults)
            initial_config = GluformerModelConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_fcn=config.d_fcn,
                num_enc_layers=config.num_enc_layers,
                num_dec_layers=config.num_dec_layers,
                len_seq=config.input_chunk_length,
                label_len=config.input_chunk_length // 3,
                len_pred=config.output_chunk_length,
                num_dynamic_features=6, # Default
                num_static_features=1, # Default
                r_drop=config.r_drop,
                activ=config.activ,
                distil=config.distil
            )
            
            self.models = []
            for i in range(NUM_COPIES):
                logger.info(f"Loading model copy {i+1}/{NUM_COPIES}")
                
                if DEVICE == "cuda":
                    device_id = i % torch.cuda.device_count()
                    device = f"cuda:{device_id}"
                else:
                    device = "cpu"
                
                wrapper = InferenceWrapper(str(model_path), device)
                # Warm up by loading initial config
                wrapper.load_if_needed(initial_config)
                self.models.append(wrapper)
            
            logger.info("All model copies loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @asynccontextmanager
    async def acquire(self, requested_copies: int = 1):
        if not self.initialized:
             logger.error("Acquire called before initialization!")
             raise RuntimeError("Models not initialized")
            
        start_time = time.time()
        self.increment_requests()
        
        num_to_acquire = min(requested_copies, len(self.models))
        if num_to_acquire <= 0: num_to_acquire = 1
        
        acquired_models = []
        try:
            for _ in range(num_to_acquire):
                model = await self.queue.get()
                acquired_models.append(model)
            
            yield acquired_models
        except Exception as e:
             logger.error(f"Error acquiring models: {e}")
             self.increment_errors()
             raise
        finally:
            for model in acquired_models:
                self.queue.put_nowait(model)
            
            duration_ms = (time.time() - start_time) * 1000
            self._fulfillment_times.append(duration_ms)
            if len(self._fulfillment_times) > self._max_stats_history:
                self._fulfillment_times.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        avg_time = 0.0
        if self._fulfillment_times:
            avg_time = sum(self._fulfillment_times) / len(self._fulfillment_times)
            
        vmem_mb = 0.0
        if DEVICE == "cuda":
            try:
                vmem_bytes = 0
                for i in range(torch.cuda.device_count()):
                    vmem_bytes += torch.cuda.memory_allocated(i)
                vmem_mb = vmem_bytes / (1024 * 1024)
            except Exception:
                pass
        
        return {
            "queue_length": self.queue.qsize(),
            "avg_fulfillment_time_ms": avg_time,
            "vmem_usage_mb": vmem_mb,
            "device": DEVICE,
            "total_requests_processed": self._total_requests,
            "total_errors": self._total_errors
        }

class BackgroundProcessor(metaclass=SingletonMeta):
    """
    Singleton processor for managing background inference and calculation workers.
    """
    def __init__(self):
        # Inference Queue Item: (priority, neg_timestamp, handle)
        # No index! Inference is FULL or nothing.
        self.inference_queue = asyncio.PriorityQueue()
        
        # Calculation Queue Item: (priority, neg_timestamp, handle, index, forecasts)
        self.calc_queue = asyncio.PriorityQueue()
        
        self.inference_workers = []
        self.calc_workers = []
        self.running = False
        
    async def start(self, num_inference_workers: int = NUM_COPIES, num_calc_workers: int = BACKGROUND_WORKERS_COUNT):
        if self.running:
            return
            
        self.running = True
        StateManager().reset_shutdown()
        logger.info(f"Starting {num_inference_workers} inference workers and {num_calc_workers} calculation workers...")
        
        for i in range(num_inference_workers):
            task = asyncio.create_task(self._inference_worker_loop(i))
            self.inference_workers.append(task)
            
        for i in range(num_calc_workers):
            task = asyncio.create_task(self._calc_worker_loop(i))
            self.calc_workers.append(task)
            
    def stop(self):
        StateManager().start_shutdown()
        self.running = False
        logger.info("Shutdown flag set, waiting for workers to exit gracefully...")
        for task in self.inference_workers + self.calc_workers:
            task.cancel()
        logger.info("Background workers stopped")

    def enqueue_inference(self, handle: str, priority: int = 1, indices: Optional[List[int]] = None):
        """
        Enqueue a task for inference.
        priority: 0 for High (Interactive), 1 for Low (Background)
        indices: Specific indices to prioritize calculation for. If None, calculates all normally.
        """
        neg_timestamp = -time.time()
        item = (priority, neg_timestamp, handle, indices)
        self.inference_queue.put_nowait(item)
        logger.debug(f"Enqueued INFERENCE: handle={handle[:8]} prio={priority} indices={'ALL' if indices is None else indices}")

    def enqueue_calc(self, handle: str, index: int, forecasts: Any, priority: int, neg_timestamp: float):
        item = (priority, neg_timestamp, handle, index, forecasts)
        self.calc_queue.put_nowait(item)
        # logger.debug(f"Enqueued CALC: handle={handle[:8]} idx={index} prio={priority}") 

    async def _inference_worker_loop(self, worker_id: int):
        logger.info(f"InfWorker {worker_id} started")
        state_mgr = StateManager()
        data_cache = DataCache()
        
        while self.running and not state_mgr.shutdown_started:
            try:
                priority, neg_timestamp, handle, indices = await self.inference_queue.get()
                
                try:
                    # 1. Check Data Cache
                    data = await data_cache.get(handle)
                    
                    if not data:
                        logger.debug(f"InfWorker {worker_id}: Handle {handle[:8]} not found. Dropping.")
                        continue
                    
                    dataset = data['dataset']
                    required_config = data.get('model_config')
                    
                    if not required_config:
                        logger.error(f"InfWorker {worker_id}: Model config missing for {handle[:8]}")
                        continue
                    
                    # 2. Check Forecast Cache (in DATA_CACHE via Polars DF)
                    # We check if the 'forecast' column is populated
                    data_df = data.get('data_df')
                    if data_df is None:
                         continue
                    
                    # Check if forecasts are present (assuming if first is present, all are)
                    # or check if any is null?
                    # Inference fills all at once.
                    is_forecast_missing = data_df['forecast'][0] is None
                    
                    full_forecasts = None

                    # 3. Run Inference (Full) if needed
                    if is_forecast_missing:
                        total_len = len(dataset)
                        logger.info(f"InfWorker {worker_id}: Running FULL inference for {handle[:8]} ({total_len} items)")
                    
                        async with ModelManager().acquire(1) as wrappers:
                            wrapper = wrappers[0]
                            loop = asyncio.get_running_loop()
                            full_forecasts_array = await loop.run_in_executor(
                                None, 
                                wrapper.run_inference, 
                                dataset, 
                                required_config,
                                BATCH_SIZE,
                                NUM_SAMPLES
                            )
                            
                            # Flatten forecasts for Polars storage: (N, 12, 10) -> (N, 120)
                            # Convert to list of lists
                            flattened = full_forecasts_array.reshape(full_forecasts_array.shape[0], -1).tolist()
                            
                            # Update cache
                            # Acquire lock to write back forecasts
                            async with data_cache.lock:
                                 if data_cache.contains_sync(handle):
                                     # Update the DataFrame column 'forecast'
                                     cached_data = data_cache.get_sync(handle)
                                     current_df = cached_data['data_df']
                                     cached_data['data_df'] = current_df.with_columns(
                                         pl.Series("forecast", flattened)
                                     )
                                     full_forecasts = full_forecasts_array
                    else:
                        # Forecasts already exist, retrieve them for enqueuing calc
                        # We need them as numpy array (N, 12, 10)
                        # Retrieve from Polars
                        flattened_lists = data_df['forecast'].to_list()
                        full_forecasts = np.array(flattened_lists).reshape(len(flattened_lists), 12, 10)

                    # 4. Enqueue Calculation
                    if indices is not None:
                        # Specific indices requested (High Prio)
                        logger.debug(f"InfWorker {worker_id}: Enqueuing specific indices: {indices}")
                        for idx in indices:
                             # Check bounds instead of dict key presence
                             if 0 <= idx < len(full_forecasts):
                                 # Extract (12, 10) array for this index
                                 # full_forecasts is (N, 12, 10)
                                 self.enqueue_calc(handle, idx, full_forecasts[idx], priority, neg_timestamp)
                    else:
                        # Background processing - enqueue all
                        logger.debug(f"InfWorker {worker_id}: Enqueuing ALL indices (background)")
                        
                        # Priority Strategy:
                        # 1. Last index (High Prio) - often requested first
                        last_idx = len(dataset) - 1
                        if 0 <= last_idx < len(full_forecasts):
                            self.enqueue_calc(handle, last_idx, full_forecasts[last_idx], 0, neg_timestamp)
                            
                        # 2. All others (Low Prio) - Reverse order
                        for idx in range(len(dataset)-2, -1, -1):
                             self.enqueue_calc(handle, idx, full_forecasts[idx], priority, neg_timestamp)

                except Exception as e:
                    logger.error(f"InfWorker {worker_id} error: {e}", exc_info=True)
                    ModelManager().increment_errors()
                finally:
                    self.inference_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"InfWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"InfWorker {worker_id} exiting gracefully")

    async def _calc_worker_loop(self, worker_id: int):
        logger.info(f"CalcWorker {worker_id} started")
        state_mgr = StateManager()
        data_cache = DataCache()
        task_registry = TaskRegistry()
        
        while self.running and not state_mgr.shutdown_started:
            try:
                priority, neg_timestamp, handle, index, forecasts = await self.calc_queue.get()
                
                try:
                    # 1. Get Data
                    data = await data_cache.get(handle)
                    if not data:
                         continue # Expired

                    # 2. Check Result Cache (in DATA_CACHE via Polars DF)
                    data_df = data.get('data_df')
                    if data_df is None: 
                        continue

                    # Check if index is already calculated
                    row = data_df.row(index, named=True)
                    if row['is_calculated'] or state_mgr.shutdown_started:
                            task_registry.notify_success(handle, index)
                            continue

                    # 3. Calculate
                    loop = asyncio.get_running_loop()
                    plot_data = await loop.run_in_executor(
                        None, calculate_plot_data, forecasts, data['dataset'], data['scalers'], index
                    )
                    
                    # 4. Store
                    # Update Polars DataFrame for this index
                    # Convert PlotData to dicts for Polars
                    plot_data_dict = plot_data.model_dump()
                    
                    async with data_cache.lock:
                        if data_cache.contains_sync(handle):
                            cached_data = data_cache.get_sync(handle)
                            current_df = cached_data['data_df']
                            
                            # We construct the update columns
                            # Note: Updating a single row in Polars using with_columns/when/then is efficient enough for this scale
                            
                            # Extract values
                            true_val_x = plot_data_dict['true_values_x']
                            true_val_y = plot_data_dict['true_values_y']
                            med_x = plot_data_dict['median_x']
                            med_y = plot_data_dict['median_y']
                            fans = plot_data_dict['fan_charts'] # List of dicts
                            
                            # Apply update
                            updated_df = current_df.with_columns([
                                pl.when(pl.col("index") == index).then(pl.lit(pl.Series([true_val_x]))).otherwise(pl.col("true_values_x")).alias("true_values_x"),
                                pl.when(pl.col("index") == index).then(pl.lit(pl.Series([true_val_y]))).otherwise(pl.col("true_values_y")).alias("true_values_y"),
                                pl.when(pl.col("index") == index).then(pl.lit(pl.Series([med_x]))).otherwise(pl.col("median_x")).alias("median_x"),
                                pl.when(pl.col("index") == index).then(pl.lit(pl.Series([med_y]))).otherwise(pl.col("median_y")).alias("median_y"),
                                # For struct list, we might need to be careful. pl.lit with object might be tricky.
                                # Alternatively, we can map the change.
                                # But pl.when().then() with list of structs is complex.
                                # Hack: Since we have the lock, we can use map_rows or similar? No, slow.
                                # Let's try the simple approach for now, but fan_charts structure is complex.
                                # If polars literal construction fails, we might need another approach.
                                pl.when(pl.col("index") == index).then(True).otherwise(pl.col("is_calculated")).alias("is_calculated")
                            ])
                            
                            # Fan charts column update is tricky due to complex type
                            # Ideally we would update it too. 
                            # For now, let's try to construct the series for fan_charts
                            # If that's too hard, we can defer it or use Object column?
                            # But we defined schema.
                            
                            # Actually, we can just reconstruct the row and vstack? No, order matters (maybe).
                            # But `index` column preserves identity.
                            
                            # Let's try to update fan_charts separately or together if possible
                            # Constructing a Series of List[Struct] with one element
                            # fans_series = pl.Series([fans], dtype=current_df['fan_charts'].dtype)
                            # updated_df = updated_df.with_columns(
                            #    pl.when(pl.col("index") == index).then(fans_series).otherwise(pl.col("fan_charts")).alias("fan_charts")
                            # )
                            
                            # If this is too complex/brittle, we can use the 'object' fallback or reconstruct the list in python.
                            # Reconstructing the list in python:
                            # 1. Get current list
                            # 2. Update item
                            # 3. Create new Series
                            # This is O(N) but N is small (<1000)
                            
                            current_fans = current_df['fan_charts'].to_list()
                            current_fans[index] = fans
                            updated_df = updated_df.with_columns(pl.Series("fan_charts", current_fans))
                            
                            cached_data['data_df'] = updated_df
                        
                    # 5. Notify
                    task_registry.notify_success(handle, index)
                    
                except Exception as e:
                    logger.error(f"CalcWorker {worker_id} error: {e}", exc_info=True)
                    task_registry.notify_error(handle, index, e)
                    ModelManager().increment_errors()
                finally:
                    self.calc_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CalcWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"CalcWorker {worker_id} exiting gracefully")
