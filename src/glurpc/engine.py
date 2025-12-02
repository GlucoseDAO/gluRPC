import logging
import asyncio
import time
import threading
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import torch
import numpy as np
from huggingface_hub import hf_hub_download


# Dependencies from glurpc
from glurpc.data_classes import GluformerModelConfig, GluformerInferenceConfig

from glurpc.state import DATA_CACHE, DATA_CACHE_LOCK, RESULT_CACHE, RESULT_CACHE_LOCK, FORECAST_CACHE, FORECAST_CACHE_LOCK, notify_success, notify_error
from glurpc.logic import ModelState, load_model, run_inference_full, calculate_plot_data, SamplingDatasetInferenceDual

logger = logging.getLogger("glurpc")

NUM_COPIES_PER_DEVICE = int(os.getenv("NUM_COPIES_PER_DEVICE", "2"))
BACKGROUND_WORKERS_COUNT = int(os.getenv("BACKGROUND_WORKERS_COUNT", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "10"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_total_copies():
    if DEVICE == "cpu":
        return 1
    return torch.cuda.device_count() * NUM_COPIES_PER_DEVICE

NUM_COPIES = get_total_copies()


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



class ModelManager:
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

MODEL_MANAGER = ModelManager()

class BackgroundProcessor:
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
        logger.info(f"Starting {num_inference_workers} inference workers and {num_calc_workers} calculation workers...")
        
        for i in range(num_inference_workers):
            task = asyncio.create_task(self._inference_worker_loop(i))
            self.inference_workers.append(task)
            
        for i in range(num_calc_workers):
            task = asyncio.create_task(self._calc_worker_loop(i))
            self.calc_workers.append(task)
            
    def stop(self):
        self.running = False
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
        while self.running:
            try:
                priority, neg_timestamp, handle, indices = await self.inference_queue.get()
                
                try:
                    # 1. Check Data Cache
                    async with DATA_CACHE_LOCK:
                        data = DATA_CACHE.get(handle)
                    
                    if not data:
                        logger.debug(f"InfWorker {worker_id}: Handle {handle[:8]} not found. Dropping.")
                        continue
                    
                    dataset = data['dataset']
                    required_config = data.get('model_config')
                    
                    if not required_config:
                        logger.error(f"InfWorker {worker_id}: Model config missing for {handle[:8]}")
                        continue
                    
                    # 2. Check Forecast Cache (Full)
                    async with FORECAST_CACHE_LOCK:
                        # Only reuse if we have FULL forecast
                        if handle in FORECAST_CACHE and len(FORECAST_CACHE[handle]) == len(dataset):
                            logger.debug(f"InfWorker {worker_id}: Full forecast exists for {handle[:8]}.")
                            full_forecasts = FORECAST_CACHE[handle]
                        else:
                            full_forecasts = None

                    # 3. Run Inference (Full) if needed
                    if full_forecasts is None:
                        total_len = len(dataset)
                        logger.info(f"InfWorker {worker_id}: Running FULL inference for {handle[:8]} ({total_len} items)")
                        
                        async with MODEL_MANAGER.acquire(1) as wrappers:
                            wrapper = wrappers[0]
                            loop = asyncio.get_running_loop()
                            full_forecasts = await loop.run_in_executor(
                                None, 
                                wrapper.run_inference, 
                                dataset, 
                                required_config,
                                BATCH_SIZE,
                                NUM_SAMPLES
                            )
                            
                        async with FORECAST_CACHE_LOCK:
                            if handle not in FORECAST_CACHE:
                                FORECAST_CACHE[handle] = {}
                            FORECAST_CACHE[handle].update(full_forecasts)
                            
                    # 4. Enqueue Calculation
                    if indices is not None:
                        # Specific indices requested (High Prio)
                        logger.debug(f"InfWorker {worker_id}: Enqueuing specific indices: {indices}")
                        for idx in indices:
                             if idx in full_forecasts:
                                 self.enqueue_calc(handle, idx, full_forecasts[idx], priority, neg_timestamp)
                    else:
                        # Background processing - enqueue all
                        logger.debug(f"InfWorker {worker_id}: Enqueuing ALL indices (background)")
                        
                        # Priority Strategy:
                        # 1. Last index (High Prio) - often requested first
                        last_idx = len(dataset) - 1
                        if last_idx in full_forecasts:
                            # Use priority 0 (High) for the last index even in background mode
                            self.enqueue_calc(handle, last_idx, full_forecasts[last_idx], 0, neg_timestamp)
                            
                        # 2. All others (Low Prio) - Reverse order
                        for idx in range(len(dataset)-2, -1, -1):
                            if idx in full_forecasts:
                                self.enqueue_calc(handle, idx, full_forecasts[idx], priority, neg_timestamp)

                except Exception as e:
                    logger.error(f"InfWorker {worker_id} error: {e}", exc_info=True)
                    MODEL_MANAGER.increment_errors()
                finally:
                    self.inference_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"InfWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)

    async def _calc_worker_loop(self, worker_id: int):
        logger.info(f"CalcWorker {worker_id} started")
        while self.running:
            try:
                priority, neg_timestamp, handle, index, forecasts = await self.calc_queue.get()
                
                try:
                    # 1. Get Data
                    async with DATA_CACHE_LOCK:
                        data = DATA_CACHE.get(handle)
                    if not data:
                         continue # Expired

                    # 2. Check Result Cache (Optimization)
                    async with RESULT_CACHE_LOCK:
                        if handle in RESULT_CACHE and index in RESULT_CACHE[handle]:
                            notify_success(handle, index)
                            continue

                    # 3. Calculate
                    loop = asyncio.get_running_loop()
                    plot_data = await loop.run_in_executor(
                        None, calculate_plot_data, forecasts, data['dataset'], data['scalers'], index
                    )
                    
                    # 4. Store
                    async with RESULT_CACHE_LOCK:
                        if handle not in RESULT_CACHE:
                            RESULT_CACHE[handle] = {}
                        RESULT_CACHE[handle][index] = plot_data
                        
                    # 5. Notify
                    notify_success(handle, index)
                    
                except Exception as e:
                    logger.error(f"CalcWorker {worker_id} error: {e}", exc_info=True)
                    notify_error(handle, index, e)
                    MODEL_MANAGER.increment_errors()
                finally:
                    self.calc_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CalcWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)

BACKGROUND_PROCESSOR = BackgroundProcessor()

