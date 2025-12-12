import logging
import asyncio
import time
import threading
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import asynccontextmanager
import torch
import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from cgm_format.interface import ProcessingWarning

# Dependencies from glurpc
from glurpc.data_classes import (
    GluformerModelConfig, 
    GluformerInferenceConfig, 
    PredictionsData, 
    LogVarsArray, 
    DartsScaler,
    DartsDataset,
    PredictionsData,
    PredictionsArray,
    FormattedWarnings,
    PlotData,
    PlotCacheEntry,
    ModelStats,
    CalcStats
)
from glurpc.config import (
    NUM_COPIES_PER_DEVICE, BACKGROUND_WORKERS_COUNT, BATCH_SIZE, NUM_SAMPLES, 
    ENABLE_CACHE_PERSISTENCE, MAX_INFERENCE_QUEUE_SIZE, MAX_CALC_QUEUE_SIZE,
    INFERENCE_TIMEOUT_GPU, INFERENCE_TIMEOUT_CPU
)
from glurpc.logic import ModelState, load_model, run_inference_full, calculate_plot_data, create_dataset_from_df, SamplingDatasetInferenceDual, reconstruct_dataset
from glurpc.state import Singleton, StateManager, InferenceCache, PlotCache, TaskRegistry, DisconnectTracker, MINIMUM_DURATION_MINUTES


logger = logging.getLogger("glurpc.engine")
inference_logger = logging.getLogger("glurpc.engine.infer")
calc_logger = logging.getLogger("glurpc.engine.calc")
preprocessing_logger = logging.getLogger("glurpc.engine.data")
locks_logger = logging.getLogger("glurpc.locks")  # App-wide lock logger

# --- Engine-specific Dynamic Configuration ---
# These are determined at runtime based on available hardware

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
"""Device to use for inference (detected dynamically)."""

# Set inference timeout based on device
INFERENCE_TIMEOUT: float = INFERENCE_TIMEOUT_GPU if DEVICE == "cuda" else INFERENCE_TIMEOUT_CPU
"""Inference timeout in seconds (device-specific)."""

logger.info(f"Device detected: {DEVICE}")
logger.info(f"Inference timeout set to: {INFERENCE_TIMEOUT}s ({INFERENCE_TIMEOUT/60:.1f} minutes)")


def get_total_copies() -> int:
    """Calculate total number of model copies based on device."""
    if DEVICE == "cpu":
        copies = 1
    else:
        copies = torch.cuda.device_count() * NUM_COPIES_PER_DEVICE
    copies = max(copies, 2) # Minimum of 2 copies
    logger.info(f"Total number of model copies: {copies} for device: {DEVICE}")
    return copies

NUM_COPIES: int = get_total_copies()
"""Total number of model copies across all devices."""

def check_queue_overload() -> Tuple[bool, str, float, float]:
    """
    Check if the system is overloaded and should reject new processing requests.
    Returns (is_overloaded, load_status) where load_status is one of: loaded, overloaded, full
    """
    bg_processor = BackgroundProcessor()
    calc_stats = bg_processor.get_calc_stats()
    
    # Calculate load status based on queue utilization
    # Use the maximum utilization across both queues
    inference_utilization = calc_stats.inference_queue_size / calc_stats.inference_queue_capacity if calc_stats.inference_queue_capacity > 0 else 0.0
    calc_utilization = calc_stats.calc_queue_size / calc_stats.calc_queue_capacity if calc_stats.calc_queue_capacity > 0 else 0.0
    max_utilization = max(inference_utilization, calc_utilization)
    
    overload = False
    if max_utilization >= 0.99:
        overload = True
        load_status = "full"
    elif max_utilization >= 0.75:
        load_status = "overloaded"
    elif max_utilization >= 0.50:
        load_status = "heavily loaded"
    elif max_utilization >= 0.25:
        load_status = "loaded"
    elif max_utilization >= 0.01:
        load_status = "lightly loaded"
    else:
        load_status = "idle"

    return overload, load_status, inference_utilization, calc_utilization

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
        locks_logger.debug(f"[InferenceWrapper] Acquiring threading lock for load_if_needed on {self.device}")
        with self._lock:
            locks_logger.debug(f"[InferenceWrapper] Acquired threading lock for load_if_needed on {self.device}")
            if self.model_state is not None:
                current_config, _ = self.model_state
                if current_config == required_config:
                    locks_logger.debug(f"[InferenceWrapper] Releasing threading lock for load_if_needed on {self.device} (no reload)")
                    return

            # Reload needed
            inference_logger.info("Model config mismatch or not loaded. Reloading model...")
            self.model_state = load_model(required_config, self.model_path, self.device)
            locks_logger.debug(f"[InferenceWrapper] Releasing threading lock for load_if_needed on {self.device} (reloaded)")

    def run_inference(
        self, 
        dataset: SamplingDatasetInferenceDual, 
        required_config: GluformerModelConfig,
        batch_size: int,
        num_samples: int
    ) -> Tuple[PredictionsArray, Optional[LogVarsArray]]:
        """
        Runs inference, ensuring model is loaded with correct config.
        Returns predictions array and logvars.
        """
        self.load_if_needed(required_config)
        
        locks_logger.debug(f"[InferenceWrapper] Acquiring threading lock for run_inference on {self.device}")
        with self._lock:
            locks_logger.debug(f"[InferenceWrapper] Acquired threading lock for run_inference on {self.device}")
            # We hold the lock to ensure the model isn't swapped out from under us
            current_state = self.model_state
            locks_logger.debug(f"[InferenceWrapper] Releasing threading lock for run_inference on {self.device}")
        
        predictions, logvars = run_inference_full(
            dataset=dataset,
            model_config=required_config,
            model_state=current_state,
            batch_size=batch_size,
            num_samples=num_samples,
            device=self.device
        )
        return predictions, logvars



class ModelManager(Singleton):
    """
    Singleton manager for ML model instances and inference requests.
    
    Model #0 is reserved exclusively for priority 0 (high-priority/interactive) requests.
    Models #1+ are in the general pool for all requests.
    """
    def __init__(self):
        self.models: List[InferenceWrapper] = []
        self.priority_queue = asyncio.Queue()  # Contains only model #0 (for priority 0 only)
        self.general_queue = asyncio.Queue()   # Contains models #1+ (for all requests)
        self.initialized = False
        self._init_lock = asyncio.Lock()
        
        # Stats
        self._fulfillment_times: List[float] = []
        self._max_stats_history = 1000
        self._inference_requests_by_priority: Dict[int, int] = {}  # priority -> count
        self._total_inference_errors = 0
        
    def increment_inference_request(self, priority: int = 1):
        """Increment inference request counter for specific priority."""
        self._inference_requests_by_priority[priority] = self._inference_requests_by_priority.get(priority, 0) + 1
        
    def increment_inference_errors(self):
        """Increment inference error counter."""
        self._total_inference_errors += 1
    
    async def initialize(self, model_name: str = "gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth"):
        if self.initialized:
            return

        locks_logger.debug("[ModelManager] Acquiring asyncio lock for initialize")
        async with self._init_lock:
            locks_logger.debug("[ModelManager] Acquired asyncio lock for initialize")
            if self.initialized:
                locks_logger.debug("[ModelManager] Releasing asyncio lock for initialize (already initialized)")
                return
                
            logger.info(f"Initializing ModelManager with model: {model_name}")
            await asyncio.to_thread(self._load_models_sync, model_name)
            
            # Model #0 goes to priority queue (reserved for priority 0 only)
            if len(self.models) > 0:
                self.priority_queue.put_nowait(self.models[0])
                logger.info("Model #0 reserved for priority 0 requests")
            
            # Models #1+ go to general queue (for all requests)
            for model in self.models[1:]:
                self.general_queue.put_nowait(model)
            
            self.initialized = True
            logger.info(f"ModelManager initialized with {len(self.models)} models (1 priority, {len(self.models)-1} general)")
            locks_logger.debug("[ModelManager] Releasing asyncio lock for initialize")

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
                model_role = "PRIORITY (reserved for priority 0)" if i == 0 else "GENERAL"
                logger.info(f"Loading model copy {i}/{NUM_COPIES-1} [{model_role}]")
                
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
    async def acquire(self, requested_copies: int = 1, priority: int = 1):
        """
        Acquire model(s) for inference.
        
        Priority 0 (high-priority): Can use general pool OR priority model #0
        Priority > 0 (background): Can ONLY use general pool
        
        Args:
            requested_copies: Number of models to acquire (currently only 1 is used)
            priority: 0 for high-priority/interactive, >0 for background
        """
        if not self.initialized:
             logger.error("Acquire called before initialization!")
             raise RuntimeError("Models not initialized")
            
        start_time = time.time()
        self.increment_inference_request(priority)
        
        # Currently we only acquire 1 model at a time
        num_to_acquire = 1
        
        acquired_models = []
        try:
            if priority == 0:
                # High priority: Try general pool first (for load balancing),
                # fall back to priority model #0 if general pool is empty
                try:
                    model = self.general_queue.get_nowait()
                    acquired_models.append(model)
                    logger.debug("Priority 0: Acquired from general pool")
                except asyncio.QueueEmpty:
                    # General pool empty, use priority model #0
                    model = await self.priority_queue.get()
                    acquired_models.append(model)
                    logger.debug("Priority 0: Acquired model #0 (priority reserved)")
            else:
                # Background: Only use general pool, never touch priority queue
                model = await self.general_queue.get()
                acquired_models.append(model)
                logger.debug(f"Priority {priority}: Acquired from general pool")
            
            yield acquired_models
        except Exception as e:
             logger.error(f"Error acquiring models: {e}")
             self.increment_inference_errors()
             raise
        finally:
            # Return models to their respective queues
            for model in acquired_models:
                if model == self.models[0]:
                    # This is model #0, return to priority queue
                    self.priority_queue.put_nowait(model)
                else:
                    # General pool model
                    self.general_queue.put_nowait(model)
            
            duration_ms = (time.time() - start_time) * 1000
            self._fulfillment_times.append(duration_ms)
            if len(self._fulfillment_times) > self._max_stats_history:
                self._fulfillment_times.pop(0)

    def get_stats(self) -> ModelStats:
        """Get model manager statistics as a Pydantic model."""
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
        else:
            # For CPU inference, report RSS (Resident Set Size) memory
            try:
                import resource
                rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On Linux, ru_maxrss is in kilobytes, on macOS it's in bytes
                # Check platform to normalize to MB
                import sys
                if sys.platform == 'darwin':
                    vmem_mb = rss_bytes / (1024 * 1024)
                else:
                    vmem_mb = rss_bytes / 1024  # Linux: KB to MB
            except Exception as e:
                logger.warning(f"Failed to get RSS memory: {e}")
        
        return ModelStats(
            available_priority_models=self.priority_queue.qsize(),
            available_general_models=self.general_queue.qsize(),
            avg_fulfillment_time_ms=avg_time,
            vmem_usage_mb=vmem_mb,
            device=DEVICE,
            inference_requests_by_priority=dict(self._inference_requests_by_priority),
            total_inference_errors=self._total_inference_errors
        )

class BackgroundProcessor(Singleton):
    """
    Singleton processor for managing background inference and calculation workers.
    
    ARCHITECTURE IMPLEMENTATION (THREADING_ARCHITECTURE.md:565-577):
    ✓ Request ID Assignment: Accepts request_id in enqueue_inference()
    ✓ Stale Job Detection: is_request_stale() checks if request_id < latest
    ✓ Last-Write-Wins: update_latest_request_id() maintains newest request per (handle, index)
    ✓ Job Cleanup: cleanup_stale_jobs() marks cancellation outcomes
    ✓ Workers Check Staleness: Before expensive operations in worker loops
    
    QUEUES:
    - Inference Queue: (priority, neg_timestamp, handle, indices, ..., request_id)
    - Calculation Queue: (priority, neg_timestamp, handle, index, ..., request_id)
    
    STALE DETECTION:
    - _latest_request_id: { (handle, index): request_id } tracks newest request
    - Workers check is_request_stale() before dataset creation and inference
    - Stale jobs are skipped, cleanup_stale_jobs() is called for metrics
    
    DEDUPLICATION:
    - _pending_inference: { handle: (priority, dataset_length, request_id) }
    - Prevents redundant inference enqueueing for same handle
    """
    def __init__(self):
        # Inference Queue Item: (priority, neg_timestamp, handle, indices, inference_df, warning_flags, expected_dataset_len, inference_config, force_calculate, request_id)
        self.inference_queue = asyncio.PriorityQueue()
        
        # Calculation Queue Item: (priority, neg_timestamp, handle, index, forecasts, version, request_id)
        self.calc_queue = asyncio.PriorityQueue()
        
        self.inference_workers = []
        self.calc_workers = []
        self.running = False
        
        # Track pending inference tasks to avoid redundant enqueueing
        # Map: handle -> (priority, dataset_length, request_id)
        self._pending_inference: Dict[str, Tuple[int, int, int]] = {}
        self._inference_lock = asyncio.Lock()
        
        # Track the latest request_id per (handle, index) for stale job detection
        # Map: (handle, index) -> request_id
        self._latest_request_id: Dict[Tuple[str, int], int] = {}
        self._request_id_lock = asyncio.Lock()
        
        # Track background tasks (e.g., delayed calculations from quick_plot)
        self._background_tasks: Set[asyncio.Task] = set()
        self._background_tasks_lock = asyncio.Lock()
        
        # Stats
        self._total_calc_runs = 0
        self._total_calc_errors = 0
        self._calc_lock = asyncio.Lock()
        
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
            
    async def stop(self):
        StateManager().start_shutdown()
        self.running = False
        logger.info("Shutdown flag set, waiting for workers to exit gracefully...")
        
        # Cancel all background tasks first
        await self.cancel_all_background_tasks()
        
        all_tasks = self.inference_workers + self.calc_workers
        for task in all_tasks:
            task.cancel()
        
        # Wait for all tasks to complete with timeout
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True),
                    timeout=5.0
                )
                logger.info("All background workers stopped successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for workers to stop, forcing shutdown")
            except Exception as e:
                logger.error(f"Error during worker shutdown: {e}")
        else:
            logger.info("No background workers to stop")
    
    async def register_background_task(self, task: asyncio.Task) -> None:
        """Register a background task for tracking and cleanup."""
        async with self._background_tasks_lock:
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
            
    async def cancel_all_background_tasks(self) -> None:
        """Cancel all registered background tasks."""
        async with self._background_tasks_lock:
            if not self._background_tasks:
                return
                
            logger.info(f"Cancelling {len(self._background_tasks)} background tasks...")
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self._background_tasks.clear()
            logger.info("All background tasks cancelled")

    async def enqueue_inference(
        self, 
        handle: str, 
        inference_df: pl.DataFrame,
        warning_flags: ProcessingWarning,
        expected_dataset_len: int,
        inference_config: GluformerInferenceConfig,
        priority: int = 1, 
        indices: Optional[List[int]] = None,
        force_calculate: bool = False,
        request_id: Optional[int] = None
    ):
        """
        Enqueue a task for inference.
        priority: 0 for High (Interactive), 1 for Low (Background)
        indices: Specific indices to prioritize calculation for. If None, calculates all normally.
        request_id: Optional request ID for tracking and stale job detection.
        
        Prevents redundant enqueueing of inference for the same handle if already pending.
        Also prevents queue flooding by checking queue size limit.
        """
        locks_logger.debug(f"[BackgroundProcessor] Acquiring asyncio lock for enqueue_inference handle={handle[:8]}")
        async with self._inference_lock:
            locks_logger.debug(f"[BackgroundProcessor] Acquired asyncio lock for enqueue_inference handle={handle[:8]}")
            # Check if inference for this handle is already pending
            if handle in self._pending_inference:
                pending_prio, pending_len, pending_req_id = self._pending_inference[handle]
                
                if priority == 0:
                    if pending_prio == 0 and pending_len == expected_dataset_len:
                        logger.debug(f"High Prio Inference for {handle[:8]} already pending (same len), reusing")
                        return
                else:
                    # Low Prio (1): Reuse if pending covers the request (len >= requested)
                    if pending_len >= expected_dataset_len:
                         logger.debug(f"Inference for {handle[:8]} already pending (len {pending_len} >= {expected_dataset_len}), skipping")
                         return

            # Check queue size limit (only for low priority background tasks)
            if priority > 0 and self.inference_queue.qsize() >= MAX_INFERENCE_QUEUE_SIZE:
                logger.warning(f"Inference queue full ({self.inference_queue.qsize()} >= {MAX_INFERENCE_QUEUE_SIZE}), dropping low-priority request for {handle[:8]}")
                return
            
            # Update pending state
            self._pending_inference[handle] = (priority, expected_dataset_len, request_id or 0)
            locks_logger.debug(f"[BackgroundProcessor] Releasing asyncio lock for enqueue_inference handle={handle[:8]}")
        
        neg_timestamp = -time.time()
        # Item: (priority, neg_timestamp, handle, indices, inference_df, warning_flags, expected_dataset_len, inference_config, force_calculate, request_id)
        item = (priority, neg_timestamp, handle, indices, inference_df, warning_flags, expected_dataset_len, inference_config, force_calculate, request_id)
        self.inference_queue.put_nowait(item)
        logger.debug(f"Enqueued INFERENCE: handle={handle[:8]} len={expected_dataset_len} prio={priority} indices={'ALL' if indices is None else indices} req_id={request_id}")

    def enqueue_calc(self, handle: str, index: int, forecasts: Any, priority: int, neg_timestamp: float, version: str, request_id: Optional[int] = None):
        item = (priority, neg_timestamp, handle, index, forecasts, version, request_id)
        self.calc_queue.put_nowait(item)
        # logger.debug(f"Enqueued CALC: handle={handle[:8]} idx={index} prio={priority} req_id={request_id}") 

    def is_processing(self, handle: str) -> bool:
        """Check if inference is currently pending for the given handle."""
        return handle in self._pending_inference
    
    def get_pending_status(self, handle: str) -> Optional[Tuple[int, int, int]]:
        """Get pending status (priority, dataset_len, request_id) for handle."""
        return self._pending_inference.get(handle)
    
    async def update_latest_request_id(self, handle: str, index: int, request_id: int) -> None:
        """
        Update the latest request_id for (handle, index).
        This enables last-write-wins semantics and stale job detection.
        """
        key = (handle, index)
        locks_logger.debug(f"[BackgroundProcessor] Acquiring lock for update_latest_request_id {handle[:8]}:{index}:{request_id}")
        async with self._request_id_lock:
            locks_logger.debug(f"[BackgroundProcessor] Acquired lock for update_latest_request_id {handle[:8]}:{index}:{request_id}")
            self._latest_request_id[key] = request_id
            locks_logger.debug(f"[BackgroundProcessor] Releasing lock for update_latest_request_id {handle[:8]}:{index}:{request_id}")
    
    async def is_request_stale(self, handle: str, index: int, request_id: Optional[int]) -> bool:
        """
        Check if a request is stale (a newer request has arrived).
        Returns True if this request should be discarded.
        
        Args:
            handle: Cache handle
            index: Data index
            request_id: Request ID to check (None means background job, never stale)
        
        Returns:
            True if the request is stale and should be skipped
        """
        if request_id is None:
            return False  # No request_id means background job, never stale
        
        key = (handle, index)
        locks_logger.debug(f"[BackgroundProcessor] Acquiring lock for is_request_stale {handle[:8]}:{index}:{request_id}")
        async with self._request_id_lock:
            locks_logger.debug(f"[BackgroundProcessor] Acquired lock for is_request_stale {handle[:8]}:{index}:{request_id}")
            latest = self._latest_request_id.get(key, 0)
            is_stale = request_id < latest
            locks_logger.debug(f"[BackgroundProcessor] Releasing lock for is_request_stale {handle[:8]}:{index}:{request_id}")
            
            if is_stale:
                logger.info(f"Request {handle[:8]}:{index}:{request_id} is stale (latest={latest}), will be skipped")
            
            return is_stale
    
    async def cleanup_stale_jobs(self, handle: str, index: int, request_id: int) -> None:
        """
        Cleanup hook for stale jobs.
        Marks cancellation outcome metrics when a job is detected as stale.
        
        This is called by workers when they detect a stale job before processing.
        """
        logger.debug(f"Cleaning up stale job: {handle[:8]}:{index}:{request_id}")
        # Future enhancement: Add metrics for stale job detection
        # For now, just log it (already done in is_request_stale)
    
    async def increment_calc_runs(self):
        """Increment calculation run counter."""
        locks_logger.debug("[BackgroundProcessor] Acquiring asyncio lock for increment_calc_runs")
        async with self._calc_lock:
            locks_logger.debug("[BackgroundProcessor] Acquired asyncio lock for increment_calc_runs")
            self._total_calc_runs += 1
            locks_logger.debug("[BackgroundProcessor] Releasing asyncio lock for increment_calc_runs")
    
    async def increment_calc_errors(self):
        """Increment calculation error counter."""
        locks_logger.debug("[BackgroundProcessor] Acquiring asyncio lock for increment_calc_errors")
        async with self._calc_lock:
            locks_logger.debug("[BackgroundProcessor] Acquired asyncio lock for increment_calc_errors")
            self._total_calc_errors += 1
            locks_logger.debug("[BackgroundProcessor] Releasing asyncio lock for increment_calc_errors")
    
    def get_calc_stats(self) -> CalcStats:
        """Get calculation worker statistics as a Pydantic model."""
        return CalcStats(
            total_calc_runs=self._total_calc_runs,
            total_calc_errors=self._total_calc_errors,
            calc_queue_size=self.calc_queue.qsize(),
            calc_queue_capacity=MAX_CALC_QUEUE_SIZE,
            inference_queue_size=self.inference_queue.qsize(),
            inference_queue_capacity=MAX_INFERENCE_QUEUE_SIZE
        )

    async def _inference_worker_loop(self, worker_id: int):
        logger.info(f"InfWorker {worker_id} started")
        state_mgr = StateManager()
        inf_cache = InferenceCache()
        
        
        while self.running and not state_mgr.shutdown_started:
            try:
                priority, neg_timestamp, handle, indices, inference_df, warning_flags, expected_dataset_len, inference_config, force_calculate, request_id = await self.inference_queue.get()
                
                try:
                    # 0. Check if request is stale before doing any work
                    # For specific indices (interactive requests), check staleness early
                    stale_indices = []
                    if indices is not None and request_id is not None:
                        for idx in indices:
                            if await self.is_request_stale(handle, idx, request_id):
                                logger.info(f"InfWorker {worker_id}: Skipping stale inference for {handle[:8]}:{idx}:{request_id}")
                                await self.cleanup_stale_jobs(handle, idx, request_id)
                                stale_indices.append(idx)
                        
                        # If all indices are stale, skip the entire job
                        if len(stale_indices) == len(indices):
                            logger.info(f"InfWorker {worker_id}: All indices stale, skipping inference job entirely")
                            continue
                        
                        # Remove stale indices from the list
                        indices = [idx for idx in indices if idx not in stale_indices]
                    
                    # 1. Pre-Run Cache Check (Avoid redundant work)
                    # "First it does the same op as request on start (fetches cacche, check if cache is non-empty, compares cached len)"
                    if not force_calculate:
                         cached_data = await inf_cache.get(handle)
                         if cached_data and cached_data.predictions is not None:
                             # Check length
                             if cached_data.dataset_length >= expected_dataset_len:
                                 logger.info(f"InfWorker {worker_id}: Cache already contains sufficient data for {handle[:8]} (cached {cached_data.dataset_length} >= requested {expected_dataset_len}). Skipping inference.")
                                 
                                 # We might still need to enqueue calculations if specific indices were requested!
                                 # Re-use cached_data for that.
                                 full_forecasts = cached_data.predictions
                                 version = cached_data.version
                                 
                                 # Proceed to Enqueue Calculation using cached data
                                 if full_forecasts is not None:
                                     self._enqueue_calculations(handle, full_forecasts, version, indices, priority, neg_timestamp, worker_id, request_id)
                                 
                                 continue
                    
                    # 2. Create Dataset (Heavy Lift)
                    logger.info(f"InfWorker {worker_id}: Creating dataset for {handle[:8]}...")
                    
                    result = await asyncio.to_thread(
                        create_dataset_from_df,
                        inference_df,
                        warning_flags,
                    )
                    
                    if not result['success']:
                        logger.error(f"InfWorker {worker_id}: Dataset creation failed: {result.get('error')}")
                        ModelManager().increment_inference_errors()
                        TaskRegistry().cancel_all_for_handle(handle)
                        continue
                        
                    dataset = result['dataset']
                    model_config_dump = result['model_config'].model_dump()
                    warning_flags = result['warning_flags']
                    scaler_target = result['scaler_target']
                    
                    # Validate length
                    # With the fix to use actual inference_df length, this should rarely trigger
                    # But we keep it as a safety check for edge cases
                    if len(dataset) < expected_dataset_len:
                        logger.error(f"InfWorker {worker_id}: Created dataset shorter than expected ({len(dataset)} < {expected_dataset_len}).")
                        logger.warning(f"Proceeding with available dataset length {len(dataset)} (expected {expected_dataset_len})")
                        # If dataset is empty, that's a real error
                        if len(dataset) == 0:
                            msg = f"Dataset is empty - cannot proceed with inference"
                            logger.error(msg)
                            ModelManager().increment_inference_errors()
                            TaskRegistry().cancel_all_for_handle(handle)
                            continue
                        # If we have some valid samples, log warning but continue with what we have
                        logger.info(f"InfWorker {worker_id}: Proceeding with available dataset length {len(dataset)} (expected {expected_dataset_len})")

                    # 3. Run Inference
                    required_config = GluformerModelConfig(**model_config_dump)
                    
                    inference_logger.info(f"InfWorker {worker_id}: Running FULL inference for {handle[:8]} ({len(dataset)} items) with priority {priority}")
                
                    async with ModelManager().acquire(1, priority=priority) as wrappers:
                        wrapper = wrappers[0]
                        full_forecasts_array, logvars = await asyncio.to_thread(
                            wrapper.run_inference, 
                            dataset, 
                            required_config,
                            BATCH_SIZE, 
                            NUM_SAMPLES 
                        )
                        
                        # Prepare result data
                        darts_dataset = DartsDataset.from_original(dataset)
                        darts_scaler = DartsScaler.from_original(scaler_target)
                        
                        result_data = PredictionsData(
                            len_pred=darts_dataset.output_chunk_length,
                            num_samples=NUM_SAMPLES,
                            time_step=inference_config.time_step,
                            first_index=-(len(dataset) - 1),
                            predictions=full_forecasts_array,
                            logvars=logvars,
                            dataset=darts_dataset,
                            target_scaler=darts_scaler,
                            model_config_dump=model_config_dump,
                            warning_flags=warning_flags.value
                        )
                        
                        # 4. Resolve Cache Write (Transaction)
                        
                        data_to_use = None
                        
                        async with inf_cache.transaction(handle) as txn:
                            cached = txn.value
                            should_write = False
                            
                            if cached is None:
                                should_write = True
                                logger.debug(f"InfWorker {worker_id}: Cache empty, writing result.")
                            elif cached.predictions is None:
                                should_write = True
                                logger.debug(f"InfWorker {worker_id}: Cache pending (no predictions), writing result.")
                            else:
                                if cached.dataset_length < result_data.dataset_length:
                                    should_write = True
                                    logger.debug(f"InfWorker {worker_id}: Cache smaller ({cached.dataset_length} < {result_data.dataset_length}), writing result.")
                                elif force_calculate:
                                    should_write = True
                                    logger.info(f"InfWorker {worker_id}: Force calculate true, overwriting cache.")
                                else:
                                    should_write = False
                                    logger.info(f"InfWorker {worker_id}: Cache larger/equal ({cached.dataset_length} >= {result_data.dataset_length}) and not forced. Discarding result.")
                            
                            if should_write:
                                txn.set(result_data)
                                data_to_use = result_data
                            else:
                                data_to_use = cached # Use cached data for calculations
                        
                        # 5. Enqueue Calculation
                        if data_to_use and data_to_use.predictions is not None:
                             self._enqueue_calculations(
                                 handle, 
                                 data_to_use.predictions, 
                                 data_to_use.version, 
                                 indices, 
                                 priority, 
                                 neg_timestamp, 
                                 worker_id,
                                 request_id
                            )

                except ValueError as e:
                    # Expected validation/data quality errors
                    logger.error(f"InfWorker {worker_id} validation error for {handle[:8]}: {e}")
                    ModelManager().increment_inference_errors()
                    TaskRegistry().cancel_all_for_handle(handle, error_msg=str(e))
                except Exception as e:
                    # Unexpected system errors
                    logger.exception(f"InfWorker {worker_id} unexpected error for {handle[:8]}: {e}")
                    ModelManager().increment_inference_errors()
                    TaskRegistry().cancel_all_for_handle(handle, error_msg=f"Internal processing error: {str(e)}")
                finally:
                    # Clear the pending flag for this handle
                    locks_logger.debug(f"[InfWorker {worker_id}] Acquiring asyncio lock for pending_inference cleanup handle={handle[:8]}")
                    async with self._inference_lock:
                        locks_logger.debug(f"[InfWorker {worker_id}] Acquired asyncio lock for pending_inference cleanup handle={handle[:8]}")
                        if handle in self._pending_inference:
                            del self._pending_inference[handle]
                        locks_logger.debug(f"[InfWorker {worker_id}] Releasing asyncio lock for pending_inference cleanup handle={handle[:8]}")
                    
                    self.inference_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"InfWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"InfWorker {worker_id} exiting gracefully")

    def _enqueue_calculations(self, handle, full_forecasts, version, indices, priority, neg_timestamp, worker_id, request_id: Optional[int] = None):
        N = len(full_forecasts)
        
        if indices is not None:
            # Specific indices requested (High Prio)
            logger.debug(f"InfWorker {worker_id}: Enqueuing specific indices: {indices}")
            for idx in indices:
                    pos_idx = idx + (N - 1)
                    if 0 <= pos_idx < N:
                        self.enqueue_calc(handle, idx, full_forecasts[pos_idx], priority, neg_timestamp, version, request_id)
        else:
            # Background processing - enqueue all
            logger.debug(f"InfWorker {worker_id}: Enqueuing ALL indices (background)")
            
            pos_last = N - 1
            if pos_last >= 0:
                self.enqueue_calc(handle, 0, full_forecasts[pos_last], 0, neg_timestamp, version, request_id)
                
            for pos_idx in range(N-2, -1, -1):
                    neg_idx = pos_idx - (N - 1)
                    self.enqueue_calc(handle, neg_idx, full_forecasts[pos_idx], priority, neg_timestamp, version, request_id)


    async def _calc_worker_loop(self, worker_id: int):
        logger.info(f"CalcWorker {worker_id} started")
        state_mgr = StateManager()
        inf_cache = InferenceCache()
        plot_cache = PlotCache()
        task_registry = TaskRegistry()
        
        while self.running and not state_mgr.shutdown_started:
            try:
                priority, neg_timestamp, handle, index, forecasts, task_version, request_id = await self.calc_queue.get()
                
                try:
                    # 0. Check if request is stale before doing any work
                    if await self.is_request_stale(handle, index, request_id):
                        logger.info(f"CalcWorker {worker_id}: Skipping stale calc job for {handle[:8]}:{index}:{request_id}")
                        await self.cleanup_stale_jobs(handle, index, request_id)
                        # Notify the specific request that it was cancelled due to staleness
                        if request_id is not None:
                            task_registry.notify_error(handle, index, Exception("Request superseded by newer request"), request_id)
                        continue
                    
                    # 1. Check Plot Cache
                    # Key for plot cache? We decided to use version + index inside PlotCache value (dict)
                    existing_plot = await plot_cache.get_plot(task_version, index)
                    if existing_plot:
                        task_registry.notify_success(handle, index, request_id)
                        continue

                    # 2. Get Data (we need scalers and dataset)
                    data : PredictionsData = await inf_cache.get(handle)
                    if not data:
                         continue # Expired

                    # Check version
                    current_version = data.version
                    if current_version != task_version:
                         logger.info(f"CalcWorker {worker_id}: Version mismatch for {handle[:8]} (Task: {task_version} != Curr: {current_version}), dropping task")
                         task_registry.notify_error(handle, index, Exception(f"Version mismatch: {task_version} vs {current_version}"), request_id)
                         continue

                    # 3. Calculate and Render
                    # Calculate directly using PredictionsData (data)
                    plot_json_str, plot_data = await asyncio.to_thread(
                        calculate_plot_data,
                        data, 
                        index,
            
                    )
                    
                    # 4. Increment calc counter
                    await self.increment_calc_runs()
                    
                    # 5. Store
                    await plot_cache.update_plot(task_version, index, plot_json_str, plot_data)
                        
                    # 6. Notify
                    task_registry.notify_success(handle, index, request_id)
                    
                except Exception as e:
                    logger.error(f"CalcWorker {worker_id} error: {e}", exc_info=True)
                    task_registry.notify_error(handle, index, e, request_id)
                    await self.increment_calc_errors()
                finally:
                    self.calc_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CalcWorker {worker_id} loop crash: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"CalcWorker {worker_id} exiting gracefully")
