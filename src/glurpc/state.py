import asyncio
import hashlib
import os
import logging
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple, Set, Union, Dict
from contextlib import asynccontextmanager

from glurpc.cache import HybridLRUCache, Singleton
from glurpc.config import (
    DEFAULT_CONFIG, NUM_SAMPLES, MAX_CACHE_SIZE, 
    DEFAULT_INPUT_CHUNK_LENGTH, DEFAULT_OUTPUT_CHUNK_LENGTH,
    MINIMUM_DURATION_MINUTES, MAXIMUM_WANTED_DURATION, STEP_SIZE_MINUTES
)
from glurpc.data_classes import PredictionsData, PlotData, PlotCacheEntry

# Module logger
logger = logging.getLogger("glurpc.state")

# App-wide lock logger for debugging lock operations
locks_logger = logging.getLogger("glurpc.locks")

# --- API Key Management ---

class APIKeyManager(Singleton):
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

class StateManager(Singleton):
    """
    Centralized state manager for application-wide flags.
    """
    def __init__(self):
        self._shutdown_started: bool = False
    
    @property
    def shutdown_started(self) -> bool:
        return self._shutdown_started
    
    def start_shutdown(self) -> None:
        """Signal that shutdown has started."""
        self._shutdown_started = True
    
    def reset_shutdown(self) -> None:
        """Reset shutdown flag (useful for testing or restart)."""
        self._shutdown_started = False

# Compute directory hash based on inference parameters
_params_str = f"{DEFAULT_CONFIG.output_chunk_length}_{NUM_SAMPLES}"
CACHE_SUBDIR = hashlib.sha256(_params_str.encode()).hexdigest()

class PlotCache(HybridLRUCache[str, PlotCacheEntry]):
    """
    Cache for plot data using PlotCacheEntry.
    Key: version (from PredictionsData)
    Value: PlotCacheEntry containing arrays of plot data and metadata
    """
    def __init__(self):
        directory = os.path.join(os.getcwd(), "cache_storage", CACHE_SUBDIR, "plots")
        super().__init__(directory, max_hot=MAX_CACHE_SIZE)

    async def get_plot(self, version: str, index: int) -> Optional[str]:
        """Get a single plot JSON string for the given version and index."""
        entry = await self.get(version)
        if entry is None:
            return None
        
        array_index = entry.slice_data.get_dataset_index(index)
        return entry.plots_jsons[array_index]

    async def update_plot(self, version: str, index: int, json_str: str, plot_data: Optional[PlotData] = None) -> None:
        """Update a single plot in the version's PlotCacheEntry safely."""
        
        def updater(entry: PlotCacheEntry) -> PlotCacheEntry:
            # Create copies of arrays
            new_plots_jsons = entry.plots_jsons.copy()
            new_plots_data = entry.plots_data.copy()
            
            # Update at the specified index
            array_index = entry.slice_data.get_dataset_index(index)
            new_plots_jsons[array_index] = json_str
            if plot_data is not None:
                new_plots_data[array_index] = plot_data
            
            # Create new entry with updated arrays
            return PlotCacheEntry(
                slice_data=entry.slice_data,
                plots_jsons=new_plots_jsons,
                plots_data=new_plots_data
            )
        
        # Entry must exist before updating individual plots
        await self.update_entry(version, updater)
    
    async def initialize_entry(self, predictions_data: PredictionsData) -> None:
        """Initialize a cache entry from PredictionsData."""
        entry = PlotCacheEntry.from_predictions_data(predictions_data)
        await self.set(predictions_data.version, entry)

class InferenceCache(HybridLRUCache[str, PredictionsData]):
    """
    Cache for inference results.
    Key: handle
    Value: PredictionsData
    """
    def __init__(self):
        directory = os.path.join(os.getcwd(), "cache_storage", CACHE_SUBDIR, "inference")
        super().__init__(directory, max_hot=MAX_CACHE_SIZE)

    async def set(self, key: str, value: PredictionsData) -> None:
        # Check if we are replacing existing data
        old_data = await self.get(key)
        if old_data:
            # Invalidate old plots associated with the old version
            old_version = getattr(old_data, 'version', None)
            if old_version:
                await PlotCache().delete(old_version)
        
        await super().set(key, value)
        # Initialize the plot cache entry for the new data
        await PlotCache().initialize_entry(value)

    @asynccontextmanager
    async def transaction(self, key: str):
        async with super().transaction(key) as txn:
            old_data = txn.value
            
            # Helper to capture new value if set is called
            new_val_holder = []
            original_set = txn.set
            
            def set_wrapper(val):
                new_val_holder.append(val)
                original_set(val)
            
            txn.set = set_wrapper
            
            yield txn
            
            # Post-yield operations (inside lock)
            if new_val_holder:
                new_data = new_val_holder[0]
                
                # Initialize plot cache for new data
                await PlotCache().initialize_entry(new_data)
                
                # Handle invalidation if version changed
                if old_data:
                    old_ver = getattr(old_data, 'version', None)
                    if old_ver and old_ver != new_data.version:
                        await PlotCache().delete(old_ver)

    async def delete(self, key: str) -> None:
        data = await self.get(key)
        if data:
            version = getattr(data, 'version', None)
            if version:
                await PlotCache().delete(version)
        await super().delete(key)

    async def clear(self) -> None:
        await super().clear()
        await PlotCache().clear()

class DisconnectTracker(Singleton):
    """
    Singleton tracker for managing request disconnect counters per handle/index.
    
    Implementation of the duplication-aware disconnect architecture:
    
    ARCHITECTURE REQUIREMENTS (THREADING_ARCHITECTURE.md:565-577):
    ✓ 1. Request ID Assignment: Each request gets a unique monotonic request_id
    ✓ 2. Per-Request Disconnect Futures: Individual requests can disconnect independently
    ✓ 3. Shared Disconnect Future: Only resolves when ALL requests for (handle, index) disconnect
    ✓ 4. Request Counter: Tracks active request count, decremented on completion/disconnect
    ✓ 5. Last-Write-Wins: Newer request_ids supersede older ones via stale detection
    ✓ 6. Unified Cancellation Hook: cancel_request() handles per-request cancellation
    
    TRACKING STRUCTURE:
    - Shared tracking: { (handle, index): {"seq": int, "count": int, "disconnect_future": Future} }
    - Per-request futures: { (handle, index, request_id): Future }
    
    FLOW:
    1. Request arrives -> register_request() -> increments count, assigns request_id
    2. Request processes -> races disconnect_future vs work completion
    3. Request completes/disconnects -> unregister_request() -> decrements count
    4. Counter reaches 0 -> shared disconnect_future resolves
    5. Individual disconnect -> per-request future resolves immediately
    
    DUPLICATE HANDLING:
    - Multiple concurrent requests for same (handle, index) share computation
    - Each has unique request_id and per-request disconnect future
    - Workers check is_request_stale() before expensive operations
    - Stale requests (request_id < latest) are skipped
    """
    def __init__(self):
        # Per (handle, index): stores current request sequence and active count
        # Structure: { (handle, index): {"seq": int, "count": int, "disconnect_future": Future} }
        self._tracking: DefaultDict[Tuple[str, int], Dict] = defaultdict(lambda: {"seq": 0, "count": 0, "disconnect_future": None})
        
        # Per-request disconnect futures: { (handle, index, request_id): Future }
        self._per_request_futures: Dict[Tuple[str, int, int], asyncio.Future] = {}
        
        self._lock = asyncio.Lock()
    
    async def register_request(self, handle: str, index: int) -> int:
        """
        Register a new request for (handle, index).
        Returns the request_id for this request.
        Creates both shared and per-request disconnect futures.
        """
        key = (handle, index)
        locks_logger.debug(f"[DisconnectTracker] Acquiring lock for register_request key={handle[:8]}:{index}")
        async with self._lock:
            locks_logger.debug(f"[DisconnectTracker] Acquired lock for register_request key={handle[:8]}:{index}")
            entry = self._tracking[key]
            entry["seq"] += 1
            entry["count"] += 1
            request_id = entry["seq"]
            
            # Create shared disconnect future if it doesn't exist
            if entry["disconnect_future"] is None or entry["disconnect_future"].done():
                loop = asyncio.get_running_loop()
                entry["disconnect_future"] = loop.create_future()
            
            # Create per-request disconnect future
            req_key = (handle, index, request_id)
            loop = asyncio.get_running_loop()
            self._per_request_futures[req_key] = loop.create_future()
            
            locks_logger.debug(f"[DisconnectTracker] Registered request_id={request_id} for {handle[:8]}:{index}, count={entry['count']}")
            locks_logger.debug(f"[DisconnectTracker] Releasing lock for register_request key={handle[:8]}:{index}")
            return request_id
    
    async def unregister_request(self, handle: str, index: int, request_id: int) -> None:
        """
        Unregister a request (called on disconnect or completion).
        When count reaches 0, the shared disconnect future is resolved.
        Also cleans up the per-request future.
        """
        key = (handle, index)
        req_key = (handle, index, request_id)
        locks_logger.debug(f"[DisconnectTracker] Acquiring lock for unregister_request key={handle[:8]}:{index} req_id={request_id}")
        async with self._lock:
            locks_logger.debug(f"[DisconnectTracker] Acquired lock for unregister_request key={handle[:8]}:{index} req_id={request_id}")
            entry = self._tracking[key]
            entry["count"] = max(0, entry["count"] - 1)
            
            locks_logger.debug(f"[DisconnectTracker] Unregistered request_id={request_id} for {handle[:8]}:{index}, count={entry['count']}")
            
            # Clean up per-request future
            if req_key in self._per_request_futures:
                per_req_future = self._per_request_futures.pop(req_key)
                if not per_req_future.done():
                    per_req_future.set_result(True)
            
            # If count reaches 0, resolve the shared disconnect future
            if entry["count"] == 0 and entry["disconnect_future"] and not entry["disconnect_future"].done():
                entry["disconnect_future"].set_result(True)
                locks_logger.debug(f"[DisconnectTracker] Shared disconnect future resolved for {handle[:8]}:{index}")
            
            locks_logger.debug(f"[DisconnectTracker] Releasing lock for unregister_request key={handle[:8]}:{index} req_id={request_id}")
    
    async def get_disconnect_future(self, handle: str, index: int, request_id: Optional[int] = None) -> asyncio.Future:
        """
        Get the disconnect future for (handle, index) or a specific request.
        
        If request_id is provided, returns the per-request disconnect future.
        Otherwise, returns the shared disconnect future that resolves when ALL requests disconnect.
        """
        key = (handle, index)
        locks_logger.debug(f"[DisconnectTracker] Acquiring lock for get_disconnect_future key={handle[:8]}:{index} req_id={request_id}")
        async with self._lock:
            locks_logger.debug(f"[DisconnectTracker] Acquired lock for get_disconnect_future key={handle[:8]}:{index} req_id={request_id}")
            
            if request_id is not None:
                # Return per-request future
                req_key = (handle, index, request_id)
                if req_key not in self._per_request_futures:
                    loop = asyncio.get_running_loop()
                    self._per_request_futures[req_key] = loop.create_future()
                locks_logger.debug(f"[DisconnectTracker] Releasing lock for get_disconnect_future key={handle[:8]}:{index} req_id={request_id}")
                return self._per_request_futures[req_key]
            else:
                # Return shared future
                entry = self._tracking[key]
                if entry["disconnect_future"] is None:
                    loop = asyncio.get_running_loop()
                    entry["disconnect_future"] = loop.create_future()
                locks_logger.debug(f"[DisconnectTracker] Releasing lock for get_disconnect_future key={handle[:8]}:{index}")
                return entry["disconnect_future"]
    
    async def get_current_seq(self, handle: str, index: int) -> int:
        """Get the current sequence number for (handle, index)."""
        key = (handle, index)
        async with self._lock:
            return self._tracking[key]["seq"]
    
    async def cancel_request(self, handle: str, index: int, request_id: int) -> None:
        """
        Cancel a specific request by resolving its per-request disconnect future.
        This allows individual request cancellation without affecting duplicates.
        """
        req_key = (handle, index, request_id)
        locks_logger.debug(f"[DisconnectTracker] Acquiring lock for cancel_request key={handle[:8]}:{index}:{request_id}")
        async with self._lock:
            locks_logger.debug(f"[DisconnectTracker] Acquired lock for cancel_request key={handle[:8]}:{index}:{request_id}")
            if req_key in self._per_request_futures:
                future = self._per_request_futures[req_key]
                if not future.done():
                    future.set_result(True)
                    logger.debug(f"Cancelled per-request future for {handle[:8]}:{index}:{request_id}")
            locks_logger.debug(f"[DisconnectTracker] Releasing lock for cancel_request key={handle[:8]}:{index}:{request_id}")
    
    async def reset(self) -> None:
        """Reset all tracking (useful for testing)."""
        locks_logger.debug("[DisconnectTracker] Acquiring lock for reset")
        async with self._lock:
            locks_logger.debug("[DisconnectTracker] Acquired lock for reset")
            # Cancel all shared disconnect futures
            for entry in self._tracking.values():
                if entry["disconnect_future"] and not entry["disconnect_future"].done():
                    entry["disconnect_future"].cancel()
            # Cancel all per-request futures
            for future in self._per_request_futures.values():
                if not future.done():
                    future.cancel()
            self._tracking.clear()
            self._per_request_futures.clear()
            locks_logger.debug("[DisconnectTracker] Releasing lock for reset")


class TaskRegistry(Singleton):
    """
    Singleton registry for tracking waiting requests and managing notifications.
    
    ARCHITECTURE IMPLEMENTATION (THREADING_ARCHITECTURE.md:565-577):
    ✓ Unified Cancellation Hook: cancel_request() with per-request granularity
    ✓ Request ID Tracking: Keys are (handle, index, request_id) triples
    ✓ Last-Write-Wins: Newer request_ids supersede older via stale detection
    ✓ Individual Cancellation: Can cancel specific request without affecting duplicates
    
    STRUCTURE:
    - Registry: { (handle, index, request_id): Future }
    - All operations are async and protected by asyncio.Lock
    
    OPERATIONS:
    - wait_for_result(): Register future and wait (with timeout/disconnect racing)
    - notify_success(): Resolve future(s) on successful completion
    - notify_error(): Reject future(s) on error
    - cancel_request(): Unified cancellation hook for per-request cancellation
    - cancel_all_for_handle(): Cancel all requests for a handle (on inference failure)
    
    All operations must be called from the event loop (async context).
    Protected by asyncio.Lock for concurrent access from multiple coroutines.
    """
    def __init__(self):
        # Changed structure to include request_id in the key
        self._registry: DefaultDict[Tuple[str, int, int], asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def reset(self) -> None:
        """
        Reset the registry state (useful for testing or when event loop changes).
        Cancels all pending futures and clears the registry.
        """
        locks_logger.debug("[TaskRegistry] Acquiring lock for reset")
        async with self._lock:
            locks_logger.debug("[TaskRegistry] Acquired lock for reset")
            error = Exception("Registry reset")
            for f in self._registry.values():
                if not f.done():
                    f.set_exception(error)
            self._registry.clear()
            locks_logger.debug("[TaskRegistry] Releasing lock for reset")
    
    @property
    def lock(self) -> asyncio.Lock:
        return self._lock
    
    def notify_success(self, handle: str, index: int, request_id: Optional[int] = None) -> None:
        """
        Notify futures waiting for this (handle, index, request_id) of success.
        If request_id is None, notifies ALL futures for this (handle, index).
        Must be called from event loop context.
        """
        if request_id is not None:
            # Notify specific request
            key = (handle, index, request_id)
            if key in self._registry:
                f = self._registry.pop(key)
                if not f.done():
                    f.set_result(True)
        else:
            # Notify all requests for this handle/index
            keys_to_remove = [k for k in list(self._registry.keys()) if k[0] == handle and k[1] == index]
            for key in keys_to_remove:
                f = self._registry.pop(key)
                if not f.done():
                    f.set_result(True)
    
    def notify_error(self, handle: str, index: int, error: Exception, request_id: Optional[int] = None) -> None:
        """
        Notify futures waiting for this (handle, index, request_id) of error.
        If request_id is None, notifies ALL futures for this (handle, index).
        Must be called from event loop context.
        """
        if request_id is not None:
            # Notify specific request
            key = (handle, index, request_id)
            if key in self._registry:
                f = self._registry.pop(key)
                if not f.done():
                    f.set_exception(error)
        else:
            # Notify all requests for this handle/index
            keys_to_remove = [k for k in list(self._registry.keys()) if k[0] == handle and k[1] == index]
            for key in keys_to_remove:
                f = self._registry.pop(key)
                if not f.done():
                    f.set_exception(error)

    async def cancel_request(self, handle: str, index: int, request_id: int, reason: str = "Request cancelled") -> None:
        """
        Unified cancellation hook for a specific request identified by (handle, index, request_id).
        This implements the architecture requirement for per-request cancellation.
        
        This function:
        1. Cancels the TaskRegistry future for this specific request
        2. Attempts to remove queued jobs from inference/calc queues (if possible)
        3. Marks the job as stale so workers will skip it
        4. Triggers the per-request disconnect future
        
        Args:
            handle: Cache handle
            index: Data index
            request_id: Request ID to cancel
            reason: Reason for cancellation (for logging)
        """
        key = (handle, index, request_id)
        
        # 1. Cancel the waiting future in TaskRegistry
        if key in self._registry:
            f = self._registry.pop(key)
            if not f.done():
                f.set_exception(Exception(f"{reason}: {handle[:8]}:{index}:{request_id}"))
                logger.debug(f"Cancelled TaskRegistry future for {handle[:8]}:{index}:{request_id} - {reason}")
        
        # 2. Trigger per-request disconnect in DisconnectTracker
        # This marks the request as disconnected for any waiting operations
        disconnect_tracker = DisconnectTracker()
        await disconnect_tracker.cancel_request(handle, index, request_id)
        
        # 3. Note: Queue removal is not possible with asyncio.PriorityQueue
        # Instead, workers check staleness before processing (already implemented)
        # The is_request_stale() check in workers will catch this
        
        logger.info(f"Cancelled request {handle[:8]}:{index}:{request_id} - {reason}")

    def cancel_all_for_handle(self, handle: str, error_msg: str = None) -> None:
        """
        Cancel all waiting futures for a specific handle.
        Must be called from event loop context.
        
        Args:
            handle: The handle to cancel tasks for
            error_msg: Optional specific error message, otherwise uses generic message
        """
        if error_msg:
            error = Exception(f"Processing failed for handle {handle[:8]}...: {error_msg}")
        else:
            error = Exception(f"Cache invalidated for handle {handle}")
        
        keys_to_remove = [k for k in list(self._registry.keys()) if k[0] == handle]
        
        for key in keys_to_remove:
            f = self._registry.pop(key)
            if not f.done():
                f.set_exception(error)
    
    def cancel_all(self) -> None:
        """
        Cancel ALL waiting futures.
        Must be called from event loop context.
        """
        error = Exception("Global cache flush")
        for f in self._registry.values():
            if not f.done():
                f.set_exception(error)
        self._registry.clear()
    
    async def wait_for_result(
        self, 
        handle: str, 
        index: int, 
        request_id: int,
        disconnect_future: Optional[asyncio.Future] = None,
        timeout: float = None
    ) -> None:
        """Register a future and wait for the result with timeout and disconnect handling.
        
        Args:
            handle: Cache handle
            index: Data index
            request_id: Request ID for this specific request
            disconnect_future: Optional disconnect future to race against
            timeout: Timeout in seconds (None = use device-specific default from engine.INFERENCE_TIMEOUT)
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            asyncio.CancelledError: If request is disconnected
        """
        # Import here to avoid circular dependency
        if timeout is None:
            from glurpc.engine import INFERENCE_TIMEOUT
            timeout = INFERENCE_TIMEOUT
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        key = (handle, index, request_id)
        locks_logger.debug(f"[TaskRegistry] Acquiring lock for wait_for_result key={handle[:8]}:{index}:{request_id}")
        async with self._lock:
            locks_logger.debug(f"[TaskRegistry] Acquired lock for wait_for_result key={handle[:8]}:{index}:{request_id}")
            self._registry[key] = future
            locks_logger.debug(f"[TaskRegistry] Releasing lock for wait_for_result key={handle[:8]}:{index}:{request_id}")
        
        try:
            # Race between: timeout, disconnect, and work completion
            tasks = [asyncio.create_task(asyncio.wait_for(future, timeout=timeout))]
            
            if disconnect_future:
                # Add the future directly - asyncio.wait() accepts both Tasks and Futures
                tasks.append(disconnect_future)
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check which task completed first
            completed_task = done.pop()
            
            if disconnect_future and completed_task == tasks[1] if len(tasks) > 1 else False:
                # Disconnect happened first
                raise asyncio.CancelledError("Client disconnected")
            
            # Otherwise, the work completed (or timed out)
            await completed_task
            
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            # Clean up the future from registry on timeout OR cancellation
            locks_logger.debug(f"[TaskRegistry] Acquiring lock for wait_for_result cleanup key={handle[:8]}:{index}:{request_id}")
            async with self._lock:
                locks_logger.debug(f"[TaskRegistry] Acquired lock for wait_for_result cleanup key={handle[:8]}:{index}:{request_id}")
                if key in self._registry:
                    del self._registry[key]
                locks_logger.debug(f"[TaskRegistry] Releasing lock for wait_for_result cleanup key={handle[:8]}:{index}:{request_id}")
            raise e


# Convenience functions for backward compatibility and cleaner code
def notify_success(handle: str, index: int, request_id: Optional[int] = None) -> None:
    """Notify futures waiting for this (handle, index, request_id)"""
    TaskRegistry().notify_success(handle, index, request_id)


def notify_error(handle: str, index: int, error: Exception, request_id: Optional[int] = None) -> None:
    """Notify futures waiting for this (handle, index, request_id) of error"""
    TaskRegistry().notify_error(handle, index, error, request_id)


async def wait_for_result(
    handle: str, 
    index: int, 
    request_id: int,
    disconnect_future: Optional[asyncio.Future] = None,
    timeout: float = None
) -> None:
    """Register a future and wait for the result with timeout."""
    await TaskRegistry().wait_for_result(handle, index, request_id, disconnect_future, timeout)
