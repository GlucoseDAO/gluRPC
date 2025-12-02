import asyncio
import datetime
import os
import pickle
import shutil
import uuid
from collections import defaultdict
from typing import Dict, Any, List, Tuple, DefaultDict, Optional

import polars as pl

# Dependencies from glurpc
from glurpc.config import (
    MAX_CACHE_SIZE,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION,
    STEP_SIZE_MINUTES,
    ENABLE_CACHE_PERSISTENCE
)
from glurpc.data_classes import RESULT_SCHEMA


class SingletonMeta(type):
    """
    A metaclass that creates a Singleton base type when called.
    """
    _instances: Dict[str, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls.__qualname__ not in cls._instances:
            cls._instances[cls.__qualname__] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls.__qualname__]


class StateManager(metaclass=SingletonMeta):
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


class DataCache(metaclass=SingletonMeta):
    """
    Singleton cache for storing dataset information, forecasts, and results.
    Stores immutable input data: { handle: { 'dataset': ..., 'scalers': ..., 'model_config': ..., 'data_df': ... } }
    Handles persistence to disk (Parquet + Pickle).
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        # Persistence Setup
        self._cache_dir = os.path.join(os.getcwd(), "cache_storage")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Metadata Index: List of {handle, start_time, end_time, ...}
        self._metadata_index: List[Dict[str, Any]] = []
        self._load_metadata_index()
    
    @property
    def lock(self) -> asyncio.Lock:
        return self._lock
        
    def _load_metadata_index(self):
        """Load the metadata index from disk."""
        index_path = os.path.join(self._cache_dir, "index.pkl")
        if os.path.exists(index_path):
            try:
                with open(index_path, "rb") as f:
                    self._metadata_index = pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache index: {e}")
                self._metadata_index = []
    
    def _save_metadata_index(self):
        """Save the metadata index to disk."""
        index_path = os.path.join(self._cache_dir, "index.pkl")
        try:
            with open(index_path, "wb") as f:
                pickle.dump(self._metadata_index, f)
        except Exception as e:
            print(f"Failed to save cache index: {e}")

    async def get(self, handle: str) -> Optional[Dict[str, Any]]:
        """Retrieve data for a handle. Checks memory first, then disk (if enabled)."""
        async with self._lock:
            # 1. Check Memory
            if handle in self._cache:
                return self._cache.get(handle)
            
            # 2. Check Disk (only if persistence enabled)
            if ENABLE_CACHE_PERSISTENCE:
                return await self._load_from_disk_internal(handle)
            
            return None

    async def _load_from_disk_internal(self, handle: str) -> Optional[Dict[str, Any]]:
        """Load from disk into memory (internal, expects lock held)."""
        handle_dir = os.path.join(self._cache_dir, handle)
        if not os.path.exists(handle_dir):
            return None
            
        try:
            # Load DF
            df_path = os.path.join(handle_dir, "data.parquet")
            if not os.path.exists(df_path): return None
            data_df = pl.read_parquet(df_path)
            
            # Load Meta
            meta_path = os.path.join(handle_dir, "meta.pkl")
            if not os.path.exists(meta_path): return None
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            
            data['data_df'] = data_df
            
            # Ensure version exists
            if 'version' not in data:
                data['version'] = str(uuid.uuid4())

            # Store in Memory (evict if needed)
            if len(self._cache) >= MAX_CACHE_SIZE:
                # FIFO Eviction from Memory Only
                key_to_remove = next(iter(self._cache))
                del self._cache[key_to_remove]
            
            self._cache[handle] = data
            return data
        except Exception as e:
            print(f"Error loading cache for {handle}: {e}")
            return None

    async def set(self, handle: str, data: Dict[str, Any]) -> None:
        """Store data for a handle, with automatic eviction if cache is full. Persists to disk if enabled."""
        async with self._lock:
            if len(self._cache) >= MAX_CACHE_SIZE:
                # FIFO Eviction
                key_to_remove = next(iter(self._cache))
                del self._cache[key_to_remove]
                # Notify any pending tasks for this handle that it's gone
                TaskRegistry().cancel_all_for_handle(key_to_remove)
            
            # Assign a unique version ID to this cache entry
            if 'version' not in data:
                data['version'] = str(uuid.uuid4())

            self._cache[handle] = data
            
            # Persist to disk (only if persistence enabled)
            if ENABLE_CACHE_PERSISTENCE:
                await self._save_to_disk_internal(handle, data)

    async def _save_to_disk_internal(self, handle: str, data: Dict[str, Any]):
        """Save to disk (internal, expects lock held)."""
        try:
            handle_dir = os.path.join(self._cache_dir, handle)
            os.makedirs(handle_dir, exist_ok=True)
            
            # Save DataFrame
            data['data_df'].write_parquet(os.path.join(handle_dir, "data.parquet"))
            
            # Save Metadata
            # Filter out non-serializable if any (data_df is removed from dict before pickle)
            meta = {k: v for k, v in data.items() if k != 'data_df'}
            with open(os.path.join(handle_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
                
            # Update Index
            entry = {
                'handle': handle,
                'timestamp': data.get('timestamp'),
                'start_time': data.get('start_time'),
                'end_time': data.get('end_time'),
                # Add content signature if available
                'content_signature': data.get('content_signature') 
            }
            
            # Remove old entry if exists
            self._metadata_index = [x for x in self._metadata_index if x['handle'] != handle]
            self._metadata_index.append(entry)
            self._save_metadata_index()
            
        except Exception as e:
            print(f"Error saving cache for {handle}: {e}")

    async def contains(self, handle: str) -> bool:
        """Check if handle exists in cache (memory or disk)."""
        async with self._lock:
            if handle in self._cache:
                return True
            return os.path.exists(os.path.join(self._cache_dir, handle))

    async def update_data_df(self, handle: str, data_df: pl.DataFrame) -> None:
        """Update the data_df for a handle. Also updates disk."""
        async with self._lock:
            if handle in self._cache:
                self._cache[handle]['data_df'] = data_df
                # Persist update (could be optimized to not rewrite everything)
                # For now, just rewrite parquet
                try:
                    handle_dir = os.path.join(self._cache_dir, handle)
                    if os.path.exists(handle_dir):
                         data_df.write_parquet(os.path.join(handle_dir, "data.parquet"))
                except Exception as e:
                     print(f"Error updating cache disk for {handle}: {e}")

    async def get_data_entry(self, handle: str) -> Optional[Dict[str, Any]]:
        """Get full data entry (for internal use)."""
        return await self.get(handle)
    
    def get_sync(self, handle: str) -> Optional[Dict[str, Any]]:
        """Synchronous get (use only when already holding lock or in sync context). Only checks memory."""
        return self._cache.get(handle)
    
    def contains_sync(self, handle: str) -> bool:
        """Synchronous contains check (use only when already holding lock). Only checks memory."""
        return handle in self._cache
    
    def set_sync(self, handle: str, data: Dict[str, Any]) -> None:
        """Synchronous set (use only when already holding lock). Only sets memory."""
        if 'version' not in data:
            data['version'] = str(uuid.uuid4())
        self._cache[handle] = data
    
    async def get_size(self) -> int:
        """Get the current size of the cache (memory + disk unique)."""
        async with self._lock:
            disk_handles = set([d for d in os.listdir(self._cache_dir) if os.path.isdir(os.path.join(self._cache_dir, d))])
            memory_handles = set(self._cache.keys())
            return len(disk_handles.union(memory_handles))
    
    def get_size_sync(self) -> int:
        """Synchronous get size (use only when already holding lock)."""
        return len(self._cache)
    
    async def find_superset(self, start_time: datetime.datetime, end_time: datetime.datetime) -> Optional[str]:
        """
        Find a cached dataset that is a superset of the requested time range.
        Criteria:
        1. Cached End Time == Requested End Time (assuming alignment by end)
        2. Cached Start Time <= Requested Start Time
        Returns the handle of the superset if found.
        """
        async with self._lock:
            for entry in self._metadata_index:
                c_start = entry.get('start_time')
                c_end = entry.get('end_time')
                
                if c_start and c_end:
                    # Allow small tolerance for float/datetime comparison? 
                    # Assuming exact match for end time as per requirements ("align by end")
                    if c_end == end_time and c_start <= start_time:
                         return entry['handle']
            return None

    async def delete_handle(self, handle: str) -> bool:
        """
        Delete a specific handle from cache (memory and disk).
        Returns True if handle was found and deleted, False otherwise.
        """
        async with self._lock:
            found = False
            
            # Remove from memory cache
            if handle in self._cache:
                del self._cache[handle]
                found = True
            
            # Remove from disk
            handle_dir = os.path.join(self._cache_dir, handle)
            if os.path.exists(handle_dir):
                shutil.rmtree(handle_dir)
                found = True
            
            # Remove from metadata index
            original_len = len(self._metadata_index)
            self._metadata_index = [x for x in self._metadata_index if x['handle'] != handle]
            if len(self._metadata_index) < original_len:
                self._save_metadata_index()
                found = True
            
            # Cancel all pending tasks for this handle
            if found:
                TaskRegistry().cancel_all_for_handle(handle)
            
            return found
    
    async def save_to_disk(self, handle: Optional[str] = None) -> int:
        """
        Save cache entries to disk on-demand.
        If handle is specified, saves only that handle.
        If handle is None, saves all in-memory cache entries.
        Returns the number of entries saved.
        """
        if not ENABLE_CACHE_PERSISTENCE:
            return 0
        
        async with self._lock:
            if handle is not None:
                # Save specific handle
                if handle in self._cache:
                    await self._save_to_disk_internal(handle, self._cache[handle])
                    return 1
                return 0
            else:
                # Save all in-memory entries
                count = 0
                for h, data in self._cache.items():
                    await self._save_to_disk_internal(h, data)
                    count += 1
                return count
    
    async def load_from_disk(self, handle: str) -> bool:
        """
        Load a specific handle from disk into memory on-demand.
        Returns True if successfully loaded, False if not found or error.
        """
        if not ENABLE_CACHE_PERSISTENCE:
            return False
        
        async with self._lock:
            # Check if already in memory
            if handle in self._cache:
                return True
            
            # Try to load from disk
            data = await self._load_from_disk_internal(handle)
            return data is not None

    async def clear_cache(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
            self._metadata_index = []
            if os.path.exists(self._cache_dir):
                shutil.rmtree(self._cache_dir)
            os.makedirs(self._cache_dir, exist_ok=True)
        
        # Cancel all pending tasks
        TaskRegistry().cancel_all()


class TaskRegistry(metaclass=SingletonMeta):
    """
    Singleton registry for tracking waiting requests and managing notifications.
    Tracks: { (handle, index): [Future, Future, ...] }
    """
    def __init__(self):
        self._registry: DefaultDict[Tuple[str, int], List[asyncio.Future]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    @property
    def lock(self) -> asyncio.Lock:
        return self._lock
    
    def notify_success(self, handle: str, index: int) -> None:
        """Notify all futures waiting for this (handle, index) of success."""
        key = (handle, index)
        if key in self._registry:
            futures = self._registry.pop(key)
            for f in futures:
                if not f.done():
                    f.set_result(True)
    
    def notify_error(self, handle: str, index: int, error: Exception) -> None:
        """Notify all futures waiting for this (handle, index) of error."""
        key = (handle, index)
        if key in self._registry:
            futures = self._registry.pop(key)
            for f in futures:
                if not f.done():
                    f.set_exception(error)

    def cancel_all_for_handle(self, handle: str) -> None:
        """Cancel all waiting futures for a specific handle."""
        keys_to_remove = [k for k in self._registry.keys() if k[0] == handle]
        error = Exception(f"Cache invalidated for handle {handle}")
        
        for key in keys_to_remove:
            if key in self._registry:
                futures = self._registry.pop(key)
                for f in futures:
                    if not f.done():
                        f.set_exception(error)

    
    def cancel_all(self) -> None:
        """Cancel ALL waiting futures."""
        error = Exception("Global cache flush")
        for futures in self._registry.values():
            for f in futures:
                if not f.done():
                    f.set_exception(error)
        self._registry.clear()
    
    async def wait_for_result(self, handle: str, index: int) -> None:
        """Register a future and wait for the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        key = (handle, index)
        async with self._lock:
            self._registry[key].append(future)
        
        await future


# Convenience functions for backward compatibility and cleaner code
def notify_success(handle: str, index: int) -> None:
    """Notify all futures waiting for this (handle, index)"""
    TaskRegistry().notify_success(handle, index)


def notify_error(handle: str, index: int, error: Exception) -> None:
    """Notify all futures waiting for this (handle, index) of error"""
    TaskRegistry().notify_error(handle, index, error)


async def wait_for_result(handle: str, index: int) -> None:
    """Register a future and wait for the result."""
    await TaskRegistry().wait_for_result(handle, index)
