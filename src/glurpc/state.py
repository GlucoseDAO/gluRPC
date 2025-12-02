import asyncio
import datetime
from collections import defaultdict
from typing import Dict, Any, List, Tuple, DefaultDict, Optional

import polars as pl

# Dependencies from glurpc
from glurpc.config import (
    MAX_CACHE_SIZE,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION,
    STEP_SIZE_MINUTES
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
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    @property
    def lock(self) -> asyncio.Lock:
        return self._lock
    
    async def get(self, handle: str) -> Optional[Dict[str, Any]]:
        """Retrieve data for a handle."""
        async with self._lock:
            return self._cache.get(handle)
    
    async def set(self, handle: str, data: Dict[str, Any]) -> None:
        """Store data for a handle, with automatic eviction if cache is full."""
        async with self._lock:
            if len(self._cache) >= MAX_CACHE_SIZE:
                # FIFO Eviction
                key_to_remove = next(iter(self._cache))
                del self._cache[key_to_remove]
            
            self._cache[handle] = data
    
    async def contains(self, handle: str) -> bool:
        """Check if handle exists in cache."""
        async with self._lock:
            return handle in self._cache
    
    async def update_data_df(self, handle: str, data_df: pl.DataFrame) -> None:
        """Update the data_df for a handle."""
        async with self._lock:
            if handle in self._cache:
                self._cache[handle]['data_df'] = data_df
    
    async def get_data_entry(self, handle: str) -> Optional[Dict[str, Any]]:
        """Get full data entry (for internal use)."""
        async with self._lock:
            return self._cache.get(handle)
    
    def get_sync(self, handle: str) -> Optional[Dict[str, Any]]:
        """Synchronous get (use only when already holding lock or in sync context)."""
        return self._cache.get(handle)
    
    def contains_sync(self, handle: str) -> bool:
        """Synchronous contains check (use only when already holding lock)."""
        return handle in self._cache
    
    def set_sync(self, handle: str, data: Dict[str, Any]) -> None:
        """Synchronous set (use only when already holding lock)."""
        self._cache[handle] = data
    
    async def get_size(self) -> int:
        """Get the current size of the cache."""
        async with self._lock:
            return len(self._cache)
    
    def get_size_sync(self) -> int:
        """Synchronous get size (use only when already holding lock)."""
        return len(self._cache)


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
