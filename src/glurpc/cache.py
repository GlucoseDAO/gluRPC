from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Generic, TypeVar, Optional, Hashable, Any, Callable

from os import makedirs
from cachetools import LRUCache as LRUMemCache
from diskcache import Index as DiskIndex
from aiorwlock import RWLock as AsyncRWLock

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

# Sentinel to mark keys as deleted in hot cache during concurrent operations
_TOMBSTONE = object()

# Module logger
cache_logger = logging.getLogger("glurpc.cache")

# App-wide lock logger for debugging lock operations
locks_logger = logging.getLogger("glurpc.locks")

class SingletonMeta(type):
    """
    A metaclass that creates a Singleton base type when called.
    """
    _instances: Dict[str, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls.__qualname__ not in cls._instances:
            cls._instances[cls.__qualname__] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls.__qualname__]


class Singleton(metaclass=SingletonMeta):
    """
    A singleton base class that can be used to create singletons.
    """
    
    @classmethod
    def free(cls) -> None:
        """Free the singleton instance (for testing)."""
        if hasattr(SingletonMeta, '_instances'):
            SingletonMeta._instances.pop(cls.__qualname__, None)

class HybridLRUCache(Generic[K, V], Singleton):
    """
    Hybrid cache:

    - Small in-memory LRU ("hot") of up to `max_hot` items.
    - Disk-backed Index as persistent store for all items.
    - Async API, safe for many concurrent tasks.
    """

    def __init__(self, directory: str, max_hot: int = 8) -> None:
        try:
            makedirs(directory, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create directory {directory}: {e}")
        self._backend = DiskIndex(directory)
        # We use Any for value type in LRUCache to accommodate _TOMBSTONE
        self._hot: LRUMemCache[K, V] = LRUMemCache(maxsize=max_hot)
        self._lock = AsyncRWLock()

    @asynccontextmanager
    async def _write_lock_logged(self, key: K, fn_name: str):
        k_str = str(key)[:10]
        locks_logger.debug(f"[{self.__class__.__name__}] Acquiring write lock for key={k_str} in {fn_name}")
        async with self._lock.writer:
            locks_logger.debug(f"[{self.__class__.__name__}] Acquired write lock for key={k_str} in {fn_name}")
            try:
                yield
            finally:
                locks_logger.debug(f"[{self.__class__.__name__}] Releasing write lock for key={k_str} in {fn_name}")

    @asynccontextmanager
    async def transaction(self, key: K):
        """
        Context manager for atomic read-modify-write operations acting as a critical section.
        Holds a write lock on the cache for the duration of the block to prevent race conditions.
        
        Usage:
            async with cache.transaction(key) as txn:
                # txn.value contains the current value (or None)
                # ... validation logic ...
                txn.set(new_value)
        """
        async with self._write_lock_logged(key, "transaction"):
            # 1. Fetch current value (Hot -> Disk)
            current_value = None
            if key in self._hot:
                val = self._hot[key]
                if val is not _TOMBSTONE:
                    current_value = val
                else:
                    current_value = None
            else:
                # Read from disk while holding write lock to ensure isolation
                # No await here because we are in critical section.
                current_value = self._backend.get(key, None)

            # 2. Setup context
            class TxnWrapper:
                def __init__(self, val: Optional[V]):
                    self.value = val
                    self._new_value: Any = None
                    self._should_update = False
                
                def set(self, value: V):
                    self._new_value = value
                    self._should_update = True
            
            ctx = TxnWrapper(current_value)
            
            yield ctx
            
            # 3. Apply changes if set() was called and no error occurred in yield
            if ctx._should_update:
                # Update hot
                self._hot[key] = ctx._new_value
                # Update disk
                self._backend[key] = ctx._new_value

    def _disk_get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._backend.get(key, default)
    
    def _disk_set(self, key: K, value: V) -> None:
        self._backend[key] = value
        return 

    def _disk_pop(self, key: K, default: Optional[V] = None) -> None:
        self._backend.pop(key, default)

    def _disk_clear(self) -> None:
        self._backend.clear()

    async def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value for key or None if not found.
        """
        # 1) Try hot LRU under read lock
        locks_logger.debug(f"[{self.__class__.__name__}] Acquiring read lock for key={str(key)[:10]} in get")
        async with self._lock.reader:
            locks_logger.debug(f"[{self.__class__.__name__}] Acquired read lock for key={str(key)[:10]} in get")
            if key in self._hot:
                val = self._hot[key]
                locks_logger.debug(f"[{self.__class__.__name__}] Releasing read lock for key={str(key)[:10]} in get (cache hit)")
                if val is _TOMBSTONE:
                    return default
                return val
            locks_logger.debug(f"[{self.__class__.__name__}] Releasing read lock for key={str(key)[:10]} in get (cache miss)")

        # 2) Miss in hot: load from disk in thread (no lock held)
        value: Optional[V] = await asyncio.to_thread(self._disk_get, key, default)

        # 3) If found on disk, promote to hot under write lock
        if value is not None:
            async with self._write_lock_logged(key, "get (from disk)"):
                # If key is not in hot (meaning not deleted or set concurrently), promote it.
                # If it IS in hot (value or TOMBSTONE), we leave it alone as that is newer.
                if key not in self._hot:
                    self._hot[key] = value
        return value

    async def set(self, key: K, value: V) -> None:
        """
        Store or overwrite value for key.
        """
        # Update hot LRU under write lock
        async with self._write_lock_logged(key, "set"):
            self._hot[key] = value

        # Persist to disk in background thread
        await asyncio.to_thread(self._disk_set, key, value)

    async def delete(self, key: K) -> None:
        """
        Remove key from cache if present (both hot and disk).
        """
        # Remove from hot first, marking as deleted to prevent re-population by concurrent reads
        async with self._write_lock_logged(key, "delete"):
            self._hot[key] = _TOMBSTONE

        # Remove from disk (ignore if missing)
        await asyncio.to_thread(self._disk_pop, key, None)

        # Remove tombstone
        async with self._write_lock_logged(key, "delete (after tombstone)"):
            if self._hot.get(key) is _TOMBSTONE:
                del self._hot[key]

    async def contains(self, key: K) -> bool:
        """
        Check if key exists (either hot or on disk).
        """
        # Quick check in hot
        locks_logger.debug(f"[{self.__class__.__name__}] Acquiring read lock for key={str(key)[:10]} in contains")
        async with self._lock.reader:
            locks_logger.debug(f"[{self.__class__.__name__}] Acquired read lock for key={str(key)[:10]} in contains")
            if key in self._hot:
                locks_logger.debug(f"[{self.__class__.__name__}] Releasing read lock for key={str(key)[:10]} in contains (found in hot)")
                if self._hot[key] is _TOMBSTONE:
                    return False
                return True
            locks_logger.debug(f"[{self.__class__.__name__}] Releasing read lock for key={str(key)[:10]} in contains (not in hot)")

        # Then check disk
        return key in self._backend

    async def clear(self) -> None:
        """
        Drop all cached entries.
        """
        locks_logger.debug(f"[{self.__class__.__name__}] Acquiring write lock for clear")
        async with self._lock.writer:
            locks_logger.debug(f"[{self.__class__.__name__}] Acquired write lock for clear")
            self._hot.clear()
            locks_logger.debug(f"[{self.__class__.__name__}] Releasing write lock for clear")
        await asyncio.to_thread(self._disk_clear)

    async def get_size(self) -> int:
        """
        Get the number of items in the disk cache (backend).
        """
        return len(self._backend)

    async def keys(self):
        """
        Async iterator over keys stored on disk.
        (Hot keys are always a subset of these.)
        """
        # Index is iterable over keys, but iteration is blocking,
        # so we wrap the whole thing in a thread and stream results.
        keys_list = await asyncio.to_thread(list, self._backend)
        for k in keys_list:
            yield k

    async def update_entry(self, key: K, updater: Callable[[V], V], default_factory: Callable[[], V] = None) -> None:
        """
        Update an entry in the cache using a callback function.
        If the key does not exist, uses default_factory to create initial value.
        This operation is atomic with respect to the cache lock.
        """
        async with self._write_lock_logged(key, "update_entry"):
            # 1. Try hot
            current_data = None
            if key in self._hot:
                current_data = self._hot[key]
                if current_data is _TOMBSTONE:
                    current_data = None
            
            # 2. If not in hot, try disk (we are holding write lock, so this blocks other ops, 
            # but ensures consistency for read-modify-write)
            if current_data is None:
                current_data = await asyncio.to_thread(self._disk_get, key, None)
            
            if current_data is None:
                if default_factory:
                    current_data = default_factory()
                else:
                    return # Or raise Error? Assuming no-op if not found and no default
            
            # 3. Apply update
            new_data = updater(current_data)
            
            # 4. Write back to hot
            self._hot[key] = new_data
            
            # 5. Write back to disk
            await asyncio.to_thread(self._disk_set, key, new_data)


