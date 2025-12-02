import asyncio
import os
from typing import Dict, Any, List, Tuple, DefaultDict
import numpy as np
from collections import defaultdict
import pandas as pd
from glurpc.data_classes import PlotData

# Config
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "128"))

# Data Cache: Stores immutable input data
# { handle: { 'dataset': ..., 'scalers': ..., 'timestamp': ... } }
DATA_CACHE: Dict[str, Any] = {}
DATA_CACHE_LOCK = asyncio.Lock()

# Forecast Cache: Stores intermediate inference results (heavy compute)
# { handle: Dict[int, np.ndarray] }
# We use a dict of indices to allow both sparse (quick plot) and full population
FORECAST_CACHE: Dict[str, Dict[int, np.ndarray]] = {}
FORECAST_CACHE_LOCK = asyncio.Lock()

# Result Cache: Stores final plots (cheap compute)
# { handle: { index: PlotData } }
RESULT_CACHE: Dict[str, Dict[int, PlotData]] = {}
RESULT_CACHE_LOCK = asyncio.Lock()

# Task Registry: Tracks waiting requests
# { (handle, index): [Future, Future, ...] }
TASK_REGISTRY: DefaultDict[Tuple[str, int], List[asyncio.Future]] = defaultdict(list)
TASK_REGISTRY_LOCK = asyncio.Lock()

# Notification Helpers

def notify_success(handle: str, index: int):
    """Notify all futures waiting for this (handle, index)"""
    key = (handle, index)
    if key in TASK_REGISTRY:
        futures = TASK_REGISTRY.pop(key)
        for f in futures:
            if not f.done():
                f.set_result(True)

def notify_error(handle: str, index: int, error: Exception):
    """Notify all futures waiting for this (handle, index) of error"""
    key = (handle, index)
    if key in TASK_REGISTRY:
        futures = TASK_REGISTRY.pop(key)
        for f in futures:
            if not f.done():
                f.set_exception(error)

async def wait_for_result(handle: str, index: int):
    """Register a future and wait for the result."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    key = (handle, index)
    async with TASK_REGISTRY_LOCK:
        TASK_REGISTRY[key].append(future)
        
    await future
