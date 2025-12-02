import asyncio
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, DefaultDict

import numpy as np
import pandas as pd

# Dependencies from glurpc
from glurpc.data_classes import PlotData, MINIMUM_DURATION_MINUTES_MODEL, MAXIMUM_WANTED_DURATION_DEFAULT, STEP_SIZE_MINUTES

# Config
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "128"))
MINIMUM_DURATION_MINUTES=int(os.getenv("MINIMUM_DURATION_MINUTES", MINIMUM_DURATION_MINUTES_MODEL))
MAXIMUM_WANTED_DURATION=int(os.getenv("MAXIMUM_WANTED_DURATION", MAXIMUM_WANTED_DURATION_DEFAULT))
if MINIMUM_DURATION_MINUTES < MINIMUM_DURATION_MINUTES_MODEL:
    raise ValueError(f"MINIMUM_DURATION_MINUTES must be greater than {MINIMUM_DURATION_MINUTES_MODEL}")
if MAXIMUM_WANTED_DURATION < MINIMUM_DURATION_MINUTES:
    raise ValueError(f"MAXIMUM_WANTED_DURATION must be greater than {MINIMUM_DURATION_MINUTES}")

# Shutdown flag for graceful worker exit
SHUTDOWN_STARTED = False

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
