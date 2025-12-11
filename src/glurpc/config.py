"""
Configuration constants for GluRPC service.
Simple, static configuration values only.
"""
import os
import logging
from glurpc.data_classes import GluformerInferenceConfig
# --- Data Processing Configuration ---

DEFAULT_CONFIG = GluformerInferenceConfig()

STEP_SIZE_MINUTES: int = DEFAULT_CONFIG.time_step
"""Time step in minutes for model input data."""

DEFAULT_INPUT_CHUNK_LENGTH: int = DEFAULT_CONFIG.input_chunk_length
"""Default input chunk length for model."""

DEFAULT_OUTPUT_CHUNK_LENGTH: int = DEFAULT_CONFIG.output_chunk_length
"""Default output chunk length for model."""
# Model requirements (based on architecture: 96 input + 12 output)
MINIMUM_DURATION_MINUTES_MODEL: int = STEP_SIZE_MINUTES * (DEFAULT_INPUT_CHUNK_LENGTH + DEFAULT_OUTPUT_CHUNK_LENGTH)
"""Minimum duration required by the model architecture (in minutes)."""

MAXIMUM_WANTED_DURATION_DEFAULT: int = MINIMUM_DURATION_MINUTES_MODEL * 2
"""Default maximum duration for data processing (in minutes)."""

# --- Cache Configuration ---
MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "128"))
"""Maximum number of datasets to cache."""

ENABLE_CACHE_PERSISTENCE: bool = os.getenv("ENABLE_CACHE_PERSISTENCE", "True").lower() in ("true", "1", "yes")
"""Enable/disable cache persistence to disk (useful to disable for testing)."""

# Runtime overridable duration limits (with validation)
MINIMUM_DURATION_MINUTES: int = int(os.getenv("MINIMUM_DURATION_MINUTES", str(MINIMUM_DURATION_MINUTES_MODEL)))
"""Minimum duration for processing (configurable via env)."""

MAXIMUM_WANTED_DURATION: int = int(os.getenv("MAXIMUM_WANTED_DURATION", str(MAXIMUM_WANTED_DURATION_DEFAULT)))
"""Maximum wanted duration for processing (configurable via env)."""

# Validation
if MINIMUM_DURATION_MINUTES < MINIMUM_DURATION_MINUTES_MODEL:
    raise ValueError(f"MINIMUM_DURATION_MINUTES must be greater than {MINIMUM_DURATION_MINUTES_MODEL}")
if MAXIMUM_WANTED_DURATION < MINIMUM_DURATION_MINUTES:
    raise ValueError(f"MAXIMUM_WANTED_DURATION must be greater than {MINIMUM_DURATION_MINUTES}")

# --- API Configuration ---
ENABLE_API_KEYS: bool = os.getenv("ENABLE_API_KEYS", "False").lower() in ("true", "1", "yes")
"""Enable/disable API key authentication."""

# --- Model and Inference Configuration ---
NUM_COPIES_PER_DEVICE: int = int(os.getenv("NUM_COPIES_PER_DEVICE", "2"))
"""Number of model copies per GPU device."""

BACKGROUND_WORKERS_COUNT: int = int(os.getenv("BACKGROUND_WORKERS_COUNT", "4"))
"""Number of background workers for calculation tasks."""

BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
"""Batch size for inference."""

NUM_SAMPLES: int = int(os.getenv("NUM_SAMPLES", "10"))
"""Number of Monte Carlo samples for uncertainty estimation."""

# --- Timeout Configuration ---
# Timeouts are device-specific (set dynamically in engine.py based on CUDA availability)
INFERENCE_TIMEOUT_GPU: float = float(os.getenv("INFERENCE_TIMEOUT_GPU", "600.0"))
"""Timeout in seconds for GPU inference (default 10 minutes)."""

INFERENCE_TIMEOUT_CPU: float = float(os.getenv("INFERENCE_TIMEOUT_CPU", "7200.0"))
"""Timeout in seconds for CPU inference (default 120 minutes - CPUs are ~100x slower)."""

# Will be set dynamically based on detected device
INFERENCE_TIMEOUT: float = INFERENCE_TIMEOUT_CPU  # Conservative default, overridden in engine.py

# --- Queue Configuration ---
MAX_INFERENCE_QUEUE_SIZE: int = int(os.getenv("MAX_INFERENCE_QUEUE_SIZE", "64"))
"""Maximum number of inference tasks allowed in the queue to prevent flooding."""

MAX_CALC_QUEUE_SIZE: int = int(os.getenv("MAX_CALC_QUEUE_SIZE", "8192"))
"""Maximum number of calculation tasks allowed in the queue to prevent flooding."""

# --- Logging Configuration ---
# Log levels: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
# Logger structure: glurpc.{module}.{task_category}
# Exception: glurpc.locks is app-wide for all lock operations
# Task categories: calc (plot calculations), infer (ML inference), data (preprocessing/dataset creation)

def _parse_log_level(level_str: str, default: int = logging.INFO) -> int:
    """Parse log level from string or int."""
    level_str = level_str.upper()
    if level_str.isdigit():
        return int(level_str)
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str, default)

LOG_LEVEL_ROOT: int = _parse_log_level(os.getenv("LOG_LEVEL_ROOT", "INFO"))
"""Root logger level (glurpc) - inherited by all subloggers."""

LOG_LEVEL_LOGIC: int = _parse_log_level(os.getenv("LOG_LEVEL_LOGIC", "INFO"))
"""Logic module logger level (glurpc.logic and subdomains: .calc, .infer, .data)."""

LOG_LEVEL_ENGINE: int = _parse_log_level(os.getenv("LOG_LEVEL_ENGINE", "INFO"))
"""Engine module logger level (glurpc.engine and subdomains: .infer, .calc, .data)."""

LOG_LEVEL_CORE: int = _parse_log_level(os.getenv("LOG_LEVEL_CORE", "INFO"))
"""Core module logger level (glurpc.core)."""

LOG_LEVEL_APP: int = _parse_log_level(os.getenv("LOG_LEVEL_APP", "INFO"))
"""App module logger level (glurpc.app)."""

LOG_LEVEL_STATE: int = _parse_log_level(os.getenv("LOG_LEVEL_STATE", "INFO"))
"""State module logger level (glurpc.state)."""

LOG_LEVEL_CACHE: int = _parse_log_level(os.getenv("LOG_LEVEL_CACHE", "INFO"))
"""Cache module logger level (glurpc.cache)."""

LOG_LEVEL_LOCKS: int = _parse_log_level(os.getenv("LOG_LEVEL_LOCKS", "ERROR"))
"""App-wide lock operations logger level (glurpc.locks) - defaults to ERROR to reduce noise."""