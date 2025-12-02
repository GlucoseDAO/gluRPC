"""
Configuration constants for GluRPC service.
Simple, static configuration values only.
"""
import os
from glurpc.data_classes import GluformerInferenceConfig
# --- Data Processing Configuration ---

DEFAULT_CONFIG = GluformerInferenceConfig()

STEP_SIZE_MINUTES: int = DEFAULT_CONFIG.time_step
"""Time step in minutes for model input data."""

# Model requirements (based on architecture: 96 input + 12 output)
MINIMUM_DURATION_MINUTES_MODEL: int = STEP_SIZE_MINUTES * (DEFAULT_CONFIG.input_chunk_length + DEFAULT_CONFIG.output_chunk_length)
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