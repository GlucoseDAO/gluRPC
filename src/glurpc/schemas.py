"""REST API request and response models for GluRPC."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from cgm_format.interface import ProcessingWarning, WarningDescription


class FormattedWarnings(BaseModel):
    """
    Structured representation of data processing warnings.
    Warnings are generated during data validation and preprocessing.
    Each warning type is represented as a boolean flag.
    """
    model_config = ConfigDict(frozen=True)
    
    has_warnings: bool = Field(
        ...,
        description="True if any warnings were raised during processing"
    )
    too_short: bool = Field(
        default=False,
        description="Data duration is too short for reliable predictions"
    )
    calibration: bool = Field(
        default=False,
        description="Calibration events detected in the data"
    )
    quality: bool = Field(
        default=False,
        description="Low quality data points detected"
    )
    imputation: bool = Field(
        default=False,
        description="Missing values were imputed"
    )
    out_of_range: bool = Field(
        default=False,
        description="Glucose values outside normal range detected"
    )
    time_duplicates: bool = Field(
        default=False,
        description="Duplicate timestamps detected in the data"
    )
    messages: List[str] = Field(
        default_factory=list,
        description="Human-readable list of detailed warning messages",
        examples=[
            [],
            ["TOO_SHORT: Data duration is too short for reliable predictions"],
            ["QUALITY: Low quality data points detected", "IMPUTATION: Missing values were imputed"]
        ]
    )
    
    @classmethod
    def from_flags(cls, warning_flags: "ProcessingWarning") -> "FormattedWarnings":
        """
        Create FormattedWarnings from ProcessingWarning bitwise flags.
        
        Args:
            warning_flags: ProcessingWarning enum flags
            
        Returns:
            FormattedWarnings instance with decoded boolean fields
        """
        from cgm_format.interface import ProcessingWarning, WarningDescription
        
        warnings_list = []
        
        # Check each flag and build messages
        too_short = bool(warning_flags & ProcessingWarning.TOO_SHORT)
        if too_short:
            warnings_list.append(f"TOO_SHORT: {WarningDescription.TOO_SHORT.value}")
        
        calibration = bool(warning_flags & ProcessingWarning.CALIBRATION)
        if calibration:
            warnings_list.append(f"CALIBRATION: {WarningDescription.CALIBRATION.value}")
        
        quality = bool(warning_flags & ProcessingWarning.QUALITY)
        if quality:
            warnings_list.append(f"QUALITY: {WarningDescription.QUALITY.value}")
        
        imputation = bool(warning_flags & ProcessingWarning.IMPUTATION)
        if imputation:
            warnings_list.append(f"IMPUTATION: {WarningDescription.IMPUTATION.value}")
        
        out_of_range = bool(warning_flags & ProcessingWarning.OUT_OF_RANGE)
        if out_of_range:
            warnings_list.append(f"OUT_OF_RANGE: {WarningDescription.OUT_OF_RANGE.value}")
        
        time_duplicates = bool(warning_flags & ProcessingWarning.TIME_DUPLICATES)
        if time_duplicates:
            warnings_list.append(f"TIME_DUPLICATES: {WarningDescription.TIME_DUPLICATES.value}")
        
        return cls(
            has_warnings=len(warnings_list) > 0,
            too_short=too_short,
            calibration=calibration,
            quality=quality,
            imputation=imputation,
            out_of_range=out_of_range,
            time_duplicates=time_duplicates,
            messages=warnings_list
        )


class RequestMetrics(BaseModel):
    """
    Tracks HTTP request counts and timing for health reporting.
    Mutable model stored on app.state.
    """

    model_config = ConfigDict(frozen=False)

    total_http_requests: int = Field(
        default=0,
        ge=0,
        description="Total number of HTTP requests received since startup",
    )
    total_http_errors: int = Field(
        default=0,
        ge=0,
        description="Total number of HTTP errors (4xx, 5xx) since startup",
    )
    request_times: List[float] = Field(
        default_factory=list,
        description="Request durations in milliseconds",
    )


class UnifiedResponse(BaseModel):
    """
    Response model for CSV processing endpoint.
    Contains a handle to the cached dataset and any processing warnings.
    """
    model_config = ConfigDict(frozen=True)
    
    handle: Optional[str] = Field(
        default=None, 
        description="Unique hash handle to reference the processed dataset in subsequent requests",
        min_length=8,
        max_length=64,
        examples=["a1b2c3d4e5f6"]
    )
    total_samples: Optional[int] = Field(
        default=None,
        description="Total number of prediction samples in the processed dataset",
        ge=1,
        examples=[144]
    )
    warnings: FormattedWarnings = Field(
        default_factory=lambda: FormattedWarnings(
            has_warnings=False,
        ),
        description="Structured processing warnings with individual boolean flags for each warning type"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if processing failed, None on success"
    )


class PlotRequest(BaseModel):
    """
    Request model for generating a plot from a cached dataset.
    Uses negative indexing: 0 is the most recent sample, -1 is second-to-last, etc.
    """
    model_config = ConfigDict(frozen=True)
    
    handle: str = Field(
        ..., 
        description="The handle returned by the process_unified endpoint",
        min_length=8,
        max_length=64,
        examples=["a1b2c3d4e5f6"]
    )
    index: int = Field(
        ...,
        le=0,
        description="Sample index to plot (non-positive: 0 is last/most recent, -1 is second-to-last, etc.)",
        examples=[0, -10, -50]
    )
    force_calculate: bool = Field(
        default=False,
        description="If True, ignores cached plot and forces recalculation"
    )


class QuickPlotResponse(BaseModel):
    """
    Response model for the quick plot endpoint.
    Contains the Plotly figure as a JSON dict (compatible with Gradio gr.Plot).
    """
    model_config = ConfigDict(frozen=True)
    
    plot_data: Dict[str, Any] = Field(
        ..., 
        description="Plotly figure as JSON dict with 'data' and 'layout' keys (use with gr.Plot in Gradio)"
    )
    warnings: FormattedWarnings = Field(
        default_factory=lambda: FormattedWarnings(
            has_warnings=False,
        ),
        description="Structured processing warnings with individual boolean flags for each warning type"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if processing or plotting failed, None on success"
    )


class ConvertResponse(BaseModel):
    """
    Response model for CSV conversion endpoint.
    Converts proprietary formats to the Unified CSV format.
    """
    model_config = ConfigDict(frozen=True)
    
    csv_content: Optional[str] = Field(
        default=None, 
        description="The converted CSV content in Unified format (sequence_id, timestamp, glucose columns)"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if conversion failed, None on success"
    )


class ProcessRequest(BaseModel):
    """
    Request model for processing CSV data.
    CSV content must be base64 encoded and in Unified format.
    """
    model_config = ConfigDict(frozen=True)
    
    csv_base64: str = Field(
        ..., 
        description="Base64 encoded CSV content in Unified format (sequence_id, timestamp, glucose)",
        min_length=1,
        examples=["c2VxdWVuY2VfaWQsdGltZXN0YW1wLGdsdWNvc2UKMSwyMDI0LTAxLTAxIDAwOjAwOjAwLDEwMAo="]
    )
    force_calculate: bool = Field(
        default=False,
        description="If True, ignores existing cache and forces reprocessing/recalculation"
    )


class CacheManagementResponse(BaseModel):
    """
    Response model for cache management operations (flush, info, delete, save, load).
    """
    model_config = ConfigDict(frozen=True)
    
    success: bool = Field(
        ...,
        description="Whether the operation succeeded"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message describing the result"
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="Number of items currently in cache"
    )
    persisted_count: int = Field(
        default=0,
        ge=0,
        description="Number of persisted items"
    )
    items_affected: Optional[int] = Field(
        default=None,
        description="Number of cache items affected by the operation"
    )


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    Provides comprehensive statistics about service health and performance.
    """
    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ...,
        description="Service status: 'ok' (healthy), 'degraded' (issues), or 'error' (critical)",
        examples=["ok", "degraded", "error"]
    )
    load_status: str = Field(
        ...,
        description="Queue load status: 'idle' (0% full), 'lightly loaded' (25-50% full), 'heavily loaded' (50-75% full), 'overloaded' (75-95% full), 'full' (>95% full)",
        examples=["idle", "lightly loaded", "heavily loaded", "overloaded", "full"]
    )
    cache_size: int = Field(
        ...,
        ge=0,
        description="Number of items currently in inference cache"
    )
    models_initialized: bool = Field(
        ...,
        description="Whether ML models are loaded and ready for inference"
    )
    available_priority_models: int = Field(
        ...,
        ge=0,
        description="Number of available (idle) models in priority queue (model #0, reserved for priority 0 requests). Should be 1 when idle, 0 when in use."
    )
    available_general_models: int = Field(
        ...,
        ge=0,
        description="Number of available (idle) models in general queue (models #1+, for all requests). Higher values indicate more capacity."
    )
    avg_fulfillment_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average time to acquire a model for inference (milliseconds)"
    )
    vmem_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="Memory usage in MB (VRAM for GPU inference, RSS for CPU inference)"
    )
    device: str = Field(
        ...,
        description="Device used for inference",
        examples=["cpu", "cuda", "cuda:0"]
    )
    total_http_requests: int = Field(
        ...,
        ge=0,
        description="Total number of HTTP requests received since startup"
    )
    total_http_errors: int = Field(
        ...,
        ge=0,
        description="Total number of HTTP errors (4xx, 5xx) since startup"
    )
    avg_request_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average (mean) HTTP request time in milliseconds"
    )
    median_request_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Median HTTP request time in milliseconds"
    )
    min_request_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Minimum HTTP request time in milliseconds"
    )
    max_request_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Maximum HTTP request time in milliseconds"
    )
    inference_requests_by_priority: Dict[int, int] = Field(
        default_factory=dict,
        description="Inference requests grouped by priority level (0=interactive, 1=background, 3=prefetch)"
    )
    total_inference_errors: int = Field(
        ...,
        ge=0,
        description="Total number of inference errors encountered since startup"
    )
    total_calc_runs: int = Field(
        ...,
        ge=0,
        description="Total number of plot calculations completed successfully"
    )
    total_calc_errors: int = Field(
        ...,
        ge=0,
        description="Total number of plot calculation errors"
    )
    inference_queue_size: int = Field(
        ...,
        ge=0,
        description="Current number of tasks waiting in inference queue"
    )
    inference_queue_capacity: int = Field(
        ...,
        ge=0,
        description="Maximum capacity of inference queue"
    )
    calc_queue_size: int = Field(
        ...,
        ge=0,
        description="Current number of tasks waiting in calculation queue"
    )
    calc_queue_capacity: int = Field(
        ...,
        ge=0,
        description="Maximum capacity of calculation queue"
    )

