from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class UnifiedResponse(BaseModel):
    """
    Response model for CSV processing endpoint.
    Contains a handle to the cached dataset and any processing warnings.
    """
    model_config = ConfigDict(frozen=True)
    
    handle: Optional[str] = Field(
        default=None, 
        description="Unique hash handle to reference the processed dataset in subsequent requests"
    )
    warnings: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dictionary of processing warnings (flags, messages)"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if processing failed"
    )

class PlotRequest(BaseModel):
    """
    Request model for generating a plot from a cached dataset.
    """
    model_config = ConfigDict(frozen=True)
    
    handle: str = Field(
        ..., 
        description="The handle returned by the process_unified endpoint",
        examples=["a1b2c3d4e5f6"]
    )
    index: int = Field(
        ..., 
        description="The index of the sample in the dataset to plot (0-based)",
        examples=[0]
    )

class QuickPlotResponse(BaseModel):
    """
    Response model for the quick plot endpoint.
    Contains the base64 encoded PNG image directly.
    """
    model_config = ConfigDict(frozen=True)
    
    plot_base64: str = Field(
        ..., 
        description="Base64 encoded PNG image of the plot"
    )
    warnings: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dictionary of processing warnings"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if processing or plotting failed"
    )

class ConvertResponse(BaseModel):
    """
    Response model for CSV conversion endpoint.
    """
    model_config = ConfigDict(frozen=True)
    
    csv_content: Optional[str] = Field(
        default=None, 
        description="The converted CSV content in Unified format"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if conversion failed"
    )

class ProcessRequest(BaseModel):
    """
    Request model for processing CSV data.
    CSV content should be base64 encoded.
    """
    model_config = ConfigDict(frozen=True)
    
    csv_base64: str = Field(
        ..., 
        description="Base64 encoded CSV content in Unified format",
        examples=["Y29sdW1uMSxjb2x1bW4yLGNvbHVtbjMKMSwxLjUsYQoyLDIuNSxiCg=="]
    )

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Service status ('ok', 'degraded', 'error')")
    cache_size: int = Field(..., description="Number of items currently in cache")
    models_initialized: bool = Field(..., description="Whether models are loaded")
    queue_length: int = Field(..., description="Current length of the model queue")
    avg_fulfillment_time_ms: float = Field(..., description="Average request fulfillment time in milliseconds")
    vmem_usage_mb: float = Field(..., description="VRAM usage in MB (if GPU available, else system RAM or 0)")
    device: str = Field(..., description="Device being used (cpu, cuda)")
    total_requests_processed: int = Field(..., description="Total number of requests processed since startup")
    total_errors: int = Field(..., description="Total number of errors encountered since startup")
