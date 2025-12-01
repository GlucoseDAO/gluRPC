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
        description="The handle returned by the process_unified endpoint"
    )
    index: int = Field(
        ..., 
        description="The index of the sample in the dataset to plot (0-based)"
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
