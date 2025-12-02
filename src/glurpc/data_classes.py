from pydantic import BaseModel, Field, ConfigDict
from typing import List
import polars as pl

class GluformerModelConfig(BaseModel):
    """
    Configuration that matches Gluformer model arguments exactly.
    Used to instantiate the model directly using **model_dump().
    """
    model_config = ConfigDict(frozen=True)
    
    d_model: int = Field(default=512, description="Model dimension")
    n_heads: int = Field(default=10, description="Number of attention heads")
    d_fcn: int = Field(default=1024, description="Fully connected layer dimension")
    num_enc_layers: int = Field(default=2, description="Number of encoder layers")
    num_dec_layers: int = Field(default=2, description="Number of decoder layers")
    
    len_seq: int = Field(..., description="Input sequence length (maps to input_chunk_length)")
    label_len: int = Field(..., description="Label length (usually len_seq // 3)")
    len_pred: int = Field(..., description="Prediction length (maps to output_chunk_length)")
    
    num_dynamic_features: int = Field(..., description="Number of dynamic features")
    num_static_features: int = Field(..., description="Number of static features")
    
    r_drop: float = Field(default=0.2, description="Dropout rate")
    activ: str = Field(default='gelu', description="Activation function")
    distil: bool = Field(default=True, description="Use distillation")

class GluformerInferenceConfig(BaseModel):
    """
    Input configuration for inference pipeline.
    Contains both processing parameters and base model architecture parameters.
    """
    model_config = ConfigDict(frozen=True)

    # Architecture defaults (can be overridden to match weights)
    d_model: int = Field(default=512, description="Model dimension")
    n_heads: int = Field(default=10, description="Number of attention heads")
    d_fcn: int = Field(default=1024, description="Fully connected layer dimension")
    num_enc_layers: int = Field(default=2, description="Number of encoder layers")
    num_dec_layers: int = Field(default=2, description="Number of decoder layers")
    
    # Sequence Lengths
    input_chunk_length: int = Field(default=96, description="Length of input sequence")
    output_chunk_length: int = Field(default=12, description="Length of output sequence")
    time_step: int = Field(default=5, description="Time step in minutes")

    # Feature Dimensions Defaults (Inferred from data during processing)
    num_dynamic_features: int = Field(default=6, description="Default number of dynamic features")
    num_static_features: int = Field(default=1, description="Default number of static features")
    
    # Data Processing
    gap_threshold: int = Field(default=45, description="Max gap in minutes to interpolate")
    min_drop_length: int = Field(default=12, description="Min length of segment to keep")
    interval_length: str = Field(default='5min', description="Interval length for interpolation")
    
    # Optional overrides for model defaults
    r_drop: float = Field(default=0.2, description="Dropout rate")
    activ: str = Field(default='gelu', description="Activation function")
    distil: bool = Field(default=True, description="Use distillation")

class FanChartData(BaseModel):
    """
    Data for a single fan chart slice (KDE distribution at a time point).
    """
    model_config = ConfigDict(frozen=True)
    
    x: List[float] = Field(..., description="X coordinates (density)")
    y: List[float] = Field(..., description="Y coordinates (value grid)")
    fillcolor: str = Field(..., description="Color string for filling")
    time_index: int = Field(..., description="Time index relative to forecast start")

class PlotData(BaseModel):
    """
    Aggregated data for rendering the prediction plot.
    """
    model_config = ConfigDict(frozen=True)
    
    true_values_x: List[int] = Field(..., description="X coordinates for true values line")
    true_values_y: List[float] = Field(..., description="Y coordinates for true values line")
    median_x: List[int] = Field(..., description="X coordinates for median forecast line")
    median_y: List[float] = Field(..., description="Y coordinates for median forecast line")
    fan_charts: List[FanChartData] = Field(..., description="List of fan chart slices")

# Polars DataFrame schema for result storage
RESULT_SCHEMA = {
    "index": pl.Int32,
    "forecast": pl.List(pl.Float64),
    # Plot Data Columns
    "true_values_x": pl.List(pl.Int32),
    "true_values_y": pl.List(pl.Float64),
    "median_x": pl.List(pl.Int32),
    "median_y": pl.List(pl.Float64),
    "fan_charts": pl.List(
        pl.Struct({
            "x": pl.List(pl.Float64),
            "y": pl.List(pl.Float64),
            "fillcolor": pl.Utf8,
            "time_index": pl.Int32
        })
    ),
    "is_calculated": pl.Boolean 
}