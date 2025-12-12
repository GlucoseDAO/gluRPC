from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Dict, List, Literal, Optional, Union, Tuple, TypedDict, Self, TYPE_CHECKING
import uuid
import polars as pl
import numpy as np
from numpy import float64, float32
from numpydantic import NDArray, Shape

if TYPE_CHECKING:
    from glucobench.utils.darts_processing import ScalerCustom
    from glucobench.utils.darts_dataset import SamplingDatasetInferenceDual
    from cgm_format.interface import ProcessingWarning
    from glucobench.lib.gluformer.model import Gluformer

class DatasetCreationSuccess(TypedDict):
    """
    Return type for successful dataset creation.
    """
    success: Literal[True]
    """Whether the dataset creation was successful"""
    dataset: "SamplingDatasetInferenceDual"
    """Darts SamplingDatasetInferenceDual object"""
    scaler_target: "ScalerCustom"
    """Darts ScalerCustom object"""
    scaler_covs: "ScalerCustom"
    """Darts ScalerCustom object"""
    model_config: "GluformerModelConfig"
    """Gluformer model config"""
    warning_flags: "ProcessingWarning"
    """ProcessingWarning flags"""

class DatasetCreationError(TypedDict):
    """
    Return type for failed dataset creation.
    """
    success: Literal[False]
    """Whether the dataset creation was successful"""
    error: str
    """Error message"""

DatasetCreationResult = Union[DatasetCreationSuccess, DatasetCreationError]

class FormattedWarnings(TypedDict):
    """
    Return type for formatted warning flags.
    """
    flags: int
    """ProcessingWarning flags"""
    has_warnings: bool
    """Whether there are any warnings"""
    messages: List[str]
    """List of warning messages"""

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
    
    # Optional overrides for model defaults
    r_drop: float = Field(default=0.2, description="Dropout rate")
    activ: str = Field(default='gelu', description="Activation function")
    distil: bool = Field(default=True, description="Use distillation")

    @property
    def interval_length(self) -> str:
        """Interval length for interpolation"""
        return f"{self.time_step}min"

class FanChartData(BaseModel):
    """
    Data for a single fan chart slice (KDE distribution at a time point).
    """
    model_config = ConfigDict(frozen=True)
    
    x: List[float] = Field(..., description="X coordinates (density)")
    y: List[float] = Field(..., description="Y coordinates (value grid)")
    fillcolor: str = Field(..., description="Color string for filling")
    time_index: int = Field(..., description="Time index relative to forecast start")

SHAPE_1D : str = "*"
SHAPE_2D : str = "*, *"
SHAPE_3D : str = "*, *, *"

PREDICTIONS_SHAPE : str = "* s, * x, * y" #num_slices x len_pred x num_samples
# Predictions
#        The predicted future target series in shape n x len_pred x num_samples, where
#        n is total number of predictions.
LOGVARS_SHAPE : str = "* s, * v, * y" #num_slices x 1 x num_samples
#    Logvar
#        The logvariance of the predicted future target series in shape n x len_pred.
PredictionsArray = NDArray[Shape[PREDICTIONS_SHAPE],Union[float64, float32]]
LogVarsArray = NDArray[Shape[LOGVARS_SHAPE],Union[float64, float32]]
class DartsScaler(BaseModel):
    """
    Scaler data from Darts library (ScalerCustom).
    """
    model_config = ConfigDict(frozen=True)
    
    min: NDArray[Shape[SHAPE_1D], Union[float64, float32]] = Field(..., description="Per feature adjustment for minimum")
    scale: NDArray[Shape[SHAPE_1D], Union[float64, float32]] = Field(..., description="Per feature relative scaling of the data")

    @classmethod
    def from_original(cls, original: "ScalerCustom") -> "DartsScaler":
        """
        Create DartsScaler from original ScalerCustom object.
        """
        return cls(
            min=original.min_,
            scale=original.scale_
        )

    def __eq__(self, other):
        if not isinstance(other, DartsScaler):
            return False
        return np.array_equal(self.min, other.min) and np.array_equal(self.scale, other.scale)


class DartsDataset(BaseModel):
    """
    Dataset from Darts library.
    Matches SamplingDatasetInferenceDual structure.
    """
    model_config = ConfigDict(frozen=True)
    
    target_series: List[NDArray[Shape[SHAPE_2D], Union[float64, float32]]] = Field(
        ..., 
        description="One or a sequence of target TimeSeries"
    )
    covariates: Optional[List[NDArray[Shape[SHAPE_2D], Union[float64, float32]]]] = Field(
        default=None, 
        description="Optionally, some future-known covariates that are used for predictions. This argument is required if the model was trained with future-known covariates"
    )
    static_covariates: Optional[List[NDArray[Shape[SHAPE_2D], Union[float64, float32]]]] = Field(
        default=None, 
        description="Static covariate data from input series"
    )
    time_index: Optional[List[List[str]]] = Field(
        default=None,
        description="Datetime timestamps for each series as ISO format strings"
    )

    n: int = Field(
        default=12, 
        description="Number of predictions into the future, could be greater than the output chunk length, in which case, the model will be called autorregressively",
        ge=1
    )
    input_chunk_length: int = Field(default=12, description="The length of the input series fed to the model")
    output_chunk_length: int = Field(default=1, description="The length of the output series emitted by the model")
    use_static_covariates: bool = Field(default=True, description="Whether to use/include static covariate data from input series")
    random_state: Optional[int] = Field(default=0, description="The random state to use for sampling")
    max_samples_per_ts: Optional[int] = Field(default=None, description="The maximum number of samples to be drawn from each time series. If None, all samples will be drawn")
    array_output_only: bool = Field(default=False, description="Whether __getitem__ returns only the arrays or adds the full TimeSeries object to the output tuple. This may cause problems with the torch collate and loader functions but works for Darts")

    @property
    def total_samples(self) -> int:
        """
        Calculate the total number of samples available in the dataset.
        This matches the logic in Darts SamplingDatasetInferenceDual.
        """
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        total_length = input_chunk_length + output_chunk_length
        
        total_count = 0
        for s in self.target_series:
             total_count += max(0, len(s) - total_length + 1)
        return total_count

    def get_sample_location(self, dataset_index: int) -> Tuple[int, int]:
        """
        Resolves a flat dataset sample index to (series_index, start_index_in_series).
        
        Args:
            dataset_index: Flat index (0..N-1).
            
        Returns:
            Tuple of (series_index, start_index_in_series).
            
        Raises:
            ValueError: If index is out of bounds.
        """
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        total_length = input_chunk_length + output_chunk_length
        
        # Calculate total samples per series
        series_counts = []
        for s in self.target_series:
             # Logic matches Darts SamplingDatasetInferenceDual
             # num_entries - total_length + 1
             count = max(0, len(s) - total_length + 1)
             series_counts.append(count)
             
        ds_len = sum(series_counts)
        
        if not (0 <= dataset_index < ds_len):
             raise ValueError(f"Dataset index {dataset_index} out of bounds for dataset length {ds_len}")
             
        current_idx = dataset_index
        for i, count in enumerate(series_counts):
            if current_idx < count:
                return (i, current_idx)
            current_idx -= count
            
        raise ValueError(f"Could not resolve sample location for index {dataset_index}")

    def get_sample_data(self, dataset_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the past target and true future for a given dataset index.
        """
        series_idx, sampling_loc = self.get_sample_location(dataset_index)
        series = self.target_series[series_idx]
        
        past_target = series[sampling_loc : sampling_loc + self.input_chunk_length]
        true_future = series[sampling_loc + self.input_chunk_length : sampling_loc + self.input_chunk_length + self.output_chunk_length]
        
        return past_target, true_future
    
    def get_sample_timestamps(self, dataset_index: int) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Retrieves timestamps for past target and true future for a given dataset index.
        Returns (past_timestamps, future_timestamps) or (None, None) if not available.
        """
        # Handle backward compatibility: old cached data won't have time_index
        if not hasattr(self, 'time_index') or self.time_index is None:
            return None, None
            
        series_idx, sampling_loc = self.get_sample_location(dataset_index)
        time_index_series = self.time_index[series_idx]
        
        past_times = time_index_series[sampling_loc : sampling_loc + self.input_chunk_length]
        future_times = time_index_series[sampling_loc + self.input_chunk_length : sampling_loc + self.input_chunk_length + self.output_chunk_length]
        
        return past_times, future_times

    @classmethod
    def from_original(cls, original: "SamplingDatasetInferenceDual") -> "DartsDataset":
        """
        Create DartsDataset from original SamplingDatasetInferenceDual object.
        """
        # target_series
        target_list = [ts.values() for ts in original.target_series]
        
        # covariates
        cov_list = None
        if original.covariates:
            cov_list = [ts.values() for ts in original.covariates]
            
        # static_covariates
        static_list = None
        if original.use_static_covariates:
            s_vals = [ts.static_covariates_values(copy=True) for ts in original.target_series]
            # Only set if all have valid static covariates
            if not any(s is None for s in s_vals):
                static_list = s_vals
        
        # Extract time_index from each series
        time_index_list = None
        if all(hasattr(ts, 'time_index') for ts in original.target_series):
            time_index_list = []
            for ts in original.target_series:
                # Convert pandas DatetimeIndex to list of ISO format strings
                timestamps = [t.isoformat() for t in ts.time_index]
                time_index_list.append(timestamps)

        return cls(
            target_series=target_list,
            covariates=cov_list,
            static_covariates=static_list,
            time_index=time_index_list,
            n=getattr(original, 'n', original.output_chunk_length),
            input_chunk_length=original.input_chunk_length,
            output_chunk_length=original.output_chunk_length,
            use_static_covariates=original.use_static_covariates,
            array_output_only=original.array_output_only
        )
    # Compare arrays helper
    @staticmethod
    def compare_arrays(
        a: List[NDArray[Shape[SHAPE_2D], Union[float64, float32]]], 
        b: List[NDArray[Shape[SHAPE_2D], Union[float64, float32]]]
    ) -> bool:
        if a is None and b is None: return True
        if a is None or b is None: return False
        
        # If list of arrays
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b): return False
            return all(np.array_equal(x, y) for x, y in zip(a, b))
        
        # If single array
        if hasattr(a, 'shape') and hasattr(b, 'shape'):
                return np.array_equal(a, b)
        
        return a == b

    def __eq__(self, other):
        if not isinstance(other, DartsDataset):
            return False
        
        # Compare non-array fields
        if self.n != other.n: return False
        if self.input_chunk_length != other.input_chunk_length: return False
        if self.output_chunk_length != other.output_chunk_length: return False
        if self.use_static_covariates != other.use_static_covariates: return False
        if self.array_output_only != other.array_output_only: return False
            
        if not self.compare_arrays(self.target_series, other.target_series): return False
        if not self.compare_arrays(self.covariates, other.covariates): return False
        if not self.compare_arrays(self.static_covariates, other.static_covariates): return False
        
        return True


class PredictionsData(BaseModel):
    model_config = ConfigDict(frozen=True)
    # validate asignment will impact performance
    len_pred: int = Field(..., description="Length of the prediction in steps, from model config")
    first_index: int = Field(..., le=0, description="First index of the dataset")
    num_samples: int = Field(..., description="Number of samples, from inference config")
    time_step: int = Field(..., description="Time step in minutes")
    
    predictions: Optional[PredictionsArray] = Field(
        default=None, 
        description="Predictions array of shape (first_index..last_index, 12, 10). None if pending.")
    logvars: Optional[LogVarsArray] = Field(default=None, description="Log variances")
    
    version: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique version identifier")
    
    # Optional fields to make PredictionsData self-contained for caching
    dataset: Optional[DartsDataset] = Field(default=None, description="Input dataset")
    target_scaler: Optional[DartsScaler] = Field(default=None, description="Target scaler")
    model_config_dump: Optional[Dict[str, Union[str, int, float, bool]]] = Field(default=None, description="Model config dump")
    warning_flags: Optional[int] = Field(default=0, description="Processing warning flags")

    @property
    def last_index(self) -> int:
        """Last index of the dataset"""
        return 0
    
    @property
    def last_index_with_ground_truth(self) -> int:
        """Last index of the predictions with ground truth"""
        return self.last_index - self.len_pred + 1

    @property
    def dataset_length(self) -> int:
        """Number of items in the dataset"""
        num_items = self.last_index - self.first_index + 1
        return num_items

    def get_dataset_index(self, index: int) -> int:
        """
        Convert a negative index (relative to dataset end) to a non-negative array index.
        
        Args:
            index: Negative index (e.g. -1 for last sample).
            
        Returns:
            Zero-based index for accessing the dataset array.
            
        Raises:
            ValueError: If index is out of bounds.
        """

        
        array_aligned_index = index - self.first_index
        
        if array_aligned_index < 0 or array_aligned_index >= self.dataset_length:
            raise ValueError(f"Index {index} is out of bounds for predictions range [{self.first_index}, {self.last_index}]")

        return array_aligned_index
    
    def det_future_aligned_index(self, index: int) -> int:
        """Shift an index to be aligned with the future prediction (beyond the dataset end)"""
        aligned_index = index + self.len_pred - 1  #align to the last prediction

        return aligned_index

    def get_dataset_sample_index(self, user_index: int) -> int:
        """
        Converts user negative index (e.g. -1) to dataset sample index (e.g. N-1).
        Assumes user_index 0 corresponds to the END of the dataset.
        """
        if self.dataset is not None:
            dataset_len = self.dataset.total_samples
        else:
            dataset_len = self.dataset_length
            
        return user_index + (dataset_len - 1)

    @model_validator(mode="after")
    def validate_predictions(self) -> "PredictionsData":
        """Validate the predictions array"""
        if self.predictions is None:
            return self
        
        num_items = self.dataset_length
        if num_items < 1:
            raise ValueError(f"Number of items in the dataset ({num_items}) should be a positive integer")
        
        if self.dataset is not None:
            if self.dataset.total_samples != self.dataset_length:
                raise ValueError(f"Number of items in the dataset ({self.dataset_length}) does not match the total number of samples in the dataset ({self.dataset.total_samples})")

        expected_slices = self.dataset_length
        
        if self.predictions.shape[0] != expected_slices:
            raise ValueError(f"Number of slices ({expected_slices}) does not match the shape of the predictions array ({self.predictions.shape[0]})")
        if self.predictions.shape[1] != self.len_pred:
            raise ValueError(f"Length of the prediction ({self.len_pred}) does not match the shape of the predictions array ({self.predictions.shape[1]})")
        if self.predictions.shape[2] != self.num_samples:
            raise ValueError(f"Number of samples ({self.num_samples}) does not match the shape of the predictions array ({self.predictions.shape[2]})")
        
        return self

    
    def strip_arrays(self) -> Self:
        """Strip the arrays from the PredictionsData"""
        stripped = PredictionsData(
            time_step=self.time_step,
            version=self.version,
            len_pred=self.len_pred,
            first_index=self.first_index,
            num_samples=self.num_samples,
            predictions=None,
            logvars=None,
            dataset=None,
            target_scaler=None,
            model_config_dump=None,
            warning_flags=None
        )
        return stripped

class PlotData(BaseModel):
    """
    Aggregated data for rendering the prediction plot.
    """
    model_config = ConfigDict(frozen=True)
    
    index: int = Field(..., le=0, description="Non-positive index of the plot")
    true_values_x: List[int] = Field(..., description="X coordinates for true values line (relative time in minutes)")
    true_values_y: List[float] = Field(..., description="Y coordinates for true values line")
    median_x: List[int] = Field(..., description="X coordinates for median forecast line (relative time in minutes)")
    median_y: List[float] = Field(..., description="Y coordinates for median forecast line")
    fan_charts: List[FanChartData] = Field(..., description="List of fan chart slices")
    true_values_timestamps: Optional[List[str]] = Field(default=None, description="Actual datetime timestamps for true values (ISO format)")
    median_timestamps: Optional[List[str]] = Field(default=None, description="Actual datetime timestamps for median forecast (ISO format)")

PlotsArray = NDArray[Shape[SHAPE_1D], Optional[str]] #json string representation of plotly's figure object
PlotsDataArray = NDArray[Shape[SHAPE_1D], Optional[PlotData]] #PlotData for each slice
class PlotCacheEntry(BaseModel):
    """
    Cache for plot data.
    """
    model_config = ConfigDict(frozen=True)
    
    slice_data: PredictionsData = Field(..., frozen=True, description="Information about the dataset version dimensions")
    plots_jsons: PlotsArray = Field(..., description="Array of plot data for each slice")
    plots_data: PlotsDataArray = Field(..., description="Array of PlotData objects for each slice")

    @classmethod
    def from_predictions_data(cls, predictions_data: PredictionsData) -> Self:
        """Create PlotCacheEntry from PredictionsData and PlotsArray"""
        stripped_plots_data = predictions_data.strip_arrays()
        return cls(
            slice_data=stripped_plots_data,
            plots_data=np.full(predictions_data.dataset_length, None, dtype=object),
            plots_jsons=np.full(predictions_data.dataset_length, None, dtype=object)
        )

# Polars DataFrame schema for result storage
# RESULT_SCHEMA = {
#     "index": pl.Int32,
#     "forecast": pl.List(pl.Float64),
#     # Plot Data Columns
#     "true_values_x": pl.List(pl.Int32),
#     "true_values_y": pl.List(pl.Float64),
#     "median_x": pl.List(pl.Int32),
#     "median_y": pl.List(pl.Float64),
#     "fan_charts": pl.List(
#         pl.Struct({
#             "x": pl.List(pl.Float64),
#             "y": pl.List(pl.Float64),
#             "fillcolor": pl.Utf8,
#             "time_index": pl.Int32
#         })
#     ),
#     "is_calculated": pl.Boolean 
# }


class ModelStats(BaseModel):
    """
    Statistics from ModelManager about model usage and performance.
    Returned by ModelManager.get_stats().
    """
    model_config = ConfigDict(frozen=True)
    
    available_priority_models: int = Field(
        ...,
        ge=0,
        description="Number of available (idle) models in priority queue (model #0). Should be 1 when idle, 0 when in use."
    )
    available_general_models: int = Field(
        ...,
        ge=0,
        description="Number of available (idle) models in general queue (models #1+). Higher values indicate more capacity."
    )
    avg_fulfillment_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average time to acquire a model (milliseconds)"
    )
    vmem_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="VRAM usage in MB"
    )
    device: str = Field(
        ...,
        description="Device being used for inference"
    )
    inference_requests_by_priority: Dict[int, int] = Field(
        default_factory=dict,
        description="Inference requests grouped by priority level"
    )
    total_inference_errors: int = Field(
        ...,
        ge=0,
        description="Total inference errors"
    )


class CalcStats(BaseModel):
    """
    Statistics from BackgroundProcessor about calculation workers.
    Returned by BackgroundProcessor.get_calc_stats().
    """
    model_config = ConfigDict(frozen=True)
    
    total_calc_runs: int = Field(
        ...,
        ge=0,
        description="Total number of plot calculations completed"
    )
    total_calc_errors: int = Field(
        ...,
        ge=0,
        description="Total number of calculation errors"
    )
    calc_queue_size: int = Field(
        ...,
        ge=0,
        description="Current size of calculation queue"
    )
    calc_queue_capacity: int = Field(
        ...,
        ge=0,
        description="Maximum capacity of calculation queue"
    )
    inference_queue_size: int = Field(
        ...,
        ge=0,
        description="Current size of inference queue"
    )
    inference_queue_capacity: int = Field(
        ...,
        ge=0,
        description="Maximum capacity of inference queue"
    )


class RequestTimeStats(BaseModel):
    """
    Statistics about HTTP request times.
    Tracks average, median, min, and max request durations.
    """
    model_config = ConfigDict(frozen=True)
    
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
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of requests tracked"
    )

