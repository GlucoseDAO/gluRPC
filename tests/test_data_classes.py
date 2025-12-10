import pytest
import numpy as np
from glurpc.data_classes import DartsDataset, PredictionsData, DartsScaler
from numpydantic import NDArray, Shape
from typing import List, Union

# Mock helpers
def create_mock_darts_dataset(n_samples=100, input_chunk=12, output_chunk=1):
    # Create target series
    # Series length needed for n_samples: len = samples + input + output - 1
    total_len = n_samples + input_chunk + output_chunk - 1
    target = np.arange(total_len, dtype=np.float64).reshape(-1, 1)
    
    return DartsDataset(
        target_series=[target],
        covariates=None,
        static_covariates=None,
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        n=output_chunk # n is typically output_chunk for inference
    )

def create_mock_multi_series_dataset(series_lengths=[50, 60], input_chunk=10, output_chunk=5):
    targets = []
    for length in series_lengths:
        targets.append(np.random.rand(length, 1).astype(np.float64))
    
    return DartsDataset(
        target_series=targets,
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk
    )

class TestDartsDataset:
    def test_dataset_equality(self):
        ds1 = create_mock_darts_dataset()
        ds2 = create_mock_darts_dataset()
        
        assert ds1 == ds2
        
        # Modify ds2
        ds2.target_series[0][0, 0] = 999
        assert ds1 != ds2

    def test_total_samples_single_series(self):
        input_chunk = 10
        output_chunk = 5
        total_window = input_chunk + output_chunk
        
        # Length 20. Samples = 20 - 15 + 1 = 6
        ds = DartsDataset(
            target_series=[np.zeros((20, 1))],
            input_chunk_length=input_chunk,
            output_chunk_length=output_chunk
        )
        assert ds.total_samples == 6

    def test_total_samples_multi_series(self):
        # Series 1: 20 len -> 6 samples
        # Series 2: 15 len -> 1 sample
        # Series 3: 14 len -> 0 samples (too short)
        ds = create_mock_multi_series_dataset(
            series_lengths=[20, 15, 14], 
            input_chunk=10, 
            output_chunk=5
        )
        assert ds.total_samples == 6 + 1 + 0 # = 7

    def test_get_sample_location(self):
        ds = create_mock_multi_series_dataset(
            series_lengths=[20, 15], # 6 samples, 1 sample
            input_chunk=10, 
            output_chunk=5
        )
        # Total samples = 7. Indices 0..6
        
        # Index 0 -> Series 0, Offset 0
        idx, offset = ds.get_sample_location(0)
        assert idx == 0
        assert offset == 0
        
        # Index 5 -> Series 0, Offset 5 (Last sample of first series)
        idx, offset = ds.get_sample_location(5)
        assert idx == 0
        assert offset == 5
        
        # Index 6 -> Series 1, Offset 0 (First sample of second series)
        idx, offset = ds.get_sample_location(6)
        assert idx == 1
        assert offset == 0
        
        # Index 7 -> Out of bounds
        with pytest.raises(ValueError):
            ds.get_sample_location(7)

    def test_get_sample_data(self):
        # Create a dataset with known values
        # Series 0: 0..19 (20 items). input=10, output=5.
        # Samples at indices 0, 1, ..., 5 (6 samples).
        # Sample 0: past=[0..9], future=[10..14]
        
        target = np.arange(20, dtype=np.float64).reshape(-1, 1)
        ds = DartsDataset(
            target_series=[target],
            input_chunk_length=10,
            output_chunk_length=5
        )
        
        # Test Sample 0
        past, future = ds.get_sample_data(0)
        assert np.array_equal(past.flatten(), np.arange(0, 10, dtype=np.float64))
        assert np.array_equal(future.flatten(), np.arange(10, 15, dtype=np.float64))
        
        # Test Sample 5 (Last one)
        # Start index = 5.
        # Past = [5..14], Future = [15..19]
        past, future = ds.get_sample_data(5)
        assert np.array_equal(past.flatten(), np.arange(5, 15, dtype=np.float64))
        assert np.array_equal(future.flatten(), np.arange(15, 20, dtype=np.float64))

class TestPredictionsData:
    def test_validation_logic(self):
        ds = create_mock_darts_dataset(n_samples=10)
        scaler = DartsScaler(min=np.array([0.0]), scale=np.array([1.0]))
        
        # Correct setup
        # Predictions shape: (n_samples, len_pred, n_mc_samples)
        preds_array = np.zeros((10, 1, 5)) 
        
        # First index for 10 samples ending at 0 is -9.
        # Length = 0 - (-9) + 1 = 10.
        pd = PredictionsData(
            len_pred=1,
            first_index=-9,
            num_samples=5,
            predictions=preds_array,
            dataset=ds,
            target_scaler=scaler
        )
        assert pd.dataset_length == 10
        
        # Mismatch dataset length and predictions shape
        with pytest.raises(ValueError, match="Number of slices"):
            PredictionsData(
                len_pred=1,
                first_index=-9, # Expects 10
                num_samples=5,
                predictions=np.zeros((11, 1, 5)), # Has 11
                dataset=ds, # Has 10
                target_scaler=scaler
            )

        # Mismatch internal dataset length check (dataset has 10, we say we are tracking 5)
        # The validator explicitly checks if self.dataset.total_samples != self.dataset_length
        with pytest.raises(ValueError, match="does not match the total number of samples"):
            PredictionsData(
                len_pred=1,
                first_index=-4, # Implies length 5
                num_samples=5,
                predictions=np.zeros((5, 1, 5)), 
                dataset=ds, # Has 10 samples
                target_scaler=scaler
            )

    def test_indexing_conversion(self):
        # Dataset length 100. Indices 0..99 in array.
        # User indices: -99 (start) to 0 (end).
        # First index = -99.
        
        pd = PredictionsData(
            len_pred=1,
            first_index=-99,
            num_samples=1,
            predictions=np.zeros((100, 1, 1))
        )
        
        assert pd.dataset_length == 100
        
        # Test 0 (End)
        # Dataset sample index should be 99
        assert pd.get_dataset_sample_index(0) == 99
        # Array index should be 99
        assert pd.get_dataset_index(0) == 99
        
        # Test -99 (Start)
        # Dataset sample index should be 0
        assert pd.get_dataset_sample_index(-99) == 0
        # Array index should be 0
        assert pd.get_dataset_index(-99) == 0
        
        # Test -1 (Second to last)
        assert pd.get_dataset_sample_index(-1) == 98
        assert pd.get_dataset_index(-1) == 98

    def test_prediction_alignment(self):
        # Scenario:
        # History: [0, 1, 2, ... 95] (96 items)
        # Prediction at t=96 (using 0..95) -> Predicts [96..107] (12 items)
        # Dataset sample 0 corresponds to window starting at 0.
        # Input: series[0:96], Target: series[96:108]
        
        # User asks for index -N (start of dataset).
        # This corresponds to the FIRST prediction made.
        # get_dataset_sample_index(-N) -> 0.
        pass
