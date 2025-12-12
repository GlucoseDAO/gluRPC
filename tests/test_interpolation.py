import base64
import logging
from pathlib import Path
from typing import List
import pytest
import polars as pl
import pandas as pd
import numpy as np

from glurpc.logic import parse_csv_content
from glurpc.config import (
    MINIMUM_DURATION_MINUTES_MODEL,
    MAXIMUM_WANTED_DURATION_DEFAULT,
)
from glurpc.data_classes import GluformerInferenceConfig
from cgm_format import FormatProcessor
from cgm_format.interface import ProcessingWarning
from glucobench.data_formatter import types as formatter_types
from glucobench.data_formatter import utils as formatter_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_interpolation")

DATA_DIR = Path(__file__).parent.parent / "data"

def get_all_csvs() -> List[Path]:
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found at {DATA_DIR}")
    
    files = []
    for f in DATA_DIR.rglob("*.csv"):
        if "parsed" in str(f):
            continue
        files.append(f)
    return sorted(files)

# Track expected unsupported files (copied from test_convert_roundtrip.py)
EXPECTED_UNSUPPORTED_FILES = {
    "Zaharia Livia 06.09.2021 01.05-25.07.2021.csv",
    "Zaharia Livia 06.09.2021 26.06-25.07 2021.csv",
}

@pytest.mark.parametrize("csv_path", get_all_csvs(), ids=lambda p: p.name)
def test_interpolation_rows(csv_path: Path):
    """
    Test that checks if formatter_utils.interpolate adds rows during the
    inference dataset creation pipeline.
    
    Pipeline:
    1. parse_csv_content
    2. Replicate create_dataset_from_df preprocessing
    3. Replicate create_inference_dataset_fast_local interpolation check
    """
    file_name = csv_path.name
    logger.info(f"Testing {file_name}...")
    
    # 1. Load Data
    content = csv_path.read_bytes()
    content_base64 = base64.b64encode(content).decode()
    
    try:
        unified_df = parse_csv_content(content_base64)
    except Exception as e:
        if file_name in EXPECTED_UNSUPPORTED_FILES:
            pytest.skip(f"Skipping known unsupported file: {file_name}")
        pytest.fail(f"Failed to parse {file_name}: {e}")

    # 2. Replicate create_dataset_from_df logic
    # See glurpc/logic.py create_dataset_from_df
    
    try:
        unified_df = FormatProcessor.interpolate_gaps(unified_df)
        unified_df = FormatProcessor.synchronize_timestamps(unified_df)
        
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=MINIMUM_DURATION_MINUTES_MODEL,
            maximum_wanted_duration=MAXIMUM_WANTED_DURATION_DEFAULT
        )
        
        if inference_df is None or len(inference_df) == 0 or warning_flags & ProcessingWarning.TOO_SHORT:
            logger.warning("Data quality insufficient for inference, skipping assertion")
            return # Skip if data is not suitable for inference, similar to original logic returning error

        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        pytest.fail(f"Preprocessing failed for {file_name}: {e}")

    # 3. Replicate create_inference_dataset_fast_local logic (partial)
    # See glurpc/logic.py create_inference_dataset_fast_local
    
    data = glucose_only_df
    config = GluformerInferenceConfig()
    
    mapping = {}
    if 'sequence_id' in data.columns: mapping['sequence_id'] = 'id'
    if 'datetime' in data.columns: mapping['datetime'] = 'time'
    if 'glucose' in data.columns: mapping['glucose'] = 'gl'
    
    if mapping:
        data = data.rename(mapping)
    
    df = data.to_pandas()
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'])
    if 'gl' in df.columns: df['gl'] = df['gl'].astype(np.float32)
    
    DataTypes = formatter_types.DataTypes
    InputTypes = formatter_types.InputTypes
    
    column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('time', DataTypes.DATE, InputTypes.TIME),
        ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET)
    ]
    
    logger.info(f"Before interpolation, shape: {df.shape}")
    
    df_interp, updated_col_def = formatter_utils.interpolate(
        df, 
        column_definition, 
        gap_threshold=config.gap_threshold,
        min_drop_length=config.min_drop_length,
        interval_length=config.interval_length
    )
    
    logger.info(f"After interpolation, shape: {df_interp.shape}")
    
    # Assert equality of number of rows
    # If new rows emerged, then something was interpolated, which shouldn't be the case 
    # after create_dataset_from_df (which should have handled gaps/sync)
    
    rows_before = df.shape[0]
    rows_after = df_interp.shape[0]
    
    if rows_before != rows_after:
        logger.error(f"Row count changed! {rows_before} -> {rows_after}")
        
        # Save problematic dataframe for examination
        output_dir = Path(__file__).parent / "interpolation_mismatch"
        output_dir.mkdir(exist_ok=True)
        
        # Save original and interpolated
        original_path = output_dir / f"{file_name}_original.csv"
        interpolated_path = output_dir / f"{file_name}_interpolated.csv"
        
        try:
            # Ensure sorting by time if possible
            if 'time' in df.columns:
                df = df.sort_values('time')
            if 'time' in df_interp.columns:
                df_interp = df_interp.sort_values('time')
                
            df.to_csv(original_path, index=False)
            df_interp.to_csv(interpolated_path, index=False)
            logger.info(f"Saved debug files to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save debug CSVs: {e}")

    assert rows_before == rows_after, \
        f"Interpolation added rows for {file_name}: {rows_before} -> {rows_after}"


@pytest.mark.parametrize("csv_path", get_all_csvs(), ids=lambda p: p.name)
@pytest.mark.parametrize("maximum_wanted_duration", [
    MINIMUM_DURATION_MINUTES_MODEL,  # Minimum
    MAXIMUM_WANTED_DURATION_DEFAULT * 2  # Maximum
], ids=["min_duration", "max_duration"])
def test_dataset_length_relationship(csv_path: Path, maximum_wanted_duration: int):
    """
    Test that measures the relationship between input data length and resulting
    dataset size after calling create_inference_dataset_fast_local.
    
    Expected relationship:
    dataset_length ≈ input_length - input_chunk_length - output_chunk_length
    
    Pipeline:
    1. parse_csv_content
    2. Replicate create_dataset_from_df preprocessing
    3. Call create_inference_dataset_fast_local
    4. Measure dataset length vs input length
    """
    file_name = csv_path.name
    logger.info(f"Testing {file_name} with max_duration={maximum_wanted_duration}...")
    
    # 1. Load Data
    content = csv_path.read_bytes()
    content_base64 = base64.b64encode(content).decode()
    
    try:
        unified_df = parse_csv_content(content_base64)
    except Exception as e:
        if file_name in EXPECTED_UNSUPPORTED_FILES:
            pytest.skip(f"Skipping known unsupported file: {file_name}")
        pytest.fail(f"Failed to parse {file_name}: {e}")

    # 2. Replicate create_dataset_from_df logic
    try:

        
        unified_df = FormatProcessor.interpolate_gaps(unified_df)
        unified_df = FormatProcessor.synchronize_timestamps(unified_df)
        
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=MINIMUM_DURATION_MINUTES_MODEL,
            maximum_wanted_duration=maximum_wanted_duration
        )
        
        if inference_df is None or len(inference_df) == 0 or warning_flags & ProcessingWarning.TOO_SHORT:
            logger.warning("Data quality insufficient for inference, skipping")
            pytest.skip(f"Data quality insufficient for {file_name}")

        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        pytest.fail(f"Preprocessing failed for {file_name}: {e}")

    input_length = len(glucose_only_df)
    logger.info(f"Input length after preprocessing: {input_length}")
    
    # 3. Call create_inference_dataset_fast_local
    config = GluformerInferenceConfig()
    
    try:
        from glurpc.logic import create_inference_dataset_fast_local
        
        dataset, model_config, scaler_target, scaler_covs = create_inference_dataset_fast_local(
            glucose_only_df, 
            config
        )
    except Exception as e:
        logger.error(f"create_inference_dataset_fast_local failed: {e}")
        pytest.fail(f"Dataset creation failed for {file_name}: {e}")
    
    # 4. Measure dataset length
    dataset_length = len(dataset)
    logger.info(f"Dataset length: {dataset_length}")
    
    # 5. Calculate expected relationship
    # The relationship for sliding window datasets is:
    # dataset_length = input_length - input_chunk_length - output_chunk_length + 1
    # 
    # Explanation: With N input points and requiring (I+O) consecutive points per window,
    # the number of valid starting positions is N - I - O + 1.
    # Example: 108 points with I=96, O=12 → 108 - 96 - 12 + 1 = 1 window
    #          109 points → 109 - 96 - 12 + 1 = 2 windows
    
    min_required_length = config.input_chunk_length + config.output_chunk_length
    
    if input_length < min_required_length:
        # Edge case: input too short, expect empty dataset
        logger.info(f"Input length {input_length} < min required {min_required_length}, expecting empty dataset")
        expected_length = 0
    else:
        # Normal case: sliding window formula
        # CONSTANT = +1 (the key insight)
        SLIDING_WINDOW_CONSTANT = 1
        expected_length = input_length - config.input_chunk_length - config.output_chunk_length + SLIDING_WINDOW_CONSTANT
    
    difference = dataset_length - expected_length
    
    logger.info(f"Input length: {input_length}")
    logger.info(f"Config: input_chunk={config.input_chunk_length}, output_chunk={config.output_chunk_length}")
    logger.info(f"Formula: dataset_length = input_length - {config.input_chunk_length} - {config.output_chunk_length} + 1")
    logger.info(f"Expected dataset length: {expected_length}")
    logger.info(f"Actual dataset length: {dataset_length}")
    logger.info(f"Difference: {difference}")
    
    # Assert exact match - the formula should be precise
    assert difference == 0, \
        f"Dataset length formula validation failed for {file_name} (max_duration={maximum_wanted_duration}):\n" \
        f"  Input length: {input_length}\n" \
        f"  Expected: {expected_length} (using formula: N - I - O + 1)\n" \
        f"  Actual: {dataset_length}\n" \
        f"  Difference: {difference}"


def get_single_test_file() -> Path:
    """Get a single representative CSV file for focused testing."""
    all_files = get_all_csvs()
    if not all_files:
        pytest.skip("No test files available")
    # Pick a mid-sized file for testing
    return all_files[len(all_files) // 2]


@pytest.mark.parametrize("max_duration_minutes", 
    [
        # Test with non-sharp minute values
        MINIMUM_DURATION_MINUTES_MODEL,  # 540 (sharp, minimum)
        541,  # non-sharp
        543,  # non-sharp
        547,  # non-sharp
        563,  # non-sharp
        599,  # non-sharp
        617,  # non-sharp
        643,  # non-sharp
        671,  # non-sharp
        723,  # non-sharp
        789,  # non-sharp
        851,  # non-sharp
        917,  # non-sharp
        983,  # non-sharp
        1037,  # non-sharp
        MAXIMUM_WANTED_DURATION_DEFAULT,  # 1080 (sharp, maximum)
    ],
    ids=lambda x: f"{x}min"
)
def test_duration_formula_correctness(max_duration_minutes: int):
    """
    Test the correctness of the formula that predicts dataset length based on
    max_duration_minutes and model config parameters.
    
    Formula to test:
    dataset_length = (max_duration_minutes / time_step) + 1 - (input_chunk_length + output_chunk_length - 1)
    
    Simplified:
    dataset_length = (max_duration_minutes / time_step) - input_chunk_length - output_chunk_length + 2
    
    Wait, let me derive the correct formula:
    - input_length ≈ max_duration_minutes / time_step
    - dataset_length = input_length - input_chunk_length - output_chunk_length + 1
    - Therefore: dataset_length = (max_duration_minutes / time_step) - input_chunk_length - output_chunk_length + 1
    
    Pipeline:
    1. Load one representative file
    2. Process with varying max_duration_minutes
    3. Validate formula prediction matches actual dataset length
    """
    csv_path = get_single_test_file()
    file_name = csv_path.name
    logger.info(f"Testing formula with {file_name}, max_duration={max_duration_minutes} minutes...")
    
    # 1. Load Data
    content = csv_path.read_bytes()
    content_base64 = base64.b64encode(content).decode()
    
    try:
        unified_df = parse_csv_content(content_base64)
    except Exception as e:
        if file_name in EXPECTED_UNSUPPORTED_FILES:
            pytest.skip(f"Skipping known unsupported file: {file_name}")
        pytest.fail(f"Failed to parse {file_name}: {e}")

    # 2. Process with specific max_duration
    try:
        unified_df = FormatProcessor.interpolate_gaps(unified_df)
        unified_df = FormatProcessor.synchronize_timestamps(unified_df)
        
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=MINIMUM_DURATION_MINUTES_MODEL,
            maximum_wanted_duration=max_duration_minutes
        )
        
        if inference_df is None or len(inference_df) == 0 or warning_flags & ProcessingWarning.TOO_SHORT:
            logger.info(f"Data insufficient for max_duration={max_duration_minutes}, skipping")
            pytest.skip(f"Data quality insufficient for {max_duration_minutes} minutes")

        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        pytest.fail(f"Preprocessing failed for {file_name}: {e}")

    input_length = len(glucose_only_df)
    
    # 3. Create dataset
    config = GluformerInferenceConfig()
    
    try:
        from glurpc.logic import create_inference_dataset_fast_local
        
        dataset, model_config, scaler_target, scaler_covs = create_inference_dataset_fast_local(
            glucose_only_df, 
            config
        )
    except Exception as e:
        logger.error(f"create_inference_dataset_fast_local failed: {e}")
        pytest.fail(f"Dataset creation failed for {file_name}: {e}")
    
    dataset_length = len(dataset)
    
    # 4. Test the formula: predict dataset_length from max_duration_minutes
    # 
    # The user's proposed formula was:
    # dataset_length = (max_duration_minutes / time_step) + 1 - (input_chunk_length + output_chunk_length - 1)
    # 
    # Let's simplify and verify:
    # = (max_duration_minutes / time_step) + 1 - input_chunk_length - output_chunk_length + 1
    # = (max_duration_minutes / time_step) - input_chunk_length - output_chunk_length + 2
    # 
    # But we know from previous tests that:
    # dataset_length = input_length - input_chunk_length - output_chunk_length + 1
    # 
    # So if input_length ≈ max_duration_minutes / time_step, then:
    # dataset_length ≈ (max_duration_minutes / time_step) - input_chunk_length - output_chunk_length + 1
    # 
    # The difference is +1 vs +2, let's test which is correct.
    
    time_step = config.time_step
    
    # User's proposed formula with integer division (no fractions)
    user_formula = (max_duration_minutes // time_step) + 1 - (config.input_chunk_length + config.output_chunk_length - 1)
    
    # Simplified form:
    # user_formula = (max_duration_minutes // time_step) - input_chunk_length - output_chunk_length + 2
    
    logger.info(f"Max duration: {max_duration_minutes} minutes")
    logger.info(f"Time step: {time_step} minutes")
    logger.info(f"Actual input length: {input_length} points")
    logger.info(f"User formula prediction: {user_formula}")
    logger.info(f"Actual dataset length: {dataset_length}")
    
    # Use the actual input_length to validate the sliding window formula
    if input_length >= config.input_chunk_length + config.output_chunk_length:
        expected_from_input = input_length - config.input_chunk_length - config.output_chunk_length + 1
    else:
        expected_from_input = 0
    
    # Assert that the sliding window formula holds (this is the ground truth)
    assert dataset_length == expected_from_input, \
        f"Sliding window formula failed for max_duration={max_duration_minutes}:\n" \
        f"  Input length: {input_length}\n" \
        f"  Expected: {expected_from_input}\n" \
        f"  Actual: {dataset_length}"
    
    # Calculate prediction error for user's formula
    user_formula_error = abs(dataset_length - user_formula)
    
    # Print for visibility in test output
    print(f"\nDuration={max_duration_minutes}min: input_len={input_length}, actual_dataset={dataset_length}")
    print(f"  User formula prediction: {user_formula} (error={user_formula_error})")
    print(f"  Max_duration // time_step: {max_duration_minutes // time_step}")
    
    # Assert STRICT 0 error - the user's formula should be exact
    # Formula: (max_duration // time_step) + 1 - (input_chunk + output_chunk - 1)
    # Simplified: (max_duration // time_step) - input_chunk - output_chunk + 2
    assert user_formula_error == 0, \
        f"User formula must have ZERO error for max_duration={max_duration_minutes}:\n" \
        f"  Formula: (max_duration // time_step) + 1 - (input_chunk + output_chunk - 1)\n" \
        f"  Formula: ({max_duration_minutes} // {time_step}) + 1 - ({config.input_chunk_length} + {config.output_chunk_length} - 1)\n" \
        f"  Predicted: {user_formula}\n" \
        f"  Actual: {dataset_length}\n" \
        f"  Error: {user_formula_error}"


