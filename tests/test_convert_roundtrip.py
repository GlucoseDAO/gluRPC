import io
import base64
import logging
from pathlib import Path
from typing import List

import polars as pl
import pytest

from glurpc.logic import convert_logic, parse_csv_content

# Setup logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert_roundtrip_test")

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


def compare_dataframes(df1: pl.DataFrame, df2: pl.DataFrame) -> bool:
    """Compare two polars dataframes for equality"""
    if df1.shape != df2.shape:
        logger.error(f"Shape mismatch: {df1.shape} vs {df2.shape}")
        return False
    
    if df1.columns != df2.columns:
        logger.error(f"Column mismatch: {df1.columns} vs {df2.columns}")
        return False
    
    # Check if dataframes are equal
    try:
        # Use equals() for polars DataFrame comparison
        if not df1.equals(df2):
            # Try to find where differences are
            for col in df1.columns:
                if not df1[col].equals(df2[col]):
                    logger.error(f"  Column '{col}' differs")
                    # Show sample of differences
                    df1_sample = df1[col].head(3)
                    df2_sample = df2[col].head(3)
                    logger.error(f"    Original: {df1_sample}")
                    logger.error(f"    Roundtrip: {df2_sample}")
            return False
        return True
    except Exception as e:
        logger.error(f"Frame comparison failed: {e}")
        return False


# Track expected unsupported files
EXPECTED_UNSUPPORTED_FILES = {
    "Zaharia Livia 06.09.2021 01.05-25.07.2021.csv",
    "Zaharia Livia 06.09.2021 26.06-25.07 2021.csv",
}


@pytest.mark.parametrize("csv_path", get_all_csvs(), ids=lambda p: p.name)
def test_convert_roundtrip(csv_path: Path):
    """
    Test that converting CSV content through convert_logic and back produces
    identical dataframes to the direct parse.
    
    For each CSV file:
    1. Direct parse: unified_df = parse_csv_content(content_base64)
    2. Roundtrip: roundtrip_df = parse_csv_content(convert_logic(content_base64).csv_content)
    3. Verify they are identical (except for the 2 unsupported files)
    """
    file_name = csv_path.name
    logger.info(f"Testing {file_name}...")
    
    # Read and encode the file
    content = csv_path.read_bytes()
    content_base64 = base64.b64encode(content).decode()
    
    # Direct parse
    try:
        unified_df = parse_csv_content(content_base64)
        logger.debug(f"  Direct parse: shape={unified_df.shape}")
    except Exception as e:
        # This file is expected to fail
        if file_name in EXPECTED_UNSUPPORTED_FILES:
            logger.info(f"  Direct parse failed as expected for unsupported file: {e}")
            
            # For files that fail direct parse, they should also fail convert_logic
            convert_response = convert_logic(content_base64)
            assert convert_response.error is not None, \
                f"File {file_name} failed direct parse but succeeded in convert_logic"
            logger.debug(f"  Convert logic also failed as expected: {convert_response.error}")
            pytest.skip(f"Unsupported file format: {file_name}")
        else:
            # Unexpected failure
            pytest.fail(f"Unexpected parse failure for {file_name}: {e}")
    
    # Convert logic
    convert_response = convert_logic(content_base64)
    assert convert_response.error is None, \
        f"Convert logic failed: {convert_response.error}"
    assert convert_response.csv_content is not None, \
        f"Convert logic succeeded but returned no csv_content"
    
    # Roundtrip parse using FormatParser (through parse_csv_content)
    try:
        # Convert the CSV content back to base64 for parse_csv_content
        csv_bytes = convert_response.csv_content.encode('utf-8')
        csv_base64 = base64.b64encode(csv_bytes).decode()
        roundtrip_df = parse_csv_content(csv_base64)
        logger.debug(f"  Roundtrip parse: shape={roundtrip_df.shape}")
    except Exception as e:
        pytest.fail(f"Roundtrip parse failed: {e}")
    
    # Compare dataframes
    if compare_dataframes(unified_df, roundtrip_df):
        logger.info(f"  ✓ Roundtrip successful - dataframes are identical")
    else:
        logger.error(f"  ✗ Dataframes differ after roundtrip")
        logger.error(f"    Original: {unified_df.shape}, columns={unified_df.columns}")
        logger.error(f"    Roundtrip: {roundtrip_df.shape}, columns={roundtrip_df.columns}")
        
        # Log first few rows for debugging
        logger.debug("Original head:")
        logger.debug(unified_df.head())
        logger.debug("Roundtrip head:")
        logger.debug(roundtrip_df.head())
        
        pytest.fail(f"Dataframes differ after roundtrip for {file_name}")

