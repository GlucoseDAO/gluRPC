# Data Corruption Trace - Concurrent Test Failure

## Summary
During concurrent REST+gRPC requests, CSV data gets corrupted with:
- Control character `0x13` (DC3/XOFF) inserted: `"2020-10-02T..."` → `"202\x13_IN"`
- Missing characters: `"2020-01-15T..."` → `"2020-15T..."`

Corrupted data is dumped to `/tmp/glurpc_debug/` when detected.

---

## Step-by-Step Data Flow

### Step 1: Test Input Creation
**File:** `tests/test_combined_service.py:593`
```python
csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
```

**Data at this point:**
- `unified_csv_content_rest`: String containing valid unified CSV format
- Source: Fixture `unified_csv_content_rest` (line 265) gets data from gRPC `ConvertToUnified` 
- `csv_base64`: Base64-encoded string (immutable Python str)

**Key observation:** At this point, the data is a single immutable string shared across all requests.

---

### Step 2A: REST Request Path

#### Step 2A.1: Test sends REST request
**File:** `tests/test_combined_service.py:597-601`
```python
response = await rest_client.post(
    "/process_unified",
    json={"csv_base64": csv_base64, "force_calculate": False},
    headers={"x-api-key": api_key}
)
```

**Data transformation:**
- AsyncClient (httpx) serializes JSON body
- `csv_base64` string → JSON string → HTTP request body bytes

#### Step 2A.2: FastAPI receives request
**File:** `src/service/combined_service.py:199`
```python
async def process_unified(request: ProcessRequest, api_key: str = Depends(require_api_key)):
```

**File:** `src/service/combined_service.py:201`
```python
logger.info(f"REST: /process_unified - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
```

**Data at this point:**
- FastAPI has deserialized JSON into `ProcessRequest` Pydantic model
- `request.csv_base64`: String field containing base64 data

#### Step 2A.3: Call parse_and_schedule
**File:** `src/service/combined_service.py:212`
```python
result = await parse_and_schedule(request.csv_base64, force_calculate=request.force_calculate)
```

**Data passed:**
- `request.csv_base64` (string) → `content_base64` parameter

---

### Step 2B: gRPC Request Path

#### Step 2B.1: Test sends gRPC request
**File:** `tests/test_combined_service.py:606-611`
```python
metadata = aio.Metadata(('x-api-key', api_key))
request = glurpc_pb2.ProcessUnifiedRequest(
    csv_base64=csv_base64,
    force_calculate=False
)
response = await grpc_stub.ProcessUnified(request, metadata=metadata)
```

**Data transformation:**
- `csv_base64` string → Protobuf message → gRPC binary frame

#### Step 2B.2: gRPC servicer receives request
**File:** `src/service/glurpc_service.py:160-163`
```python
async def ProcessUnified(
    self,
    request: glurpc_pb2.ProcessUnifiedRequest,
    context: aio.ServicerContext
) -> glurpc_pb2.ProcessUnifiedResponse:
```

**File:** `src/service/glurpc_service.py:167`
```python
logger.info(f"gRPC: ProcessUnified called - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
```

**Data at this point:**
- gRPC has deserialized protobuf into `ProcessUnifiedRequest`
- `request.csv_base64`: String field containing base64 data

#### Step 2B.3: Call parse_and_schedule
**File:** `src/service/glurpc_service.py:182`
```python
result = await parse_and_schedule(request.csv_base64, force_calculate=request.force_calculate)
```

**Data passed:**
- `request.csv_base64` (string) → `content_base64` parameter

---

### Step 3: parse_and_schedule (Common Path)
**File:** `src/glurpc/core.py:112-127`
```python
async def parse_and_schedule(
    content_base64: str, 
    maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION,
    inference_config: GluformerInferenceConfig = DEFAULT_CONFIG,
    force_calculate: bool = False,
    priority: int = 1,
    request_id: Optional[int] = None
) -> UnifiedResponse:
    logger.info(f"Action: parse_and_schedule started (force={force_calculate}, req_id={request_id})")
    inf_cache = InferenceCache()
    bg_processor = BackgroundProcessor()
    
    try:
        # 1. Convert to unified format (Parse Only)
        # We need handle and unified_df to proceed.
        handle, unified_df = await asyncio.to_thread(logic.get_handle_and_df, content_base64)
```

**Data transformation:**
- `content_base64` string passed to `asyncio.to_thread`
- Executes `logic.get_handle_and_df` in thread pool

---

### Step 4: get_handle_and_df
**File:** `src/glurpc/logic.py:311-325`
```python
def get_handle_and_df(content_base64: str) -> Tuple[str, pl.DataFrame]:
    """
    Parses CSV and computes canonical hash (handle).
    Returns (handle, unified_df).
    """
    try:
        unified_df = parse_csv_content(content_base64)
        handle = compute_handle(unified_df)
        preprocessing_logger.info(f"Parsed CSV. Unified Shape: {unified_df.shape} Handle: {handle[:8]}")
        return handle, unified_df
    except ValueError as e:
        # Re-raise ValueError as-is (already formatted)
        raise
    except Exception as e:
        raise ValueError(f"Failed to process file: {str(e)}")
```

**Data passed:**
- `content_base64` (string) → `parse_csv_content()`

---

### Step 5: parse_csv_content
**File:** `src/glurpc/logic.py:273-314`
```python
def parse_csv_content(content_base64: str) -> pl.DataFrame:
    preprocessing_logger.debug("Starting CSV content parsing")
    try:
        preprocessing_logger.debug(f"Parsing base64 content (length: {len(content_base64)} chars)")
        unified_df = FormatParser.parse_base64(content_base64)
        preprocessing_logger.info(f"Successfully parsed CSV: shape={unified_df.shape}, columns={unified_df.columns}")
        return unified_df
    except UnknownFormatError as e:
        # Expected error - invalid format
        preprocessing_logger.warning(f"Unknown file format: {e}")
        raise e
    except (MalformedDataError, ColumnOrderError, ColumnTypeError, 
            MissingColumnError, ExtraColumnError, ZeroValidInputError) as e:
        # Expected errors - known data quality/structure issues
        error_name = type(e).__name__.replace("Error", "")
        preprocessing_logger.warning(f"Data validation error ({error_name}): {e}")
        
        # Dump corrupted input to file for debugging
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = os.path.join(tempfile.gettempdir(), "glurpc_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save base64 content
            base64_file = os.path.join(debug_dir, f"corrupted_base64_{timestamp}.txt")
            with open(base64_file, 'w') as f:
                f.write(content_base64)
            
            # Try to decode and save raw content
            try:
                raw_data = base64.b64decode(content_base64)
                raw_file = os.path.join(debug_dir, f"corrupted_raw_{timestamp}.csv")
                with open(raw_file, 'wb') as f:
                    f.write(raw_data)
                preprocessing_logger.error(f"Corrupted input dumped to: {base64_file} and {raw_file}")
            except Exception as decode_err:
                preprocessing_logger.error(f"Could not decode base64 for dump: {decode_err}. Base64 saved to: {base64_file}")
        except Exception as dump_err:
            preprocessing_logger.error(f"Failed to dump corrupted input: {dump_err}")
        
        raise e
```

**Critical point:** This is where corruption is DETECTED and DUMPED.

**Data passed:**
- `content_base64` (string) → `FormatParser.parse_base64()`

---

### Step 6: FormatParser.parse_base64 (External Library)
**File:** `.venv/lib/python3.13/site-packages/cgm_format/interface/cgm_interface.py:325-347`
```python
@classmethod
def parse_base64(cls, base64_data: str) -> UnifiedFormat:
    """Parse CGM data from base64 encoded string."""
    try:
        raw_data = b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}")
    
    return cls.parse_from_bytes(raw_data)
```

**Data transformation:**
- `base64_data` (string) → `b64decode()` → `raw_data` (bytes)
- This is where corruption COULD be introduced during base64 decode

**Key observation:** `b64decode()` is from Python's standard library - unlikely to be the source.

---

### Step 7: parse_from_bytes (External Library)
**File:** `.venv/lib/python3.13/site-packages/cgm_format/interface/cgm_interface.py:256-276`
```python
@classmethod
def parse_from_bytes(cls, raw_data: bytes) -> UnifiedFormat:
    """Convenience method to parse raw bytes directly to unified format."""
    text_data = cls.decode_raw_data(raw_data)
    format_type = cls.detect_format(text_data)
    return cls.parse_to_unified(text_data, format_type)
```

**This eventually calls Polars to parse the CSV, which detects the corruption:**
- Invalid datetime value like `"202\x13_IN"` or `"2020-15T10:01:09.000"`
- Raises `MalformedDataError` which propagates back to Step 5

---

## Corruption Analysis

### Where Corruption COULD Occur:

1. **Test fixture sharing (UNLIKELY)**
   - `csv_base64` is an immutable Python string
   - Same string reference used by all concurrent tasks
   - Python strings cannot be mutated

2. **JSON serialization in httpx/FastAPI (POSSIBLE)**
   - httpx AsyncClient serializes request
   - FastAPI deserializes request body
   - Potential buffer reuse or threading issue?

3. **Protobuf serialization in gRPC (POSSIBLE)**
   - gRPC serializes/deserializes protobuf messages
   - Potential buffer reuse in async gRPC?

4. **Base64 decoding (UNLIKELY)**
   - Standard library `base64.b64decode()`
   - Well-tested, unlikely to have race conditions

5. **String passing through asyncio.to_thread (POSSIBLE)**
   - Data passed from async context to thread pool
   - Potential GIL interaction issue?

### Evidence from Dumps:

- Corruption happens BEFORE `FormatParser.parse_base64()` is called
- The base64 string itself is corrupted (as evidenced by the decoded raw CSV)
- Control character `0x13` is NOT part of HTTP/gRPC protocol
- Pattern suggests buffer overwrite: `"2020-10-02..."` → `"202\x13_IN"`
- Missing bytes: `"2020-01-15T..."` → `"2020-15T..."`

### Most Likely Culprit:

The corruption appears to be in **request deserialization/body parsing** in either:
- FastAPI's request body parsing (Pydantic model creation)
- gRPC's protobuf deserialization

Both frameworks may have shared buffers or caching that isn't properly isolated between concurrent requests.

---

## Next Steps for Investigation:

1. **Add logging before deserialization:**
   - Log raw HTTP body bytes in FastAPI before Pydantic parsing
   - Log raw gRPC message bytes before protobuf parsing

2. **Check for shared state in FastAPI/Pydantic:**
   - `ProcessRequest` Pydantic model may have class-level state
   - Check if FastAPI is reusing request parsers

3. **Check for shared state in gRPC:**
   - Async gRPC channel/stub may reuse buffers
   - Check if protobuf deserialization has shared state

4. **Test with unique base64 per request:**
   - Create fresh base64 encoding for each request
   - If corruption stops, confirms shared string is not the issue
