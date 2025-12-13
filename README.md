# GluRPC

**Production-ready REST and gRPC API server for real-time glucose prediction using the Gluformer transformer model.**

**NEW**: Now includes gRPC interface compatible with SingularityNET! See [GRPC_SERVICE_README.md](GRPC_SERVICE_README.md) for details.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [REST Service](#rest-service)
  - [gRPC Service (SNET Compatible)](#grpc-service-snet-compatible)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Data Requirements](#data-requirements)
- [Deployment Scenarios](#deployment-scenarios)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Testing](#testing)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

GluRPC is a high-performance service that provides both REST (FastAPI) and gRPC interfaces for processing continuous glucose monitoring (CGM) data and generating blood glucose predictions using the Gluformer model. The service handles multiple CGM device formats (Dexcom, FreeStyle Libre), performs quality checks, and generates visual predictions with uncertainty quantification through Monte Carlo Dropout.

The gRPC interface is compatible with **SingularityNET** (SNET), enabling blockchain-based payment channels and AI marketplace integration.

## Features

- 🔄 **Dual Interface**: Both REST (FastAPI) and gRPC endpoints with identical functionality
- 🌐 **SNET Compatible**: gRPC service compatible with SingularityNET marketplace
- 🔄 **Multi-format CGM Support**: Auto-detects and parses Dexcom, FreeStyle Libre, and unified CSV formats
- 🧠 **Transformer-based Predictions**: Uses pre-trained Gluformer models from HuggingFace
- 📊 **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence intervals (10 samples)
- ⚡ **Intelligent Caching**: 
  - SHA256-based content-addressable storage
  - Two-tier caching (memory + disk persistence)
  - Superset matching for efficient reuse
  - Configurable cache size (default: 128 datasets)
- 🔍 **Quality Assurance**: Comprehensive data quality checks with detailed warnings
- 📈 **Interactive Visualizations**: Plotly-based prediction plots with fan charts for uncertainty
- ⚙️ **Background Processing**: Async workers with priority-based task scheduling
- 🛡️ **Disconnect Detection**: Graceful cancellation on client disconnect (REST & gRPC)
- 🔐 **Optional API Key Authentication**: Secure endpoint access control
- 📝 **Detailed Logging**: Timestamped logs with full pipeline traceability
- 🚀 **GPU Acceleration**: Multi-GPU support with model pooling

---

## Installation

### Requirements

- Python 3.11+
- `uv` package manager
- (Optional) CUDA-compatible GPU for faster inference

### Setup

```bash
# Clone repository
cd gluRPC

# Install dependencies using uv
uv sync

# For development (includes pytest and other dev tools)
uv sync --extra dev
```

---

## Quick Start

### Combined Service (Recommended)

**Run both REST and gRPC in the same process** for maximum efficiency:

```bash
# Start combined service (shares models and cache between REST and gRPC)
uv run python run_glurpc_service.py --combined

# This starts:
# - gRPC on port 7003
# - REST on port 8000
# - Both sharing the same model instances and cache
```

📖 **See [COMBINED_SERVICE.md](COMBINED_SERVICE.md) for detailed documentation.**

### REST Service

#### 1. Start the Server

**Basic startup** (production):
```bash
uv run uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

**With reload** (development):
```bash
uv run uvicorn glurpc.app:app --host 0.0.0.0 --port 8000 --reload
```

**Using the entry point**:
```bash
uv run glurpc-server
```

#### 2. Verify Installation

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "cache_size": 0,
  "models_initialized": true,
  "queue_length": 2,
  "avg_fulfillment_time_ms": 0.0,
  "vmem_usage_mb": 3584.2,
  "device": "cuda",
  "total_requests_processed": 0,
  "total_errors": 0
}
```

#### 3. Interactive API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### 4. Quick Test

Convert a sample file:
```bash
curl -X POST "http://localhost:8000/convert_to_unified" \
  -F "file=@your_cgm_data.csv"
```

Generate a quick plot:
```bash
# Base64 encode your CSV
CSV_BASE64=$(cat your_unified_data.csv | base64 -w 0)

# Get prediction plot
curl -X POST "http://localhost:8000/quick_plot" \
  -H "Content-Type: application/json" \
  -d "{\"csv_base64\": \"$CSV_BASE64\"}" | jq -r '.plot_base64' | base64 -d > prediction.png
```

---

### gRPC Service (SNET Compatible)

The gRPC service provides the same functionality as REST but uses Protocol Buffers and is compatible with SingularityNET for blockchain-based payments.

#### 1. Build Protocol Buffers

```bash
./buildproto.sh
# Or manually:
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/glurpc.proto
```

#### 2. Start Combined REST + gRPC Service

```bash
# Run both REST (8000) and gRPC (7003) services
python run_glurpc_service.py

# Or gRPC only
python run_glurpc_service.py --grpc-only

# Or with SNET daemon for blockchain payments
python run_glurpc_service.py --daemon-config snetd_configs/snetd.ropsten.json
```

#### 3. Test gRPC Service

```bash
# Test with sample CGM data
python test_glurpc_service.py data/sample.csv

# Auto mode (non-interactive)
python test_glurpc_service.py data/sample.csv auto
```

#### 4. Quick gRPC Example (Python Client)

```python
import grpc
from service.service_spec import glurpc_pb2, glurpc_pb2_grpc

# Connect to gRPC service
channel = grpc.insecure_channel('localhost:7003')
stub = glurpc_pb2_grpc.GlucosePredictionStub(channel)

# Check health
health_response = stub.CheckHealth(glurpc_pb2.HealthRequest())
print(f"Status: {health_response.status}")

# Process and predict (requires API key in metadata)
metadata = [('x-api-key', 'your-api-key')]
request = glurpc_pb2.ProcessUnifiedRequest(
    csv_base64="<base64_encoded_csv>",
    force_calculate=False
)
response = stub.ProcessUnified(request, metadata=metadata)
print(f"Handle: {response.handle}")
```

**For detailed gRPC documentation**, see:
- [GRPC_SERVICE_README.md](GRPC_SERVICE_README.md) - Complete gRPC service guide
- [SNET_SERVICE_COMPARISON.md](SNET_SERVICE_COMPARISON.md) - Comparison with SNET example-service

---

## Configuration

### Environment Variables

All configuration options can be set via environment variables:

#### Cache Configuration

```bash
# Maximum number of datasets to cache (default: 128)
export MAX_CACHE_SIZE=128

# Enable/disable cache persistence to disk (default: True)
export ENABLE_CACHE_PERSISTENCE=True
```

#### Security Configuration

```bash
# Enable/disable API key authentication (default: False)
export ENABLE_API_KEYS=False

# If enabled, create api_keys_list file with one key per line:
# echo "your-secret-api-key-1" > api_keys_list
# echo "your-secret-api-key-2" >> api_keys_list
```

#### Data Processing Configuration

```bash
# Minimum data duration in minutes (default: 540 = 9 hours)
# Must be >= 540 (model requirement: 96 input points + 12 output points at 5min intervals)
export MINIMUM_DURATION_MINUTES=540

# Maximum wanted duration in minutes (default: 1080 = 18 hours)
# Larger datasets provide more prediction samples
export MAXIMUM_WANTED_DURATION=1080
```

#### Model and Inference Configuration

```bash
# Number of model copies per GPU device (default: 2)
# Increase for higher throughput, decrease if running out of VRAM
export NUM_COPIES_PER_DEVICE=2

# Number of background workers for calculations (default: 4)
export BACKGROUND_WORKERS_COUNT=4

# Batch size for inference (default: 32)
# Larger = faster but more memory
export BATCH_SIZE=32

# Number of Monte Carlo Dropout samples (default: 10)
# More samples = better uncertainty estimates but slower
export NUM_SAMPLES=10
```

### Configuration File

Alternatively, create a `.env` file in the project root:

```env
# .env
MAX_CACHE_SIZE=128
ENABLE_CACHE_PERSISTENCE=True
ENABLE_API_KEYS=True
NUM_COPIES_PER_DEVICE=2
BACKGROUND_WORKERS_COUNT=4
BATCH_SIZE=32
NUM_SAMPLES=10
```

---

## API Reference

### Authentication

Protected endpoints require an API key when authentication is enabled (`ENABLE_API_KEYS=True`):

```bash
curl -X POST "http://localhost:8000/process_unified" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"csv_base64": "..."}'
```

**Protected Endpoints**: `/process_unified`, `/draw_a_plot`, `/quick_plot`, `/cache_management`  
**Public Endpoints**: `/convert_to_unified`, `/health`

---

### 1. Convert to Unified Format

**Endpoint**: `POST /convert_to_unified`  
**Authentication**: None (public)  
**Content-Type**: `multipart/form-data`

Converts any supported CGM format (Dexcom, FreeStyle Libre, etc.) to the standardized Unified format.

**Request**:
```bash
curl -X POST "http://localhost:8000/convert_to_unified" \
  -F "file=@dexcom_export.csv"
```

**Response**:
```json
{
  "csv_content": "sequence_id,event_type,quality,datetime,glucose,notes,transmitter_id,transmitter_time\n1,EGV,OK,2025-12-01T08:00:00,120.0,,,",
  "error": null
}
```

**Supported Formats**:
- Dexcom G6/G7 standard export
- FreeStyle Libre AGP reports  
- Unified CSV format (pass-through)

---

### 2. Process and Cache Dataset

**Endpoint**: `POST /process_unified`  
**Authentication**: Required (if enabled)  
**Content-Type**: `application/json`

Processes a CSV file, performs quality checks, creates inference dataset, and caches it for future plot requests. Returns a unique handle for the dataset.

**Request**:
```json
{
  "csv_base64": "c2VxdWVuY2VfaWQsZXZlbnRfdHlwZS4uLg==",
  "force_calculate": false
}
```

**Parameters**:
- `csv_base64` (string, required): Base64-encoded unified CSV content
- `force_calculate` (boolean, optional): If `true`, bypasses cache and forces reprocessing. Default: `false`

**Response**:
```json
{
  "handle": "0742f5d8d69da1a6f05a0ad493072ab5af4e7c212474acc54c43f89460662e80",
  "warnings": {
    "flags": 0,
    "has_warnings": false,
    "messages": []
  },
  "error": null
}
```

**Cache Behavior**:
- **Direct Cache Hit**: Returns existing handle immediately (~10ms)
- **Superset Match**: If a larger dataset covering the same time range exists, returns that handle
- **Cache Miss**: Processes data, enqueues background inference, returns handle (~1-3s)

**Warning Flags**:
| Flag | Description |
|------|-------------|
| `TOO_SHORT` | Insufficient data duration for predictions |
| `CALIBRATION` | Sensor calibration events detected |
| `QUALITY` | Data quality issues (gaps, noise) |
| `IMPUTATION` | Gaps filled via interpolation |
| `OUT_OF_RANGE` | Glucose values outside normal range (40-400 mg/dL) |
| `TIME_DUPLICATES` | Duplicate timestamps detected and resolved |

**Example with curl**:
```bash
CSV_BASE64=$(cat unified_data.csv | base64 -w 0)
curl -X POST "http://localhost:8000/process_unified" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d "{\"csv_base64\": \"$CSV_BASE64\", \"force_calculate\": false}"
```

---

### 3. Generate Prediction Plot

**Endpoint**: `POST /draw_a_plot`  
**Authentication**: Required (if enabled)  
**Content-Type**: `application/json`

Generates a prediction plot for a specific sample in a cached dataset. Returns a PNG image.

**Request**:
```json
{
  "handle": "0742f5d8d69da1a6f05a0ad493072ab5af4e7c212474acc54c43f89460662e80",
  "index": 0
}
```

**Parameters**:
- `handle` (string, required): Dataset handle from `/process_unified`
- `index` (integer, required): Sample index in the dataset
  - `0` = Last sample (most recent)
  - `-1` = Second-to-last sample
  - `-(N-1)` = First sample (where N is dataset length)

**Response**: PNG image (binary, `image/png`)

**Plot Components**:
- **Blue line**: Historical glucose values (last 12 points = 1 hour) + actual future values (next 12 points = 1 hour)
- **Red line**: Median predicted glucose (next 12 points = 1 hour)
- **Blue gradient fan charts**: Prediction uncertainty distribution from Monte Carlo Dropout (10 samples)
  - Darker = earlier predictions
  - Lighter = later predictions
  - Width indicates uncertainty

**Timing**:
- If plot already calculated: ~100-500ms
- If inference needed: ~5-15 seconds (first request for a dataset)
- Subsequent requests for same index: instant (cached)

**Example**:
```bash
curl -X POST "http://localhost:8000/draw_a_plot" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"handle":"0742f5d8d69da1a6f05a0ad493072ab5af4e7c212474acc54c43f89460662e80","index":0}' \
  --output prediction.png
```

**Error Responses**:
- `404 Not Found`: Handle doesn't exist or has been evicted from cache
- `400 Bad Request`: Index out of range
- `500 Internal Server Error`: Calculation or rendering failed

---

### 4. Quick Plot (One-Shot)

**Endpoint**: `POST /quick_plot`  
**Authentication**: Required (if enabled)  
**Content-Type**: `application/json`

Processes data and immediately returns a base64-encoded plot for the last available sample. Combines `/process_unified` and `/draw_a_plot` in a single request.

**Request**:
```json
{
  "csv_base64": "c2VxdWVuY2VfaWQsZXZlbnRfdHlwZS4uLg==",
  "force_calculate": false
}
```

**Parameters**:
- `csv_base64` (string, required): Base64-encoded unified CSV content
- `force_calculate` (boolean, optional): Bypass cache if `true`. Default: `false`

**Response**:
```json
{
  "plot_base64": "iVBORw0KGgoAAAANSUhEUgAAA+gAAAJYCAYAAADxHswlAAAgAElEQVR4nOzdd5wV1f...",
  "warnings": {
    "flags": 0,
    "has_warnings": false,
    "messages": []
  },
  "error": null
}
```

**Use Case**: One-off predictions without needing to manage handles. Ideal for:
- Testing
- Simple integrations
- Single-use predictions

**Timing**:
- First request: ~6-18 seconds (processing + inference + calculation + rendering)
- Cached request: ~100-500ms (cache hit + rendering)

**Example**:
```bash
CSV_BASE64=$(cat unified_data.csv | base64 -w 0)
RESPONSE=$(curl -X POST "http://localhost:8000/quick_plot" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d "{\"csv_base64\": \"$CSV_BASE64\"}")

# Extract base64 plot and decode to file
echo $RESPONSE | jq -r '.plot_base64' | base64 -d > prediction.png
```

---

### 5. Cache Management

**Endpoint**: `POST /cache_management`  
**Authentication**: Required (if enabled)  
**Content-Type**: Query parameters

Manage the cache: flush, get info, delete specific handles, save/load from disk.

**Actions**:

#### Flush (Clear All)
```bash
curl -X POST "http://localhost:8000/cache_management?action=flush" \
  -H "X-API-Key: your-key"
```

**Response**:
```json
{
  "success": true,
  "message": "Cache flushed successfully",
  "cache_size": 0,
  "persisted_count": 0,
  "items_affected": null
}
```

#### Get Cache Info
```bash
curl -X POST "http://localhost:8000/cache_management?action=info" \
  -H "X-API-Key: your-key"
```

**Response**:
```json
{
  "success": true,
  "message": "Cache info retrieved",
  "cache_size": 42,
  "persisted_count": 42,
  "items_affected": null
}
```

#### Delete Specific Handle
```bash
curl -X POST "http://localhost:8000/cache_management?action=delete&handle=0742f5d8d69da1a6..." \
  -H "X-API-Key: your-key"
```

**Response**:
```json
{
  "success": true,
  "message": "Handle 0742f5d8... deleted successfully",
  "cache_size": 41,
  "persisted_count": 41,
  "items_affected": 1
}
```

#### Save to Disk
```bash
# Save all in-memory cache
curl -X POST "http://localhost:8000/cache_management?action=save" \
  -H "X-API-Key: your-key"

# Save specific handle
curl -X POST "http://localhost:8000/cache_management?action=save&handle=0742f5d8..." \
  -H "X-API-Key: your-key"
```

#### Load from Disk
```bash
curl -X POST "http://localhost:8000/cache_management?action=load&handle=0742f5d8..." \
  -H "X-API-Key: your-key"
```

**Parameters**:
- `action` (string, required): Operation to perform (`flush`, `info`, `delete`, `save`, `load`)
- `handle` (string, optional): Required for `delete` and `load`, optional for `save`

---

### 6. Health Check

**Endpoint**: `GET /health`  
**Authentication**: None (public)

Returns comprehensive server status, cache metrics, and performance statistics.

**Request**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "ok",
  "cache_size": 42,
  "models_initialized": true,
  "queue_length": 2,
  "avg_fulfillment_time_ms": 123.45,
  "vmem_usage_mb": 3584.2,
  "device": "cuda",
  "total_requests_processed": 1234,
  "total_errors": 5
}
```

**Fields**:
- `status`: Service status (`ok`, `degraded`, `error`)
- `cache_size`: Number of datasets currently cached (memory + disk)
- `models_initialized`: Whether ML models are loaded and ready
- `queue_length`: Number of models available in pool
- `avg_fulfillment_time_ms`: Average time to acquire a model from pool
- `vmem_usage_mb`: GPU VRAM usage in MB (0 if CPU)
- `device`: Inference device (`cpu`, `cuda`, `cuda:0`, etc.)
- `total_requests_processed`: Request counter since startup
- `total_errors`: Error counter since startup

**Use Case**:
- Health checks for load balancers
- Monitoring dashboards
- Service readiness probes

---

## Data Requirements

### Input Format

The service accepts CSV files from:
- **Dexcom G6/G7**: Standard export format
- **FreeStyle Libre**: AGP reports
- **Unified Format**: Custom standardized schema

### Unified CSV Schema

```csv
sequence_id,event_type,quality,datetime,glucose,notes,transmitter_id,transmitter_time
1,EGV,OK,2025-12-01T08:00:00,120.0,,,
1,EGV,OK,2025-12-01T08:05:00,125.0,,,
```

**Required columns**:
- `sequence_id`: Integer identifier for continuous sequences
- `event_type`: Event type (e.g., "EGV" for estimated glucose value)
- `quality`: Data quality indicator
- `datetime`: ISO 8601 timestamp
- `glucose`: Glucose value in mg/dL

### Minimum Data Requirements

- **Duration**: At least 540 minutes (9 hours) of continuous data
- **Interval**: 5-minute sampling (automatically interpolated if needed)
- **Prediction Window**: 
  - Input: 96 points (8 hours of history at 5min intervals)
  - Output: 12 points (1 hour prediction at 5min intervals)

### Data Processing Pipeline

1. **Format Detection**: Auto-detect input format
2. **Parsing**: Convert to unified format
3. **Gap Interpolation**: Fill gaps up to 15 minutes
4. **Timestamp Synchronization**: Align to 5-minute intervals
5. **Quality Validation**: Check duration and data quality
6. **Feature Engineering**: Extract temporal features (hour, day, etc.)
7. **Segmentation**: Split into continuous sequences
8. **Scaling**: Standardize values
9. **Dataset Creation**: Create Darts TimeSeries dataset

---

## Deployment

### Development / Local Testing

For local development and testing:

```bash
# Simple startup with auto-reload
uv run uvicorn glurpc.app:app --reload

# Or with explicit parameters
uv run uvicorn glurpc.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload \
  --log-level debug
```

**Configuration**: Default settings, no authentication required.

### Production Deployment

For production deployment scenarios (systemd, Docker, Kubernetes, AWS, multi-GPU), see the **[deployment/](deployment/)** directory.

**Available deployment configurations:**
- Single-server with systemd and Nginx
- Docker and Docker Compose
- Kubernetes with auto-scaling
- AWS ECS (Fargate)
- Multi-GPU server setup

⚠️ **Note**: All deployment configurations are untested examples and should be customized and tested thoroughly before production use. See [`deployment/README.md`](deployment/README.md) for details

---

## Project Structure

```
gluRPC/
├── src/glurpc/
│   ├── app.py              # FastAPI application & endpoints
│   ├── core.py             # Core business logic & orchestration
│   ├── engine.py           # Model management & background workers
│   ├── logic.py            # Data processing & ML inference
│   ├── state.py            # Cache & task coordination
│   ├── schemas.py          # Pydantic request/response models
│   ├── config.py           # Configuration & environment variables
│   └── data_classes.py     # Domain data models
├── tests/
│   ├── test_integration.py         # Integration tests
│   └── test_integration_load.py    # Load/stress tests
├── logs/                   # Timestamped log files (auto-created)
├── cache_storage/          # Persistent cache (auto-created)
├── files/                  # Generated plots (gitignored)
├── data/                   # Sample data (gitignored)
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Dependency lock file
├── README.md               # This file
└── HLA.md                  # High-Level Architecture document
```

---

## Performance

### Benchmarks

**Hardware**: NVIDIA RTX 4090 (24GB VRAM), AMD Ryzen 9 5950X

| Operation | First Request | Cached Request |
|-----------|--------------|----------------|
| Convert CSV | 50-200ms | N/A |
| Process & Cache | 1-3s | 10-50ms (cache hit) |
| Generate Plot | 5-15s | 100-500ms |
| Quick Plot | 6-18s | 100-500ms |

**Throughput**:
- With 2 GPUs (4 model copies): ~40-60 plots/minute
- Cache hit rate >80% in typical usage: ~200-300 plots/minute

### Resource Usage

- **Memory**: ~8-12GB (2 model copies @ 2GB each + cache)
- **VRAM**: ~4-6GB per GPU (2 model copies)
- **CPU**: 2-4 cores recommended
- **Disk**: ~10MB per cached dataset

### Cache Performance

- **Memory Cache Hit**: ~10ms
- **Disk Cache Hit**: ~50-100ms (load from Parquet)
- **Cache Miss**: ~1-3s (processing)
- **Superset Match**: ~50ms (metadata lookup + return)

---

## Testing

### Run All Tests

```bash
uv run pytest tests/
```

### Run with Coverage

```bash
uv run pytest tests/ --cov=src/glurpc --cov-report=html
```

### Run Specific Test

```bash
uv run pytest tests/test_integration.py::test_quick_plot
```

### Load Testing

```bash
# Run load test with 10 concurrent requests
uv run pytest tests/test_integration_load.py::test_concurrent_quick_plot -v
```

---

## Logging

### Log Location

Logs are written to `logs/glurpc_YYYYMMDD_HHMMSS.log` with the following information:

- Data processing pipeline steps
- Dataset shapes at each transformation
- Scaler parameters (min/scale values)
- Model predictions statistics
- Cache operations
- Errors with full stack traces

### Log Levels

- **DEBUG**: Detailed execution traces (disabled for `calc` logger to reduce verbosity)
- **INFO**: Request/response logging, pipeline steps
- **WARNING**: Data quality issues, cache misses
- **ERROR**: Failures with stack traces

### Example Log Entries

```
2025-12-01 08:26:40,843 - glurpc - INFO - Action: process_and_cache started (force=False)
2025-12-01 08:26:40,873 - glurpc - INFO - Action: process_and_cache - generated handle=0742f5d8..., df_shape=(3889, 9)
2025-12-01 08:26:40,902 - glurpc.logic - INFO - Dataset creation successful: 3707 samples, warnings=0
2025-12-01 08:26:45,332 - glurpc - INFO - InfWorker 0: Running FULL inference for 0742f5d8... (3707 items)
2025-12-01 08:27:02,118 - glurpc - INFO - Action: generate_plot_from_handle completed - png_size=125432 bytes
```

---

## Troubleshooting

### Model Download Issues

If HuggingFace download fails:

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Or manually download and place in cache
huggingface-cli download Livia-Zaharia/gluformer_models \
  gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth
```

### Memory Issues

If running out of memory:

```bash
# Reduce cache size
export MAX_CACHE_SIZE=32

# Reduce batch size
export BATCH_SIZE=16

# Reduce model copies
export NUM_COPIES_PER_DEVICE=1
```

### CUDA Out of Memory

```bash
# Use smaller batch size
export BATCH_SIZE=16

# Reduce model copies per GPU
export NUM_COPIES_PER_DEVICE=1

# Or use CPU (slower)
# Models will auto-detect and use CPU if no GPU available
```

### Plot Generation Failures

Ensure kaleido is properly installed:

```bash
uv add kaleido==0.2.1
```

### API Key Issues

Check API keys file:

```bash
# File should exist and contain keys (one per line)
cat api_keys_list

# Ensure no extra whitespace or comments
sed '/^#/d; /^$/d' api_keys_list
```

### Cache Persistence Issues

Check disk permissions:

```bash
# Ensure cache directory is writable
mkdir -p cache_storage
chmod 755 cache_storage

# Disable persistence if issues persist
export ENABLE_CACHE_PERSISTENCE=False
```

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests (`uv run pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Development Guidelines

- Use type hints for all functions
- Add docstrings for public APIs
- Follow existing code style
- Add tests for new features
- Update documentation

---

## License

See LICENSE file for details.

---

## Citation

If you use this service in your research, please cite:

```bibtex
@software{glurpc2025,
  title={GluRPC: REST API for Glucose Prediction},
  author={GlucoseDAO Contributors},
  year={2025},
  url={https://github.com/glucosedao/gluRPC}
}
```

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: 
  - [HLA.md](HLA.md) - High-Level Architecture
  - [API Docs](http://localhost:8000/docs) - Interactive Swagger UI
- **Contact**: Project maintainers

---

## Acknowledgments

- **[GlucoBench](https://github.com/IrinaStatsLab/GlucoBench)** - Curated list of Continuous Glucose Monitoring datasets with prediction benchmarks
- **[Gluformer](https://github.com/mrsergazinov/gluformer)** - Transformer-based model for glucose prediction
- **[CGM-Format](https://github.com/GlucoseDAO/cgm_format)** - Library for parsing CGM data
- **GlucoseDAO community** for contributions and feedback
