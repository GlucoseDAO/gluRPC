# GluRPC

**REST API server for real-time glucose prediction using the Gluformer transformer model.**

## Overview

GluRPC is a production-ready FastAPI service that processes continuous glucose monitoring (CGM) data and provides blood glucose predictions using the Gluformer model. The service handles multiple CGM device formats (Dexcom, FreeStyle Libre), performs quality checks, and generates visual predictions with uncertainty quantification.

## Features

- 🔄 **Multi-format CGM Support**: Auto-detects and parses Dexcom, FreeStyle Libre, and unified CSV formats
- 🧠 **Transformer-based Predictions**: Uses pre-trained Gluformer models from HuggingFace
- 📊 **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence intervals
- ⚡ **Efficient Caching**: SHA256-based dataset caching for multiple simultaneous users (up to 128)
- 🔍 **Quality Assurance**: Comprehensive data quality checks with detailed warnings
- 📈 **Interactive Visualizations**: Plotly-based prediction plots with true values overlay
- 📝 **Detailed Logging**: Timestamped logs with full pipeline traceability

## Installation

### Requirements

- Python 3.11+
- `uv` package manager

### Setup

```bash
# Clone repository
cd gluRPC

# Install dependencies
uv sync

# For development (includes pytest)
uv sync --extra dev
```

## Quick Start

### Start the Server

```bash
uv run uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

Or use the entry point:

```bash
uv run glurpc-server
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Convert CSV to Unified Format

**Endpoint**: `POST /convert_to_unified`

Converts any supported CGM format to the standardized Unified format.

**Request**: Multipart form-data with file upload

**Response**:
```json
{
  "csv_content": "sequence_id,event_type,quality,datetime,glucose,...",
  "error": null
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/convert_to_unified" \
  -F "file=@dexcom_export.csv"
```

---

### 2. Process and Cache Dataset

**Endpoint**: `POST /process_unified`

Processes a CSV file, performs quality checks, and caches the prepared dataset for inference.

**Request**:
```json
{
  "csv_base64": "<base64-encoded-csv-content>"
}
```

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

**Warning Types**:
- `TOO_SHORT`: Insufficient data duration
- `CALIBRATION`: Sensor calibration events detected
- `QUALITY`: Data quality issues
- `IMPUTATION`: Gaps filled via interpolation
- `OUT_OF_RANGE`: Values outside normal glucose range
- `TIME_DUPLICATES`: Duplicate timestamps found

---

### 3. Generate Prediction Plot

**Endpoint**: `POST /draw_a_plot`

Generates a prediction plot for a specific sample in a cached dataset.

**Request**:
```json
{
  "handle": "0742f5d8...",
  "index": 10
}
```

**Response**: PNG image (binary)

**Example**:
```bash
curl -X POST "http://localhost:8000/draw_a_plot" \
  -H "Content-Type: application/json" \
  -d '{"handle":"0742f5d8...","index":10}' \
  --output prediction.png
```

**Plot Details**:
- **Blue line**: Historical glucose values (last hour) + actual future values
- **Red line**: Median predicted glucose (next hour)
- **Blue gradient fans**: Prediction uncertainty distribution (24 stochastic samples)

---

### 4. Quick Plot (One-Shot)

**Endpoint**: `POST /quick_plot`

Processes data and immediately returns a plot for the last available sample. Dataset is not cached.

**Request**:
```json
{
  "csv_base64": "<base64-encoded-csv-content>"
}
```

**Response**:
```json
{
  "plot_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "warnings": {...},
  "error": null
}
```

**Use Case**: One-off predictions without needing to manage handles.

---

### 5. Health Check

**Endpoint**: `GET /health`

Returns server status and cache size.

**Response**:
```json
{
  "status": "ok",
  "cache_size": 5
}
```

## Data Requirements

### Input Format

The service accepts CSV files from:
- **Dexcom G6/G7**: Standard export format
- **FreeStyle Libre**: AGP reports
- **Unified Format**: Custom standardized schema

### Minimum Data Requirements

- **Duration**: At least 15 minutes of continuous data
- **Interval**: 5-minute sampling (automatically interpolated if needed)
- **Prediction Window**: 
  - Input: 96 points (8 hours of history)
  - Output: 12 points (1 hour prediction)

## Project Structure

```
gluRPC/
├── src/glurpc/
│   ├── app.py              # FastAPI application
│   ├── core.py             # Core inference logic
│   ├── schemas.py          # Pydantic request/response models
│   └── data_classes.py     # Gluformer config models
├── tests/
│   └── test_integration.py # Integration tests
├── data/                   # Sample data (gitignored)
├── logs/                   # Timestamped log files (gitignored)
├── files/                  # Generated plots (gitignored)
├── pyproject.toml          # Project dependencies
└── README.md
```

## Configuration

### Model Configuration

Default model: `gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth`

Model is automatically downloaded from HuggingFace on first startup.

### Inference Parameters

- **Input chunk**: 96 timesteps (8 hours @ 5min intervals)
- **Output chunk**: 12 timesteps (1 hour @ 5min intervals)
- **Stochastic samples**: 24 (for uncertainty quantification)
- **Gap threshold**: 45 minutes (max interpolation gap)
- **Cache size**: 128 simultaneous datasets

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/glurpc

# Run specific test
uv run pytest tests/test_integration.py::test_quick_plot
```

## Logging

Logs are written to `logs/glurpc_YYYYMMDD_HHMMSS.log` with the following information:

- Data processing pipeline steps
- Dataset shapes at each transformation
- Scaler parameters (min/scale values)
- Model predictions statistics
- Cache operations
- Errors with full stack traces

**Example log entry**:
```
2025-12-01 08:26:40,843 - glurpc - INFO - Input DF Shape: (3889, 9)
2025-12-01 08:26:40,873 - glurpc - INFO - Interpolated DF Shape: (3921, 10)
2025-12-01 08:26:40,902 - glurpc - INFO - Target Scaler Fitted: Min=[-0.14498141], Scale=[0.00371747]
2025-12-01 08:26:40,910 - glurpc - INFO - Created Dataset with 3707 samples
```

## Production Deployment

### Environment Variables

```bash
# Optional overrides
export GLURPC_HOST="0.0.0.0"
export GLURPC_PORT="8000"
export GLURPC_LOG_LEVEL="INFO"
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "uvicorn", "glurpc.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=GluRPC Glucose Prediction Service
After=network.target

[Service]
Type=simple
User=glurpc
WorkingDirectory=/opt/glurpc
ExecStart=/usr/local/bin/uv run uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance

- **Inference Time**: ~3-5 seconds per batch (16 samples)
- **Memory Usage**: ~2GB (model loaded)
- **Cache Eviction**: FIFO when 128 datasets reached
- **Concurrency**: Supports multiple simultaneous requests

## Dependencies

### Core
- `fastapi`: REST API framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation
- `darts`: Time series library
- `torch`: PyTorch for model inference

### Data Processing
- `polars`: Fast DataFrame operations
- `pandas`: Legacy compatibility
- `cgm-format`: CGM data parsing
- `glucosedao-glucobench`: Gluformer model utilities

### Visualization
- `plotly`: Interactive plots
- `kaleido`: Static image export
- `scipy`: Statistical functions (KDE)

## Troubleshooting

### Model Download Issues

If HuggingFace download fails:
```bash
# Manually download and specify path
export HF_HOME=/path/to/cache
```

### Memory Issues

Reduce cache size or batch size in `core.py`:
```python
MAX_CACHE_SIZE = 32  # Default: 128
batch_size = 8       # Default: 16
```

### Plot Generation Failures

Ensure kaleido is properly installed:
```bash
uv add kaleido==0.2.1
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests (`uv run pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## License

See LICENSE file for details.

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

## Support

- **Issues**: GitHub Issues
- **Documentation**: See IMPLEMENTATION.md for technical details
- **Contact**: [Project maintainers]

## Acknowledgments

- Gluformer model by Livia Zaharia
- CGM-Format library for data parsing
- GlucoseDAO community
