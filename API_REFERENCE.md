# GluRPC API Reference

**Version:** 1.0  
**Base URL:** `http://your-server:8000`  
**Protocol:** REST API with JSON payloads

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Concepts](#common-concepts)
4. [Endpoints](#endpoints)
   - [POST /convert_to_unified](#post-convert_to_unified)
   - [POST /process_unified](#post-process_unified)
   - [POST /draw_a_plot](#post-draw_a_plot)
   - [POST /quick_plot](#post-quick_plot)
   - [POST /cache_management](#post-cache_management)
   - [GET /health](#get-health)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Client Implementation Examples](#client-implementation-examples)
8. [Best Practices](#best-practices)

---

## Overview

GluRPC is a glucose prediction service that processes Continuous Glucose Monitoring (CGM) data and generates forecasts with visualizations. The API provides:

- **Data conversion** from proprietary CGM formats to a unified format
- **Data processing** and inference caching for efficient repeated access
- **Interactive plotting** with Plotly-compatible JSON outputs
- **Quick plot** endpoint for immediate results
- **Cache management** for persistence and optimization
- **Health monitoring** with detailed service metrics

The service uses ML models to predict future glucose values and provides interactive visualizations showing historical data, predictions, and confidence intervals.

---

## Authentication

### API Key Authentication

The service supports optional API key authentication via the `X-API-Key` header.

**When enabled** (via `ENABLE_API_KEYS` environment variable):
- All endpoints except `/health` require a valid API key
- Pass the API key in the request header: `X-API-Key: your-api-key-here`
- Invalid or missing keys return `401 Unauthorized` or `403 Forbidden`

**When disabled** (default for development):
- All endpoints are accessible without authentication

```http
X-API-Key: your-api-key-here
```

**Endpoints requiring authentication (when enabled):**
- `/process_unified`
- `/draw_a_plot`
- `/quick_plot`
- `/cache_management`

**Endpoints always public:**
- `/convert_to_unified`
- `/health`

---

## Common Concepts

### Handles

A **handle** is a unique hash string (8-64 characters) that identifies a processed dataset in the cache. After uploading and processing CSV data via `/process_unified`, you receive a handle that can be used to request plots for different samples without re-uploading the data.

**Example handle:** `"a1b2c3d4e5f6789012345678"`

### Sample Indexing

The service uses **negative indexing** for samples:
- `0` = most recent/last sample (end of timeline)
- `-1` = second-to-last sample
- `-10` = 10 samples back from the end
- `-N` = N samples back from the end

This allows intuitive access to recent predictions, which is typically what users want to see first.

### Unified CSV Format

The service expects CSV data in a standardized format with three columns:

```csv
sequence_id,timestamp,glucose
1,2024-01-01 00:00:00,100
1,2024-01-01 00:05:00,105
1,2024-01-01 00:10:00,110
```

- **sequence_id**: Integer identifier for the data sequence
- **timestamp**: ISO format datetime string (YYYY-MM-DD HH:MM:SS)
- **glucose**: Glucose value in mg/dL

### Warnings System

Processing warnings indicate potential data quality issues without failing the request. All warnings are returned as structured `FormattedWarnings` objects with:

- `has_warnings`: Boolean indicating if any warnings exist
- Individual boolean flags: `too_short`, `calibration`, `quality`, `imputation`, `out_of_range`, `time_duplicates`
- `messages`: Array of human-readable warning descriptions

### Force Calculate

The `force_calculate` boolean parameter (default: `false`) allows bypassing cache:
- `false`: Use cached results if available (faster)
- `true`: Recalculate even if cached (useful after model updates or for testing)

### Overload Protection

The service monitors queue status and rejects requests when overloaded:
- Returns HTTP `503 Service Unavailable`
- Includes `Retry-After: 30` header
- Load status levels: idle → lightly loaded → heavily loaded → overloaded → full

---

## Endpoints

### POST /convert_to_unified

Convert proprietary CGM formats (e.g., Dexcom Clarity exports) to the unified CSV format.

**Authentication:** Not required

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** File upload with field name `file`

**cURL Example:**
```bash
curl -X POST http://localhost:8000/convert_to_unified \
  -F "file=@Clarity_Export.csv"
```

**Python Example:**
```python
import requests

with open("Clarity_Export.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/convert_to_unified",
        files={"file": ("data.csv", f, "text/csv")}
    )
    
data = response.json()
if data["error"] is None:
    unified_csv = data["csv_content"]
    print("Conversion successful!")
else:
    print(f"Error: {data['error']}")
```

**Response:** `ConvertResponse`
```json
{
  "csv_content": "sequence_id,timestamp,glucose\n1,2024-01-01 00:00:00,100\n...",
  "error": null
}
```

**Fields:**
- `csv_content` (string, optional): Converted CSV content in unified format
- `error` (string, optional): Error message if conversion failed, null on success

**Status Codes:**
- `200 OK`: Success (check `error` field for conversion errors)

---

### POST /process_unified

Upload and process CSV data (in unified format) to generate predictions and cache the results.

**Authentication:** Required (when enabled)

**Request:** `ProcessRequest`
```json
{
  "csv_base64": "c2VxdWVuY2VfaWQsdGltZXN0YW1wLGdsdWNvc2UKMS...",
  "force_calculate": false
}
```

**Fields:**
- `csv_base64` (string, required): Base64-encoded CSV content in unified format
- `force_calculate` (boolean, optional, default: false): Force recalculation bypassing cache

**Python Example:**
```python
import base64
import requests

# Read and encode CSV
with open("unified_data.csv", "rb") as f:
    csv_content = f.read()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')

# Send request
response = requests.post(
    "http://localhost:8000/process_unified",
    json={"csv_base64": csv_base64, "force_calculate": False},
    headers={"X-API-Key": "your-api-key"}  # If auth enabled
)

data = response.json()
if data["error"] is None:
    handle = data["handle"]
    total_samples = data["total_samples"]
    print(f"Processing complete! Handle: {handle}, Samples: {total_samples}")
    
    if data["warnings"]["has_warnings"]:
        print("Warnings:", data["warnings"]["messages"])
else:
    print(f"Error: {data['error']}")
```

**Response:** `UnifiedResponse`
```json
{
  "handle": "a1b2c3d4e5f6789012345678",
  "total_samples": 144,
  "warnings": {
    "has_warnings": true,
    "too_short": false,
    "calibration": false,
    "quality": true,
    "imputation": true,
    "out_of_range": false,
    "time_duplicates": false,
    "messages": [
      "QUALITY: Low quality data points detected",
      "IMPUTATION: Missing values were imputed"
    ]
  },
  "error": null
}
```

**Fields:**
- `handle` (string, optional): Unique identifier for this dataset in cache
- `total_samples` (integer, optional): Number of prediction samples available (≥ 1)
- `warnings` (FormattedWarnings): Processing warnings
- `error` (string, optional): Error message if processing failed

**Status Codes:**
- `200 OK`: Success (check `error` field)
- `503 Service Unavailable`: Service overloaded (retry after 30s)

---

### POST /draw_a_plot

Generate a Plotly JSON plot for a specific sample from a cached dataset.

**Authentication:** Required (when enabled)

**Request:** `PlotRequest`
```json
{
  "handle": "a1b2c3d4e5f6789012345678",
  "index": 0,
  "force_calculate": false
}
```

**Fields:**
- `handle` (string, required): Handle from `/process_unified` response
- `index` (integer, required): Sample index (≤ 0, where 0 is most recent)
- `force_calculate` (boolean, optional, default: false): Force plot recalculation

**Python Example:**
```python
import requests
import plotly.graph_objects as go

response = requests.post(
    "http://localhost:8000/draw_a_plot",
    json={
        "handle": "a1b2c3d4e5f6789012345678",
        "index": 0,  # Most recent sample
        "force_calculate": False
    },
    headers={"X-API-Key": "your-api-key"}
)

if response.status_code == 200:
    plot_data = response.json()
    
    # Use with Plotly
    fig = go.Figure(plot_data)
    fig.show()
    
    # Or save to file
    fig.write_html("glucose_plot.html")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

**Response:** Plotly JSON (dict)
```json
{
  "data": [
    {
      "type": "scatter",
      "x": ["2024-01-01 00:00:00", "2024-01-01 00:05:00", ...],
      "y": [100, 105, 110, ...],
      "name": "Historical Glucose",
      "mode": "lines+markers"
    },
    {
      "type": "scatter",
      "x": ["2024-01-01 12:00:00", "2024-01-01 12:05:00", ...],
      "y": [120, 125, 130, ...],
      "name": "Predicted Glucose",
      "mode": "lines"
    }
  ],
  "layout": {
    "title": "Glucose Prediction",
    "xaxis": {"title": "Time"},
    "yaxis": {"title": "Glucose (mg/dL)"}
  }
}
```

**Response Format:**
- Standard Plotly JSON with `data` (traces) and `layout` objects
- Compatible with Gradio `gr.Plot` component
- Can be directly used with `plotly.graph_objects.Figure(plot_data)`

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid index or parameters
- `404 Not Found`: Handle not found in cache
- `499 Client Closed Request`: Client disconnected (internal)
- `503 Service Unavailable`: Service overloaded
- `504 Gateway Timeout`: Request timeout

**Features:**
- **Concurrent request deduplication**: Multiple requests for same (handle, index) share computation
- **Disconnect detection**: Server cancels computation if client disconnects
- **Plot caching**: Results are cached for repeated access

---

### POST /quick_plot

Upload CSV and immediately get a plot for the most recent sample. Combines `/process_unified` + `/draw_a_plot` in one request.

**Authentication:** Required (when enabled)

**Request:** `ProcessRequest`
```json
{
  "csv_base64": "c2VxdWVuY2VfaWQsdGltZXN0YW1wLGdsdWNvc2UKMS...",
  "force_calculate": false
}
```

**Fields:**
- `csv_base64` (string, required): Base64-encoded CSV in unified format
- `force_calculate` (boolean, optional, default: false): Force recalculation

**Python Example:**
```python
import base64
import requests
import plotly.graph_objects as go

# Read and encode CSV
with open("unified_data.csv", "rb") as f:
    csv_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/quick_plot",
    json={"csv_base64": csv_base64},
    headers={"X-API-Key": "your-api-key"}
)

data = response.json()
if data["error"] is None:
    # Extract plot
    fig = go.Figure(data["plot_data"])
    fig.show()
    
    # Check warnings
    if data["warnings"]["has_warnings"]:
        print("Warnings:", data["warnings"]["messages"])
else:
    print(f"Error: {data['error']}")
```

**Response:** `QuickPlotResponse`
```json
{
  "plot_data": {
    "data": [...],
    "layout": {...}
  },
  "warnings": {
    "has_warnings": false,
    "too_short": false,
    "calibration": false,
    "quality": false,
    "imputation": false,
    "out_of_range": false,
    "time_duplicates": false,
    "messages": []
  },
  "error": null
}
```

**Fields:**
- `plot_data` (dict): Plotly figure JSON with `data` and `layout` keys
- `warnings` (FormattedWarnings): Processing warnings
- `error` (string, optional): Error message if failed

**Status Codes:**
- `200 OK`: Success (check `error` field)
- `503 Service Unavailable`: Service overloaded

---

### POST /cache_management

Manage the inference and plot caches (flush, info, delete, save, load).

**Authentication:** Required

**Request:** Query parameters
- `action` (string, required): Action to perform: `flush`, `info`, `delete`, `save`, `load`
- `handle` (string, optional): Required for `delete` and `load` actions

**Actions:**

#### 1. Flush Cache
Clear all cached data (memory and disk).

```bash
curl -X POST "http://localhost:8000/cache_management?action=flush" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "message": "Cache flushed successfully",
  "cache_size": 0,
  "persisted_count": 0,
  "items_affected": null
}
```

#### 2. Get Cache Info
Retrieve cache statistics.

```bash
curl -X POST "http://localhost:8000/cache_management?action=info" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "message": "Cache info retrieved (Inference: 5, Plots: 12)",
  "cache_size": 5,
  "persisted_count": 17,
  "items_affected": null
}
```

#### 3. Delete Specific Handle
Remove a specific dataset from cache.

```bash
curl -X POST "http://localhost:8000/cache_management?action=delete&handle=a1b2c3d4" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "message": "Handle a1b2c3d4 deleted successfully",
  "cache_size": 4,
  "persisted_count": 4,
  "items_affected": 1
}
```

#### 4. Save Cache
No-op (cache is auto-persisted).

```bash
curl -X POST "http://localhost:8000/cache_management?action=save" \
  -H "X-API-Key: your-api-key"
```

#### 5. Load Handle
Verify/load a handle from disk to memory.

```bash
curl -X POST "http://localhost:8000/cache_management?action=load&handle=a1b2c3d4" \
  -H "X-API-Key: your-api-key"
```

**Python Example:**
```python
import requests

# Get cache info
response = requests.post(
    "http://localhost:8000/cache_management?action=info",
    headers={"X-API-Key": "your-api-key"}
)
info = response.json()
print(f"Cache size: {info['cache_size']} items")

# Delete a specific handle
response = requests.post(
    "http://localhost:8000/cache_management",
    params={"action": "delete", "handle": "a1b2c3d4"},
    headers={"X-API-Key": "your-api-key"}
)
result = response.json()
print(f"Deleted: {result['success']}")
```

**Response:** `CacheManagementResponse`
- `success` (boolean): Whether operation succeeded
- `message` (string): Status description
- `cache_size` (integer): Items in inference cache
- `persisted_count` (integer): Total persisted items
- `items_affected` (integer, optional): Number of items affected

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid action or missing required parameters

---

### GET /health

Health check endpoint with comprehensive service statistics.

**Authentication:** Not required

**Request:**
```bash
curl http://localhost:8000/health
```

**Python Example:**
```python
import requests

response = requests.get("http://localhost:8000/health")
health = response.json()

print(f"Status: {health['status']}")
print(f"Load: {health['load_status']}")
print(f"Cache size: {health['cache_size']}")
print(f"Models initialized: {health['models_initialized']}")
print(f"Avg request time: {health['avg_request_time_ms']:.2f}ms")
```

**Response:** `HealthResponse`
```json
{
  "status": "ok",
  "load_status": "lightly loaded",
  "cache_size": 5,
  "models_initialized": true,
  "available_priority_models": 1,
  "available_general_models": 2,
  "avg_fulfillment_time_ms": 12.5,
  "vmem_usage_mb": 2048.0,
  "device": "cuda:0",
  "total_http_requests": 150,
  "total_http_errors": 2,
  "avg_request_time_ms": 125.5,
  "median_request_time_ms": 98.0,
  "min_request_time_ms": 5.2,
  "max_request_time_ms": 3500.0,
  "inference_requests_by_priority": {
    "0": 50,
    "1": 80,
    "3": 20
  },
  "total_inference_errors": 0,
  "total_calc_runs": 145,
  "total_calc_errors": 0,
  "inference_queue_size": 2,
  "inference_queue_capacity": 100,
  "calc_queue_size": 1,
  "calc_queue_capacity": 50
}
```

**Key Fields:**
- `status`: Service health: `ok`, `degraded`, or `error`
- `load_status`: Queue load: `idle`, `lightly loaded`, `heavily loaded`, `overloaded`, `full`
- `cache_size`: Number of cached datasets
- `models_initialized`: Whether ML models are ready
- `available_priority_models`: Number of idle priority models (should be 1 when idle)
- `available_general_models`: Number of idle general models (higher = more capacity)
- `device`: Inference device (cpu, cuda, cuda:0, etc.)
- `vmem_usage_mb`: Memory usage in MB (VRAM for GPU, RSS for CPU)
- `avg_request_time_ms`: Average HTTP request latency
- `inference_queue_size` / `inference_queue_capacity`: Current queue utilization
- `total_http_requests` / `total_http_errors`: Request statistics

**Use Cases:**
- Monitor service health in dashboards
- Poll during long operations to show progress
- Check load status before submitting requests
- Debug performance issues

**Status Codes:**
- `200 OK`: Always returns 200 (check `status` field for actual health)

---

## Data Models

### FormattedWarnings

Structured warnings from data processing.

```python
{
  "has_warnings": bool,          # True if any warnings exist
  "too_short": bool,             # Data too short for reliable predictions
  "calibration": bool,           # Calibration events detected
  "quality": bool,               # Low quality data points
  "imputation": bool,            # Missing values imputed
  "out_of_range": bool,          # Values outside normal range
  "time_duplicates": bool,       # Duplicate timestamps
  "messages": List[str]          # Human-readable descriptions
}
```

### UnifiedResponse

Response from `/process_unified`.

```python
{
  "handle": Optional[str],              # Unique cache identifier (8-64 chars)
  "total_samples": Optional[int],       # Number of samples (≥1)
  "warnings": FormattedWarnings,        # Processing warnings
  "error": Optional[str]                # Error message or null
}
```

### PlotRequest

Request for `/draw_a_plot`.

```python
{
  "handle": str,                  # Handle from process_unified
  "index": int,                   # Sample index (≤0)
  "force_calculate": bool         # Force recalculation (default: false)
}
```

### QuickPlotResponse

Response from `/quick_plot`.

```python
{
  "plot_data": dict,              # Plotly JSON (data + layout)
  "warnings": FormattedWarnings,  # Processing warnings
  "error": Optional[str]          # Error message or null
}
```

### ConvertResponse

Response from `/convert_to_unified`.

```python
{
  "csv_content": Optional[str],   # Converted CSV in unified format
  "error": Optional[str]          # Error message or null
}
```

### ProcessRequest

Request for `/process_unified` and `/quick_plot`.

```python
{
  "csv_base64": str,              # Base64-encoded CSV
  "force_calculate": bool         # Force recalculation (default: false)
}
```

### CacheManagementResponse

Response from `/cache_management`.

```python
{
  "success": bool,                    # Operation succeeded
  "message": Optional[str],           # Status message
  "cache_size": int,                  # Inference cache size
  "persisted_count": int,             # Total persisted items
  "items_affected": Optional[int]     # Items affected by operation
}
```

### HealthResponse

Response from `/health`.

```python
{
  "status": str,                      # "ok", "degraded", or "error"
  "load_status": str,                 # Queue load level
  "cache_size": int,                  # Cached datasets
  "models_initialized": bool,         # Models ready
  "available_priority_models": int,   # Idle priority models (1 when idle, 0 when busy)
  "available_general_models": int,    # Idle general models (higher = more capacity)
  "avg_fulfillment_time_ms": float,   # Model acquisition time
  "vmem_usage_mb": float,             # Memory usage (VRAM for GPU, RSS for CPU)
  "device": str,                      # Inference device
  "total_http_requests": int,         # Total requests
  "total_http_errors": int,           # Total errors
  "avg_request_time_ms": float,       # Average request time
  "median_request_time_ms": float,    # Median request time
  "min_request_time_ms": float,       # Minimum request time
  "max_request_time_ms": float,       # Maximum request time
  "inference_requests_by_priority": dict,  # Requests by priority
  "total_inference_errors": int,      # Inference errors
  "total_calc_runs": int,             # Plot calculations
  "total_calc_errors": int,           # Calculation errors
  "inference_queue_size": int,        # Current inference queue
  "inference_queue_capacity": int,    # Max inference queue
  "calc_queue_size": int,             # Current calc queue
  "calc_queue_capacity": int          # Max calc queue
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Success (check response for `error` field) |
| 400 | Bad Request | Invalid parameters, missing required fields |
| 401 | Unauthorized | API key missing (when auth enabled) |
| 403 | Forbidden | Invalid API key |
| 404 | Not Found | Handle not found in cache |
| 499 | Client Closed Request | Client disconnected (internal use) |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Server overloaded, retry later |
| 504 | Gateway Timeout | Request took too long |

### Error Response Format

Most endpoints return errors within the response body:

```json
{
  "error": "Handle not found in cache",
  "... other fields ..."
}
```

Some endpoints (e.g., overload protection) use HTTP errors:

```json
{
  "detail": "Service overloaded. Queue is heavily loaded. Please retry later."
}
```

### Retry Strategy

For `503 Service Unavailable`:
- Wait at least 30 seconds (check `Retry-After` header)
- Implement exponential backoff
- Poll `/health` to check `load_status` before retrying

For `504 Gateway Timeout`:
- Retry with same parameters
- Plot calculations are deduplicated, so retries won't cause duplicate work

For `499 Client Closed Request`:
- Don't retry automatically (client cancelled intentionally)

---

## Client Implementation Examples

### Complete Workflow Example (Python)

```python
import base64
import time
import requests
from typing import Optional
import plotly.graph_objects as go

class GluRPCClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def convert_csv(self, file_path: str) -> Optional[str]:
        """Convert proprietary CSV to unified format."""
        with open(file_path, 'rb') as f:
            response = self.session.post(
                f"{self.base_url}/convert_to_unified",
                files={"file": (file_path, f, "text/csv")}
            )
        response.raise_for_status()
        data = response.json()
        if data['error']:
            raise ValueError(f"Conversion failed: {data['error']}")
        return data['csv_content']
    
    def process_csv(self, csv_content: str, force: bool = False) -> tuple:
        """Process CSV and get handle."""
        csv_base64 = base64.b64encode(csv_content.encode()).decode()
        response = self.session.post(
            f"{self.base_url}/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": force}
        )
        response.raise_for_status()
        data = response.json()
        if data['error']:
            raise ValueError(f"Processing failed: {data['error']}")
        return data['handle'], data['total_samples'], data['warnings']
    
    def get_plot(self, handle: str, index: int = 0, force: bool = False) -> dict:
        """Get plot for specific sample."""
        response = self.session.post(
            f"{self.base_url}/draw_a_plot",
            json={"handle": handle, "index": index, "force_calculate": force}
        )
        response.raise_for_status()
        return response.json()
    
    def quick_plot(self, csv_content: str, force: bool = False) -> tuple:
        """Get immediate plot from CSV."""
        csv_base64 = base64.b64encode(csv_content.encode()).decode()
        response = self.session.post(
            f"{self.base_url}/quick_plot",
            json={"csv_base64": csv_base64, "force_calculate": force}
        )
        response.raise_for_status()
        data = response.json()
        if data['error']:
            raise ValueError(f"Quick plot failed: {data['error']}")
        return data['plot_data'], data['warnings']
    
    def get_health(self) -> dict:
        """Get service health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def wait_for_service(self, timeout: int = 60) -> bool:
        """Wait for service to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                health = self.get_health()
                if health['models_initialized'] and health['status'] == 'ok':
                    return True
            except:
                pass
            time.sleep(1)
        return False

# Example usage
def main():
    client = GluRPCClient("http://localhost:8000", api_key="your-key")
    
    # Wait for service
    if not client.wait_for_service():
        print("Service not ready!")
        return
    
    # Convert proprietary format
    unified_csv = client.convert_csv("Clarity_Export.csv")
    
    # Process and cache
    handle, num_samples, warnings = client.process_csv(unified_csv)
    print(f"Processed {num_samples} samples. Handle: {handle}")
    
    if warnings['has_warnings']:
        print("Warnings:", warnings['messages'])
    
    # Get plot for most recent sample
    plot_data = client.get_plot(handle, index=0)
    fig = go.Figure(plot_data)
    fig.show()
    
    # Get plots for multiple samples
    for i in range(-5, 1):  # Last 6 samples
        plot_data = client.get_plot(handle, index=i)
        fig = go.Figure(plot_data)
        fig.write_html(f"plot_sample_{i}.html")
    
    # Or use quick plot for one-off analysis
    plot_data, warnings = client.quick_plot(unified_csv)
    fig = go.Figure(plot_data)
    fig.write_html("quick_analysis.html")

if __name__ == "__main__":
    main()
```

### JavaScript/TypeScript Example

```typescript
class GluRPCClient {
  constructor(
    private baseUrl: string,
    private apiKey?: string
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async request(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<any> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  async processCSV(csvContent: string, force = false): Promise<{
    handle: string;
    totalSamples: number;
    warnings: any;
  }> {
    const csvBase64 = btoa(csvContent);
    const data = await this.request('/process_unified', {
      method: 'POST',
      body: JSON.stringify({
        csv_base64: csvBase64,
        force_calculate: force,
      }),
    });

    if (data.error) {
      throw new Error(data.error);
    }

    return {
      handle: data.handle,
      totalSamples: data.total_samples,
      warnings: data.warnings,
    };
  }

  async getPlot(
    handle: string,
    index = 0,
    force = false
  ): Promise<any> {
    return this.request('/draw_a_plot', {
      method: 'POST',
      body: JSON.stringify({
        handle,
        index,
        force_calculate: force,
      }),
    });
  }

  async quickPlot(csvContent: string, force = false): Promise<{
    plotData: any;
    warnings: any;
  }> {
    const csvBase64 = btoa(csvContent);
    const data = await this.request('/quick_plot', {
      method: 'POST',
      body: JSON.stringify({
        csv_base64: csvBase64,
        force_calculate: force,
      }),
    });

    if (data.error) {
      throw new Error(data.error);
    }

    return {
      plotData: data.plot_data,
      warnings: data.warnings,
    };
  }

  async getHealth(): Promise<any> {
    return this.request('/health', { method: 'GET' });
  }
}

// Example usage
async function main() {
  const client = new GluRPCClient('http://localhost:8000', 'your-api-key');

  // Process CSV
  const csvContent = '...'; // Load your CSV
  const { handle, totalSamples, warnings } = await client.processCSV(csvContent);
  
  console.log(`Processed ${totalSamples} samples`);
  
  if (warnings.has_warnings) {
    console.warn('Warnings:', warnings.messages);
  }

  // Get plot
  const plotData = await client.getPlot(handle, 0);
  
  // Use with Plotly.js
  Plotly.newPlot('plot-div', plotData.data, plotData.layout);
}
```

### React Component Example

```tsx
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

interface GlucosePlotProps {
  baseUrl: string;
  apiKey?: string;
  csvContent: string;
}

export const GlucosePlot: React.FC<GlucosePlotProps> = ({
  baseUrl,
  apiKey,
  csvContent,
}) => {
  const [plotData, setPlotData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  useEffect(() => {
    const fetchPlot = async () => {
      try {
        setLoading(true);
        setError(null);

        const csvBase64 = btoa(csvContent);
        const headers: HeadersInit = {
          'Content-Type': 'application/json',
        };
        
        if (apiKey) {
          headers['X-API-Key'] = apiKey;
        }

        const response = await fetch(`${baseUrl}/quick_plot`, {
          method: 'POST',
          headers,
          body: JSON.stringify({ csv_base64: csvBase64 }),
        });

        const data = await response.json();

        if (data.error) {
          setError(data.error);
        } else {
          setPlotData(data.plot_data);
          if (data.warnings.has_warnings) {
            setWarnings(data.warnings.messages);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchPlot();
  }, [baseUrl, apiKey, csvContent]);

  if (loading) return <div>Loading plot...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!plotData) return null;

  return (
    <div>
      {warnings.length > 0 && (
        <div className="warnings">
          <h4>Data Warnings:</h4>
          <ul>
            {warnings.map((msg, i) => (
              <li key={i}>{msg}</li>
            ))}
          </ul>
        </div>
      )}
      <Plot
        data={plotData.data}
        layout={plotData.layout}
        config={{ responsive: true }}
      />
    </div>
  );
};
```

---

## Best Practices

### 1. Caching Strategy

**Use handles for repeated access:**
```python
# ✅ Good: Process once, plot multiple times
handle, num_samples, _ = client.process_csv(csv_content)
for i in range(-10, 1):
    plot = client.get_plot(handle, i)
    # Use plot...

# ❌ Bad: Re-processing for each plot
for i in range(-10, 1):
    plot, _ = client.quick_plot(csv_content)
```

**When to use force_calculate:**
- Testing after code changes
- After model updates
- Debugging cache issues
- When data format changes
- NOT for normal operation

### 2. Error Handling

**Always check for errors in response body:**
```python
response = requests.post(url, json=payload)
response.raise_for_status()  # Check HTTP status
data = response.json()

if data.get('error'):  # Check application error
    handle_error(data['error'])
```

**Handle overload gracefully:**
```python
from time import sleep

def request_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, json=payload)
        
        if response.status_code == 503:
            retry_after = int(response.headers.get('Retry-After', 30))
            print(f"Service overloaded, waiting {retry_after}s...")
            sleep(retry_after)
            continue
        
        response.raise_for_status()
        return response.json()
    
    raise Exception("Max retries exceeded")
```

### 3. Health Monitoring

**Poll health during long operations:**
```python
import asyncio

async def monitor_health(client, stop_event):
    while not stop_event.is_set():
        health = client.get_health()
        if health['status'] != 'ok':
            print(f"Warning: Service status is {health['status']}")
        await asyncio.sleep(5)

async def main():
    stop = asyncio.Event()
    health_task = asyncio.create_task(monitor_health(client, stop))
    
    # Do work...
    handle, _, _ = client.process_csv(csv)
    
    stop.set()
    await health_task
```

**Check load before submitting:**
```python
def safe_submit(client, csv_content):
    health = client.get_health()
    
    if health['load_status'] in ['overloaded', 'full']:
        print("Service too busy, waiting...")
        time.sleep(30)
        return safe_submit(client, csv_content)  # Retry
    
    return client.process_csv(csv_content)
```

### 4. Warnings

**Always check and display warnings:**
```python
handle, num_samples, warnings = client.process_csv(csv_content)

if warnings['has_warnings']:
    print("⚠️  Data Quality Warnings:")
    for msg in warnings['messages']:
        print(f"  - {msg}")
    
    # Specific checks
    if warnings['too_short']:
        print("❌ Dataset too short for reliable predictions!")
    if warnings['quality']:
        print("⚠️  Low quality data detected - predictions may be less accurate")
```

### 5. Performance

**Use async for multiple plots:**
```python
import asyncio
import aiohttp

async def fetch_plots(client, handle, indices):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_plot(session, client.base_url, handle, i)
            for i in indices
        ]
        return await asyncio.gather(*tasks)

async def fetch_plot(session, base_url, handle, index):
    async with session.post(
        f"{base_url}/draw_a_plot",
        json={"handle": handle, "index": index}
    ) as response:
        return await response.json()

# Fetch 10 plots concurrently
plots = asyncio.run(fetch_plots(client, handle, range(-9, 1)))
```

**Cancel requests when no longer needed:**
```python
import signal

# JavaScript
const controller = new AbortController();
fetch(url, { signal: controller.signal });
// Later: controller.abort();

# Python with requests doesn't support cancellation well
# Use async with aiohttp for cancellation support
```

### 6. Data Format

**Ensure CSV is properly formatted:**
```python
import pandas as pd

def validate_unified_csv(csv_content: str) -> bool:
    """Validate CSV matches unified format."""
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Check required columns
    required = ['sequence_id', 'timestamp', 'glucose']
    if not all(col in df.columns for col in required):
        return False
    
    # Check data types
    if not pd.api.types.is_integer_dtype(df['sequence_id']):
        return False
    
    # Check timestamp format
    try:
        pd.to_datetime(df['timestamp'])
    except:
        return False
    
    # Check glucose values
    if not pd.api.types.is_numeric_dtype(df['glucose']):
        return False
    
    return True
```

### 7. Timeout Configuration

**Set appropriate timeouts:**
```python
# Short timeout for health checks
health = requests.get(url + "/health", timeout=5)

# Medium timeout for processing
process = requests.post(
    url + "/process_unified",
    json=payload,
    timeout=60  # 1 minute
)

# Long timeout for plots (may need inference)
plot = requests.post(
    url + "/draw_a_plot",
    json=payload,
    timeout=120  # 2 minutes
)
```

### 8. Logging

**Log important events:**
```python
import logging

logger = logging.getLogger(__name__)

def process_with_logging(client, csv_content):
    logger.info("Starting CSV processing...")
    
    try:
        handle, num_samples, warnings = client.process_csv(csv_content)
        logger.info(f"Processed {num_samples} samples. Handle: {handle}")
        
        if warnings['has_warnings']:
            logger.warning(f"Data warnings: {warnings['messages']}")
        
        return handle
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

---

## Appendix: Testing

The integration tests demonstrate realistic usage patterns. Key test scenarios:

1. **Basic flow:** Convert → Process → Plot
2. **Cache behavior:** Test with and without force_calculate
3. **Concurrent requests:** Multiple plot requests with cancellation
4. **Health polling:** Monitor service during operations
5. **Edge cases:** Invalid indices, missing handles, overload conditions

See `tests/test_integration.py` for complete examples.

---

**Last Updated:** 2025-12-12  
**API Version:** 1.0  
**Server:** GluRPC Glucose Prediction Service
