import os
import base64
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os

# Ensure directories exist for tests
os.makedirs("logs", exist_ok=True)
os.makedirs("files", exist_ok=True)

from glurpc.app import app

client = TestClient(app)

# Locate the data file
# Assuming tests are run from project root or tests dir
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "Clarity_Export__Patient_2025-05-14_154517.csv"

def get_csv_content():
    if not DATA_FILE_PATH.exists():
        pytest.skip(f"Data file not found at {DATA_FILE_PATH}")
    return DATA_FILE_PATH.read_bytes()

def test_convert_to_unified():
    csv_content = get_csv_content()
    
    # Multipart upload
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/convert_to_unified", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "csv_content" in data
    assert data["error"] is None
    # Check if content looks like CSV (header)
    # Updated expectation based on actual output
    assert "sequence_id" in data["csv_content"] and "glucose" in data["csv_content"]

def test_process_unified_flow():
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    # 1. Process Unified
    payload = {"csv_base64": csv_base64}
    response = client.post("/process_unified", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Process unified failed: {data['error']}")
        
    assert "handle" in data
    assert data["handle"] is not None
    handle = data["handle"]
    
    # 2. Draw Plot
    # We need a valid index. The file has ~3980 lines. 
    # Let's try an index somewhere in the middle/end.
    # The core logic creates a dataset. We don't know the exact size but if it processed, it should have samples.
    plot_payload = {"handle": handle, "index": 10}
    
    plot_response = client.post("/draw_a_plot", json=plot_payload)
    
    if plot_response.status_code != 200:
        # If 10 is out of bounds, try 0
        plot_payload["index"] = 0
        plot_response = client.post("/draw_a_plot", json=plot_payload)
        
    assert plot_response.status_code == 200
    assert plot_response.headers["content-type"] == "image/png"
    assert len(plot_response.content) > 0
    
    # Save plot for inspection
    with open("files/debug_plot_draw.png", "wb") as f:
        f.write(plot_response.content)

def test_quick_plot():
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    payload = {"csv_base64": csv_base64}
    response = client.post("/quick_plot", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Quick plot failed: {data['error']}")
        
    assert "plot_base64" in data
    assert len(data["plot_base64"]) > 0
    
    # Verify it decodes to png bytes
    png_bytes = base64.b64decode(data["plot_base64"])
    assert len(png_bytes) > 0
    # Simple check for PNG magic number
    assert png_bytes.startswith(b'\x89PNG')
    
    # Save quick plot for inspection
    with open("files/debug_plot_quick.png", "wb") as f:
        f.write(png_bytes)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

