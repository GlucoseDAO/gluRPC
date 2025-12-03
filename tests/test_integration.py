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

# Locate the data file
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "Clarity_Export__Patient_2025-05-14_154517.csv"

def get_csv_content():
    if not DATA_FILE_PATH.exists():
        pytest.skip(f"Data file not found at {DATA_FILE_PATH}")
    return DATA_FILE_PATH.read_bytes()

@pytest.fixture(scope="module")
def client():
    # Use context manager to trigger startup/shutdown events
    with TestClient(app) as c:
        yield c

def test_convert_to_unified(client):
    csv_content = get_csv_content()
    
    # Multipart upload
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/convert_to_unified", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "csv_content" in data
    assert data["error"] is None
    assert "sequence_id" in data["csv_content"] and "glucose" in data["csv_content"]

def test_process_unified_flow(client):
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
    
    # 2. Draw Plot (now returns Plotly JSON dict)
    plot_payload = {"handle": handle, "index": -10}
    
    plot_response = client.post("/draw_a_plot", json=plot_payload)
    
    if plot_response.status_code != 200:
        plot_payload["index"] = 0
        plot_response = client.post("/draw_a_plot", json=plot_payload)
        
    assert plot_response.status_code == 200
    assert plot_response.headers["content-type"] == "application/json"
    
    plot_dict = plot_response.json()
    assert isinstance(plot_dict, dict)
    assert "data" in plot_dict  # Plotly figure has 'data' and 'layout' keys
    assert "layout" in plot_dict
    assert len(plot_dict["data"]) > 0  # Should have traces

def test_quick_plot(client):
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    payload = {"csv_base64": csv_base64}
    response = client.post("/quick_plot", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Quick plot failed: {data['error']}")
        
    assert "plot_data" in data
    assert isinstance(data["plot_data"], dict)
    
    plot_dict = data["plot_data"]
    assert "data" in plot_dict  # Plotly figure has 'data' and 'layout' keys
    assert "layout" in plot_dict
    assert len(plot_dict["data"]) > 0  # Should have traces

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_z_cache_state(client):
    """
    Verify cache state after all tests have run.
    Named with 'z_' prefix to ensure it runs last.
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    
    # Validate cache size is at least 1
    cache_size = data.get("cache_size", 0)
    assert cache_size >= 1, f"Expected cache_size >= 1, got {cache_size}"
    
    # Validate total requests processed is at least 5
    # (convert_to_unified, process_unified, draw_a_plot, quick_plot, health)
    total_requests = data.get("total_requests_processed", 0)
    assert total_requests >= 5, f"Expected total_requests_processed >= 5, got {total_requests}"
    
    # Validate no errors occurred
    total_errors = data.get("total_errors", 0)
    assert total_errors == 0, f"Expected total_errors == 0, got {total_errors}"
