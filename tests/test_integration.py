import os
import base64
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import plotly.graph_objects as go

# Ensure directories exist for tests
os.makedirs("logs", exist_ok=True)
os.makedirs("files", exist_ok=True)
os.makedirs("test_outputs", exist_ok=True)

from glurpc.app import app

# Locate the data file
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "Clarity_Export__Patient_2025-05-14_154517.csv"
import logging

# Switch locks logger to DEBUG mode for this test
locks_logger = logging.getLogger("glurpc.locks")
original_level = locks_logger.level
locks_logger.setLevel(logging.DEBUG)

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

def test_a1_process_unified_flow_cached_before(client):
    """Test with cache reuse BEFORE forced calc (runs first to capture old cache, 'a1_' prefix)"""
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    # 1. Process Unified (will use cache if available)
    payload = {"csv_base64": csv_base64}
    response = client.post("/process_unified", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Process unified failed: {data['error']}")
        
    assert "handle" in data
    assert data["handle"] is not None
    handle = data["handle"]
    
    # 2. Draw Plot (will use plot cache if available - might have OLD buggy plots)
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
    
    # Save the plot as HTML and SVG (from cache - potentially old/buggy)
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_process_unified_flow_cached_before.html")
    fig.write_image(output_dir / "test_process_unified_flow_cached_before.svg")
    print(f"Saved cached (before) plot to {output_dir / 'test_process_unified_flow_cached_before.html'} and .svg")

def test_a1_quick_plot_cached_before(client):
    """Test with cache reuse BEFORE forced calc (runs first to capture old cache, 'a1_' prefix)"""
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
    
    # Save the plot as HTML and SVG (from cache - potentially old/buggy)
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_quick_plot_cached_before.html")
    fig.write_image(output_dir / "test_quick_plot_cached_before.svg")
    print(f"Saved cached (before) plot to {output_dir / 'test_quick_plot_cached_before.html'} and .svg")

def test_a2_process_unified_flow_forced(client):
    """Test with forced calculation (runs second due to 'a2_' prefix)"""
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    # 1. Process Unified with force_calculate=True
    payload = {"csv_base64": csv_base64, "force_calculate": True}
    response = client.post("/process_unified", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Process unified failed: {data['error']}")
        
    assert "handle" in data
    assert data["handle"] is not None
    handle = data["handle"]
    
    # 2. Draw Plot with force_calculate=True to bypass plot cache
    plot_payload = {"handle": handle, "index": -10, "force_calculate": True}
    
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
    
    # Save the plot as HTML and SVG
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_process_unified_flow_forced.html")
    fig.write_image(output_dir / "test_process_unified_flow_forced.svg")
    print(f"Saved forced calc plot to {output_dir / 'test_process_unified_flow_forced.html'} and .svg")

def test_a2_quick_plot_forced(client):
    """Test with forced calculation (runs second due to 'a2_' prefix)"""
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    payload = {"csv_base64": csv_base64, "force_calculate": True}
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
    
    # Save the plot as HTML and SVG
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_quick_plot_forced.html")
    fig.write_image(output_dir / "test_quick_plot_forced.svg")
    print(f"Saved forced calc plot to {output_dir / 'test_quick_plot_forced.html'} and .svg")

def test_a3_process_unified_flow_cached_after(client):
    """Test with cache reuse AFTER forced calc (runs third to verify new cache, 'a3_' prefix)"""
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    # 1. Process Unified (will use cache)
    payload = {"csv_base64": csv_base64}
    response = client.post("/process_unified", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("error"):
        pytest.fail(f"Process unified failed: {data['error']}")
        
    assert "handle" in data
    assert data["handle"] is not None
    handle = data["handle"]
    
    # 2. Draw Plot (will use plot cache)
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
    
    # Save the plot as HTML and SVG (from NEW cache after forced calc)
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_process_unified_flow_cached_after.html")
    fig.write_image(output_dir / "test_process_unified_flow_cached_after.svg")
    print(f"Saved cached (after) plot to {output_dir / 'test_process_unified_flow_cached_after.html'} and .svg")

def test_a3_quick_plot_cached_after(client):
    """Test with cache reuse AFTER forced calc (runs third to verify new cache, 'a3_' prefix)"""
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
    
    # Save the plot as HTML and SVG (from NEW cache after forced calc)
    fig = go.Figure(plot_dict)
    output_dir = Path("test_outputs")
    fig.write_html(output_dir / "test_quick_plot_cached_after.html")
    fig.write_image(output_dir / "test_quick_plot_cached_after.svg")
    print(f"Saved cached (after) plot to {output_dir / 'test_quick_plot_cached_after.html'} and .svg")

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
    
    # Validate total HTTP requests processed is at least 5
    # (convert_to_unified, process_unified, draw_a_plot, quick_plot, health)
    total_requests = data.get("total_http_requests", 0)
    assert total_requests >= 5, f"Expected total_http_requests >= 5, got {total_requests}"
    
    # Validate HTTP errors - allow for up to 4 expected 400s from index fallback logic
    # (We now have 2 cached runs + 2 forced runs, each may try index -10 and fallback to 0)
    total_http_errors = data.get("total_http_errors", 0)
    assert total_http_errors <= 4, f"Expected total_http_errors <= 4 (allowing for index fallback in all tests), got {total_http_errors}"
    
    # Validate no inference errors
    total_inference_errors = data.get("total_inference_errors", 0)
    assert total_inference_errors == 0, f"Expected total_inference_errors == 0, got {total_inference_errors}"
    
    # Validate request time statistics are present
    assert "avg_request_time_ms" in data, "avg_request_time_ms not in health response"
    assert "median_request_time_ms" in data, "median_request_time_ms not in health response"
    assert "min_request_time_ms" in data, "min_request_time_ms not in health response"
    assert "max_request_time_ms" in data, "max_request_time_ms not in health response"
    
    # Validate request time values are reasonable
    avg_time = data.get("avg_request_time_ms", 0)
    median_time = data.get("median_request_time_ms", 0)
    min_time = data.get("min_request_time_ms", 0)
    max_time = data.get("max_request_time_ms", 0)
    
    assert avg_time >= 0, f"avg_request_time_ms should be >= 0, got {avg_time}"
    assert median_time >= 0, f"median_request_time_ms should be >= 0, got {median_time}"
    assert min_time >= 0, f"min_request_time_ms should be >= 0, got {min_time}"
    assert max_time >= min_time, f"max_request_time_ms ({max_time}) should be >= min_request_time_ms ({min_time})"
    
    # If requests were made, times should be positive
    if total_requests > 0:
        assert avg_time > 0, f"avg_request_time_ms should be > 0 when requests were made, got {avg_time}"
        assert max_time > 0, f"max_request_time_ms should be > 0 when requests were made, got {max_time}"
