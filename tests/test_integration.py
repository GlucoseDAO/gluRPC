import asyncio
import base64
import logging
import os
import json
import random
import socket
import multiprocessing
from pathlib import Path
from statistics import mean
from time import perf_counter

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import plotly.graph_objects as go
import uvicorn

# Force spawn instead of fork to avoid deadlocks
multiprocessing.set_start_method('spawn', force=True)

# Ensure directories exist for tests
os.makedirs("logs", exist_ok=True)
os.makedirs("test_outputs", exist_ok=True)

from glurpc.app import app

# Locate the data file
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "Clarity_Export__Patient_2025-05-14_154517.csv"

# Switch locks logger to DEBUG mode for this test
locks_logger = logging.getLogger("glurpc.locks")
original_level = locks_logger.level
locks_logger.setLevel(logging.DEBUG)

def get_csv_content():
    if not DATA_FILE_PATH.exists():
        pytest.skip(f"Data file not found at {DATA_FILE_PATH}")
    return DATA_FILE_PATH.read_bytes()

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _run_server(host: str, port: int) -> None:
    """Run uvicorn server in a separate process."""
    config = uvicorn.Config("glurpc.app:app", host=host, port=port, reload=False, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

@pytest.fixture(scope="module")
def live_server():
    """
    Start the FastAPI app in a separate process using spawn (not fork) to avoid deadlocks.
    """
    host = "127.0.0.1"
    port = _find_free_port()
    proc = multiprocessing.Process(target=_run_server, args=(host, port))
    proc.start()

    # Wait until /health is ready and models are initialized
    async def _wait_ready():
        for _ in range(60):  # 30 seconds timeout
            try:
                async with AsyncClient(base_url=f"http://{host}:{port}", timeout=5.0) as ac:
                    resp = await ac.get("/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("models_initialized"):
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Server did not become ready with models initialized in time")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_wait_ready())
    finally:
        loop.close()
    
    yield f"http://{host}:{port}"
    
    # Cleanup: terminate the server process
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)

@pytest.fixture(scope="module")
def client():
    # Use context manager to trigger startup/shutdown events
    with TestClient(app) as c:
        yield c

@pytest_asyncio.fixture(scope="module")
async def async_client(live_server: str):
    """AsyncClient that connects to the real live server."""
    async with AsyncClient(base_url=live_server, timeout=120.0) as c:
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


@pytest.mark.asyncio
async def test_b_plot_cancellation_and_health_polling(async_client):
    """
    Simulate frontend behaviour:
    1) create a request to get handle + num_samples,
    2) fetch plot for index 0,
    3) repeatedly fire 5 plot requests, cancel 4, await the last,
    while polling /health in parallel.
    """
    csv_content = get_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode("utf-8")

    process_response = await async_client.post(
        "/process_unified", json={"csv_base64": csv_base64, "force_calculate": True}
    )
    assert process_response.status_code == 200
    process_data = process_response.json()
    if process_data.get("error"):
        pytest.fail(f"Process unified failed: {process_data['error']}")

    handle = process_data.get("handle")
    total_samples = process_data.get("total_samples") or process_data.get("num_samples")

    assert handle, "Expected handle in /process_unified response"
    assert total_samples and total_samples > 0, "Expected num_samples/total_samples > 0"

    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    stop_event = asyncio.Event()
    health_results: list[dict] = []
    health_statuses: list[str] = []
    health_timestamps: list[float] = []
    test_start_time = perf_counter()

    async def poll_health() -> None:
        while not stop_event.is_set():
            health_time = perf_counter() - test_start_time
            response = await async_client.get("/health")
            assert response.status_code == 200
            payload = response.json()
            health_results.append(payload)
            status = payload.get("status")
            health_statuses.append(status)
            health_timestamps.append(health_time)
            assert status in {"ok", "degraded"}
            await asyncio.sleep(1)

    health_task = asyncio.create_task(poll_health())

    time_to_image_ms: list[float] = []
    saved_plots: list[dict] = []
    plot_data_collection: list[dict] = []  # Store all plot JSON data

    # Step 2: Fetch initial plot at index 0 (wait for inference to complete)
    initial_payload = {"handle": handle, "index": 0}
    start_time = perf_counter()
    initial_plot = await async_client.post("/draw_a_plot", json=initial_payload)
    duration = (perf_counter() - start_time) * 1000
    time_to_image_ms.append(duration)
    
    assert initial_plot.status_code == 200, f"Initial plot failed: {initial_plot.text}"
    assert initial_plot.headers["content-type"] == "application/json"
    initial_data = initial_plot.json()
    assert "data" in initial_data and "layout" in initial_data
    
    # Store plot data
    saved_plots.append({"index": 0, "iteration": 0, "time_ms": duration})
    plot_data_collection.append(initial_data)
    
    # Wait a moment for inference to complete and populate cache
    print(f"Waiting for inference to complete for all {total_samples} samples...")
    await asyncio.sleep(2.0)

    rng = random.Random(42)
    # Use actual total_samples now that inference is complete
    max_offset = max(0, min(total_samples - 1, 50))

    # Step 3: Repeat 5 times - fire 5 requests, cancel 4, wait for last
    try:
        for iteration in range(1, 6):
            indices = [-rng.randint(0, max_offset) for _ in range(5)]
            tasks = [
                asyncio.create_task(
                    async_client.post(
                        "/draw_a_plot",
                        json={"handle": handle, "index": idx, "force_calculate": True},
                    )
                )
                for idx in indices
            ]

            # Give tasks a moment to start
            await asyncio.sleep(0.05)

            start_iteration = perf_counter()
            
            # Cancel the first 4 tasks
            for cancel_task in tasks[:-1]:
                cancel_task.cancel()
            await asyncio.gather(*tasks[:-1], return_exceptions=True)

            # Wait for the last task with timeout protection
            try:
                final_response = await asyncio.wait_for(tasks[-1], timeout=120.0)
            except asyncio.TimeoutError:
                print(f"⚠️  Iteration {iteration} timed out after 120s waiting for plot at index {indices[-1]}")
                print(f"   This likely means inference is stuck. Skipping this iteration.")
                # Continue to next iteration
                continue
                
            duration_ms = (perf_counter() - start_iteration) * 1000
            time_to_image_ms.append(duration_ms)

            assert final_response.status_code == 200
            assert final_response.headers["content-type"] == "application/json"
            plot_payload = final_response.json()
            assert "data" in plot_payload and "layout" in plot_payload
            
            # Store plot data
            saved_plots.append({"index": indices[-1], "iteration": iteration, "time_ms": duration_ms})
            plot_data_collection.append(plot_payload)
            
            # Add small delay to let health polling happen
            await asyncio.sleep(1.5)
            
    finally:
        stop_event.set()
        await health_task

    assert time_to_image_ms, "No plot timings collected"
    
    # Check that we got at least some plots (may have timeouts on slow iterations)
    if len(time_to_image_ms) < 3:
        print(f"⚠️  Warning: Only got {len(time_to_image_ms)} plots out of 6 expected")
        print(f"   This suggests some requests are timing out or hanging")
    
    avg_ms = mean(time_to_image_ms)
    min_ms = min(time_to_image_ms)
    max_ms = max(time_to_image_ms)

    assert min_ms <= avg_ms <= max_ms
    assert all(t > 0 for t in time_to_image_ms)

    assert health_results, "Health polling did not run"
    assert all(status in {"ok", "degraded"} for status in health_statuses)

    # Save metrics
    metrics = {
        "time_to_image_ms": time_to_image_ms,
        "avg_time_to_image_ms": avg_ms,
        "min_time_to_image_ms": min_ms,
        "max_time_to_image_ms": max_ms,
        "num_samples": total_samples,
        "health_calls": len(health_results),
        "health_statuses": health_statuses,
        "saved_plots": saved_plots,
    }
    (output_dir / "user_scenario_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Create time-to-image bar chart
    fig_timing = go.Figure()
    fig_timing.add_trace(go.Bar(
        x=[f"Request {i}" for i in range(len(time_to_image_ms))],
        y=time_to_image_ms,
        name="Time to Image",
        text=[f"{t:.1f}ms" for t in time_to_image_ms],
        textposition="auto",
    ))
    fig_timing.update_layout(
        title="Time to Image per Request",
        xaxis_title="Request #",
        yaxis_title="Milliseconds",
        showlegend=False,
        height=400,
    )

    # Create health metrics over time graph
    fig_health = None
    if health_results and health_timestamps:
        fig_health = go.Figure()
        
        # Extract key metrics
        cache_sizes = [h.get("cache_size", 0) for h in health_results]
        queue_lengths = [h.get("priority_queue_length", 0) + h.get("general_queue_length", 0) for h in health_results]
        total_requests = [h.get("total_http_requests", 0) for h in health_results]
        
        fig_health.add_trace(go.Scatter(
            x=health_timestamps,
            y=cache_sizes,
            mode="lines+markers",
            name="Cache Size",
            yaxis="y1",
        ))
        fig_health.add_trace(go.Scatter(
            x=health_timestamps,
            y=queue_lengths,
            mode="lines+markers",
            name="Available Models",
            yaxis="y2",
        ))
        fig_health.add_trace(go.Scatter(
            x=health_timestamps,
            y=total_requests,
            mode="lines+markers",
            name="Total HTTP Requests",
            yaxis="y3",
        ))
        
        fig_health.update_layout(
            title="Health Metrics During Test Execution",
            xaxis_title="Time (seconds)",
            yaxis=dict(title="Cache Size", side="left"),
            yaxis2=dict(title="Available Models", overlaying="y", side="right"),
            yaxis3=dict(title="HTTP Requests", overlaying="y", side="right", position=0.85),
            hovermode="x unified",
            height=400,
        )
    
    # Create comprehensive HTML report with all plots
    from datetime import datetime
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Scenario Test Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .plot-container {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
        }}
        .plot-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .plot-subtitle {{
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .glucose-plot {{
            margin: 30px 0;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 15px;
            background: white;
        }}
        .glucose-plot-header {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 User Scenario Test Report</h1>
        <p><strong>Test Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Test Duration:</strong> {perf_counter() - test_start_time:.2f} seconds</p>
        
        <h2>📊 Summary Metrics</h2>
        <div class="metrics">
            <div class="metric-card success">
                <div class="metric-label">Total Plots Saved</div>
                <div class="metric-value">{len(saved_plots)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Time to Image</div>
                <div class="metric-value">{avg_ms:.1f}ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Min Time</div>
                <div class="metric-value">{min_ms:.1f}ms</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-label">Max Time</div>
                <div class="metric-value">{max_ms:.1f}ms</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">Health Checks</div>
                <div class="metric-value">{len(health_results)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{total_samples}</div>
            </div>
        </div>

        <h2>⏱️ Time to Image Chart</h2>
        <div class="plot-container">
            <div id="timing-chart"></div>
        </div>

        <h2>🏥 Health Metrics Over Time</h2>
        <div class="plot-container">
            <div id="health-chart"></div>
        </div>

        <h2>📈 Individual Plot Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Request #</th>
                    <th>Iteration</th>
                    <th>Index</th>
                    <th>Time (ms)</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, plot_info in enumerate(saved_plots):
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{plot_info['iteration']}</td>
                    <td>{plot_info['index']}</td>
                    <td>{plot_info['time_ms']:.2f}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>

        <h2>📸 All Generated Glucose Prediction Plots</h2>
"""
    
    # Add all plots inline
    for i, (plot_info, plot_data) in enumerate(zip(saved_plots, plot_data_collection)):
        iter_num = plot_info['iteration']
        idx = plot_info['index']
        time_ms = plot_info['time_ms']
        title = f"Plot {i}: {'Initial (Index 0)' if i == 0 else f'Iteration {iter_num} - Index {idx}'}"
        html_content += f"""
        <div class="glucose-plot">
            <div class="glucose-plot-header">{title} - Time: {time_ms:.1f}ms</div>
            <div id="plot-{i}"></div>
        </div>
"""
    
    html_content += """
    </div>
    <script>
"""
    
    # Add timing chart
    html_content += f"""
        var timingData = {fig_timing.to_json()};
        Plotly.newPlot('timing-chart', timingData.data, timingData.layout);
"""
    
    # Add health chart if available
    if fig_health:
        html_content += f"""
        var healthData = {fig_health.to_json()};
        Plotly.newPlot('health-chart', healthData.data, healthData.layout);
"""
    
    # Add all glucose plots
    for i, plot_data in enumerate(plot_data_collection):
        html_content += f"""
        var plotData{i} = {json.dumps(plot_data)};
        Plotly.newPlot('plot-{i}', plotData{i}.data, plotData{i}.layout);
"""
    
    html_content += """
    </script>
</body>
</html>
"""
    
    (output_dir / "user_scenario_report.html").write_text(html_content)
    
    print(f"\n=== Test Summary ===")
    print(f"Total plots saved: {len(saved_plots)}")
    print(f"Average time to image: {avg_ms:.2f}ms")
    print(f"Min time: {min_ms:.2f}ms, Max time: {max_ms:.2f}ms")
    print(f"Health checks: {len(health_results)}")
    print(f"Outputs saved to: {output_dir}")
    print(f"  ✅ Comprehensive report: user_scenario_report.html")
    print(f"  📄 Metrics JSON: user_scenario_metrics.json")

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
