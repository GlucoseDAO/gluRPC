"""
Combined integration tests for both REST and gRPC endpoints.
Starts the combined service once and tests both access methods.
All tests are async for consistency.
"""
import asyncio
import base64
import json
import logging
import socket
import time
import multiprocessing
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
import grpc
from grpc import aio
from httpx import AsyncClient
import plotly.graph_objects as go

from service.service_spec import glurpc_pb2, glurpc_pb2_grpc
from service import registry


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_combined")

# Ensure output directory exists
Path("test_outputs").mkdir(exist_ok=True)

# Run all async tests in this module on the same event loop
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _find_free_port() -> int:
    """Find a free port to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_combined_server(grpc_port: int, rest_port: int, log_file: str) -> None:
    """Run combined gRPC+REST server in a separate process."""
    import logging
    
    # Setup logging to file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logger.info(f"Starting combined server: gRPC on {grpc_port}, REST on {rest_port}")
    
    try:
        from service.combined_service import main_combined
        asyncio.run(main_combined(grpc_port=grpc_port, rest_port=rest_port))
    except KeyboardInterrupt:
        logger.info("Server interrupted, shutting down")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        raise


@pytest.fixture(scope="module")
def combined_server():
    """
    Start the combined gRPC+REST server once for the whole module.
    Uses 'spawn' method to avoid fork() deadlocks.
    """
    grpc_port = _find_free_port()
    rest_port = _find_free_port()
    log_file = Path(__file__).parent.parent / "test_outputs" / "test_combined_server.log"
    log_file.parent.mkdir(exist_ok=True)
    
    print(f"Starting combined server: gRPC on {grpc_port}, REST on {rest_port}")
    print(f"Logging to {log_file}")
    
    # Use spawn to avoid fork() deadlocks
    ctx = multiprocessing.get_context('spawn')
    proc = ctx.Process(target=_run_combined_server, args=(grpc_port, rest_port, str(log_file)), daemon=True)
    proc.start()
    
    # Wait for both servers to be ready
    max_wait_time = 120  # 2 minutes for model loading
    
    # Wait for REST server
    rest_ready = False
    for attempt in range(max_wait_time):
        try:
            import httpx
            with httpx.Client(base_url=f"http://127.0.0.1:{rest_port}", timeout=1.0) as client:
                resp = client.get("/health")
                if resp.status_code == 200:
                    rest_ready = True
                    print(f"REST server ready after {attempt + 1} seconds")
                    break
        except Exception:
            pass
        
        if attempt % 10 == 0 and attempt > 0:
            print(f"Waiting for REST server... {attempt + 1}s elapsed")
        time.sleep(1)
    
    # Wait for gRPC server
    grpc_ready = False
    if rest_ready:
        channel = grpc.insecure_channel(f'localhost:{grpc_port}')
        for attempt in range(20):  # REST is ready, gRPC should be quick
            try:
                grpc.channel_ready_future(channel).result(timeout=1)
                grpc_ready = True
                print(f"gRPC server ready after {attempt + 1} additional seconds")
                break
            except grpc.FutureTimeoutError:
                pass
            time.sleep(1)
        channel.close()
    
    if not rest_ready or not grpc_ready:
        print(f"Server startup failed. REST: {rest_ready}, gRPC: {grpc_ready}")
        print(f"Check logs at: {log_file}")
        proc.terminate()
        proc.join(timeout=5)
        pytest.fail(f"Combined server did not become ready. REST: {rest_ready}, gRPC: {grpc_ready}. Check {log_file}")
    
    yield {"grpc_port": grpc_port, "rest_port": rest_port, "rest_base_url": f"http://127.0.0.1:{rest_port}"}
    
    proc.terminate()
    proc.join(timeout=5)


@pytest_asyncio.fixture(scope="module")
async def rest_client(combined_server):
    """AsyncClient for REST endpoints."""
    async with AsyncClient(base_url=combined_server["rest_base_url"], timeout=120.0) as client:
        yield client


@pytest_asyncio.fixture(scope="module")
async def grpc_channel(combined_server):
    """Async gRPC channel."""
    port = combined_server["grpc_port"]
    channel = aio.insecure_channel(f'localhost:{port}')
    
    # Wait for channel to be ready
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Async gRPC channel could not connect")
    
    yield channel
    await channel.close()


@pytest_asyncio.fixture(scope="module")
async def grpc_stub(grpc_channel):
    """Async gRPC stub."""
    return glurpc_pb2_grpc.GlucosePredictionStub(grpc_channel)


@pytest.fixture(scope="module")
def api_key():
    """Get API key from file or use test key."""
    api_key_file = Path(__file__).parent.parent / "api_keys_list"
    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            return f.readline().strip()
    return "test_key"


@pytest.fixture(scope="module")
def sample_csv_file():
    """Path to sample CSV file for testing."""
    data_dir = Path(__file__).parent.parent / "data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    
    pytest.skip("No sample CSV file found in data/ directory")


@pytest.fixture(scope="module")
def sample_csv_content(sample_csv_file):
    """Raw CSV content."""
    return Path(sample_csv_file).read_bytes()


# ============================================================================
# Health Check Tests (both REST and gRPC)
# ============================================================================

class TestHealthEndpoints:
    """Test health endpoints on both REST and gRPC."""
    
    async def test_rest_health(self, rest_client):
        """Test REST /health endpoint."""
        response = await rest_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert data["models_initialized"] is True
        assert data["cache_size"] >= 0
        assert data["device"] != ""
    
    async def test_grpc_health(self, grpc_stub):
        """Test gRPC CheckHealth endpoint."""
        request = glurpc_pb2.HealthRequest()
        response = await grpc_stub.CheckHealth(request)
        
        assert response.status in ["ok", "degraded"]
        assert response.models_initialized is True
        assert response.cache_size >= 0
        assert response.device != ""


# ============================================================================
# Conversion Tests (both REST and gRPC)
# ============================================================================

class TestConversionEndpoints:
    """Test conversion endpoints on both REST and gRPC."""
    
    async def test_rest_convert(self, rest_client, sample_csv_content):
        """Test REST /convert_to_unified endpoint."""
        files = {"file": ("test.csv", sample_csv_content, "text/csv")}
        response = await rest_client.post("/convert_to_unified", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None or data["error"] == ""
        assert len(data["csv_content"]) > 0
        assert "sequence_id" in data["csv_content"] or "glucose" in data["csv_content"]
    
    async def test_grpc_convert(self, grpc_stub, sample_csv_content):
        """Test gRPC ConvertToUnified endpoint."""
        request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv_content)
        response = await grpc_stub.ConvertToUnified(request)
        
        assert response.error == "" or response.error is None
        assert len(response.csv_content) > 0
        assert "sequence_id" in response.csv_content or "glucose" in response.csv_content


# ============================================================================
# Processing Tests (both REST and gRPC with shared cache)
# ============================================================================

@pytest_asyncio.fixture(scope="module")
async def unified_csv_content_rest(grpc_stub, sample_csv_content):
    """Convert sample file to unified format via gRPC."""
    request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv_content)
    response = await grpc_stub.ConvertToUnified(request)
    
    if response.error:
        pytest.fail(f"ConvertToUnified failed: {response.error}")
    
    return response.csv_content


@pytest.fixture(scope="module")
def dataset_handle_rest(combined_server, unified_csv_content_rest, api_key):
    """Process CSV via REST and get handle."""
    import httpx
    
    csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
    
    with httpx.Client(base_url=combined_server["rest_base_url"], timeout=120.0) as client:
        response = client.post(
            "/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        if data.get("error"):
            pytest.fail(f"ProcessUnified failed: {data['error']}")
        
        return data["handle"]


@pytest_asyncio.fixture(scope="module")
async def dataset_handle_grpc(grpc_stub, unified_csv_content_rest, api_key):
    """Process CSV via gRPC and get handle."""
    csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
    metadata = aio.Metadata(('x-api-key', api_key))
    
    request = glurpc_pb2.ProcessUnifiedRequest(
        csv_base64=csv_base64,
        force_calculate=False
    )
    
    response = await grpc_stub.ProcessUnified(request, metadata=metadata)
    
    if response.error:
        pytest.fail(f"ProcessUnified failed: {response.error}")
    
    return response.handle


class TestProcessingEndpoints:
    """Test processing endpoints on both REST and gRPC."""
    
    async def test_rest_process(self, rest_client, unified_csv_content_rest, api_key):
        """Test REST /process_unified endpoint."""
        csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
        response = await rest_client.post(
            "/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("error") is None or data.get("error") == ""
        assert len(data["handle"]) > 0
        assert data["total_samples"] > 0
    
    async def test_grpc_process(self, grpc_stub, unified_csv_content_rest, api_key):
        """Test gRPC ProcessUnified endpoint."""
        csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        response = await grpc_stub.ProcessUnified(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.handle) > 0
        assert response.total_samples > 0
    
    async def test_cache_shared_between_rest_and_grpc(
        self, rest_client, grpc_stub, unified_csv_content_rest, api_key
    ):
        """
        Test that cache is shared between REST and gRPC.
        Process via REST, then retrieve via gRPC (and vice versa).
        """
        csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
        
        # Process via REST
        rest_response = await rest_client.post(
            "/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        assert rest_response.status_code == 200
        rest_handle = rest_response.json()["handle"]
        
        # Process same data via gRPC (should hit cache)
        metadata = aio.Metadata(('x-api-key', api_key))
        grpc_request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        grpc_response = await grpc_stub.ProcessUnified(grpc_request, metadata=metadata)
        grpc_handle = grpc_response.handle
        
        # Handles should be identical (same data = same handle)
        assert rest_handle == grpc_handle, "Cache should be shared between REST and gRPC"


# ============================================================================
# Plotting Tests (both REST and gRPC)
# ============================================================================

class TestPlottingEndpoints:
    """Test plotting endpoints on both REST and gRPC."""
    
    async def test_rest_draw_plot(self, rest_client, dataset_handle_rest, api_key):
        """Test REST /draw_a_plot endpoint."""
        response = await rest_client.post(
            "/draw_a_plot",
            json={"handle": dataset_handle_rest, "index": 0, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        
        assert response.status_code == 200
        plot_dict = response.json()
        assert "data" in plot_dict
        assert "layout" in plot_dict
        assert len(plot_dict["data"]) > 0
        
        # Save plot
        fig = go.Figure(plot_dict)
        output_dir = Path("test_outputs")
        fig.write_html(output_dir / "test_combined_rest_plot.html")
        print(f"Saved REST plot to {output_dir / 'test_combined_rest_plot.html'}")
    
    async def test_grpc_draw_plot(self, grpc_stub, dataset_handle_grpc, api_key):
        """Test gRPC DrawPlot endpoint."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.PlotRequest(
            handle=dataset_handle_grpc,
            index=0,
            force_calculate=False
        )
        
        response = await grpc_stub.DrawPlot(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.plot_json) > 0
        
        plot_dict = json.loads(response.plot_json)
        assert "data" in plot_dict
        assert "layout" in plot_dict
        
        # Save plot
        fig = go.Figure(plot_dict)
        output_dir = Path("test_outputs")
        fig.write_html(output_dir / "test_combined_grpc_plot.html")
        print(f"Saved gRPC plot to {output_dir / 'test_combined_grpc_plot.html'}")
    
    async def test_rest_quick_plot(self, rest_client, unified_csv_content_rest, api_key):
        """Test REST /quick_plot endpoint."""
        csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
        response = await rest_client.post(
            "/quick_plot",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("error") is None or data.get("error") == ""
        assert "plot_data" in data
        
        plot_dict = data["plot_data"]
        assert "data" in plot_dict
        assert "layout" in plot_dict
        
        # Save plot
        fig = go.Figure(plot_dict)
        output_dir = Path("test_outputs")
        fig.write_html(output_dir / "test_combined_rest_quick_plot.html")
        print(f"Saved REST quick plot to {output_dir / 'test_combined_rest_quick_plot.html'}")
    
    async def test_grpc_quick_plot(self, grpc_stub, unified_csv_content_rest, api_key):
        """Test gRPC QuickPlot endpoint."""
        csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.QuickPlotRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        response = await grpc_stub.QuickPlot(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.plot_json) > 0
        
        plot_dict = json.loads(response.plot_json)
        assert "data" in plot_dict
        assert "layout" in plot_dict
        
        # Save plot
        fig = go.Figure(plot_dict)
        output_dir = Path("test_outputs")
        fig.write_html(output_dir / "test_combined_grpc_quick_plot.html")
        print(f"Saved gRPC quick plot to {output_dir / 'test_combined_grpc_quick_plot.html'}")


# ============================================================================
# Cache Management Tests (both REST and gRPC)
# ============================================================================

class TestCacheManagement:
    """Test cache management on both REST and gRPC."""
    
    async def test_rest_cache_info(self, rest_client, api_key):
        """Test REST /cache_management?action=info endpoint."""
        response = await rest_client.post(
            "/cache_management?action=info",
            headers={"x-api-key": api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cache_size"] >= 0
    
    async def test_grpc_cache_info(self, grpc_stub, api_key):
        """Test gRPC ManageCache endpoint."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.CacheManagementRequest(
            action="info",
            handle=""
        )
        
        response = await grpc_stub.ManageCache(request, metadata=metadata)
        
        assert response.success is True
        assert response.cache_size >= 0


# ============================================================================
# End-to-End Integration Test
# ============================================================================

@pytest.mark.integration
class TestCombinedEndToEnd:
    """End-to-end test using both REST and gRPC in combination."""
    
    async def test_full_workflow_mixed_protocols(
        self, rest_client, grpc_stub, sample_csv_content, api_key
    ):
        """
        Test complete workflow mixing REST and gRPC:
        1. Convert via gRPC
        2. Process via REST
        3. Plot via gRPC
        4. Verify cache via REST
        """
        # Step 1: Convert via gRPC
        grpc_convert_request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv_content)
        grpc_convert_response = await grpc_stub.ConvertToUnified(grpc_convert_request)
        assert grpc_convert_response.error == "" or grpc_convert_response.error is None
        unified_csv = grpc_convert_response.csv_content
        
        # Step 2: Process via REST
        csv_base64 = base64.b64encode(unified_csv.encode()).decode()
        rest_process_response = await rest_client.post(
            "/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        assert rest_process_response.status_code == 200
        process_data = rest_process_response.json()
        assert process_data.get("error") is None or process_data.get("error") == ""
        handle = process_data["handle"]
        
        # Step 3: Plot via gRPC
        metadata = aio.Metadata(('x-api-key', api_key))
        grpc_plot_request = glurpc_pb2.PlotRequest(
            handle=handle,
            index=0,
            force_calculate=False
        )
        grpc_plot_response = await grpc_stub.DrawPlot(grpc_plot_request, metadata=metadata)
        assert grpc_plot_response.error == "" or grpc_plot_response.error is None
        plot_dict = json.loads(grpc_plot_response.plot_json)
        assert "data" in plot_dict
        assert len(plot_dict["data"]) > 0
        
        # Step 4: Verify cache via REST
        rest_cache_response = await rest_client.post(
            "/cache_management?action=info",
            headers={"x-api-key": api_key}
        )
        assert rest_cache_response.status_code == 200
        cache_data = rest_cache_response.json()
        assert cache_data["success"] is True
        assert cache_data["cache_size"] > 0
        
        print(f"✅ Mixed protocol workflow completed successfully!")
        print(f"   gRPC Convert → REST Process → gRPC Plot → REST Cache Check")
        print(f"   Cache size: {cache_data['cache_size']}")


# ============================================================================
# Concurrent Access Test
# ============================================================================
@pytest.mark.asyncio
async def test_concurrent_rest_and_grpc_access(
    rest_client, grpc_stub, unified_csv_content_rest, api_key
):
    """
    Test that REST and gRPC can be used concurrently without conflicts.
    Fire multiple requests via both protocols simultaneously.
    """
    csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
    
    async def rest_process():
        """Process via REST."""
        response = await rest_client.post(
            "/process_unified",
            json={"csv_base64": csv_base64, "force_calculate": False},
            headers={"x-api-key": api_key}
        )
        return response.status_code == 200
    
    async def grpc_process():
        """Process via gRPC."""
        metadata = aio.Metadata(('x-api-key', api_key))
        request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        response = await grpc_stub.ProcessUnified(request, metadata=metadata)
        return response.error == "" or response.error is None
    
    # Fire 10 REST requests and 10 gRPC requests concurrently
    rest_tasks = [rest_process() for _ in range(5)]
    grpc_tasks = [grpc_process() for _ in range(5)]
    
    # Wait for all
    results = await asyncio.gather(*rest_tasks, *grpc_tasks, return_exceptions=True)
    
    # Check that most succeeded (allow some to fail due to concurrency)
    successes = sum(1 for r in results if r is True)
    assert successes == len(results), f"Some requests failed: {successes}/{len(results)} succeeded"

    print(f"Concurrent access test: {successes}/{len(results)} requests succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

