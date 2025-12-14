"""
Integration tests for combined REST and gRPC endpoints.
Tests communicate with a running Docker container (docker-compose up).
All tests are async for consistency.
"""
import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
import grpc
from grpc import aio
from httpx import AsyncClient
import plotly.graph_objects as go

from service.service_spec import glurpc_pb2, glurpc_pb2_grpc


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_combined_integration")

# Ensure output directory exists
Path("test_outputs").mkdir(exist_ok=True)

# Run all async tests in this module on the same event loop
pytestmark = pytest.mark.asyncio(loop_scope="module")

# Docker container configuration
DOCKER_GRPC_HOST = "localhost"
DOCKER_GRPC_PORT = 7003
DOCKER_REST_HOST = "localhost"
DOCKER_REST_PORT = 8000
DOCKER_REST_BASE_URL = f"http://{DOCKER_REST_HOST}:{DOCKER_REST_PORT}"


@pytest_asyncio.fixture(scope="module")
async def rest_client():
    """AsyncClient for REST endpoints on Docker container."""
    async with AsyncClient(base_url=DOCKER_REST_BASE_URL, timeout=180.0) as client:
        # Wait for REST server to be ready
        max_attempts = 60
        for attempt in range(max_attempts):
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    logger.info(f"REST server ready at {DOCKER_REST_BASE_URL}")
                    break
            except Exception as e:
                if attempt % 10 == 0 and attempt > 0:
                    logger.info(f"Waiting for REST server... {attempt}s elapsed")
                if attempt == max_attempts - 1:
                    pytest.fail(f"REST server not ready after {max_attempts}s: {e}")
                await asyncio.sleep(1)
        
        yield client


@pytest_asyncio.fixture(scope="module")
async def grpc_channel():
    """Async gRPC channel for Docker container."""
    channel = aio.insecure_channel(f'{DOCKER_GRPC_HOST}:{DOCKER_GRPC_PORT}')
    
    # Wait for channel to be ready
    max_attempts = 60
    for attempt in range(max_attempts):
        try:
            await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
            logger.info(f"gRPC server ready at {DOCKER_GRPC_HOST}:{DOCKER_GRPC_PORT}")
            break
        except asyncio.TimeoutError:
            if attempt % 10 == 0 and attempt > 0:
                logger.info(f"Waiting for gRPC server... {attempt}s elapsed")
            if attempt == max_attempts - 1:
                pytest.fail(f"gRPC server not ready after {max_attempts}s")
            await asyncio.sleep(1)
    
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
            # Use the first file that's not too large
            for csv_file in sorted(csv_files):
                if csv_file.stat().st_size < 1_000_000:  # < 1MB
                    logger.info(f"Using sample CSV: {csv_file.name}")
                    return str(csv_file)
            # If all are large, just use the first one
            logger.info(f"Using sample CSV: {csv_files[0].name}")
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
        logger.info(f"REST Health: {data}")
    
    async def test_grpc_health(self, grpc_stub):
        """Test gRPC CheckHealth endpoint."""
        request = glurpc_pb2.HealthRequest()
        response = await grpc_stub.CheckHealth(request)
        
        assert response.status in ["ok", "degraded"]
        assert response.models_initialized is True
        assert response.cache_size >= 0
        assert response.device != ""
        logger.info(f"gRPC Health: status={response.status}, device={response.device}, cache={response.cache_size}")


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
        logger.info(f"REST Convert: converted {len(sample_csv_content)} bytes to {len(data['csv_content'])} chars")
    
    async def test_grpc_convert(self, grpc_stub, sample_csv_content):
        """Test gRPC ConvertToUnified endpoint."""
        request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv_content)
        response = await grpc_stub.ConvertToUnified(request)
        
        assert response.error == "" or response.error is None
        assert len(response.csv_content) > 0
        assert "sequence_id" in response.csv_content or "glucose" in response.csv_content
        logger.info(f"gRPC Convert: converted {len(sample_csv_content)} bytes to {len(response.csv_content)} chars")


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
    
    logger.info(f"Unified CSV content prepared: {len(response.csv_content)} chars")
    return response.csv_content


@pytest_asyncio.fixture(scope="module")
async def dataset_handle_rest(rest_client, unified_csv_content_rest, api_key):
    """Process CSV via REST and get handle."""
    csv_base64 = base64.b64encode(unified_csv_content_rest.encode()).decode()
    
    response = await rest_client.post(
        "/process_unified",
        json={"csv_base64": csv_base64, "force_calculate": False},
        headers={"x-api-key": api_key}
    )
    
    assert response.status_code == 200
    data = response.json()
    if data.get("error"):
        pytest.fail(f"ProcessUnified failed: {data['error']}")
    
    logger.info(f"REST dataset handle: {data['handle']}, total_samples: {data['total_samples']}")
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
    
    logger.info(f"gRPC dataset handle: {response.handle}, total_samples: {response.total_samples}")
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
        logger.info(f"REST process: handle={data['handle'][:20]}..., samples={data['total_samples']}")
    
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
        logger.info(f"gRPC process: handle={response.handle[:20]}..., samples={response.total_samples}")
    
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
        logger.info(f"✅ Cache shared: REST and gRPC returned same handle: {rest_handle[:20]}...")


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
        output_file = output_dir / "test_docker_rest_plot.html"
        fig.write_html(output_file)
        logger.info(f"Saved REST plot to {output_file}")
    
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
        output_file = output_dir / "test_docker_grpc_plot.html"
        fig.write_html(output_file)
        logger.info(f"Saved gRPC plot to {output_file}")
    
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
        output_file = output_dir / "test_docker_rest_quick_plot.html"
        fig.write_html(output_file)
        logger.info(f"Saved REST quick plot to {output_file}")
    
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
        output_file = output_dir / "test_docker_grpc_quick_plot.html"
        fig.write_html(output_file)
        logger.info(f"Saved gRPC quick plot to {output_file}")


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
        logger.info(f"REST cache info: size={data['cache_size']}")
    
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
        logger.info(f"gRPC cache info: size={response.cache_size}")


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
        logger.info("Starting mixed protocol workflow test...")
        
        # Step 1: Convert via gRPC
        logger.info("Step 1: Converting via gRPC...")
        grpc_convert_request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv_content)
        grpc_convert_response = await grpc_stub.ConvertToUnified(grpc_convert_request)
        assert grpc_convert_response.error == "" or grpc_convert_response.error is None
        unified_csv = grpc_convert_response.csv_content
        logger.info(f"  ✓ Converted {len(sample_csv_content)} bytes to {len(unified_csv)} chars")
        
        # Step 2: Process via REST
        logger.info("Step 2: Processing via REST...")
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
        logger.info(f"  ✓ Processed: handle={handle[:20]}..., samples={process_data['total_samples']}")
        
        # Step 3: Plot via gRPC
        logger.info("Step 3: Plotting via gRPC...")
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
        logger.info(f"  ✓ Generated plot with {len(plot_dict['data'])} traces")
        
        # Step 4: Verify cache via REST
        logger.info("Step 4: Verifying cache via REST...")
        rest_cache_response = await rest_client.post(
            "/cache_management?action=info",
            headers={"x-api-key": api_key}
        )
        assert rest_cache_response.status_code == 200
        cache_data = rest_cache_response.json()
        assert cache_data["success"] is True
        assert cache_data["cache_size"] > 0
        logger.info(f"  ✓ Cache size: {cache_data['cache_size']}")
        
        logger.info("✅ Mixed protocol workflow completed successfully!")
        logger.info("   gRPC Convert → REST Process → gRPC Plot → REST Cache Check")


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
    logger.info("Testing concurrent REST and gRPC access...")
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
    
    # Fire 5 REST requests and 5 gRPC requests concurrently
    rest_tasks = [rest_process() for _ in range(5)]
    grpc_tasks = [grpc_process() for _ in range(5)]
    
    # Wait for all
    results = await asyncio.gather(*rest_tasks, *grpc_tasks, return_exceptions=True)
    
    # Check that all succeeded
    successes = sum(1 for r in results if r is True)
    assert successes == len(results), f"Some requests failed: {successes}/{len(results)} succeeded"

    logger.info(f"✅ Concurrent access test: {successes}/{len(results)} requests succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
