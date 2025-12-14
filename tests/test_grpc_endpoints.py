"""
Pytest tests for gRPC service.
Tests the gRPC endpoints with a running server.
All tests are async.
"""
import pytest
import pytest_asyncio
import base64
import json
import grpc
from grpc import aio
from pathlib import Path
import asyncio
import time
import sys
import socket
import multiprocessing
from typing import Optional

from service.service_spec import glurpc_pb2, glurpc_pb2_grpc
from service import registry


# Run all async tests in this module on the same event loop
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _find_free_port() -> int:
    """Find a free port to bind the gRPC server to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_grpc_server(port: int, log_file: str) -> None:
    """Run gRPC server in a separate process for the module lifespan."""
    import logging
    
    # Setup logging to file - avoid duplicate handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
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
    
    logger.info(f"Starting async gRPC test server on port {port}")
    
    try:
        from service.glurpc_service import main_async
        asyncio.run(main_async(port=port))
    except KeyboardInterrupt:
        logger.info("Server interrupted, shutting down")
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        raise


@pytest.fixture(scope="module")
def grpc_server():
    """
    Start the gRPC server once for the whole module in its own process.
    Uses 'spawn' method to avoid fork() deadlocks with multi-threaded code.
    """
    port = _find_free_port()
    log_file = Path(__file__).parent.parent / "test_outputs" / "test_grpc_server.log"
    log_file.parent.mkdir(exist_ok=True)
    
    print(f"Starting gRPC test server on port {port}, logging to {log_file}")
    
    # Use spawn to avoid fork() deadlocks with asyncio/threads
    ctx = multiprocessing.get_context('spawn')
    proc = ctx.Process(target=_run_grpc_server, args=(port, str(log_file)), daemon=True)
    proc.start()
    
    # Wait for server to be ready - model loading can take 60+ seconds
    channel = grpc.insecure_channel(f'localhost:{port}')
    ready = False
    max_wait_time = 120  # 2 minutes for model loading
    
    for attempt in range(max_wait_time):
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            ready = True
            print(f"gRPC server ready after {attempt + 1} seconds")
            break
        except grpc.FutureTimeoutError:
            if attempt % 10 == 0:
                print(f"Waiting for server... {attempt + 1}s elapsed (check {log_file} for details)")
            time.sleep(1)
    
    channel.close()
    
    if not ready:
        print(f"Server failed to start. Check logs at: {log_file}")
        proc.terminate()
        proc.join(timeout=5)
        pytest.fail(f"gRPC server did not become ready in {max_wait_time}s. Check {log_file} for details")
    
    yield port
    
    proc.terminate()
    proc.join(timeout=5)


@pytest_asyncio.fixture(scope="module")
async def grpc_channel(grpc_server):
    """Create async gRPC channel to test server."""
    port = grpc_server
    channel = aio.insecure_channel(f'localhost:{port}')
    
    # Verify channel is ready
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Async gRPC channel could not connect to server")
    
    yield channel
    await channel.close()


@pytest_asyncio.fixture(scope="module")
async def grpc_stub(grpc_channel):
    """Create async gRPC stub."""
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
    # Look for sample data in data/ directory
    data_dir = Path(__file__).parent.parent / "data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    
    pytest.skip("No sample CSV file found in data/ directory")


@pytest_asyncio.fixture(scope="module")
async def unified_csv_content(grpc_stub, sample_csv_file):
    """Convert sample file to unified format."""
    with open(sample_csv_file, 'rb') as f:
        file_content = f.read()
    
    request = glurpc_pb2.ConvertToUnifiedRequest(file_content=file_content)
    response = await grpc_stub.ConvertToUnified(request)
    
    if response.error:
        pytest.fail(f"ConvertToUnified failed: {response.error}")
    
    return response.csv_content


@pytest_asyncio.fixture(scope="module")
async def dataset_handle(grpc_stub, unified_csv_content, api_key):
    """Process CSV and get dataset handle."""
    csv_base64 = base64.b64encode(unified_csv_content.encode()).decode()
    metadata = aio.Metadata(('x-api-key', api_key))
    
    request = glurpc_pb2.ProcessUnifiedRequest(
        csv_base64=csv_base64,
        force_calculate=False
    )
    
    response = await grpc_stub.ProcessUnified(request, metadata=metadata)
    
    if response.error:
        pytest.fail(f"ProcessUnified failed: {response.error}")
    
    return response.handle


class TestGrpcHealthCheck:
    """Tests for CheckHealth endpoint."""
    
    async def test_health_check(self, grpc_stub):
        """Test CheckHealth returns valid response."""
        request = glurpc_pb2.HealthRequest()
        response = await grpc_stub.CheckHealth(request)
        
        assert response.status in ["ok", "degraded", "error"]
        assert response.load_status != ""
        assert response.cache_size >= 0
        assert response.device != ""
        
    async def test_health_check_models_initialized(self, grpc_stub):
        """Test that models are initialized."""
        request = glurpc_pb2.HealthRequest()
        response = await grpc_stub.CheckHealth(request)
        
        # Models should be initialized for a running service
        assert response.models_initialized is True


class TestGrpcConversion:
    """Tests for ConvertToUnified endpoint."""
    
    async def test_convert_to_unified_success(self, grpc_stub, sample_csv_file):
        """Test successful conversion of CSV file."""
        with open(sample_csv_file, 'rb') as f:
            file_content = f.read()
        
        request = glurpc_pb2.ConvertToUnifiedRequest(file_content=file_content)
        response = await grpc_stub.ConvertToUnified(request)
        
        assert response.error == "" or response.error is None
        assert len(response.csv_content) > 0
        assert "sequence_id" in response.csv_content or "glucose" in response.csv_content
    
    async def test_convert_to_unified_invalid_data(self, grpc_stub):
        """Test conversion with invalid data returns error."""
        request = glurpc_pb2.ConvertToUnifiedRequest(file_content=b"invalid data")
        response = await grpc_stub.ConvertToUnified(request)
        
        # Must return a response object
        assert isinstance(response, glurpc_pb2.ConvertToUnifiedResponse)
        # Must have an error message for invalid data
        assert response.error is not None and response.error != "", \
            f"Expected error for invalid data, but got: error='{response.error}', csv_content length={len(response.csv_content)}"


class TestGrpcProcessing:
    """Tests for ProcessUnified endpoint."""
    
    async def test_process_unified_success(self, grpc_stub, unified_csv_content, api_key):
        """Test successful processing of unified CSV."""
        csv_base64 = base64.b64encode(unified_csv_content.encode()).decode()
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        response = await grpc_stub.ProcessUnified(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.handle) > 0
        assert response.total_samples > 0
        assert isinstance(response.warnings.has_warnings, bool)
    
    async def test_process_unified_requires_auth(self, grpc_stub, unified_csv_content):
        """Test that ProcessUnified requires authentication."""
        csv_base64 = base64.b64encode(unified_csv_content.encode()).decode()
        
        request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        # Without metadata, should fail or succeed based on ENABLE_API_KEYS
        try:
            response = await grpc_stub.ProcessUnified(request)
            # If API keys are disabled, this will succeed
            assert isinstance(response, glurpc_pb2.ProcessUnifiedResponse)
        except grpc.RpcError as e:
            # If API keys are enabled, should get UNAUTHENTICATED or PERMISSION_DENIED
            assert e.code() in [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED]
    
    async def test_process_unified_cache_hit(self, grpc_stub, unified_csv_content, api_key):
        """Test that repeated processing returns cached result."""
        csv_base64 = base64.b64encode(unified_csv_content.encode()).decode()
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        # First request
        response1 = await grpc_stub.ProcessUnified(request, metadata=metadata)
        handle1 = response1.handle
        
        # Second request (should hit cache)
        response2 = await grpc_stub.ProcessUnified(request, metadata=metadata)
        handle2 = response2.handle
        
        # Should return same handle
        assert handle1 == handle2


class TestGrpcPlotting:
    """Tests for DrawPlot endpoint."""
    
    async def test_draw_plot_success(self, grpc_stub, dataset_handle, api_key):
        """Test successful plot generation."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.PlotRequest(
            handle=dataset_handle,
            index=0,  # Most recent
            force_calculate=False
        )
        
        response = await grpc_stub.DrawPlot(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.plot_json) > 0
        
        # Verify it's valid JSON
        plot_data = json.loads(response.plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
    
    async def test_draw_plot_invalid_handle(self, grpc_stub, api_key):
        """Test DrawPlot with invalid handle returns error."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.PlotRequest(
            handle="invalid_handle_123456789",
            index=0,
            force_calculate=False
        )
        
        try:
            response = await grpc_stub.DrawPlot(request, metadata=metadata)
            # Should have error in response
            assert response.error != ""
        except grpc.RpcError as e:
            # Or fail with NOT_FOUND
            assert e.code() == grpc.StatusCode.NOT_FOUND
    
    async def test_draw_plot_requires_auth(self, grpc_stub, dataset_handle):
        """Test that DrawPlot requires authentication."""
        request = glurpc_pb2.PlotRequest(
            handle=dataset_handle,
            index=0,
            force_calculate=False
        )
        
        # Without metadata, should fail or succeed based on ENABLE_API_KEYS
        try:
            response = await grpc_stub.DrawPlot(request)
            assert isinstance(response, glurpc_pb2.PlotResponse)
        except grpc.RpcError as e:
            assert e.code() in [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED]


class TestGrpcQuickPlot:
    """Tests for QuickPlot endpoint."""
    
    async def test_quick_plot_success(self, grpc_stub, unified_csv_content, api_key):
        """Test successful quick plot generation."""
        csv_base64 = base64.b64encode(unified_csv_content.encode()).decode()
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.QuickPlotRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        
        response = await grpc_stub.QuickPlot(request, metadata=metadata)
        
        assert response.error == "" or response.error is None
        assert len(response.plot_json) > 0
        
        # Verify it's valid JSON
        plot_data = json.loads(response.plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Check warnings structure
        assert isinstance(response.warnings.has_warnings, bool)


class TestGrpcCacheManagement:
    """Tests for ManageCache endpoint."""
    
    async def test_cache_info(self, grpc_stub, api_key):
        """Test cache info retrieval."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.CacheManagementRequest(
            action="info",
            handle=""
        )
        
        response = await grpc_stub.ManageCache(request, metadata=metadata)
        
        assert response.success is True
        assert response.cache_size >= 0
        assert "info" in response.message.lower()
    
    async def test_cache_delete_invalid_handle(self, grpc_stub, api_key):
        """Test deleting non-existent handle."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.CacheManagementRequest(
            action="delete",
            handle="nonexistent_handle_12345"
        )
        
        response = await grpc_stub.ManageCache(request, metadata=metadata)
        
        # Should either succeed with items_affected=0 or fail
        if response.success:
            assert response.items_affected == 0
    
    async def test_cache_invalid_action(self, grpc_stub, api_key):
        """Test invalid cache action returns error."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        request = glurpc_pb2.CacheManagementRequest(
            action="invalid_action",
            handle=""
        )
        
        try:
            response = await grpc_stub.ManageCache(request, metadata=metadata)
            assert response.success is False
            assert "unknown" in response.message.lower() or "invalid" in response.message.lower()
        except grpc.RpcError as e:
            assert e.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.integration
class TestGrpcEndToEnd:
    """End-to-end integration tests."""
    
    async def test_full_workflow(self, grpc_stub, sample_csv_file, api_key):
        """Test complete workflow: convert -> process -> plot."""
        metadata = aio.Metadata(('x-api-key', api_key))
        
        # Step 1: Convert
        with open(sample_csv_file, 'rb') as f:
            file_content = f.read()
        
        convert_request = glurpc_pb2.ConvertToUnifiedRequest(file_content=file_content)
        convert_response = await grpc_stub.ConvertToUnified(convert_request)
        assert convert_response.error == "" or convert_response.error is None
        
        # Step 2: Process
        csv_base64 = base64.b64encode(convert_response.csv_content.encode()).decode()
        process_request = glurpc_pb2.ProcessUnifiedRequest(
            csv_base64=csv_base64,
            force_calculate=False
        )
        process_response = await grpc_stub.ProcessUnified(process_request, metadata=metadata)
        assert process_response.error == "" or process_response.error is None
        handle = process_response.handle
        
        # Step 3: Plot
        plot_request = glurpc_pb2.PlotRequest(
            handle=handle,
            index=0,
            force_calculate=False
        )
        plot_response = await grpc_stub.DrawPlot(plot_request, metadata=metadata)
        assert plot_response.error == "" or plot_response.error is None
        
        # Verify plot is valid JSON
        plot_data = json.loads(plot_response.plot_json)
        assert 'data' in plot_data
        assert len(plot_data['data']) > 0

