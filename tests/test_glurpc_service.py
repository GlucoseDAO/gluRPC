import sys
import base64
import json
import grpc
from pathlib import Path

# Import generated gRPC modules
from service.service_spec import glurpc_pb2, glurpc_pb2_grpc
from service import registry


def test_convert_to_unified(stub, file_path: str):
    """Test ConvertToUnified endpoint."""
    print("\n=== Testing ConvertToUnified ===")
    
    # Read file content
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    request = glurpc_pb2.ConvertToUnifiedRequest(file_content=file_content)
    response = stub.ConvertToUnified(request)
    
    if response.error:
        print(f"❌ Error: {response.error}")
        return None
    else:
        print(f"✅ Success: Converted to unified CSV ({len(response.csv_content)} bytes)")
        return response.csv_content


def test_process_unified(stub, csv_content: str, api_key: str):
    """Test ProcessUnified endpoint."""
    print("\n=== Testing ProcessUnified ===")
    
    # Encode to base64
    csv_base64 = base64.b64encode(csv_content.encode()).decode()
    
    # Create metadata with API key
    metadata = [('x-api-key', api_key)]
    
    request = glurpc_pb2.ProcessUnifiedRequest(
        csv_base64=csv_base64,
        force_calculate=False
    )
    
    try:
        response = stub.ProcessUnified(request, metadata=metadata)
        
        if response.error:
            print(f"❌ Error: {response.error}")
            return None
        else:
            print(f"✅ Success:")
            print(f"  Handle: {response.handle}")
            print(f"  Total Samples: {response.total_samples}")
            print(f"  Has Warnings: {response.warnings.has_warnings}")
            if response.warnings.has_warnings:
                print(f"  Warning Messages:")
                for msg in response.warnings.messages:
                    print(f"    - {msg}")
            return response.handle
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return None


def test_draw_plot(stub, handle: str, index: int, api_key: str):
    """Test DrawPlot endpoint."""
    print("\n=== Testing DrawPlot ===")
    
    # Create metadata with API key
    metadata = [('x-api-key', api_key)]
    
    request = glurpc_pb2.PlotRequest(
        handle=handle,
        index=index,
        force_calculate=False
    )
    
    try:
        response = stub.DrawPlot(request, metadata=metadata)
        
        if response.error:
            print(f"❌ Error: {response.error}")
            return None
        else:
            # Parse plot JSON
            plot_data = json.loads(response.plot_json)
            print(f"✅ Success:")
            print(f"  Plot data keys: {list(plot_data.keys())}")
            if 'data' in plot_data:
                print(f"  Number of traces: {len(plot_data['data'])}")
            return plot_data
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return None


def test_quick_plot(stub, csv_content: str, api_key: str):
    """Test QuickPlot endpoint."""
    print("\n=== Testing QuickPlot ===")
    
    # Encode to base64
    csv_base64 = base64.b64encode(csv_content.encode()).decode()
    
    # Create metadata with API key
    metadata = [('x-api-key', api_key)]
    
    request = glurpc_pb2.QuickPlotRequest(
        csv_base64=csv_base64,
        force_calculate=False
    )
    
    try:
        response = stub.QuickPlot(request, metadata=metadata)
        
        if response.error:
            print(f"❌ Error: {response.error}")
            return None
        else:
            # Parse plot JSON
            plot_data = json.loads(response.plot_json)
            print(f"✅ Success:")
            print(f"  Plot data keys: {list(plot_data.keys())}")
            if 'data' in plot_data:
                print(f"  Number of traces: {len(plot_data['data'])}")
            print(f"  Has Warnings: {response.warnings.has_warnings}")
            if response.warnings.has_warnings:
                print(f"  Warning Messages:")
                for msg in response.warnings.messages:
                    print(f"    - {msg}")
            return plot_data
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return None


def test_check_health(stub):
    """Test CheckHealth endpoint."""
    print("\n=== Testing CheckHealth ===")
    
    request = glurpc_pb2.HealthRequest()
    
    try:
        response = stub.CheckHealth(request)
        print(f"✅ Success:")
        print(f"  Status: {response.status}")
        print(f"  Load Status: {response.load_status}")
        print(f"  Cache Size: {response.cache_size}")
        print(f"  Models Initialized: {response.models_initialized}")
        print(f"  Device: {response.device}")
        print(f"  Total HTTP Requests: {response.total_http_requests}")
        print(f"  Total HTTP Errors: {response.total_http_errors}")
        print(f"  Inference Queue Size: {response.inference_queue_size}/{response.inference_queue_capacity}")
        return response
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return None


def test_manage_cache(stub, action: str, api_key: str, handle: str = None):
    """Test ManageCache endpoint."""
    print(f"\n=== Testing ManageCache (action={action}) ===")
    
    # Create metadata with API key
    metadata = [('x-api-key', api_key)]
    
    request = glurpc_pb2.CacheManagementRequest(
        action=action,
        handle=handle or ""
    )
    
    try:
        response = stub.ManageCache(request, metadata=metadata)
        
        if response.success:
            print(f"✅ Success: {response.message}")
            print(f"  Cache Size: {response.cache_size}")
            print(f"  Items Affected: {response.items_affected}")
        else:
            print(f"❌ Failed: {response.message}")
        return response
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return None


def main():
    """Main test runner."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_glurpc_service.py <test_file.csv> [api_key] [endpoint] [auto]")
        print("  test_file.csv: Path to test CGM data file")
        print("  api_key: Optional API key (default: reads from api_keys_list)")
        print("  endpoint: Optional gRPC endpoint (default: localhost:7003)")
        print("  auto: Optional 'auto' flag for non-interactive mode")
        sys.exit(1)
    
    test_file = sys.argv[1]
    
    # Get API key
    if len(sys.argv) > 2 and sys.argv[2] != "auto":
        api_key = sys.argv[2]
        endpoint = sys.argv[3] if len(sys.argv) > 3 else None
        test_flag = len(sys.argv) > 4 and sys.argv[4] == "auto"
    else:
        # Try to read from api_keys_list
        api_key_file = Path(__file__).parent / "api_keys_list"
        if api_key_file.exists():
            with open(api_key_file, 'r') as f:
                api_key = f.readline().strip()
        else:
            api_key = "test_key"
        endpoint = None
        test_flag = len(sys.argv) > 2 and sys.argv[2] == "auto"
    
    # Get endpoint
    if not endpoint:
        if test_flag:
            endpoint = f"localhost:{registry['glurpc_service']['grpc']}"
        else:
            endpoint = input(f"Endpoint (localhost:{registry['glurpc_service']['grpc']}): ") or f"localhost:{registry['glurpc_service']['grpc']}"
    
    print(f"\n🔌 Connecting to gRPC service at {endpoint}")
    print(f"🔑 Using API key: {api_key[:8]}...")
    
    # Open gRPC channel
    channel = grpc.insecure_channel(endpoint)
    stub = glurpc_pb2_grpc.GlucosePredictionStub(channel)
    
    # Test 1: Health Check
    test_check_health(stub)
    
    # Test 2: Convert to Unified
    csv_content = test_convert_to_unified(stub, test_file)
    if not csv_content:
        print("\n❌ ConvertToUnified failed, cannot proceed with other tests")
        return
    
    # Test 3: Process Unified
    handle = test_process_unified(stub, csv_content, api_key)
    if not handle:
        print("\n❌ ProcessUnified failed, cannot proceed with DrawPlot test")
    else:
        # Test 4: Draw Plot
        test_draw_plot(stub, handle, 0, api_key)  # Index 0 = most recent
        
        # Test 5: Cache Info
        test_manage_cache(stub, "info", api_key)
    
    # Test 6: Quick Plot (end-to-end)
    test_quick_plot(stub, csv_content, api_key)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()

