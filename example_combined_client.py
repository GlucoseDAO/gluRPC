#!/usr/bin/env python3
"""
Example client showing how to interact with both gRPC and REST endpoints
from the combined service.

This demonstrates that both protocols work simultaneously and share the same cache.
"""

import sys
import grpc
import requests
import base64
from pathlib import Path

# Add service to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from service.service_spec import glurpc_pb2, glurpc_pb2_grpc


def create_sample_csv() -> bytes:
    """Create a sample CGM CSV file for testing."""
    csv_content = """timestamp,glucose
2024-01-01 00:00:00,120
2024-01-01 00:05:00,125
2024-01-01 00:10:00,130
2024-01-01 00:15:00,128
2024-01-01 00:20:00,122
2024-01-01 00:25:00,118
2024-01-01 00:30:00,115
2024-01-01 00:35:00,120
2024-01-01 00:40:00,125
2024-01-01 00:45:00,130
"""
    return csv_content.encode()


def example_rest_workflow(rest_port: int = 8000, api_key: str = None):
    """Demonstrate REST API workflow."""
    print("\n" + "="*60)
    print("REST API Workflow Example")
    print("="*60)
    
    # Step 1: Convert to unified format (public endpoint)
    print("\n1. Converting CSV to unified format...")
    sample_csv = create_sample_csv()
    
    files = {'file': ('test.csv', sample_csv, 'text/csv')}
    response = requests.post(
        f"http://localhost:{rest_port}/convert_to_unified",
        files=files
    )
    
    if response.status_code != 200:
        print(f"❌ Convert failed: {response.text}")
        return None
    
    unified_csv = response.json().get('csv_content', '')
    print(f"✅ Converted to unified format ({len(unified_csv)} bytes)")
    
    # If API keys are enabled, you need a valid key for the next steps
    if not api_key:
        print("\n⚠️  Skipping authenticated endpoints (no API key provided)")
        return None
    
    headers = {"X-API-Key": api_key}
    
    # Step 2: Process unified CSV (authenticated)
    print("\n2. Processing unified CSV and caching...")
    csv_base64 = base64.b64encode(unified_csv.encode()).decode()
    
    response = requests.post(
        f"http://localhost:{rest_port}/process_unified",
        json={"csv_base64": csv_base64, "force_calculate": False},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"❌ Process failed: {response.text}")
        return None
    
    result = response.json()
    handle = result.get('handle')
    total_samples = result.get('total_samples', 0)
    has_warnings = result.get('warnings', {}).get('has_warnings', False)
    
    print(f"✅ Processed and cached: handle={handle[:16]}...")
    print(f"   Total samples: {total_samples}")
    print(f"   Warnings: {'Yes' if has_warnings else 'No'}")
    
    # Step 3: Draw a plot (authenticated)
    print("\n3. Generating plot for sample index 0...")
    
    response = requests.post(
        f"http://localhost:{rest_port}/draw_a_plot",
        json={"handle": handle, "index": 0, "force_calculate": False},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"❌ Plot generation failed: {response.text}")
        return None
    
    plot_json = response.json()
    print(f"✅ Plot generated with {len(plot_json.get('data', []))} traces")
    
    return handle


def example_grpc_workflow(grpc_port: int = 7003, api_key: str = None):
    """Demonstrate gRPC API workflow."""
    print("\n" + "="*60)
    print("gRPC API Workflow Example")
    print("="*60)
    
    channel = grpc.insecure_channel(f"localhost:{grpc_port}")
    stub = glurpc_pb2_grpc.GlucosePredictionStub(channel)
    
    # Step 1: Convert to unified format (public endpoint)
    print("\n1. Converting CSV to unified format...")
    sample_csv = create_sample_csv()
    
    request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv)
    response = stub.ConvertToUnified(request)
    
    if response.error:
        print(f"❌ Convert failed: {response.error}")
        channel.close()
        return None
    
    unified_csv = response.csv_content
    print(f"✅ Converted to unified format ({len(unified_csv)} bytes)")
    
    # If API keys are enabled, you need a valid key for the next steps
    if not api_key:
        print("\n⚠️  Skipping authenticated endpoints (no API key provided)")
        channel.close()
        return None
    
    # For gRPC, pass API key in metadata
    metadata = [('x-api-key', api_key)]
    
    # Step 2: Process unified CSV (authenticated)
    print("\n2. Processing unified CSV and caching...")
    csv_base64 = base64.b64encode(unified_csv.encode()).decode()
    
    request = glurpc_pb2.ProcessUnifiedRequest(
        csv_base64=csv_base64,
        force_calculate=False
    )
    response = stub.ProcessUnified(request, metadata=metadata)
    
    if response.error:
        print(f"❌ Process failed: {response.error}")
        channel.close()
        return None
    
    handle = response.handle
    total_samples = response.total_samples
    has_warnings = response.warnings.has_warnings
    
    print(f"✅ Processed and cached: handle={handle[:16]}...")
    print(f"   Total samples: {total_samples}")
    print(f"   Warnings: {'Yes' if has_warnings else 'No'}")
    
    # Step 3: Draw a plot (authenticated)
    print("\n3. Generating plot for sample index 0...")
    
    request = glurpc_pb2.PlotRequest(
        handle=handle,
        index=0,
        force_calculate=False
    )
    response = stub.DrawPlot(request, metadata=metadata)
    
    if response.error:
        print(f"❌ Plot generation failed: {response.error}")
        channel.close()
        return None
    
    plot_json = response.plot_json
    print(f"✅ Plot generated ({len(plot_json)} bytes)")
    
    channel.close()
    return handle


def verify_shared_cache(rest_handle: str, grpc_handle: str):
    """Verify that both REST and gRPC share the same cache."""
    print("\n" + "="*60)
    print("Cache Sharing Verification")
    print("="*60)
    
    if rest_handle and grpc_handle:
        if rest_handle == grpc_handle:
            print("✅ CONFIRMED: Both REST and gRPC generated the SAME handle!")
            print(f"   Handle: {rest_handle[:16]}...")
            print("   This proves they share the same cache and core logic.")
        else:
            print("ℹ️  Different handles generated (different CSV content or timing)")
            print(f"   REST handle: {rest_handle[:16]}...")
            print(f"   gRPC handle: {grpc_handle[:16]}...")
    else:
        print("⚠️  Could not verify cache sharing (missing handles)")


def main():
    """Run example workflows."""
    print("\n" + "="*60)
    print("Combined Service Client Example")
    print("="*60)
    print("\nThis example demonstrates:")
    print("1. Both REST and gRPC APIs work simultaneously")
    print("2. They share the same core logic and cache")
    print("3. Authentication works with API keys (if enabled)")
    
    # Check if API keys are enabled (you would set this via environment)
    # For this example, we'll try without API keys first
    api_key = None  # Set to your API key if ENABLE_API_KEYS=true
    
    # Run REST workflow
    rest_handle = example_rest_workflow(rest_port=8000, api_key=api_key)
    
    # Run gRPC workflow
    grpc_handle = example_grpc_workflow(grpc_port=7003, api_key=api_key)
    
    # Verify cache sharing
    verify_shared_cache(rest_handle, grpc_handle)
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
