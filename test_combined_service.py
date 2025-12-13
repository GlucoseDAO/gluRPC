#!/usr/bin/env python3
"""
Test script for the combined gRPC + REST service.
Tests that both endpoints are accessible and share the same core logic.
"""

import sys
import time
import requests
import grpc
import base64
from pathlib import Path

# Add service to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from service.service_spec import glurpc_pb2, glurpc_pb2_grpc


def test_rest_health(rest_port: int = 8000):
    """Test REST health endpoint."""
    print(f"\n{'='*60}")
    print("Testing REST /health endpoint...")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"http://localhost:{rest_port}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Models Initialized: {data.get('models_initialized')}")
            print(f"Cache Size: {data.get('cache_size')}")
            print(f"Load Status: {data.get('load_status')}")
            print("✅ REST health check PASSED")
            return True
        else:
            print(f"❌ REST health check FAILED: {response.text}")
            return False
    except Exception as e:
        print(f"❌ REST health check FAILED: {e}")
        return False


def test_grpc_health(grpc_port: int = 7003):
    """Test gRPC health endpoint."""
    print(f"\n{'='*60}")
    print("Testing gRPC CheckHealth endpoint...")
    print(f"{'='*60}")
    
    try:
        channel = grpc.insecure_channel(f"localhost:{grpc_port}")
        stub = glurpc_pb2_grpc.GlucosePredictionStub(channel)
        
        request = glurpc_pb2.HealthRequest()
        response = stub.CheckHealth(request, timeout=5)
        
        print(f"Status: {response.status}")
        print(f"Models Initialized: {response.models_initialized}")
        print(f"Cache Size: {response.cache_size}")
        print(f"Load Status: {response.load_status}")
        print("✅ gRPC health check PASSED")
        
        channel.close()
        return True
    except Exception as e:
        print(f"❌ gRPC health check FAILED: {e}")
        return False


def test_rest_convert(rest_port: int = 8000):
    """Test REST convert endpoint with sample data."""
    print(f"\n{'='*60}")
    print("Testing REST /convert_to_unified endpoint...")
    print(f"{'='*60}")
    
    # Create a minimal CSV file
    sample_csv = b"timestamp,glucose\n2024-01-01 00:00:00,120\n2024-01-01 00:05:00,125\n"
    
    try:
        files = {'file': ('test.csv', sample_csv, 'text/csv')}
        response = requests.post(
            f"http://localhost:{rest_port}/convert_to_unified",
            files=files,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('error'):
                print(f"Conversion returned error: {data['error']}")
            else:
                print(f"CSV Content Length: {len(data.get('csv_content', ''))}")
                print("✅ REST convert PASSED")
                return True
        
        print(f"❌ REST convert FAILED: {response.text}")
        return False
    except Exception as e:
        print(f"❌ REST convert FAILED: {e}")
        return False


def test_grpc_convert(grpc_port: int = 7003):
    """Test gRPC convert endpoint with sample data."""
    print(f"\n{'='*60}")
    print("Testing gRPC ConvertToUnified endpoint...")
    print(f"{'='*60}")
    
    # Create a minimal CSV file
    sample_csv = b"timestamp,glucose\n2024-01-01 00:00:00,120\n2024-01-01 00:05:00,125\n"
    
    try:
        channel = grpc.insecure_channel(f"localhost:{grpc_port}")
        stub = glurpc_pb2_grpc.GlucosePredictionStub(channel)
        
        request = glurpc_pb2.ConvertToUnifiedRequest(file_content=sample_csv)
        response = stub.ConvertToUnified(request, timeout=10)
        
        if response.error:
            print(f"Conversion returned error: {response.error}")
        else:
            print(f"CSV Content Length: {len(response.csv_content)}")
            print("✅ gRPC convert PASSED")
            channel.close()
            return True
        
        channel.close()
        return False
    except Exception as e:
        print(f"❌ gRPC convert FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Combined gRPC + REST Service Test Suite")
    print("="*60)
    print("\nWaiting for services to be ready...")
    time.sleep(2)
    
    results = []
    
    # Test REST endpoints
    results.append(("REST Health", test_rest_health()))
    results.append(("REST Convert", test_rest_convert()))
    
    # Test gRPC endpoints
    results.append(("gRPC Health", test_grpc_health()))
    results.append(("gRPC Convert", test_grpc_convert()))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
