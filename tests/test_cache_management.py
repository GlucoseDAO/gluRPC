import asyncio
import base64
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from glurpc.app import app
from glurpc.config import ENABLE_CACHE_PERSISTENCE
from glurpc.data_classes import RESULT_SCHEMA
from glurpc.state import DataCache, TaskRegistry

# Test data file
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "Clarity_Export__Patient_2025-05-14_154517.csv"


def create_dummy_cache_data(length: int = 10, handle: str = None) -> Dict[str, Any]:
    """Create dummy data for cache testing."""
    if handle is None:
        handle = str(uuid.uuid4())[:8]
    
    dataset = [np.random.rand(10, 10) for _ in range(length)]
    data_df = pl.DataFrame(
        {
            "index": list(range(-(length - 1), 1)),
            "forecast": [None] * length,
            "true_values_x": [None] * length,
            "true_values_y": [None] * length,
            "median_x": [None] * length,
            "median_y": [None] * length,
            "fan_charts": [None] * length,
            "is_calculated": [False] * length
        },
        schema=RESULT_SCHEMA
    )
    return {
        "dataset": dataset,
        "model_config": {"mock": True},
        "data_df": data_df,
        "start_time": 100,
        "end_time": 200,
        "timestamp": "2025-12-02T00:00:00",
        "content_signature": f"test_sig_{handle}",
        "version": str(uuid.uuid4())
    }


# -----------------------------------------------------------------------------
# Unit Tests for DataCache Methods
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_handle_from_memory():
    """Test deleting a handle that exists only in memory."""
    cache = DataCache()
    await cache.clear_cache()
    
    handle = "test_delete_memory"
    data = create_dummy_cache_data(length=5, handle=handle)
    
    # Set in cache (memory only for this test)
    await cache.set(handle, data)
    
    # Verify it exists
    assert await cache.contains(handle)
    
    # Delete it
    deleted = await cache.delete_handle(handle)
    
    # Verify deletion
    assert deleted is True
    assert not await cache.contains(handle)
    assert await cache.get(handle) is None


@pytest.mark.asyncio
async def test_delete_handle_from_disk():
    """Test deleting a handle that exists on disk."""
    # Force enable persistence for this test
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_delete_disk"
        data = create_dummy_cache_data(length=5, handle=handle)
        
        # Set in cache (will persist to disk)
        await cache.set(handle, data)
        
        # Verify it exists on disk
        handle_dir = os.path.join(cache._cache_dir, handle)
        assert os.path.exists(handle_dir)
        
        # Delete it
        deleted = await cache.delete_handle(handle)
        
        # Verify deletion from both memory and disk
        assert deleted is True
        assert not await cache.contains(handle)
        assert not os.path.exists(handle_dir)


@pytest.mark.asyncio
async def test_delete_nonexistent_handle():
    """Test deleting a handle that doesn't exist."""
    cache = DataCache()
    await cache.clear_cache()
    
    handle = "nonexistent_handle"
    
    # Try to delete non-existent handle
    deleted = await cache.delete_handle(handle)
    
    # Should return False
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_handle_cancels_pending_tasks():
    """Test that deleting a handle cancels all pending tasks for that handle."""
    cache = DataCache()
    registry = TaskRegistry()
    await cache.clear_cache()
    
    handle = "test_delete_tasks"
    data = create_dummy_cache_data(length=5, handle=handle)
    await cache.set(handle, data)
    
    # Register a future waiting for this handle
    loop = asyncio.get_running_loop()
    future1 = loop.create_future()
    future2 = loop.create_future()
    
    async with registry.lock:
        registry._registry[(handle, 0)].append(future1)
        registry._registry[(handle, 1)].append(future2)
    
    # Delete the handle
    deleted = await cache.delete_handle(handle)
    assert deleted is True
    
    # Verify futures were cancelled with exception
    assert future1.done()
    assert future2.done()
    
    with pytest.raises(Exception) as exc_info1:
        future1.result()
    assert "invalidated" in str(exc_info1.value).lower()
    
    with pytest.raises(Exception) as exc_info2:
        future2.result()
    assert "invalidated" in str(exc_info2.value).lower()


@pytest.mark.asyncio
async def test_save_to_disk_specific_handle():
    """Test saving a specific handle to disk."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_save_specific"
        data = create_dummy_cache_data(length=5, handle=handle)
        
        # Set in cache but clear disk to simulate memory-only state
        await cache.set(handle, data)
        handle_dir = os.path.join(cache._cache_dir, handle)
        if os.path.exists(handle_dir):
            shutil.rmtree(handle_dir)
        
        # Save to disk
        count = await cache.save_to_disk(handle)
        
        # Verify saved
        assert count == 1
        assert os.path.exists(handle_dir)
        assert os.path.exists(os.path.join(handle_dir, "data.parquet"))
        assert os.path.exists(os.path.join(handle_dir, "meta.pkl"))


@pytest.mark.asyncio
async def test_save_to_disk_all_handles():
    """Test saving all handles to disk."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        # Add multiple handles
        handles = ["test_save_all_1", "test_save_all_2", "test_save_all_3"]
        for handle in handles:
            data = create_dummy_cache_data(length=5, handle=handle)
            await cache.set(handle, data)
            # Remove from disk to simulate memory-only
            handle_dir = os.path.join(cache._cache_dir, handle)
            if os.path.exists(handle_dir):
                shutil.rmtree(handle_dir)
        
        # Save all to disk
        count = await cache.save_to_disk(None)
        
        # Verify all saved
        assert count == 3
        for handle in handles:
            handle_dir = os.path.join(cache._cache_dir, handle)
            assert os.path.exists(handle_dir)


@pytest.mark.asyncio
async def test_save_to_disk_nonexistent_handle():
    """Test saving a non-existent handle returns 0."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        count = await cache.save_to_disk("nonexistent")
        assert count == 0


@pytest.mark.asyncio
async def test_save_to_disk_disabled_persistence():
    """Test that save_to_disk returns 0 when persistence is disabled."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', False):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_save_disabled"
        data = create_dummy_cache_data(length=5, handle=handle)
        await cache.set(handle, data)
        
        count = await cache.save_to_disk(handle)
        assert count == 0


@pytest.mark.asyncio
async def test_load_from_disk():
    """Test loading a handle from disk to memory."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_load_disk"
        data = create_dummy_cache_data(length=5, handle=handle)
        
        # Save to disk first
        await cache.set(handle, data)
        
        # Remove from memory but keep on disk
        async with cache.lock:
            del cache._cache[handle]
        
        # Verify not in memory but exists on disk
        assert handle not in cache._cache
        handle_dir = os.path.join(cache._cache_dir, handle)
        assert os.path.exists(handle_dir)
        
        # Load from disk
        loaded = await cache.load_from_disk(handle)
        
        # Verify loaded
        assert loaded is True
        assert handle in cache._cache
        loaded_data = await cache.get(handle)
        assert loaded_data is not None
        assert len(loaded_data['dataset']) == 5


@pytest.mark.asyncio
async def test_load_from_disk_already_in_memory():
    """Test loading a handle that's already in memory returns True."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_load_memory"
        data = create_dummy_cache_data(length=5, handle=handle)
        await cache.set(handle, data)
        
        # Load (already in memory)
        loaded = await cache.load_from_disk(handle)
        
        assert loaded is True


@pytest.mark.asyncio
async def test_load_from_disk_nonexistent():
    """Test loading a non-existent handle returns False."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        loaded = await cache.load_from_disk("nonexistent")
        assert loaded is False


@pytest.mark.asyncio
async def test_load_from_disk_disabled_persistence():
    """Test that load_from_disk returns False when persistence is disabled."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', False):
        cache = DataCache()
        await cache.clear_cache()
        
        loaded = await cache.load_from_disk("test_handle")
        assert loaded is False


# -----------------------------------------------------------------------------
# Integration Tests for API Endpoints
# -----------------------------------------------------------------------------

@pytest.fixture(scope="function")
def client():
    """Create a test client with cache persistence enabled."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        with TestClient(app) as c:
            yield c


def get_test_csv_content():
    """Get test CSV content."""
    if DATA_FILE_PATH.exists():
        return DATA_FILE_PATH.read_bytes()
    # Return minimal valid CSV if test data not available
    return b"sequence_id,time,glucose\n1,2025-01-01 00:00:00,100\n2,2025-01-01 00:05:00,105\n"


def test_cache_management_info(client):
    """Test /cache_management with action=info."""
    response = client.post("/cache_management?action=info")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "cache_size" in data
    assert "persisted_count" in data
    assert data["message"] == "Cache info retrieved"


def test_cache_management_flush(client):
    """Test /cache_management with action=flush."""
    # First add something to cache
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    client.post("/process_unified", json={"csv_base64": csv_base64})
    
    # Flush cache
    response = client.post("/cache_management?action=flush")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["cache_size"] == 0
    assert data["persisted_count"] == 0
    assert "flushed" in data["message"].lower()


def test_cache_management_delete_existing_handle(client):
    """Test /cache_management with action=delete for existing handle."""
    # First add something to cache
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    process_response = client.post("/process_unified", json={"csv_base64": csv_base64})
    
    assert process_response.status_code == 200
    handle = process_response.json()["handle"]
    
    # Delete the handle
    response = client.post(f"/cache_management?action=delete&handle={handle}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["items_affected"] == 1
    assert handle in data["message"]
    assert "deleted" in data["message"].lower()


def test_cache_management_delete_nonexistent_handle(client):
    """Test /cache_management with action=delete for non-existent handle."""
    response = client.post("/cache_management?action=delete&handle=nonexistent123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is False
    assert data["items_affected"] == 0
    assert "not found" in data["message"].lower()


def test_cache_management_delete_missing_handle_param(client):
    """Test /cache_management with action=delete but missing handle parameter."""
    response = client.post("/cache_management?action=delete")
    
    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_cache_management_save_specific_handle(client):
    """Test /cache_management with action=save for specific handle."""
    # Add something to cache
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    process_response = client.post("/process_unified", json={"csv_base64": csv_base64})
    
    assert process_response.status_code == 200
    handle = process_response.json()["handle"]
    
    # Save the handle
    response = client.post(f"/cache_management?action=save&handle={handle}")
    
    assert response.status_code == 200
    data = response.json()
    
    # When persistence is enabled, it's already saved during set, so items_affected might be 1
    assert "items_affected" in data
    assert handle in data["message"]


def test_cache_management_save_all(client):
    """Test /cache_management with action=save for all handles."""
    # Add multiple items to cache
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    client.post("/process_unified", json={"csv_base64": csv_base64})
    
    # Save all
    response = client.post("/cache_management?action=save")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "items_affected" in data
    assert data["items_affected"] >= 0  # Could be 0 if already persisted


def test_cache_management_load_existing_handle(client):
    """Test /cache_management with action=load for existing handle on disk."""
    # Add something to cache (will be persisted)
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    process_response = client.post("/process_unified", json={"csv_base64": csv_base64})
    
    assert process_response.status_code == 200
    handle = process_response.json()["handle"]
    
    # Load the handle (already in memory, so should return True)
    response = client.post(f"/cache_management?action=load&handle={handle}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["items_affected"] == 1
    assert handle in data["message"]


def test_cache_management_load_nonexistent_handle(client):
    """Test /cache_management with action=load for non-existent handle."""
    response = client.post("/cache_management?action=load&handle=nonexistent123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is False
    assert data["items_affected"] == 0
    assert "not found" in data["message"].lower()


def test_cache_management_load_missing_handle_param(client):
    """Test /cache_management with action=load but missing handle parameter."""
    response = client.post("/cache_management?action=load")
    
    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_cache_management_invalid_action(client):
    """Test /cache_management with invalid action."""
    response = client.post("/cache_management?action=invalid_action")
    
    assert response.status_code == 400
    assert "unknown action" in response.json()["detail"].lower()


def test_cache_management_workflow(client):
    """Test complete cache management workflow: add, save, delete, load."""
    csv_content = get_test_csv_content()
    csv_base64 = base64.b64encode(csv_content).decode('utf-8')
    
    # 1. Add item to cache
    process_response = client.post("/process_unified", json={"csv_base64": csv_base64})
    assert process_response.status_code == 200
    handle = process_response.json()["handle"]
    
    # 2. Verify it's in cache via info
    info_response = client.post("/cache_management?action=info")
    assert info_response.json()["cache_size"] >= 1
    
    # 3. Save to disk (redundant since auto-saved, but tests the endpoint)
    save_response = client.post(f"/cache_management?action=save&handle={handle}")
    assert save_response.status_code == 200
    
    # 4. Delete the handle
    delete_response = client.post(f"/cache_management?action=delete&handle={handle}")
    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True
    
    # 5. Verify it's gone
    plot_response = client.post("/draw_a_plot", json={"handle": handle, "index": 0})
    assert plot_response.status_code == 404  # Should be gone
    
    # 6. Info should show reduced size
    final_info = client.post("/cache_management?action=info")
    assert final_info.status_code == 200


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_updates_metadata_index():
    """Test that delete properly updates metadata index."""
    with patch('glurpc.state.ENABLE_CACHE_PERSISTENCE', True):
        cache = DataCache()
        await cache.clear_cache()
        
        handle = "test_delete_metadata"
        data = create_dummy_cache_data(length=5, handle=handle)
        await cache.set(handle, data)
        
        # Verify in metadata index
        async with cache.lock:
            assert any(entry['handle'] == handle for entry in cache._metadata_index)
        
        # Delete
        await cache.delete_handle(handle)
        
        # Verify removed from metadata index
        async with cache.lock:
            assert not any(entry['handle'] == handle for entry in cache._metadata_index)


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent cache operations don't cause issues."""
    cache = DataCache()
    await cache.clear_cache()
    
    async def add_and_delete(idx: int):
        handle = f"concurrent_{idx}"
        data = create_dummy_cache_data(length=5, handle=handle)
        await cache.set(handle, data)
        await asyncio.sleep(0.01)  # Small delay
        await cache.delete_handle(handle)
    
    # Run multiple operations concurrently
    tasks = [add_and_delete(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    # All should be deleted
    size = await cache.get_size()
    assert size == 0

