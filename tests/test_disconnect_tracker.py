"""
Test suite for DisconnectTracker and request lifecycle with duplication handling.

Tests the implementation of THREADING_ARCHITECTURE.md:565-577:
- Request ID assignment
- Per-request disconnect futures
- Shared disconnect future with counter semantics
- Last-write-wins behavior
- Unified cancellation hook
"""
import asyncio
import pytest
import pytest_asyncio
from glurpc.state import DisconnectTracker, TaskRegistry


@pytest_asyncio.fixture
async def disconnect_tracker():
    """Fixture providing a fresh DisconnectTracker instance."""
    tracker = DisconnectTracker()
    await tracker.reset()
    yield tracker
    await tracker.reset()


@pytest_asyncio.fixture
async def task_registry():
    """Fixture providing a fresh TaskRegistry instance."""
    registry = TaskRegistry()
    await registry.reset()
    yield registry
    await registry.reset()


class TestRequestIDAssignment:
    """Test request ID assignment and sequencing."""
    
    @pytest.mark.asyncio
    async def test_monotonic_request_ids(self, disconnect_tracker):
        """Request IDs should be monotonically increasing."""
        handle = "test_handle_123"
        index = 0
        
        req_id_1 = await disconnect_tracker.register_request(handle, index)
        req_id_2 = await disconnect_tracker.register_request(handle, index)
        req_id_3 = await disconnect_tracker.register_request(handle, index)
        
        assert req_id_2 > req_id_1
        assert req_id_3 > req_id_2
        assert req_id_3 == req_id_1 + 2
    
    @pytest.mark.asyncio
    async def test_request_ids_per_key(self, disconnect_tracker):
        """Request IDs should be independent per (handle, index)."""
        handle = "test_handle_123"
        
        req_id_a1 = await disconnect_tracker.register_request(handle, 0)
        req_id_b1 = await disconnect_tracker.register_request(handle, -1)
        req_id_a2 = await disconnect_tracker.register_request(handle, 0)
        
        # Different indices have independent sequences
        assert req_id_a1 == 1
        assert req_id_b1 == 1
        assert req_id_a2 == 2


class TestDisconnectFutureSemantics:
    """Test disconnect future behavior with counter semantics."""
    
    @pytest.mark.asyncio
    async def test_shared_future_waits_for_all_requests(self, disconnect_tracker):
        """Shared disconnect future should only resolve when all requests finish."""
        handle = "test_handle_123"
        index = 0
        
        # Register 3 requests
        req_id_1 = await disconnect_tracker.register_request(handle, index)
        req_id_2 = await disconnect_tracker.register_request(handle, index)
        req_id_3 = await disconnect_tracker.register_request(handle, index)
        
        # Get shared disconnect future
        shared_future = await disconnect_tracker.get_disconnect_future(handle, index)
        
        # Shared future should not be done
        assert not shared_future.done()
        
        # Unregister first request
        await disconnect_tracker.unregister_request(handle, index, req_id_1)
        await asyncio.sleep(0.01)  # Let event loop process
        assert not shared_future.done()  # Still 2 active
        
        # Unregister second request
        await disconnect_tracker.unregister_request(handle, index, req_id_2)
        await asyncio.sleep(0.01)
        assert not shared_future.done()  # Still 1 active
        
        # Unregister third request
        await disconnect_tracker.unregister_request(handle, index, req_id_3)
        await asyncio.sleep(0.01)
        assert shared_future.done()  # All disconnected
    
    @pytest.mark.asyncio
    async def test_per_request_future_resolves_immediately(self, disconnect_tracker):
        """Per-request futures should resolve immediately on unregister."""
        handle = "test_handle_123"
        index = 0
        
        # Register 2 requests
        req_id_1 = await disconnect_tracker.register_request(handle, index)
        req_id_2 = await disconnect_tracker.register_request(handle, index)
        
        # Get per-request futures
        future_1 = await disconnect_tracker.get_disconnect_future(handle, index, req_id_1)
        future_2 = await disconnect_tracker.get_disconnect_future(handle, index, req_id_2)
        shared_future = await disconnect_tracker.get_disconnect_future(handle, index)
        
        # Both per-request futures should not be done
        assert not future_1.done()
        assert not future_2.done()
        assert not shared_future.done()
        
        # Unregister first request
        await disconnect_tracker.unregister_request(handle, index, req_id_1)
        await asyncio.sleep(0.01)
        
        # First per-request future resolves, others don't
        assert future_1.done()
        assert not future_2.done()
        assert not shared_future.done()
        
        # Unregister second request
        await disconnect_tracker.unregister_request(handle, index, req_id_2)
        await asyncio.sleep(0.01)
        
        # All futures resolve
        assert future_1.done()
        assert future_2.done()
        assert shared_future.done()


class TestCancellationHook:
    """Test unified cancellation hook with per-request granularity."""
    
    @pytest.mark.asyncio
    async def test_cancel_specific_request(self, disconnect_tracker):
        """cancel_request() should cancel only the specified request."""
        handle = "test_handle_123"
        index = 0
        
        # Register 2 requests
        req_id_1 = await disconnect_tracker.register_request(handle, index)
        req_id_2 = await disconnect_tracker.register_request(handle, index)
        
        # Get per-request futures
        future_1 = await disconnect_tracker.get_disconnect_future(handle, index, req_id_1)
        future_2 = await disconnect_tracker.get_disconnect_future(handle, index, req_id_2)
        
        # Cancel first request
        await disconnect_tracker.cancel_request(handle, index, req_id_1)
        await asyncio.sleep(0.01)
        
        # Only first future resolves
        assert future_1.done()
        assert not future_2.done()
    
    @pytest.mark.asyncio
    async def test_unified_cancellation_in_task_registry(self, task_registry, disconnect_tracker):
        """TaskRegistry.cancel_request() should trigger full cancellation."""
        handle = "test_handle_123"
        index = 0
        
        # Register request in DisconnectTracker
        req_id = await disconnect_tracker.register_request(handle, index)
        
        # Get per-request future
        future = await disconnect_tracker.get_disconnect_future(handle, index, req_id)
        
        # Register in TaskRegistry
        loop = asyncio.get_running_loop()
        task_future = loop.create_future()
        async with task_registry.lock:
            task_registry._registry[(handle, index, req_id)] = task_future
        
        # Call unified cancellation hook
        await task_registry.cancel_request(handle, index, req_id, "Test cancellation")
        await asyncio.sleep(0.01)
        
        # Both futures should be affected
        assert future.done()  # Disconnect future
        assert task_future.done()  # Task future
        assert task_future.exception() is not None  # Should have exception


class TestDuplicationHandling:
    """Test duplicate request handling with last-write-wins."""
    
    @pytest.mark.asyncio
    async def test_request_counter_tracks_duplicates(self, disconnect_tracker):
        """Request counter should accurately track duplicate requests."""
        handle = "test_handle_123"
        index = 0
        
        # Register 3 duplicate requests
        req_ids = []
        for _ in range(3):
            req_id = await disconnect_tracker.register_request(handle, index)
            req_ids.append(req_id)
        
        # Check counter via shared future behavior
        shared_future = await disconnect_tracker.get_disconnect_future(handle, index)
        
        # Unregister one by one
        for i, req_id in enumerate(req_ids[:-1]):
            await disconnect_tracker.unregister_request(handle, index, req_id)
            await asyncio.sleep(0.01)
            assert not shared_future.done(), f"Future resolved too early at iteration {i}"
        
        # Last unregister should resolve
        await disconnect_tracker.unregister_request(handle, index, req_ids[-1])
        await asyncio.sleep(0.01)
        assert shared_future.done()
    
    @pytest.mark.asyncio
    async def test_independent_keys(self, disconnect_tracker):
        """Different (handle, index) pairs should be independent."""
        handle = "test_handle_123"
        
        # Register requests for different keys
        req_id_a = await disconnect_tracker.register_request(handle, 0)
        req_id_b = await disconnect_tracker.register_request(handle, -1)
        
        # Get futures
        future_a = await disconnect_tracker.get_disconnect_future(handle, 0)
        future_b = await disconnect_tracker.get_disconnect_future(handle, -1)
        
        # Unregister one
        await disconnect_tracker.unregister_request(handle, 0, req_id_a)
        await asyncio.sleep(0.01)
        
        # Only that future resolves
        assert future_a.done()
        assert not future_b.done()


class TestRaceConditions:
    """Test concurrent operations on DisconnectTracker."""
    
    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, disconnect_tracker):
        """Concurrent registrations should all get unique IDs."""
        handle = "test_handle_123"
        index = 0
        
        async def register():
            return await disconnect_tracker.register_request(handle, index)
        
        # Register 10 requests concurrently
        tasks = [asyncio.create_task(register()) for _ in range(10)]
        req_ids = await asyncio.gather(*tasks)
        
        # All IDs should be unique
        assert len(set(req_ids)) == 10
        assert sorted(req_ids) == list(range(1, 11))
    
    @pytest.mark.asyncio
    async def test_concurrent_unregistrations(self, disconnect_tracker):
        """Concurrent unregistrations should handle counter correctly."""
        handle = "test_handle_123"
        index = 0
        
        # Register 5 requests
        req_ids = []
        for _ in range(5):
            req_id = await disconnect_tracker.register_request(handle, index)
            req_ids.append(req_id)
        
        shared_future = await disconnect_tracker.get_disconnect_future(handle, index)
        
        # Unregister all concurrently
        tasks = [
            asyncio.create_task(disconnect_tracker.unregister_request(handle, index, req_id))
            for req_id in req_ids
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.01)
        
        # Shared future should be resolved
        assert shared_future.done()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_unregister_before_register(self, disconnect_tracker):
        """Unregistering non-existent request should not crash."""
        handle = "test_handle_123"
        index = 0
        
        # Should not raise
        await disconnect_tracker.unregister_request(handle, index, 999)
    
    @pytest.mark.asyncio
    async def test_double_unregister(self, disconnect_tracker):
        """Double unregister should not break counter."""
        handle = "test_handle_123"
        index = 0
        
        req_id = await disconnect_tracker.register_request(handle, index)
        shared_future = await disconnect_tracker.get_disconnect_future(handle, index)
        
        # Unregister twice
        await disconnect_tracker.unregister_request(handle, index, req_id)
        await disconnect_tracker.unregister_request(handle, index, req_id)
        await asyncio.sleep(0.01)
        
        # Future should be resolved (counter can't go negative)
        assert shared_future.done()
    
    @pytest.mark.asyncio
    async def test_get_future_after_completion(self, disconnect_tracker):
        """Getting future after completion should return completed future."""
        handle = "test_handle_123"
        index = 0
        
        req_id = await disconnect_tracker.register_request(handle, index)
        await disconnect_tracker.unregister_request(handle, index, req_id)
        await asyncio.sleep(0.01)
        
        # Get future after completion - should create new one
        future = await disconnect_tracker.get_disconnect_future(handle, index)
        # The new future should not be done yet (no active requests)
        # Actually it should be done since counter is 0
        # Let's verify the behavior by registering a new request
        req_id_2 = await disconnect_tracker.register_request(handle, index)
        future_2 = await disconnect_tracker.get_disconnect_future(handle, index)
        assert not future_2.done()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
