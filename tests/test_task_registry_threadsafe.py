import asyncio
import pytest
import pytest_asyncio
import threading
import time
from glurpc.state import TaskRegistry


@pytest_asyncio.fixture
async def registry():
    """Fixture that provides a clean TaskRegistry for each test."""
    reg = TaskRegistry()
    await reg.reset()
    yield reg
    await reg.reset()


@pytest.mark.asyncio
async def test_task_registry_notify_from_thread(registry):
    """
    Test that notify_success can be called from a worker thread safely.
    
    Note: In production, notify_* are only called from event loop.
    This test verifies they work even from threads (using CPython atomics).
    """
    handle = "test_handle"
    index = 0
    
    # Create a future that will wait for notification
    wait_task = asyncio.create_task(registry.wait_for_result(handle, index, timeout=5.0))
    
    # Give the wait task time to register
    await asyncio.sleep(0.1)
    
    # Notify from a worker thread (simulating edge case)
    def notify_from_thread():
        time.sleep(0.2)  # Simulate some work
        registry.notify_success(handle, index)
    
    thread = threading.Thread(target=notify_from_thread)
    thread.start()
    
    # Wait should complete successfully
    await wait_task
    thread.join()
    
    # Verify the future completed
    assert wait_task.done()


@pytest.mark.asyncio
async def test_task_registry_notify_error_from_thread(registry):
    """Test that notify_error can be called from a worker thread safely."""
    handle = "test_handle_error"
    index = 1
    
    # Create a future that will wait for notification
    wait_task = asyncio.create_task(registry.wait_for_result(handle, index, timeout=5.0))
    
    # Give the wait task time to register
    await asyncio.sleep(0.1)
    
    # Notify error from a worker thread
    test_error = ValueError("Test error from thread")
    
    def notify_error_from_thread():
        time.sleep(0.2)
        registry.notify_error(handle, index, test_error)
    
    thread = threading.Thread(target=notify_error_from_thread)
    thread.start()
    
    # Wait should raise the error
    with pytest.raises(ValueError, match="Test error from thread"):
        await wait_task
    
    thread.join()


@pytest.mark.asyncio
async def test_task_registry_cancel_all_for_handle_from_thread(registry):
    """Test that cancel_all_for_handle can be called from a worker thread safely."""
    handle = "test_handle_cancel"
    
    # Create multiple futures for the same handle
    wait_tasks = [
        asyncio.create_task(registry.wait_for_result(handle, i, timeout=5.0))
        for i in range(5)
    ]
    
    # Give the wait tasks time to register
    await asyncio.sleep(0.1)
    
    # Cancel from a worker thread
    def cancel_from_thread():
        time.sleep(0.2)
        registry.cancel_all_for_handle(handle)
    
    thread = threading.Thread(target=cancel_from_thread)
    thread.start()
    
    # All waits should raise exceptions
    for task in wait_tasks:
        with pytest.raises(Exception, match=f"Cache invalidated for handle {handle}"):
            await task
    
    thread.join()


@pytest.mark.asyncio
async def test_task_registry_notify_from_event_loop(registry):
    """Test the normal case: notify called from event loop (same thread)."""
    handle = "test_handle_loop"
    index = 0
    
    # Create a future that will wait for notification
    wait_task = asyncio.create_task(registry.wait_for_result(handle, index, timeout=5.0))
    
    # Give the wait task time to register
    await asyncio.sleep(0.1)
    
    # Notify from event loop (the normal production case)
    registry.notify_success(handle, index)
    
    # Wait should complete successfully
    await wait_task
    
    # Verify the future completed
    assert wait_task.done()


@pytest.mark.asyncio
async def test_task_registry_multiple_threads_concurrent(registry):
    """Test multiple threads notifying different handles concurrently."""
    num_threads = 10
    handles = [f"handle_{i}" for i in range(num_threads)]
    
    # Create wait tasks for each handle
    wait_tasks = [
        asyncio.create_task(registry.wait_for_result(handle, 0, timeout=5.0))
        for handle in handles
    ]
    
    # Give the wait tasks time to register
    await asyncio.sleep(0.1)
    
    # Create threads that will notify success
    def notify_from_thread(handle: str, delay: float):
        time.sleep(delay)
        registry.notify_success(handle, 0)
    
    threads = [
        threading.Thread(target=notify_from_thread, args=(handle, 0.1 * i))
        for i, handle in enumerate(handles)
    ]
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # All waits should complete
    await asyncio.gather(*wait_tasks)
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    # All tasks should be done
    assert all(task.done() for task in wait_tasks)


@pytest.mark.asyncio
async def test_task_registry_reset_cancels_pending(registry):
    """Test that reset() properly cancels all pending futures."""
    handle = "test_reset"
    
    # Create multiple pending wait tasks
    wait_tasks = [
        asyncio.create_task(registry.wait_for_result(handle, i, timeout=5.0))
        for i in range(3)
    ]
    
    # Give them time to register
    await asyncio.sleep(0.1)
    
    # Reset should cancel them all
    await registry.reset()
    
    # All tasks should complete with exceptions
    for task in wait_tasks:
        with pytest.raises(Exception, match="Registry reset"):
            await task

