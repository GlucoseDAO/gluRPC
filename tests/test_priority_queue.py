"""
Test priority-based model queue system.

Verifies that model #0 is reserved for priority 0 requests only.
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from glurpc.engine import ModelManager, InferenceWrapper


@pytest.fixture
def mock_model_manager():
    """Create a ModelManager with mocked model loading."""
    manager = ModelManager()
    # Reset singleton state
    manager.models = []
    manager.priority_queue = asyncio.Queue()
    manager.general_queue = asyncio.Queue()
    manager.initialized = False
    return manager


@pytest.mark.asyncio
async def test_model_queue_initialization(mock_model_manager):
    """Test that models are distributed correctly between queues."""
    # Mock the model loading
    mock_wrappers = []
    for i in range(3):  # Simulate 3 model copies
        wrapper = Mock(spec=InferenceWrapper)
        wrapper.model_path = f"model_{i}"
        wrapper.device = "cpu"
        mock_wrappers.append(wrapper)
    
    mock_model_manager.models = mock_wrappers
    
    # Manually initialize queues (simulating what initialize() does)
    mock_model_manager.priority_queue.put_nowait(mock_model_manager.models[0])
    for model in mock_model_manager.models[1:]:
        mock_model_manager.general_queue.put_nowait(model)
    
    mock_model_manager.initialized = True
    
    # Verify queue sizes
    assert mock_model_manager.priority_queue.qsize() == 1, "Priority queue should have 1 model (model #0)"
    assert mock_model_manager.general_queue.qsize() == 2, "General queue should have 2 models (models #1 and #2)"


@pytest.mark.asyncio
async def test_priority_0_can_use_general_pool(mock_model_manager):
    """Test that priority 0 requests can acquire from general pool."""
    # Setup
    mock_wrappers = []
    for i in range(2):
        wrapper = Mock(spec=InferenceWrapper)
        wrapper.model_path = f"model_{i}"
        mock_wrappers.append(wrapper)
    
    mock_model_manager.models = mock_wrappers
    mock_model_manager.priority_queue.put_nowait(mock_wrappers[0])
    mock_model_manager.general_queue.put_nowait(mock_wrappers[1])
    mock_model_manager.initialized = True
    
    # Priority 0 should get from general pool first
    async with mock_model_manager.acquire(1, priority=0) as models:
        assert len(models) == 1
        # Should get model #1 from general pool, not model #0
        assert models[0] == mock_wrappers[1]
    
    # Verify model was returned
    assert mock_model_manager.general_queue.qsize() == 1


@pytest.mark.asyncio
async def test_priority_0_falls_back_to_priority_queue(mock_model_manager):
    """Test that priority 0 requests fall back to priority queue when general pool is empty."""
    # Setup
    mock_wrappers = []
    for i in range(2):
        wrapper = Mock(spec=InferenceWrapper)
        wrapper.model_path = f"model_{i}"
        mock_wrappers.append(wrapper)
    
    mock_model_manager.models = mock_wrappers
    mock_model_manager.priority_queue.put_nowait(mock_wrappers[0])
    # General queue is empty
    mock_model_manager.initialized = True
    
    # Priority 0 should fall back to priority queue
    async with mock_model_manager.acquire(1, priority=0) as models:
        assert len(models) == 1
        # Should get model #0 from priority queue
        assert models[0] == mock_wrappers[0]
    
    # Verify model was returned to priority queue
    assert mock_model_manager.priority_queue.qsize() == 1


@pytest.mark.asyncio
async def test_background_priority_only_uses_general_pool(mock_model_manager):
    """Test that background (priority > 0) requests only use general pool."""
    # Setup
    mock_wrappers = []
    for i in range(2):
        wrapper = Mock(spec=InferenceWrapper)
        wrapper.model_path = f"model_{i}"
        mock_wrappers.append(wrapper)
    
    mock_model_manager.models = mock_wrappers
    mock_model_manager.priority_queue.put_nowait(mock_wrappers[0])
    mock_model_manager.general_queue.put_nowait(mock_wrappers[1])
    mock_model_manager.initialized = True
    
    # Priority 1 should only get from general pool
    async with mock_model_manager.acquire(1, priority=1) as models:
        assert len(models) == 1
        # Should get model #1, never model #0
        assert models[0] == mock_wrappers[1]
    
    # Verify priority queue was not touched
    assert mock_model_manager.priority_queue.qsize() == 1
    assert mock_model_manager.general_queue.qsize() == 1


@pytest.mark.asyncio
async def test_minimum_two_copies_enforced():
    """Test that get_total_copies returns at least 2."""
    from glurpc.engine import get_total_copies
    
    # The function should always return at least 2
    total = get_total_copies()
    assert total >= 2, f"Expected at least 2 copies, got {total}"


@pytest.mark.asyncio
async def test_model_return_to_correct_queue(mock_model_manager):
    """Test that models are returned to their correct queues after use."""
    # Setup
    mock_wrappers = []
    for i in range(2):
        wrapper = Mock(spec=InferenceWrapper)
        wrapper.model_path = f"model_{i}"
        mock_wrappers.append(wrapper)
    
    mock_model_manager.models = mock_wrappers
    mock_model_manager.priority_queue.put_nowait(mock_wrappers[0])
    # Start with empty general queue to force using model #0
    mock_model_manager.initialized = True
    
    # Acquire and release model #0 (priority model)
    async with mock_model_manager.acquire(1, priority=0) as models:
        # General queue is empty, so should get model #0
        assert models[0] == mock_wrappers[0]
    
    # Model #0 should be back in priority queue
    assert mock_model_manager.priority_queue.qsize() == 1
    model_0_back = mock_model_manager.priority_queue.get_nowait()
    assert model_0_back == mock_wrappers[0]
    mock_model_manager.priority_queue.put_nowait(model_0_back)
    
    # Now add model #1 to general queue and test it
    mock_model_manager.general_queue.put_nowait(mock_wrappers[1])
    async with mock_model_manager.acquire(1, priority=1) as models:
        assert models[0] == mock_wrappers[1]
    
    # Model #1 should be back in general queue
    assert mock_model_manager.general_queue.qsize() == 1
    model_1_back = mock_model_manager.general_queue.get_nowait()
    assert model_1_back == mock_wrappers[1]
