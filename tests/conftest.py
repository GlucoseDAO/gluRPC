"""Pytest configuration for GluRPC tests."""
import os
import signal
import sys

# Disable cache persistence for tests (prevent loading corrupt cache from disk)
# MUST be set BEFORE importing any glurpc modules
os.environ["ENABLE_CACHE_PERSISTENCE"] = "False"

# Ensure clean test environment
os.makedirs("logs", exist_ok=True)
os.makedirs("files", exist_ok=True)


def pytest_configure(config):
    """Configure pytest with proper signal handling."""
    # Allow Ctrl+C to work during tests
    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    # Set default timeout for tests
    config.addinivalue_line(
        "markers", "timeout: mark test to run with a timeout"
    )

