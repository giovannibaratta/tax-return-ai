"""Pytest configuration and global fixtures for test isolation."""

import os
import shutil
import socket

import pytest

# Override cache directory for tests to prevent modifying user's .pii_cache
TEST_CACHE_DIR = "database/test_pii_cache"
os.environ["PII_CACHE_DIR"] = TEST_CACHE_DIR


@pytest.fixture(autouse=True)
def disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture that automatically blocks external network access during tests."""
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        if isinstance(address, tuple) and len(address) >= 2:
            host = str(address[0])
            if host in ("127.0.0.1", "localhost", "::1"):
                return original_connect(self, address)
        raise RuntimeError(f"Network access is blocked during tests! Attempted connect to {address}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def pytest_configure(config) -> None:
    """Create test cache directory if it doesn't exist.

    Args:
        config: Pytest config object.
    """
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def pytest_unconfigure(config) -> None:
    """Clean up test cache directory after tests finish.

    Args:
        config: Pytest config object.
    """
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)
