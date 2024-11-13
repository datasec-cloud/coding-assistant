# tests/test_core/test_context/test_cache_manager.py

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import json
import hashlib
from unittest.mock import MagicMock, patch

from coding_assistant.core.context.cache_manager import (
    CacheManager, CacheLevel, CacheMode, CacheStrategy,
    HierarchicalCache, CacheEntry, CacheConfig
)
from coding_assistant.core.io.transaction import TransactionManager, OperationType


@pytest.fixture
def base_dir(tmp_path):
    """Create a temporary base directory for tests"""
    return tmp_path / "cache_test"


@pytest.fixture
def mock_transaction_manager():
    """Create a mock TransactionManager"""
    mock_tm = MagicMock(spec=TransactionManager)
    # Mock the transaction context manager
    mock_tm.transaction.return_value.__enter__.return_value = "transaction_id"
    mock_tm.transaction.return_value.__exit__.return_value = False
    mock_tm.add_operation.return_value = True  # Assume add_operation always succeeds
    return mock_tm


@pytest.fixture
def cache_config():
    """Create a test cache configuration"""
    config = CacheConfig()
    config.ttl = {
        CacheLevel.SYSTEM: timedelta(hours=24),
        CacheLevel.DOMAIN: timedelta(hours=12),
        CacheLevel.COMPONENT: timedelta(hours=6)
    }
    config.max_entries = {
        CacheLevel.SYSTEM: 1000,
        CacheLevel.DOMAIN: 5000,
        CacheLevel.COMPONENT: 10000
    }
    config.learning = {
        "pattern_detection": True,
        "confidence_threshold": 0.85,
        "min_pattern_occurrences": 3
    }
    config.resilience = {
        "max_retries": 3,
        "retry_delay": 1.0,  # seconds
        "circuit_breaker": {
            "error_threshold": 5,
            "timeout": 60  # seconds
        }
    }
    config.mode = CacheMode.HYBRID
    config.strategy = CacheStrategy.WRITE_THROUGH
    config.prefetch_threshold = 0.7
    return config

@pytest.fixture
def cache_manager_instance(base_dir, cache_config, mock_transaction_manager, mocker):
    """Create a CacheManager instance for testing"""
    # Mock time.sleep to speed up background threads
    mocker.patch('time.sleep', return_value=None)
    
    manager = CacheManager(base_dir, transaction_manager=mock_transaction_manager)
    
    # Inject the test cache configuration
    manager.config = cache_config
    manager.cache.config = cache_config
    
    return manager

@pytest.fixture
def cache_manager(cache_manager_instance):
    """Ensure proper shutdown after tests"""
    yield cache_manager_instance
    cache_manager_instance.shutdown()


class TestCacheBasicOperations:
    def test_set_and_get(self, cache_manager):
        """Test basic set and get operations"""
        key = "test_key"
        value = "test_value"

        # Test set operation
        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True

        # Test get operation
        retrieved_value = cache_manager.get(key, CacheLevel.SYSTEM)
        assert retrieved_value == value

    def test_get_nonexistent(self, cache_manager):
        """Test getting non-existent keys"""
        assert cache_manager.get("nonexistent", CacheLevel.SYSTEM) is None
        assert cache_manager.get("", CacheLevel.DOMAIN) is None
        # 'None' is handled gracefully and returns None, no exception
        assert cache_manager.get(None, CacheLevel.COMPONENT) is None

    def test_set_with_metadata(self, cache_manager):
        """Test setting values with metadata"""
        key = "meta_key"
        value = "test_value"
        metadata = {"version": "1.0"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM, metadata=metadata) is True
        retrieved_value = cache_manager.get(key, CacheLevel.SYSTEM)
        assert retrieved_value == value

        # Verify that metadata is correctly stored
        cache_entry = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        assert cache_entry is not None
        assert cache_entry.metadata == metadata


class TestCachePersistence:
    def test_persistence_basics(self, cache_manager, base_dir):
        """Test basic persistence operations"""
        # Set a value
        assert cache_manager.set("persist_key", "persist_value", CacheLevel.SYSTEM) is True

        # Verify directory structure
        cache_dir = base_dir / "cache" / CacheLevel.SYSTEM.value
        assert cache_dir.exists()

        # Verify persistent file
        cache_file = cache_dir / "persist_key.cache"
        assert cache_file.exists()

    def test_persistent_storage_integrity(self, cache_manager, base_dir):
        """Test integrity of persistent storage"""
        key = "integrity_key"
        value = {"test": "data"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        retrieved_value = cache_manager.get(key, CacheLevel.SYSTEM)
        assert retrieved_value == value

        # Verify file content and checksum
        cache_file = base_dir / "cache" / CacheLevel.SYSTEM.value / f"{key}.cache"
        assert cache_file.exists()
        with cache_file.open('r') as f:
            data = json.load(f)
            assert data["value"] == value

            # Recalculate checksum
            serialized_value = json.dumps(value, sort_keys=True)
            expected_checksum = hashlib.sha256(serialized_value.encode()).hexdigest()
            assert data["checksum"] == expected_checksum


class TestCacheEviction:
    def test_size_based_eviction(self, cache_manager, cache_config):
        """Test size-based eviction"""
        # Fill cache to trigger eviction
        max_entries = cache_config.max_entries[CacheLevel.SYSTEM]
        for i in range(max_entries + 1000):  # Exceed max entries
            cache_manager.set(f"key_{i}", f"value_{i}", CacheLevel.SYSTEM)

        # First entries should be evicted (LRU)
        assert cache_manager.get("key_0", CacheLevel.SYSTEM) is None

        # Last entry should still exist
        last_key = f"key_{max_entries + 999}"
        assert cache_manager.get(last_key, CacheLevel.SYSTEM) == f"value_{max_entries + 999}"

        # Verify cache size constraint
        assert len(cache_manager.cache._caches[CacheLevel.SYSTEM]) <= max_entries

    def test_ttl_eviction(self, cache_manager):
        """Test time-based eviction"""
        key = "ttl_test"
        value = "test_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        # Fast-forward time beyond TTL by manually adjusting the entry's timestamps
        cache_entry = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        cache_entry.created_at = datetime.now() - timedelta(hours=25)
        cache_entry.accessed_at = datetime.now() - timedelta(hours=25)

        # Now, attempt to get the value should trigger eviction
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None


class TestCacheResilience:
    def test_circuit_breaker_activation(self, cache_manager, cache_config):
        """Test circuit breaker activation after repeated errors"""
        level = CacheLevel.SYSTEM

        # Simulate storage initialization failure by patching _initialize_storage to raise Exception
        with patch.object(HierarchicalCache, '_initialize_storage', side_effect=Exception("Storage Error")):
            # Re-initialize storage to trigger the error
            with pytest.raises(Exception):
                cache_manager.cache._initialize_storage()

        # Manually increment error counts to trigger circuit breaker
        for _ in range(cache_config.resilience["circuit_breaker"]["error_threshold"]):
            cache_manager.cache._handle_storage_error(level)

        # Check if circuit breaker is open
        assert cache_manager.cache._circuit_broken[level] is True

        # Attempt to set a value should fail due to circuit breaker
        assert cache_manager.set("circuit_key", "circuit_value", level) is False

    def test_circuit_breaker_recovery(self, cache_manager, cache_config):
        """Test circuit breaker recovery after timeout"""
        level = CacheLevel.SYSTEM

        # Manually open circuit breaker
        cache_manager.cache._circuit_broken[level] = True
        cache_manager.cache._last_error_reset[level] = datetime.now() - timedelta(seconds=cache_config.resilience["circuit_breaker"]["timeout"] + 1)

        # Attempt to set a value should proceed as circuit breaker timeout has passed
        with patch.object(HierarchicalCache, '_handle_storage_error', return_value=None):
            assert cache_manager.set("recover_key", "recover_value", level) is True
            # Circuit breaker should now be closed
            assert cache_manager.cache._circuit_broken[level] is False


class TestCachePatterns:
    def test_pattern_detection_and_prefetch(self, cache_manager, cache_config):
        """Test pattern detection and prefetching based on learned patterns"""
        key = "pattern_key"
        value = "pattern_value"

        # Access the key multiple times to trigger pattern detection
        for _ in range(cache_config.learning["min_pattern_occurrences"]):
            cache_manager.set(key, value, CacheLevel.DOMAIN, pattern_id="pattern_1")
            cache_manager.get(key, CacheLevel.DOMAIN)

        # Check if the pattern count has been updated
        pattern_stats = cache_manager.get_stats().get("patterns", {})
        pattern_key = f"domain:{key}"
        assert pattern_key in pattern_stats
        assert pattern_stats[pattern_key]["count"] >= cache_config.learning["min_pattern_occurrences"]
        assert pattern_stats[pattern_key]["confidence"] >= cache_config.learning["confidence_threshold"]

        # Check if prefetch is enabled based on the pattern
        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_key in recommendations
        assert recommendations[pattern_key] >= cache_config.learning["confidence_threshold"]


class TestCacheManagerShutdown:
    def test_shutdown(self, cache_manager):
        """Test graceful shutdown of CacheManager"""
        # Ensure shutdown doesn't raise exceptions
        try:
            cache_manager.shutdown()
        except Exception as e:
            pytest.fail(f"Shutdown raised an exception: {e}")

    def test_cleanup_thread_running(self, cache_manager):
        """Test if cleanup and monitor threads are not running after shutdown"""
        # After shutdown, threads should not be alive
        assert not cache_manager._cleanup_thread.is_alive()
        assert not cache_manager._monitor_thread.is_alive()


class TestInvalidation:
    def test_invalidate_specific_key(self, cache_manager):
        """Test invalidating a specific cache key"""
        key = "invalidate_key"
        value = "invalidate_value"

        assert cache_manager.set(key, value, CacheLevel.COMPONENT) is True
        assert cache_manager.get(key, CacheLevel.COMPONENT) == value

        cache_manager.invalidate(key, CacheLevel.COMPONENT)
        assert cache_manager.get(key, CacheLevel.COMPONENT) is None

    def test_invalidate_all_levels(self, cache_manager):
        """Test invalidating a key across all cache levels"""
        key = "multi_level_key"
        value = "multi_level_value"

        for level in CacheLevel:
            assert cache_manager.set(key, value, level) is True
            assert cache_manager.get(key, level) == value

        cache_manager.invalidate(key)
        for level in CacheLevel:
            assert cache_manager.get(key, level) is None


class TestUpdateConfig:
    def test_update_cache_config(self, cache_manager, cache_config):
        """Test updating cache configuration at runtime"""
        new_max_entries = {
            CacheLevel.SYSTEM: 2000,
            CacheLevel.DOMAIN: 6000,
            CacheLevel.COMPONENT: 12000
        }
        update = {"max_entries": new_max_entries}
        assert cache_manager.update_config(update) is True

        # Verify that the configuration has been updated
        assert cache_manager.config.max_entries == new_max_entries


class TestPreloadPattern:
    def test_preload_pattern(self, cache_manager, cache_config):
        """Test preloading data for known access patterns"""
        pattern_id = "preload_pattern"
        data = {
            "preload_key_1": "preload_value_1",
            "preload_key_2": "preload_value_2"
        }

        assert cache_manager.preload_pattern(pattern_id, data, CacheLevel.DOMAIN) is True

        for key, value in data.items():
            assert cache_manager.get(key, CacheLevel.DOMAIN) == value

        # Check if pattern stats are updated
        pattern_stats = cache_manager.get_stats().get("patterns", {})
        for key in data.keys():
            pattern_key = f"domain:{key}"
            assert pattern_key in pattern_stats
            assert pattern_stats[pattern_key]["count"] >= 1


class TestCacheStats:
    def test_get_stats(self, cache_manager, cache_config):
        """Test retrieval of cache statistics"""
        # Populate cache with some entries
        for i in range(10):
            cache_manager.set(f"stat_key_{i}", f"stat_value_{i}", CacheLevel.COMPONENT)

        stats = cache_manager.get_stats()
        assert "component" in stats
        assert stats["component"]["total_entries"] >= 10
        assert stats["component"]["valid_entries"] >= 10
        assert stats["component"]["total_hits"] == 0  # No gets yet

        # Access some entries
        for i in range(5):
            cache_manager.get(f"stat_key_{i}", CacheLevel.COMPONENT)

        stats = cache_manager.get_stats()
        assert stats["component"]["total_hits"] == 5
        assert stats["component"]["avg_hits"] == 0.5  # 5 hits / 10 entries


class TestConcurrentAccess:
    def test_concurrent_set_and_get(self, cache_manager):
        """Test concurrent set and get operations to ensure thread safety"""
        key = "concurrent_key"
        value = "concurrent_value"

        def setter():
            for _ in range(100):
                cache_manager.set(key, value, CacheLevel.DOMAIN)

        def getter():
            for _ in range(100):
                retrieved = cache_manager.get(key, CacheLevel.DOMAIN)
                if retrieved is not None:
                    assert retrieved == value

        threads = []
        for _ in range(5):
            t_set = threading.Thread(target=setter)
            t_get = threading.Thread(target=getter)
            threads.extend([t_set, t_get])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Final verification
        assert cache_manager.get(key, CacheLevel.DOMAIN) == value
        assert len(cache_manager.cache._caches[CacheLevel.DOMAIN]) <= cache_config.max_entries[CacheLevel.DOMAIN]


class TestCacheManagerErrorHandling:
    def test_set_with_invalid_key(self, cache_manager):
        """Test setting a value with an invalid key"""
        invalid_keys = ["", None, 123, {"key": "value"}]
        value = "test_value"

        for key in invalid_keys:
            if isinstance(key, str) and key:
                # Valid key
                assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
            else:
                # Invalid key should return False
                assert cache_manager.set(key, value, CacheLevel.SYSTEM) is False

    def test_get_with_invalid_key(self, cache_manager):
        """Test getting a value with an invalid key"""
        invalid_keys = ["", None, 123, {"key": "value"}]

        for key in invalid_keys:
            assert cache_manager.get(key, CacheLevel.SYSTEM) is None
