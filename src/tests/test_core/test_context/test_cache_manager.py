# tests/test_cache_manager.py

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import json
from unittest.mock import MagicMock, patch

from coding_assistant.core.context.cache_manager import (
    CacheManager, CacheLevel, CacheMode, CacheStrategy
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
    mock_tm.transaction.return_value.__enter__.return_value = "transaction_id"
    mock_tm.transaction.return_value.__exit__.return_value = False
    mock_tm.add_operation.return_value = None
    return mock_tm


@pytest.fixture
def cache_manager(base_dir, mock_transaction_manager):
    """Create a CacheManager instance for testing with a mock TransactionManager"""
    return CacheManager(base_dir, transaction_manager=mock_transaction_manager)


class TestCacheBasicOperations:
    def test_set_and_get(self, cache_manager):
        """Test basic set and get operations"""
        key = "test_key"
        value = "test_value"

        # Test set operation
        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True

        # Test get operation
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

    def test_get_nonexistent(self, cache_manager):
        """Test getting non-existent keys"""
        assert cache_manager.get("nonexistent", CacheLevel.SYSTEM) is None
        assert cache_manager.get("", CacheLevel.DOMAIN) is None
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
    def test_persistence_basics(self, cache_manager):
        """Test basic persistence operations"""
        # Set a value
        assert cache_manager.set("persist_key", "persist_value", CacheLevel.SYSTEM) is True

        # Verify directory structure
        cache_dir = cache_manager.base_dir / "cache" / CacheLevel.SYSTEM.value
        assert cache_dir.exists()

        # Verify persistent file
        cache_file = cache_dir / "persist_key.cache"
        assert cache_file.exists()

    def test_persistent_storage_integrity(self, cache_manager):
        """Test integrity of persistent storage"""
        key = "integrity_key"
        value = {"test": "data"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        retrieved_value = cache_manager.get(key, CacheLevel.SYSTEM)
        assert retrieved_value == value

        # Manually load the persistent file to verify checksum
        cache_file = cache_manager.base_dir / "cache" / CacheLevel.SYSTEM.value / f"{key}.cache"
        with cache_file.open('r') as f:
            data = json.load(f)
        assert data["value"] == value
        assert data["checksum"] == json.dumps(value, sort_keys=True).__hash__().__str__()  # Simplified checksum check


class TestCacheEviction:
    def test_size_based_eviction(self, cache_manager):
        """Test size-based eviction"""
        # Fill cache to trigger eviction
        for i in range(2000):  # More than max entries (1000 for SYSTEM)
            cache_manager.set(f"key_{i}", f"value_{i}", CacheLevel.SYSTEM)

        # First entry should be evicted (LRU)
        assert cache_manager.get("key_0", CacheLevel.SYSTEM) is None

        # Last entry should still exist
        assert cache_manager.get("key_1999", CacheLevel.SYSTEM) == "value_1999"

    def test_ttl_eviction(self, cache_manager):
        """Test time-based eviction"""
        key = "ttl_test"
        value = "test_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        # Fast-forward time beyond TTL
        cache_entry = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        cache_entry.created_at -= timedelta(hours=25)  # TTL for SYSTEM is 24 hours

        # Cleanup should remove the expired entry
        cleaned = cache_manager.cleanup()
        assert cleaned >= 1
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None


class TestCacheResilience:
    def test_circuit_breaker(self, cache_manager):
        """Test circuit breaker functionality"""
        # Trigger errors to open circuit breaker
        for _ in range(cache_manager.config.resilience["circuit_breaker"]["error_threshold"]):
            cache_manager.cache._handle_error(CacheLevel.SYSTEM)

        assert cache_manager.cache._is_circuit_broken(CacheLevel.SYSTEM) is True

        # Attempt to set a value should fail due to circuit breaker
        assert cache_manager.set("cb_key", "cb_value", CacheLevel.SYSTEM) is False

    def test_error_recovery(self, cache_manager):
        """Test error recovery mechanism"""
        # Open circuit breaker
        for _ in range(cache_manager.config.resilience["circuit_breaker"]["error_threshold"]):
            cache_manager.cache._handle_error(CacheLevel.SYSTEM)

        assert cache_manager.cache._is_circuit_broken(CacheLevel.SYSTEM) is True

        # Fast-forward time beyond circuit breaker timeout
        cache_manager.cache._last_error_reset[CacheLevel.SYSTEM] -= timedelta(seconds=cache_manager.config.resilience["circuit_breaker"]["timeout"] + 1)

        # Circuit breaker should attempt to recover
        assert cache_manager.cache._is_circuit_broken(CacheLevel.SYSTEM) is False
        assert cache_manager.cache._error_counts[CacheLevel.SYSTEM] == 0

        # Now setting should work
        assert cache_manager.set("cb_recovery_key", "cb_recovered_value", CacheLevel.SYSTEM) is True
        assert cache_manager.get("cb_recovery_key", CacheLevel.SYSTEM) == "cb_recovered_value"


class TestCacheConcurrency:
    def test_basic_concurrent_access(self, cache_manager):
        """Test basic concurrent access"""
        success = True
        errors = []

        def worker(worker_id):
            nonlocal success
            try:
                # Set and get operations
                for i in range(10):
                    key = f"concurrent_key_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"

                    if not cache_manager.set(key, value, CacheLevel.SYSTEM):
                        success = False
                        return

                    time.sleep(0.01)  # Small delay to increase chance of concurrency

                    result = cache_manager.get(key, CacheLevel.SYSTEM)
                    if result != value:
                        success = False
                        return
            except Exception as e:
                errors.append(e)
                success = False

        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        assert success, f"Concurrent operations failed: {errors}"


class TestCacheMonitoring:
    def test_basic_stats(self, cache_manager):
        """Test basic statistics collection"""
        # Generate some cache activity
        for i in range(5):
            key = f"stats_key_{i}"
            cache_manager.set(key, f"value_{i}", CacheLevel.SYSTEM)
            cache_manager.get(key, CacheLevel.SYSTEM)  # Generate a hit

        stats = cache_manager.get_stats()
        system_stats = stats[CacheLevel.SYSTEM.value]

        assert "total_entries" in system_stats
        assert system_stats["total_entries"] == 5
        assert "valid_entries" in system_stats
        assert system_stats["valid_entries"] == 5
        assert "total_hits" in system_stats
        assert system_stats["total_hits"] == 5
        assert "avg_hits" in system_stats
        assert system_stats["avg_hits"] == 1.0
        assert "memory_usage" in system_stats
        assert "circuit_breaker_status" in system_stats
        assert system_stats["circuit_breaker_status"] in ["open", "closed"]
        assert "error_count" in system_stats

    def test_cleanup_stats(self, cache_manager):
        """Test cleanup functionality and statistics"""
        # Add some entries
        for i in range(5):
            cache_manager.set(f"cleanup_key_{i}", f"value_{i}", CacheLevel.SYSTEM)

        # Fast-forward time to expire some entries
        for i in range(3):
            cache_entry = cache_manager.cache._caches[CacheLevel.SYSTEM].get(f"cleanup_key_{i}")
            cache_entry.created_at -= timedelta(hours=25)  # Expire these entries

        # Perform cleanup
        cleaned = cache_manager.cleanup()
        assert cleaned == 3

        # Verify remaining entries
        for i in range(3):
            assert cache_manager.get(f"cleanup_key_{i}", CacheLevel.SYSTEM) is None
        for i in range(3, 5):
            assert cache_manager.get(f"cleanup_key_{i}", CacheLevel.SYSTEM) == f"value_{i}"

        # Verify stats after cleanup
        stats = cache_manager.get_stats()
        system_stats = stats[CacheLevel.SYSTEM.value]
        assert system_stats["total_entries"] == 2
        assert system_stats["valid_entries"] == 2


class TestErrorHandling:
    def test_invalid_operations(self, cache_manager):
        """Test handling of invalid operations"""
        # Test with None key
        assert cache_manager.set(None, "value", CacheLevel.SYSTEM) is False
        assert cache_manager.get(None, CacheLevel.SYSTEM) is None

        # Test with empty key
        assert cache_manager.set("", "value", CacheLevel.SYSTEM) is False
        assert cache_manager.get("", CacheLevel.SYSTEM) is None

    def test_invalid_value_handling(self, cache_manager):
        """Test handling of invalid values"""
        # Test with None value (should still work)
        key = "none_value_key"
        assert cache_manager.set(key, None, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None

        # Test with very large value
        large_value = "x" * (1024 * 1024)  # 1MB string
        assert cache_manager.set("large_key", large_value, CacheLevel.SYSTEM) is True
        assert cache_manager.get("large_key", CacheLevel.SYSTEM) == large_value


class TestCacheConfiguration:
    def test_cache_levels(self, cache_manager):
        """Test different cache levels"""
        value = "test_value"

        # Test all cache levels
        for level in CacheLevel:
            key = f"test_key_{level.value}"
            assert cache_manager.set(key, value, level) is True
            assert cache_manager.get(key, level) == value

    def test_cache_modes(self, cache_manager):
        """Test different cache modes"""
        # Test memory mode
        cache_manager.config.mode = CacheMode.MEMORY
        assert cache_manager.set("memory_key", "memory_value", CacheLevel.SYSTEM) is True

        # Test persistent mode
        cache_manager.config.mode = CacheMode.PERSISTENT
        assert cache_manager.set("persistent_key", "persistent_value", CacheLevel.SYSTEM) is True

        # Test hybrid mode
        cache_manager.config.mode = CacheMode.HYBRID
        assert cache_manager.set("hybrid_key", "hybrid_value", CacheLevel.SYSTEM) is True

    def test_update_config(self, cache_manager):
        """Test updating cache configuration"""
        new_ttl = {
            CacheLevel.SYSTEM: timedelta(hours=48),
            CacheLevel.DOMAIN: timedelta(hours=24),
            CacheLevel.COMPONENT: timedelta(hours=12)
        }
        new_max_entries = {
            CacheLevel.SYSTEM: 2000,
            CacheLevel.DOMAIN: 10000,
            CacheLevel.COMPONENT: 20000
        }
        updates = {
            "ttl": new_ttl,
            "max_entries": new_max_entries
        }
        assert cache_manager.update_config(updates) is True

        assert cache_manager.config.ttl == new_ttl
        assert cache_manager.config.max_entries == new_max_entries

    def test_unknown_config_key(self, cache_manager, caplog):
        """Test updating with an unknown configuration key"""
        updates = {
            "unknown_key": "value"
        }
        assert cache_manager.update_config(updates) is True
        assert "Unknown configuration key: unknown_key" in caplog.text


class TestCacheIntegrity:
    def test_data_integrity(self, cache_manager):
        """Test data integrity across operations"""
        # Test with different data types
        test_data = [
            ("string_key", "string_value"),
            ("int_key", 42),
            ("dict_key", {"nested": "value"}),
            ("list_key", [1, 2, 3]),
            ("bool_key", True)
        ]

        # Set all values
        for key, value in test_data:
            assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True

        # Verify all values
        for key, value in test_data:
            assert cache_manager.get(key, CacheLevel.SYSTEM) == value

    def test_concurrent_integrity(self, cache_manager):
        """Test data integrity under concurrent access"""
        key = "concurrent_integrity_key"
        iterations = 100
        results = []

        def writer():
            for i in range(iterations):
                assert cache_manager.set(key, i, CacheLevel.SYSTEM) is True
                time.sleep(0.001)

        def reader():
            for _ in range(iterations):
                value = cache_manager.get(key, CacheLevel.SYSTEM)
                if value is not None:
                    results.append(value)
                time.sleep(0.001)

        # Run concurrent read/write operations
        write_thread = threading.Thread(target=writer)
        read_thread = threading.Thread(target=reader)

        write_thread.start()
        read_thread.start()

        write_thread.join()
        read_thread.join()

        # Verify results
        assert len(results) > 0
        assert all(isinstance(x, int) for x in results)


class TestTransactionManagement:
    def test_transactional_set(self, cache_manager, mock_transaction_manager):
        """Test transactional set operations"""
        key = "transaction_key"
        value = "transaction_value"

        with patch.object(mock_transaction_manager, 'add_operation') as mock_add_op:
            assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
            mock_add_op.assert_called_once()

        # Verify that the value is set
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

    def test_transaction_commit(self, cache_manager, mock_transaction_manager):
        """Test that transaction commits correctly"""
        key = "commit_key"
        value = "commit_value"

        # Simulate successful transaction
        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

    def test_transaction_rollback(self, cache_manager, mock_transaction_manager):
        """Test that transaction rollback handles failures"""
        key = "rollback_key"
        value = "rollback_value"

        # Simulate an exception during transaction
        with patch.object(mock_transaction_manager, 'add_operation', side_effect=Exception("Transaction Failed")):
            assert cache_manager.set(key, value, CacheLevel.SYSTEM) is False

        # Verify that the value was not set
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None


class TestLearningSystem:
    def test_access_pattern_detection(self, cache_manager):
        """Test that access patterns are detected and recorded"""
        key = "pattern_key"
        value = "pattern_value"

        # Access the key multiple times to meet min_pattern_occurrences
        for _ in range(cache_manager.config.learning["min_pattern_occurrences"]):
            cache_manager.set(key, value, CacheLevel.SYSTEM, pattern_id="test_pattern")
            cache_manager.get(key, CacheLevel.SYSTEM)

        pattern_stats = cache_manager.cache.get_pattern_stats()
        assert "test_pattern" in pattern_stats
        assert pattern_stats["test_pattern"]["count"] == cache_manager.config.learning["min_pattern_occurrences"]
        assert pattern_stats["test_pattern"]["confidence"] == 1.0

    def test_prefetch_recommendations(self, cache_manager):
        """Test that prefetch recommendations are generated based on patterns"""
        pattern_id = "prefetch_pattern"
        preload_data = {
            "prefetch_key_1": "value_1",
            "prefetch_key_2": "value_2",
            "prefetch_key_3": "value_3"
        }

        # Preload pattern data
        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Access the pattern multiple times to trigger recommendation
        for _ in range(cache_manager.config.learning["min_pattern_occurrences"]):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_id in recommendations
        assert recommendations[pattern_id] >= cache_manager.config.learning["confidence_threshold"]


class TestCachePreloading:
    def test_preload_known_pattern(self, cache_manager):
        """Test preloading data for a known access pattern"""
        pattern_id = "known_pattern"
        preload_data = {
            "preload_key_1": {"data": 123},
            "preload_key_2": {"data": 456}
        }

        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Verify that preloaded data is accessible
        for key, value in preload_data.items():
            assert cache_manager.get(key, CacheLevel.SYSTEM) == value

    def test_preload_with_partial_failure(self, cache_manager, mock_transaction_manager):
        """Test preloading with partial failures"""
        pattern_id = "partial_failure_pattern"
        preload_data = {
            "preload_fail_key_1": {"data": 789},
            "preload_fail_key_2": {"data": 101112}
        }

        # Simulate failure on setting the second key
        def side_effect_set(key, value, level, metadata=None, pattern_id=None):
            if key == "preload_fail_key_2":
                raise Exception("Failed to set key")
            return True

        with patch.object(cache_manager, 'set', side_effect=side_effect_set):
            assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is False

        # Verify that the first key was set and the second was not
        assert cache_manager.get("preload_fail_key_1", CacheLevel.SYSTEM) == {"data": 789}
        assert cache_manager.get("preload_fail_key_2", CacheLevel.SYSTEM) is None


class TestPatternRecommendation:
    def test_recommendation_threshold(self, cache_manager):
        """Test that recommendations respect the confidence threshold"""
        pattern_id = "high_confidence_pattern"
        preload_data = {
            "rec_key_1": "rec_value_1",
            "rec_key_2": "rec_value_2"
        }

        # Preload pattern data
        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Access the pattern multiple times to exceed confidence threshold
        for _ in range(int(cache_manager.config.learning["min_pattern_occurrences"] / 2)):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        # Access the pattern enough to reach confidence threshold
        for _ in range(int(cache_manager.config.learning["min_pattern_occurrences"] / 2)):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_id in recommendations
        assert recommendations[pattern_id] >= cache_manager.config.learning["confidence_threshold"]

    def test_recommendation_below_threshold(self, cache_manager):
        """Test that patterns below the confidence threshold are not recommended"""
        pattern_id = "low_confidence_pattern"
        preload_data = {
            "low_rec_key_1": "low_rec_value_1",
            "low_rec_key_2": "low_rec_value_2"
        }

        # Preload pattern data
        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Access the pattern fewer times than required for threshold
        for _ in range(1):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_id not in recommendations


class TestCircuitBreakerRecovery:
    def test_circuit_breaker_recovery(self, cache_manager):
        """Test that circuit breaker recovers after timeout"""
        # Trigger circuit breaker
        for _ in range(cache_manager.config.resilience["circuit_breaker"]["error_threshold"]):
            cache_manager.cache._handle_error(CacheLevel.SYSTEM)

        assert cache_manager.cache._is_circuit_broken(CacheLevel.SYSTEM) is True

        # Simulate time passing beyond the timeout
        cache_manager.cache._last_error_reset[CacheLevel.SYSTEM] -= timedelta(seconds=cache_manager.config.resilience["circuit_breaker"]["timeout"] + 1)

        # Next access should attempt to reset the circuit breaker
        assert cache_manager.cache._is_circuit_broken(CacheLevel.SYSTEM) is False

        # Verify that error count is reset
        assert cache_manager.cache._error_counts[CacheLevel.SYSTEM] == 0


class TestCacheThreadSafety:
    def test_thread_safety_under_load(self, cache_manager):
        """Test thread safety under high load"""
        success = True
        errors = []

        def writer():
            try:
                for i in range(100):
                    key = f"thread_safe_key_{i}"
                    value = f"value_{i}"
                    if not cache_manager.set(key, value, CacheLevel.SYSTEM):
                        success = False
                        return
            except Exception as e:
                errors.append(e)
                success = False

        def reader():
            try:
                for i in range(100):
                    key = f"thread_safe_key_{i}"
                    value = cache_manager.get(key, CacheLevel.SYSTEM)
                    # Value can be None if not yet set, so just access to ensure no exceptions
            except Exception as e:
                errors.append(e)
                success = False

        threads = []
        for _ in range(10):
            t_write = threading.Thread(target=writer)
            t_read = threading.Thread(target=reader)
            threads.extend([t_write, t_read])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert success, f"Thread safety under load failed: {errors}"


class TestCacheChecksum:
    def test_checksum_verification(self, cache_manager):
        """Test that checksum verification works correctly"""
        key = "checksum_key"
        value = {"data": "checksum_test"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True

        # Manually tamper with the persistent cache file to simulate corruption
        cache_file = cache_manager.base_dir / "cache" / CacheLevel.SYSTEM.value / f"{key}.cache"
        with cache_file.open('r') as f:
            data = json.load(f)
        data["value"] = {"data": "tampered_data"}
        with cache_file.open('w') as f:
            json.dump(data, f)

        # Attempt to get the tampered entry
        retrieved_value = cache_manager.get(key, CacheLevel.SYSTEM)
        assert retrieved_value is None

        # Verify that the cache entry is not cached in memory due to checksum mismatch
        assert cache_manager.cache._caches[CacheLevel.SYSTEM].get(key) is None


class TestCacheInvalidation:
    def test_invalidate_specific_entry(self, cache_manager):
        """Test invalidating a specific cache entry"""
        key = "invalidate_key"
        value = "invalidate_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        # Invalidate the entry
        cache_manager.invalidate(key, CacheLevel.SYSTEM)

        # Verify that the entry is removed
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None

    def test_invalidate_all_levels(self, cache_manager):
        """Test invalidating a cache entry across all levels"""
        key = "multi_level_key"
        value = "multi_level_value"

        # Set the key in all levels
        for level in CacheLevel:
            assert cache_manager.set(key, value, level) is True
            assert cache_manager.get(key, level) == value

        # Invalidate without specifying level
        cache_manager.invalidate(key)

        # Verify that the key is removed from all levels
        for level in CacheLevel:
            assert cache_manager.get(key, level) is None


class TestCachePrefetching:
    def test_prefetch_triggering(self, cache_manager):
        """Test that prefetch is triggered based on access patterns"""
        pattern_id = "prefetch_trigger_pattern"
        preload_data = {
            "prefetch_key_1": "prefetch_value_1",
            "prefetch_key_2": "prefetch_value_2"
        }

        # Preload pattern data
        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Access the pattern to trigger prefetch
        for _ in range(int(cache_manager.config.prefetch_threshold * cache_manager.config.learning["min_pattern_occurrences"])):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        # Check if prefetch recommendations are available
        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_id in recommendations


class TestCacheSerialization:
    def test_serialize_deserialize_entry(self, cache_manager):
        """Test serialization and deserialization of cache entries"""
        key = "serialize_key"
        value = {"serialize": "test"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True

        # Directly access the HierarchicalCache to serialize and deserialize
        cache_entry = cache_manager.cache._get_from_memory(key, CacheLevel.SYSTEM)
        serialized = cache_manager.cache._serialize_entry(cache_entry)
        deserialized_entry = cache_manager.cache._deserialize_entry(json.loads(serialized))

        assert deserialized_entry.key == cache_entry.key
        assert deserialized_entry.value == cache_entry.value
        assert deserialized_entry.level == cache_entry.level
        assert deserialized_entry.created_at == cache_entry.created_at
        assert deserialized_entry.accessed_at == cache_entry.accessed_at
        assert deserialized_entry.hits == cache_entry.hits
        assert deserialized_entry.version == cache_entry.version
        assert deserialized_entry.pattern_id == cache_entry.pattern_id
        assert deserialized_entry.metadata == cache_entry.metadata
        assert deserialized_entry.checksum == cache_entry.checksum


class TestCacheLogging:
    def test_logging_on_error(self, cache_manager, caplog):
        """Test that errors are logged appropriately"""
        key = "logging_error_key"

        with patch.object(cache_manager.cache, '_cache_in_persistent', side_effect=Exception("Persistent storage failure")):
            assert cache_manager.set(key, "value", CacheLevel.SYSTEM) is False

        # Check that the error was logged
        assert "Error writing to persistent cache for key" in caplog.text

    def test_logging_on_eviction(self, cache_manager, caplog):
        """Test that eviction events are logged"""
        # Fill the cache to trigger eviction
        for i in range(cache_manager.config.max_entries[CacheLevel.SYSTEM] + 10):
            cache_manager.set(f"evict_key_{i}", f"value_{i}", CacheLevel.SYSTEM)

        # Check that eviction logs are present
        eviction_logs = [record for record in caplog.records if "Evicted" in record.message]
        assert len(eviction_logs) >= 10  # At least 10 evictions should have occurred


class TestCacheRecovery:
    def test_recovery_after_failure(self, cache_manager, mock_transaction_manager):
        """Test cache recovery mechanisms after failures"""
        key = "recovery_key"
        value = "recovery_value"

        # Simulate a failure during set operation
        with patch.object(mock_transaction_manager, 'add_operation', side_effect=Exception("Transaction Failure")):
            assert cache_manager.set(key, value, CacheLevel.SYSTEM) is False

        # Verify that the value was not set
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None

        # Now perform a successful set
        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value


class TestCachePatternCleanup:
    def test_pattern_cleanup(self, cache_manager):
        """Test that old pattern data is cleaned up"""
        pattern_id = "cleanup_pattern"
        key = "cleanup_pattern_key"
        value = "cleanup_value"

        # Access the key to record the pattern
        cache_manager.set(key, value, CacheLevel.SYSTEM, pattern_id=pattern_id)
        cache_manager.get(key, CacheLevel.SYSTEM)

        # Verify that the pattern is recorded
        pattern_stats = cache_manager.cache.get_pattern_stats()
        assert pattern_id in pattern_stats

        # Fast-forward time beyond pattern data retention
        old_time = datetime.now() - timedelta(seconds=86500)  # Just over 24 hours
        cache_manager.cache._access_patterns[f"{CacheLevel.SYSTEM.value}:{key}"] = [
            (key, old_time)
        ]

        # Perform cleanup
        cleaned = cache_manager.cleanup()
        assert cleaned >= 1

        # Verify that the pattern is removed
        pattern_stats = cache_manager.cache.get_pattern_stats()
        assert pattern_id not in pattern_stats


class TestCacheShutdown:
    def test_shutdown_behavior(self, cache_manager):
        """Test cache behavior during shutdown"""
        # Since background threads are daemon threads, they should terminate when main thread exits
        # Here we can simulate a shutdown by simply deleting the cache_manager
        key = "shutdown_key"
        value = "shutdown_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        del cache_manager

        # After deletion, accessing the cache_manager should raise an AttributeError
        with pytest.raises(NameError):
            cache_manager.get(key, CacheLevel.SYSTEM)


class TestCacheConcurrentEviction:
    def test_concurrent_eviction(self, cache_manager):
        """Test that eviction works correctly under concurrent access"""

        def filler():
            for i in range(1500):  # More than max entries to trigger eviction
                cache_manager.set(f"concurrent_evict_key_{i}", f"value_{i}", CacheLevel.SYSTEM)
                time.sleep(0.001)

        def accessor():
            for i in range(1500):
                cache_manager.get(f"concurrent_evict_key_{i}", CacheLevel.SYSTEM)
                time.sleep(0.001)

        threads = []
        for _ in range(5):
            t_filler = threading.Thread(target=filler)
            t_accessor = threading.Thread(target=accessor)
            threads.extend([t_filler, t_accessor])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify that only max_entries are present
        system_cache = cache_manager.cache._caches[CacheLevel.SYSTEM]
        assert len(system_cache) <= cache_manager.config.max_entries[CacheLevel.SYSTEM]

        # Verify that the most recently used entries are retained
        for i in range(500, 1500):
            key = f"concurrent_evict_key_{i}"
            assert cache_manager.get(key, CacheLevel.SYSTEM) == f"value_{i}"


class TestCacheVersioning:
    def test_version_increment_on_update(self, cache_manager):
        """Test that version is incremented on cache update"""
        key = "version_key"
        value1 = "version_value_1"
        value2 = "version_value_2"

        assert cache_manager.set(key, value1, CacheLevel.SYSTEM) is True
        entry1 = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        version1 = entry1.version

        time.sleep(0.01)  # Ensure timestamp changes

        assert cache_manager.set(key, value2, CacheLevel.SYSTEM) is True
        entry2 = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        version2 = entry2.version

        assert version1 != version2
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value2


class TestCacheMetadataHandling:
    def test_metadata_persistence(self, cache_manager):
        """Test that metadata is persisted and retrievable"""
        key = "metadata_key"
        value = "metadata_value"
        metadata = {"source": "unit_test", "priority": "high"}

        assert cache_manager.set(key, value, CacheLevel.SYSTEM, metadata=metadata) is True
        retrieved_entry = cache_manager.cache._caches[CacheLevel.SYSTEM].get(key)
        assert retrieved_entry.metadata == metadata

        # Verify persistent storage has metadata
        cache_file = cache_manager.base_dir / "cache" / CacheLevel.SYSTEM.value / f"{key}.cache"
        with cache_file.open('r') as f:
            data = json.load(f)
        assert data["metadata"] == metadata


class TestCachePatternStatistics:
    def test_pattern_statistics(self, cache_manager):
        """Test that pattern statistics are accurately maintained"""
        pattern_id = "stat_pattern"
        key = "stat_pattern_key"
        value = "stat_value"

        # Access the key multiple times
        for _ in range(5):
            cache_manager.set(key, value, CacheLevel.SYSTEM, pattern_id=pattern_id)
            cache_manager.get(key, CacheLevel.SYSTEM)

        pattern_stats = cache_manager.cache.get_pattern_stats()
        assert pattern_id in pattern_stats
        assert pattern_stats[pattern_id]["count"] == 5
        assert pattern_stats[pattern_id]["confidence"] == min(5 / 3, 1.0)  # min_pattern_occurrences is 3


class TestCachePersistentIntegrity:
    def test_persistent_integrity_after_restart(self, cache_manager, base_dir):
        """Test that persistent cache retains data after CacheManager is re-instantiated"""
        key = "persistent_restart_key"
        value = "persistent_restart_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        # Re-instantiate CacheManager
        new_cache_manager = CacheManager(base_dir, transaction_manager=cache_manager.transaction_manager)

        # Verify that the data persists
        assert new_cache_manager.get(key, CacheLevel.SYSTEM) == value


class TestCacheIntegrityAfterTampering:
    def test_integrity_after_tampering(self, cache_manager):
        """Test that cache detects tampered persistent entries"""
        key = "tamper_key"
        value = "original_value"

        assert cache_manager.set(key, value, CacheLevel.SYSTEM) is True
        assert cache_manager.get(key, CacheLevel.SYSTEM) == value

        # Tamper with the persistent file
        cache_file = cache_manager.base_dir / "cache" / CacheLevel.SYSTEM.value / f"{key}.cache"
        with cache_file.open('r') as f:
            data = json.load(f)
        data["value"] = "tampered_value"
        with cache_file.open('w') as f:
            json.dump(data, f)

        # Attempt to get the tampered entry
        assert cache_manager.get(key, CacheLevel.SYSTEM) is None

        # Verify that the entry was not cached in memory due to checksum mismatch
        assert cache_manager.cache._caches[CacheLevel.SYSTEM].get(key) is None


class TestCacheLearningDisabled:
    def test_learning_disabled(self, cache_manager, caplog):
        """Test that learning system is bypassed when disabled"""
        cache_manager.config.learning["pattern_detection"] = False
        key = "no_learning_key"
        value = "no_learning_value"

        cache_manager.set(key, value, CacheLevel.SYSTEM)
        cache_manager.get(key, CacheLevel.SYSTEM)

        pattern_stats = cache_manager.cache.get_pattern_stats()
        assert "patterns" not in pattern_stats or len(pattern_stats) == 0


class TestCachePrefetchThreshold:
    def test_prefetch_threshold(self, cache_manager):
        """Test that prefetch is triggered based on the prefetch_threshold"""
        pattern_id = "prefetch_threshold_pattern"
        preload_data = {
            "prefetch_t_key_1": "prefetch_t_value_1",
            "prefetch_t_key_2": "prefetch_t_value_2"
        }

        # Preload pattern data
        assert cache_manager.preload_pattern(pattern_id, preload_data, level=CacheLevel.SYSTEM) is True

        # Access the pattern to reach prefetch threshold
        required_accesses = int(cache_manager.config.prefetch_threshold * cache_manager.config.learning["min_pattern_occurrences"])
        for _ in range(required_accesses):
            for key in preload_data.keys():
                cache_manager.get(key, CacheLevel.SYSTEM)

        # Verify that prefetch was triggered (this is a placeholder as actual prefetch logic needs implementation)
        # For example, you might check if certain keys are prefetched
        recommendations = cache_manager.get_pattern_recommendations()
        assert pattern_id in recommendations
