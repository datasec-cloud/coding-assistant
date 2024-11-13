"""Tests for the cache manager functionality"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from coding_assistant.core.context.cache_manager import CacheManager, CacheLevel, CacheMode, CacheStrategy

@pytest.fixture
def base_dir(tmp_path):
    """Create a temporary base directory for tests"""
    return tmp_path / "cache_test"

@pytest.fixture
def cache_manager(base_dir):
    """Create a CacheManager instance for testing"""
    return CacheManager(base_dir)

def test_basic_cache_operations(cache_manager):
    """Test basic cache operations"""
    # Test setting values at different levels
    assert cache_manager.set("key1", "value1", CacheLevel.SYSTEM)
    assert cache_manager.set("key2", "value2", CacheLevel.DOMAIN)
    assert cache_manager.set("key3", "value3", CacheLevel.COMPONENT)
    
    # Test retrieving values
    assert cache_manager.get("key1", CacheLevel.SYSTEM) == "value1"
    assert cache_manager.get("key2", CacheLevel.DOMAIN) == "value2"
    assert cache_manager.get("key3", CacheLevel.COMPONENT) == "value3"

def test_cache_invalidation(cache_manager):
    """Test cache invalidation"""
    # Set some values
    cache_manager.set("test_key", "test_value", CacheLevel.SYSTEM)
    cache_manager.set("test_key", "test_value", CacheLevel.DOMAIN)
    
    # Test selective invalidation
    cache_manager.invalidate("test_key", CacheLevel.DOMAIN)
    assert cache_manager.get("test_key", CacheLevel.SYSTEM) == "test_value"
    assert cache_manager.get("test_key", CacheLevel.DOMAIN) is None

def test_cache_ttl(cache_manager, monkeypatch):
    """Test cache TTL functionality"""
    # Set initial value
    cache_manager.set("ttl_key", "ttl_value", CacheLevel.SYSTEM)
    assert cache_manager.get("ttl_key", CacheLevel.SYSTEM) == "ttl_value"
    
    # Mock time to advance past TTL
    future = datetime.now() + timedelta(hours=25)  # Past system TTL
    monkeypatch.setattr('datetime.datetime', lambda: future)
    
    # Value should be expired
    assert cache_manager.get("ttl_key", CacheLevel.SYSTEM) is None

def test_cache_persistence(base_dir):
    """Test cache persistence across instances"""
    # First instance
    cache1 = CacheManager(base_dir)
    cache1.set("persist_key", "persist_value", CacheLevel.SYSTEM)
    
    # Second instance
    cache2 = CacheManager(base_dir)
    assert cache2.get("persist_key", CacheLevel.SYSTEM) == "persist_value"

def test_cache_stats(cache_manager):
    """Test cache statistics"""
    # Generate some cache activity
    cache_manager.set("stats_key1", "value1", CacheLevel.SYSTEM)
    cache_manager.get("stats_key1", CacheLevel.SYSTEM)
    cache_manager.get("stats_key1", CacheLevel.SYSTEM)
    
    stats = cache_manager.get_stats()
    assert stats[CacheLevel.SYSTEM.value]["total_entries"] > 0
    assert stats[CacheLevel.SYSTEM.value]["valid_entries"] > 0

def test_cache_patterns(cache_manager):
    """Test pattern detection and recommendations"""
    # Create pattern of access
    for _ in range(5):
        cache_manager.set("pattern_key", "value", CacheLevel.SYSTEM, pattern_id="test_pattern")
        cache_manager.get("pattern_key", CacheLevel.SYSTEM)
    
    recommendations = cache_manager.get_pattern_recommendations()
    assert len(recommendations) > 0

def test_cache_error_handling(cache_manager):
    """Test cache error handling"""
    # Test invalid cache level
    with pytest.raises(KeyError):
        cache_manager.set("key", "value", "invalid_level")
    
    # Test None key
    assert cache_manager.get(None, CacheLevel.SYSTEM) is None

def test_cache_cleanup(cache_manager):
    """Test cache cleanup"""
    # Add some entries
    cache_manager.set("cleanup_key1", "value1", CacheLevel.SYSTEM)
    cache_manager.set("cleanup_key2", "value2", CacheLevel.DOMAIN)
    
    # Perform cleanup
    cleaned = cache_manager.cache.cleanup()
    assert cleaned >= 0  # Should return number of cleaned entries

def test_cache_config_update(cache_manager):
    """Test cache configuration updates"""
    updates = {
        "prefetch_threshold": 0.8,
        "mode": CacheMode.HYBRID
    }
    
    assert cache_manager.update_config(updates)
    assert cache_manager.config.prefetch_threshold == 0.8