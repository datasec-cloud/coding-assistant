# src/coding_assistant/core/context/cache_manager.py

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict
from coding_assistant.core.io.transaction import TransactionManager, OperationType
import hashlib


class CacheLevel(Enum):
    SYSTEM = "system"          # Long-lived, globally relevant data
    DOMAIN = "domain"          # Business logic and rule cache
    COMPONENT = "component"    # Short-lived, operation-specific data


class CacheMode(Enum):
    MEMORY = "memory"          # Memory-only for fastest access
    PERSISTENT = "persistent"  # Disk-based for durability
    HYBRID = "hybrid"          # Combined for optimal performance


class CacheStrategy(Enum):
    WRITE_THROUGH = "write_through"  # Immediate disk sync
    WRITE_BACK = "write_back"        # Delayed disk sync
    WRITE_AROUND = "write_around"    # Skip cache for writes


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata and patterns"""
    key: str
    value: Any
    level: CacheLevel
    created_at: datetime
    accessed_at: datetime
    hits: int = 0
    version: str = ""
    pattern_id: Optional[str] = None  # For learning system
    metadata: Dict = None
    checksum: Optional[str] = None

    def update_stats(self) -> None:
        """Update entry statistics"""
        self.accessed_at = datetime.now()
        self.hits += 1

    def calculate_checksum(self) -> str:
        """Calculate content checksum for integrity checks"""
        content = json.dumps(self.value, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class CacheConfig:
    """Enhanced cache configuration"""
    def __init__(self):
        # TTL configuration
        self.ttl = {
            CacheLevel.SYSTEM: timedelta(hours=24),
            CacheLevel.DOMAIN: timedelta(hours=12),
            CacheLevel.COMPONENT: timedelta(hours=6)
        }
        # Size limits
        self.max_entries = {
            CacheLevel.SYSTEM: 1000,
            CacheLevel.DOMAIN: 5000,
            CacheLevel.COMPONENT: 10000
        }
        # Performance settings
        self.mode = CacheMode.HYBRID
        self.strategy = CacheStrategy.WRITE_THROUGH
        self.prefetch_threshold = 0.7  # Trigger prefetch at 70% hit rate

        # Resilience configuration
        self.resilience = {
            "max_retries": 3,
            "retry_delay": 1.0,  # seconds
            "circuit_breaker": {
                "error_threshold": 5,
                "timeout": 60  # seconds
            }
        }

        # Learning configuration
        self.learning = {
            "pattern_detection": True,
            "confidence_threshold": 0.85,
            "min_pattern_occurrences": 3
        }


class HierarchicalCache:
    """Enhanced hierarchical cache with resilience and learning capabilities"""

    def __init__(self, base_dir: Path, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)

        # Memory caches with LRU implementation
        self._caches: Dict[CacheLevel, OrderedDict] = {
            level: OrderedDict() for level in CacheLevel
        }

        # Thread safety
        self._locks: Dict[CacheLevel, threading.RLock] = {
            level: threading.RLock() for level in CacheLevel
        }

        # Pattern tracking
        self._pattern_counts: Dict[str, int] = {}
        self._access_patterns: Dict[str, List[Tuple[str, datetime]]] = {}

        # Circuit breaker state
        self._error_counts: Dict[CacheLevel, int] = {
            level: 0 for level in CacheLevel
        }
        self._circuit_broken: Dict[CacheLevel, bool] = {
            level: False for level in CacheLevel
        }
        self._last_error_reset: Dict[CacheLevel, datetime] = {
            level: datetime.now() for level in CacheLevel
        }

        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize persistent storage with resilience"""
        for level in CacheLevel:
            cache_dir = self.base_dir / level.value
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to initialize storage for {level}: {e}")
                self._handle_storage_error(level)

    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is valid (not expired)"""
        if not entry:
            return False
        ttl = self.config.ttl.get(entry.level, timedelta(hours=24))
        return (datetime.now() - entry.created_at) <= ttl
    
    def _handle_storage_error(self, level: CacheLevel):
        """Handle storage initialization errors"""
        self._error_counts[level] += 1
        if self._error_counts[level] >= self.config.resilience["max_retries"]:
            self._circuit_broken[level] = True
            self.logger.critical(f"Circuit broken for {level} due to storage errors")

    def get(self, key: str, level: CacheLevel = CacheLevel.COMPONENT,
            use_fallback: bool = True) -> Optional[Any]:
        """Get value from cache with resilience and learning"""
        if not isinstance(key, str) or not key:
            self.logger.warning(f"Invalid key provided for get: {key}")
            return None

        if self._is_circuit_broken(level):
            return self._handle_circuit_broken(key, level) if use_fallback else None

        try:
            with self._locks[level]:
                # Try memory cache first
                entry = self._get_from_memory(key, level)
                if entry and self._is_valid(entry):
                    self._record_access_pattern(key, level)
                    return self._update_entry_stats(entry).value

                # Try persistent cache if in hybrid mode
                if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
                    entry = self._get_from_persistent(key, level)
                    if entry and self._is_valid(entry):
                        # Verify checksum
                        if entry.checksum != entry.calculate_checksum():
                            self.logger.error(f"Checksum mismatch for key: {key}")
                            self._handle_error(level)
                            return None
                        self._cache_in_memory(entry)
                        return self._update_entry_stats(entry).value

                # Fallback to higher levels if not found
                if use_fallback and level != CacheLevel.SYSTEM:
                    higher_level = CacheLevel.SYSTEM if level == CacheLevel.DOMAIN else CacheLevel.DOMAIN
                    return self.get(key, higher_level, use_fallback=True)

                return None

        except Exception as e:
            self.logger.error(f"Error retrieving from cache {level}: {e}")
            self._handle_error(level)
            return None

    def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.COMPONENT,
            metadata: Dict = None, pattern_id: Optional[str] = None) -> bool:
        """Set value in cache with transaction support"""
        if not isinstance(key, str) or not key:
            self.logger.warning(f"Invalid key provided for set: {key}")
            return False

        if self._is_circuit_broken(level):
            return False

        try:
            with self._locks[level]:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    level=level,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    pattern_id=pattern_id,
                    metadata=metadata or {},
                    version=datetime.now().isoformat()
                )
                entry.checksum = entry.calculate_checksum()

                self._cache_in_memory(entry)
                if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
                    self._cache_in_persistent(entry)

                # Update pattern tracking
                if pattern_id:
                    self._update_pattern_stats(pattern_id, key)

                return True

        except Exception as e:
            self.logger.error(f"Error setting cache entry {key}: {e}")
            self._handle_error(level)
            return False


    def _transactional_set(self, transaction_id: str, entry: CacheEntry):
        """Set cache entry with transaction support"""
        self.transaction_manager.add_operation(
            transaction_id,
            OperationType.CACHE_UPDATE,
            f"{entry.level.value}:{entry.key}",
            {
                "value": entry.value,
                "metadata": entry.metadata,
                "pattern_id": entry.pattern_id
            },
            backup=True
        )
        self._cache_in_memory(entry)
        if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
            self._cache_in_persistent(entry)

    def _non_transactional_set(self, entry: CacheEntry):
        """Set cache entry without transaction support"""
        self._cache_in_memory(entry)
        if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
            self._cache_in_persistent(entry)

    def _is_circuit_broken(self, level: CacheLevel) -> bool:
        """Check if circuit breaker is active for level"""
        if not self._circuit_broken[level]:
            return False

        # Check if we should reset circuit breaker
        if (datetime.now() - self._last_error_reset[level]).total_seconds() > \
           self.config.resilience["circuit_breaker"]["timeout"]:
            self._circuit_broken[level] = False
            self._error_counts[level] = 0
            self._last_error_reset[level] = datetime.now()
            return False

        return True

    def _handle_error(self, level: CacheLevel):
        """Handle cache operation errors"""
        self._error_counts[level] += 1
        if self._error_counts[level] >= self.config.resilience["circuit_breaker"]["error_threshold"]:
            self._circuit_broken[level] = True
            self._last_error_reset[level] = datetime.now()
            self.logger.warning(f"Circuit breaker opened for level {level.value}")

    def _record_access_pattern(self, key: str, level: CacheLevel):
        """Record access patterns for learning system"""
        if not self.config.learning["pattern_detection"]:
            return

        pattern_key = f"{level.value}:{key}"
        if pattern_key not in self._access_patterns:
            self._access_patterns[pattern_key] = []

        self._access_patterns[pattern_key].append((key, datetime.now()))
        self._analyze_patterns(pattern_key)

    def _analyze_patterns(self, pattern_key: str):
        """Analyze access patterns for learning system"""
        if len(self._access_patterns[pattern_key]) >= self.config.learning["min_pattern_occurrences"]:
            # Analyze temporal patterns
            accesses = self._access_patterns[pattern_key]
            intervals = [
                (accesses[i+1][1] - accesses[i][1]).total_seconds()
                for i in range(len(accesses)-1)
            ]

            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 3600:  # If average access interval is less than 1 hour
                    self._pattern_counts[pattern_key] = self._pattern_counts.get(pattern_key, 0) + 1

    def _update_pattern_stats(self, pattern_id: str, key: str):
        """Update pattern statistics"""
        if pattern_id in self._pattern_counts:
            self._pattern_counts[pattern_id] += 1
            if self._pattern_counts[pattern_id] >= self.config.learning["min_pattern_occurrences"]:
                self.logger.info(f"Pattern {pattern_id} reached threshold, enabling prefetch")

    def get_pattern_stats(self) -> Dict[str, Dict]:
        """Get pattern statistics for analysis"""
        return {
            pattern: {
                "count": count,
                "confidence": min(count / self.config.learning["min_pattern_occurrences"], 1.0)
            }
            for pattern, count in self._pattern_counts.items()
        }

    def cleanup(self) -> int:
        """Remove expired entries and update pattern statistics"""
        cleaned = 0
        try:
            for level in CacheLevel:
                with self._locks[level]:
                    # Clean memory cache
                    expired = [
                        key for key, entry in self._caches[level].items()
                        if not self._is_valid(entry)
                    ]
                    for key in expired:
                        self._caches[level].pop(key)
                    cleaned += len(expired)

                    # Clean persistent cache
                    if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
                        cleaned += self._clean_persistent_cache(level)

            return cleaned
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
            return cleaned

    def _clean_pattern_data(self):
        """Clean old pattern tracking data"""
        now = datetime.now()
        for pattern_key in list(self._access_patterns.keys()):
            # Remove patterns older than 24 hours
            self._access_patterns[pattern_key] = [
                (key, time) for key, time in self._access_patterns[pattern_key]
                if (now - time).total_seconds() < 86400
            ]
            if not self._access_patterns[pattern_key]:
                del self._access_patterns[pattern_key]

    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        stats = {}
        for level in CacheLevel:
            with self._locks[level]:
                entries = list(self._caches[level].values())
                valid_entries = [e for e in entries if self._is_valid(e)]
                stats[level.value] = {
                    "total_entries": len(entries),
                    "valid_entries": len(valid_entries),
                    "total_hits": sum(e.hits for e in entries),
                    "avg_hits": sum(e.hits for e in entries) / len(entries) if entries else 0,
                    "memory_usage": sum(len(self._serialize_entry(e)) for e in entries),
                    "circuit_breaker_status": "open" if self._circuit_broken[level] else "closed",
                    "error_count": self._error_counts[level]
                }

        stats["patterns"] = self.get_pattern_stats()
        return stats

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry for storage"""
        return json.dumps(entry.__dict__, default=str).encode()

    def _deserialize_entry(self, data: Dict) -> CacheEntry:
        """Create cache entry from serialized data"""
        return CacheEntry(
            key=data['key'],
            value=data['value'],
            level=CacheLevel(data['level']),
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            hits=data.get('hits', 0),
            version=data.get('version', ''),
            pattern_id=data.get('pattern_id'),
            metadata=data.get('metadata'),
            checksum=data.get('checksum')
        )

    def _get_from_memory(self, key: str, level: CacheLevel) -> Optional[CacheEntry]:
        """Retrieve entry from memory cache"""
        return self._caches[level].get(key)

    def _get_from_persistent(self, key: str, level: CacheLevel) -> Optional[CacheEntry]:
        """Retrieve entry from persistent storage"""
        cache_file = self.base_dir / level.value / f"{key}.cache"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open('r') as f:
                data = json.load(f)
            return self._deserialize_entry(data)
        except Exception as e:
            self.logger.error(f"Failed to load cache entry from {cache_file}: {e}")
            return None

    def _cache_in_memory(self, entry: CacheEntry):
        """Cache entry in memory with LRU eviction"""
        cache = self._caches[entry.level]
        if entry.key in cache:
            cache.pop(entry.key)
        cache[entry.key] = entry
        if len(cache) > self.config.max_entries[entry.level]:
            evicted_key, evicted_entry = cache.popitem(last=False)
            self.logger.debug(f"Evicted {evicted_key} from memory cache at level {entry.level}")

    def _cache_in_persistent(self, entry: CacheEntry):
        """Cache entry in persistent storage"""
        cache_file = self.base_dir / entry.level.value / f"{entry.key}.cache"
        try:
            with cache_file.open('w') as f:
                json.dump(entry.__dict__, f, default=str)
        except Exception as e:
            self.logger.error(f"Failed to persist cache entry {entry.key}: {e}")
            self._handle_error(entry.level)

    def _clean_persistent_cache(self, level: CacheLevel) -> int:
        """Clean expired entries from persistent storage"""
        cleaned = 0
        cache_dir = self.base_dir / level.value
        for cache_file in cache_dir.glob("*.cache"):
            try:
                with cache_file.open('r') as f:
                    data = json.load(f)
                entry = self._deserialize_entry(data)
                if not self._is_valid(entry):
                    cache_file.unlink()
                    cleaned += 1
            except Exception as e:
                self.logger.error(f"Error cleaning persistent cache {cache_file}: {e}")
                continue
        return cleaned

    def _update_entry_stats(self, entry: CacheEntry) -> CacheEntry:
        """Update statistics for a cache entry"""
        entry.update_stats()
        if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
            self._cache_in_persistent(entry)
        return entry

    def _handle_circuit_broken(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Enhanced circuit breaker with recovery"""
        # Placeholder for asynchronous recovery handling
        # Implement as needed, e.g., fallback mechanisms or notifications
        self.logger.warning(f"Circuit breaker is open for level {level}. Cannot retrieve key {key}.")
        return None

    def _clean_pattern_data(self):
        """Clean old pattern tracking data"""
        now = datetime.now()
        for pattern_key in list(self._access_patterns.keys()):
            # Remove patterns older than 24 hours
            self._access_patterns[pattern_key] = [
                (key, time) for key, time in self._access_patterns[pattern_key]
                if (now - time).total_seconds() < 86400
            ]
            if not self._access_patterns[pattern_key]:
                del self._access_patterns[pattern_key]


class CacheManager:
    """Manager class coordinating cache operations with monitoring"""

    def __init__(self, base_dir: Path, transaction_manager: Optional[TransactionManager] = None):
        self.base_dir = Path(base_dir)
        self.config = CacheConfig()

        # Initialize cache
        self.cache = HierarchicalCache(
            self.base_dir / "cache",
            self.config,
            transaction_manager
        )

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Start background threads for cleanup and monitoring
        cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        cleanup_thread.start()

        monitor_thread = threading.Thread(target=self._monitor_health, daemon=True)
        monitor_thread.start()

    def get(self, key: str, level: CacheLevel = CacheLevel.COMPONENT) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key, level)

    def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.COMPONENT,
            metadata: Dict = None, pattern_id: Optional[str] = None) -> bool:
        """Set value in cache"""
        return self.cache.set(key, value, level, metadata, pattern_id)

    def invalidate(self, key: str, level: Optional[CacheLevel] = None) -> None:
        """Invalidate cache entries"""
        if level:
            with self._locks[level]:
                self._caches[level].pop(key, None)
                if self.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
                    self._remove_persistent_entry(key, level)
        else:
            for cache_level in CacheLevel:
                self.invalidate(key, cache_level)
    
    def _remove_persistent_entry(self, key: str, level: CacheLevel):
        """Remove entry from persistent storage"""
        try:
            path = self.cache.base_dir / level.value / f"{key}.cache"
            if path.exists():
                path.unlink()
        except Exception as e:
            self.logger.error(f"Error removing persistent entry {key}: {e}")

    def clear_level(self, level: CacheLevel) -> None:
        """Clear all entries at a specific cache level"""
        with self.cache._locks[level]:
            # Clear memory cache
            self.cache._caches[level].clear()

            # Clear persistent cache
            if self.cache.config.mode in [CacheMode.PERSISTENT, CacheMode.HYBRID]:
                cache_dir = self.cache.base_dir / level.value
                for cache_file in cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        self.logger.error(f"Error removing cache file {cache_file}: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()

    def _periodic_cleanup(self):
        """Periodic cleanup of expired cache entries"""
        while True:
            try:
                cleaned = self.cache.cleanup()
                if cleaned > 0:
                    self.logger.info(f"Cleaned {cleaned} expired cache entries")
                threading.Event().wait(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                threading.Event().wait(60)  # Retry after a minute on error

    def _monitor_health(self):
        """Monitor cache health and performance"""
        while True:
            try:
                stats = self.cache.get_stats()
                for level, level_stats in stats.items():
                    if level != "patterns":  # Skip pattern stats
                        self._check_health_metrics(level, level_stats)
                threading.Event().wait(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                threading.Event().wait(60)

    def _check_health_metrics(self, level: str, stats: Dict):
        """Check cache health metrics and log warnings"""
        # Check circuit breaker status
        if stats["circuit_breaker_status"] == "open":
            self.logger.warning(f"Circuit breaker open for {level}")

        # Check hit ratios
        if stats["total_entries"] > 0:
            hit_ratio = stats["total_hits"] / stats["total_entries"]
            if hit_ratio < 0.5:  # Less than 50% hit ratio
                self.logger.warning(f"Low hit ratio ({hit_ratio:.2f}) for {level}")

        # Check memory usage
        memory_threshold = 1024 * 1024 * 100  # 100MB
        if stats["memory_usage"] > memory_threshold:
            self.logger.warning(f"High memory usage for {level}: {stats['memory_usage']} bytes")

    def update_config(self, updates: Dict) -> bool:
        """Update cache configuration"""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif key in self.config.ttl:
                    self.config.ttl[key] = value
                elif key in self.config.max_entries:
                    self.config.max_entries[key] = value
                elif key in self.config.resilience:
                    self.config.resilience[key] = value
                elif key in self.config.learning:
                    self.config.learning[key] = value
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating cache config: {e}")
            return False

    def preload_pattern(self, pattern_id: str, data: Dict[str, Any],
                       level: CacheLevel = CacheLevel.COMPONENT) -> bool:
        """Preload data for known access pattern"""
        try:
            success = True
            for key, value in data.items():
                if not self.set(key, value, level, pattern_id=pattern_id):
                    success = False
            return success
        except Exception as e:
            self.logger.error(f"Error preloading pattern {pattern_id}: {e}")
            return False

    def get_pattern_recommendations(self) -> Dict[str, float]:
        """Get recommended patterns based on learned behavior"""
        pattern_stats = self.cache.get_pattern_stats()
        return {
            pattern: stats["confidence"]
            for pattern, stats in pattern_stats.items()
            if stats["confidence"] >= self.cache.config.learning["confidence_threshold"]
        }
