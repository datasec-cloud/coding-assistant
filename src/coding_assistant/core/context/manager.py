# src/coding_assistant/core/context/manager.py
from typing import Dict, Optional, List, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging
import yaml
from enum import Enum

from coding_assistant.core.context.cache_manager import CacheManager, CacheLevel, CacheMode
from coding_assistant.core.context.version_manager import ContextVersionManager, VersionType, VersionInfo
from coding_assistant.core.context.validation import ContextValidator, ValidationError
from coding_assistant.core.io.handler import IOHandler
from coding_assistant.core.io.transaction import TransactionManager, OperationType
from coding_assistant.core.recovery.recovery_manager import RecoveryManager, Alert, AlertSeverity, RecoveryStatus
import dataclasses
import asyncio


class ContextCategory(str, Enum):
    SYSTEM = "system"
    DOMAIN = "domain"
    COMPONENT = "component"
    MODIFICATION = "modification"


class ContextManager:
    """Enhanced context manager with recovery integration"""

    def __init__(self, session_id: str, base_dir: Optional[Path] = None):
        self.session_id = session_id
        self.base_dir = Path(base_dir) if base_dir else Path("data")
        self.context_dir = self.base_dir / "context" / session_id
        self.logger = logging.getLogger(__name__)

        # Initialize transaction manager first as it's required by other components
        self.transaction_manager = TransactionManager(self.base_dir / "transactions")

        # Initialize component managers with transaction support
        self.cache_manager = CacheManager(
            self.base_dir / "cache",
            transaction_manager=self.transaction_manager
        )
        self.version_manager = ContextVersionManager(self.context_dir)
        self.io_handler = IOHandler(self.base_dir)

        # Internal state
        self.current_context: Optional[Dict] = None
        self.current_version: Optional[str] = None

        # Pattern tracking
        self._access_patterns: Dict[str, List[datetime]] = {}

        # Initialize recovery manager
        self.recovery_manager = RecoveryManager(self.base_dir / "recovery")

        # Register alert handlers
        self._register_alert_handlers()

        # Initialize storage and context
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage directories and base context"""
        try:
            self.context_dir.mkdir(parents=True, exist_ok=True)

            # Load or create initial context
            context = self.load_context()
            if not context:
                context = self._create_initial_context()
                self.save_context(
                    context,
                    "Initial context creation",
                    VersionType.MAJOR,
                    {"action": "initialization"}
                )
        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}")
            raise

    def _register_alert_handlers(self):
        """Register handlers for different alert levels"""
        self.recovery_manager.register_alert_handler(
            AlertSeverity.CRITICAL,
            self._handle_critical_alert
        )
        self.recovery_manager.register_alert_handler(
            AlertSeverity.HIGH,
            self._handle_high_alert
        )

    def _handle_critical_alert(self, alert: Alert):
        """Handle critical system alerts"""
        try:
            self.logger.critical(f"Critical alert: {alert.message}")

            if self.current_context:
                # Update context with failure
                self.current_context["component_context"]["failure_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "critical_alert",
                    "component": alert.component,
                    "message": alert.message,
                    "recovery_id": alert.recovery_id,
                    "details": alert.details
                })

                # Persist updated context
                self.save_context(
                    self.current_context,
                    f"Critical alert: {alert.message}",
                    VersionType.MINOR
                )

        except Exception as e:
            self.logger.error(f"Error handling critical alert: {e}")

    def _handle_high_alert(self, alert: Alert):
        """Handle high severity system alerts"""
        try:
            self.logger.error(f"High severity alert: {alert.message}")

            if self.current_context:
                # Update metrics and state
                self.current_context["component_context"]["performance_metrics"]["alerts_high"] = \
                    self.current_context["component_context"]["performance_metrics"].get("alerts_high", 0) + 1

                # Add to history
                self.current_context["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "high_alert",
                    "component": alert.component,
                    "message": alert.message,
                    "recovery_id": alert.recovery_id
                })

        except Exception as e:
            self.logger.error(f"Error handling high alert: {e}")

    async def handle_context_failure(self, error: Exception,
                                   severity: AlertSeverity = AlertSeverity.HIGH) -> str:
        """Handle context manager failures"""
        try:
            return await self.recovery_manager.handle_failure(
                "context_manager",
                error,
                severity=severity,
                details={
                    "session_id": self.session_id,
                    "current_version": self.current_version,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            self.logger.error(f"Error handling context failure: {e}")
            raise

    async def _recover_context(self, recovery_id: str) -> bool:
        """Recover context after failure"""
        try:
            # Get latest valid version
            versions = self.version_manager.list_versions(limit=5)

            for version in versions:
                try:
                    # Attempt to load and validate version
                    context = self.version_manager.get_version(version.id)
                    if context and not self._validate_context(context):
                        continue

                    # Restore valid version
                    self.current_context = context
                    self.current_version = version.id

                    # Clear cache
                    cache_key = f"context_{self.session_id}"
                    self.cache_manager.invalidate(cache_key, CacheLevel.SYSTEM)

                    return True

                except Exception as e:
                    self.logger.error(f"Error recovering version {version.id}: {e}")
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Error in context recovery: {e}")
            return False

    def _create_initial_context(self) -> Dict:
        """Create new initial context with enhanced structure"""
        now = datetime.now().isoformat()
        return {
            "metadata": {
                "version": "2.0.0",
                "created_at": now,
                "session_id": self.session_id,
                "last_modified": now,
                "execution_mode": "mixed"
            },
            "system_context": {
                "architecture_state": "initial",
                "global_constraints": {},
                "cache_policy": {
                    "mode": CacheMode.HYBRID.value,
                    "ttl": {
                        "system": "24h",
                        "domain": "12h",
                        "component": "6h"
                    },
                    "strategy": "write_through",
                    "prefetch": True,
                    "replication": True
                },
                "resilience_config": {
                    "circuit_breaker": {
                        "enabled": True,
                        "thresholds": {
                            "errors": 5,
                            "timeouts": 3
                        }
                    },
                    "fallback": {
                        "strategy": "cache",
                        "options": ["cache", "degraded", "manual"]
                    }
                },
                "learning_state": {
                    "pattern_detection": True,
                    "confidence_threshold": 0.85,
                    "feedback_collection": True,
                    "pattern_storage": True
                }
            },
            "domain_context": {
                "business_rules": [],
                "validation_rules": [],
                "cached_patterns": [],
                "impact_analysis": {
                    "scope": "local",
                    "risk_level": "low"
                },
                "risk_assessment": {
                    "level": "low",
                    "mitigation_strategy": ""
                }
            },
            "component_context": {
                "local_state": {},
                "dependencies": [],
                "cached_results": {},
                "failure_history": [],
                "performance_metrics": {
                    "response_time": [],
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "parallel_efficiency": 0.0
                }
            },
            "modification_context": {
                "change_scope": [],
                "validation_rules": {
                    "architecture": {"mode": "sync", "required": True},
                    "domain": {"mode": "async", "required": True},
                    "code": {"mode": "sync", "required": True},
                    "qa": {"mode": "async", "required": True}
                },
                "cache_strategy": {
                    "mode": CacheMode.MEMORY.value,
                    "ttl": "6h",
                    "invalidation": []
                },
                "fallback_options": ["cache", "degraded"],
                "learning_targets": {
                    "pattern_detection": True,
                    "feedback_collection": True
                },
                "parallel_execution": {
                    "enabled": True,
                    "max_threads": 4,
                    "dependencies": []
                }
            },
            "history": []
        }

    def load_context(self) -> Optional[Dict]:
        """Load context with enhanced caching and validation"""
        try:
            # Generate cache key
            cache_key = f"context_{self.session_id}"

            # Try loading from cache
            cached_context = self.cache_manager.get(cache_key, CacheLevel.SYSTEM)
            if cached_context:
                self.logger.debug("Context loaded from cache")
                self.current_context = cached_context
                self._record_access_pattern("load_context")
                return cached_context

            # Load from file system
            context_file = self.context_dir / "context.yaml"
            if not context_file.exists():
                return None

            with context_file.open('r') as f:
                context = yaml.safe_load(f)

            # Validate context
            validation_errors = ContextValidator.validate_context(context)
            if validation_errors:
                self._handle_validation_errors(validation_errors)
                return None

            # Update cache and internal state
            self.cache_manager.set(
                cache_key,
                context,
                CacheLevel.SYSTEM,
                metadata={
                    "version": context["metadata"]["version"],
                    "last_modified": context["metadata"]["last_modified"]
                }
            )

            self.current_context = context
            return context

        except Exception as e:
            self.logger.error(f"Error loading context: {e}")
            return None

    def save_context(self, context: Dict, description: str,
                    version_type: VersionType = VersionType.MINOR,
                    changes: Optional[Dict] = None) -> bool:
        """Save context with enhanced versioning and caching"""
        try:
            with self.transaction_manager.transaction() as transaction_id:
                # Validate context
                validation_errors = ContextValidator.validate_context(context)
                if validation_errors:
                    self._handle_validation_errors(validation_errors)
                    return False

                # Create new version
                version_id = self.version_manager.create_version(
                    context,
                    version_type,
                    description,
                    changes or {}
                )
                if not version_id:
                    return False

                # Update current context file
                context_file = self.context_dir / "context.yaml"
                context['metadata']['last_modified'] = datetime.now().isoformat()
                context['metadata']['version'] = version_id

                # Add operation to transaction
                self.transaction_manager.add_operation(
                    transaction_id,
                    OperationType.FILE_WRITE,
                    str(context_file),
                    {"content": context},
                    backup=True
                )

                # Write to file
                with context_file.open('w') as f:
                    yaml.safe_dump(context, f)

                # Update cache with pattern tracking
                cache_key = f"context_{self.session_id}"
                pattern_id = f"context_update_{version_type.value}"
                self.cache_manager.set(
                    cache_key,
                    context,
                    CacheLevel.SYSTEM,
                    metadata={
                        "version": version_id,
                        "last_modified": context['metadata']['last_modified']
                    },
                    pattern_id=pattern_id
                )

                # Update internal state
                self.current_context = context
                self.current_version = version_id

                # Record access pattern
                self._record_access_pattern("save_context")

                return True

        except Exception as e:
            self.logger.error(f"Error saving context: {e}")
            return False

    def update_context(self, updates: Dict, path: List[str],
                      description: str) -> bool:
        """Update specific parts of the context with pattern detection"""
        try:
            if self.current_context is None:
                self.logger.error("No context loaded")
                return False

            with self.transaction_manager.transaction() as transaction_id:
                # Navigate to update point
                current = self.current_context
                for p in path[:-1]:
                    if p not in current:
                        current[p] = {}
                    current = current[p]

                # Apply update
                last_key = path[-1]
                old_value = current.get(last_key)
                current[last_key] = updates

                # Record in history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "path": "/".join(path),
                    "updates": updates,
                    "description": description,
                    "previous_value": old_value
                }
                self.current_context.setdefault("history", []).append(history_entry)

                # Create pattern ID for cache
                category = path[0] if path else "unknown"
                pattern_id = f"context_update_{category}"

                # Save updated context
                return self.save_context(
                    self.current_context,
                    description,
                    VersionType.MINOR,
                    {
                        "path": path,
                        "updates": updates,
                        "previous_value": old_value,
                        "pattern_id": pattern_id
                    }
                )

        except Exception as e:
            self.logger.error(f"Error updating context: {e}")
            return False

    def _record_access_pattern(self, operation: str):
        """Record access patterns for learning system"""
        pattern_key = f"{self.session_id}_{operation}"
        if pattern_key not in self._access_patterns:
            self._access_patterns[pattern_key] = []
        self._access_patterns[pattern_key].append(datetime.now())

        # Analyze patterns periodically
        if len(self._access_patterns[pattern_key]) >= 5:  # Threshold for analysis
            self._analyze_access_pattern(pattern_key)

    def _analyze_access_pattern(self, pattern_key: str):
        """Analyze access patterns for potential optimizations"""
        accesses = self._access_patterns[pattern_key]
        if len(accesses) < 2:
            return

        # Calculate average interval between accesses
        intervals = [
            (accesses[i+1] - accesses[i]).total_seconds()
            for i in range(len(accesses)-1)
        ]
        avg_interval = sum(intervals) / len(intervals)

        # If frequent access pattern detected, update cache policy
        if avg_interval < 300:  # Less than 5 minutes
            self.logger.info(f"Frequent access pattern detected for {pattern_key}")
            self._optimize_cache_for_pattern(pattern_key)

    def _optimize_cache_for_pattern(self, pattern_key: str):
        """Optimize cache settings for detected patterns"""
        if pattern_key.startswith(f"{self.session_id}_load_context"):
            # Optimize for frequent context loading
            self.cache_manager.update_config({
                "prefetch_threshold": 0.5,  # More aggressive prefetching
                "mode": CacheMode.HYBRID  # Ensure hybrid caching
            })
        elif pattern_key.startswith(f"{self.session_id}_save_context"):
            # Optimize for frequent updates
            self.cache_manager.update_config({
                "strategy": "write_back",  # Delayed writes for better performance
                "max_entries": {
                    CacheLevel.SYSTEM: 2000,  # Increase cache size
                    CacheLevel.DOMAIN: 7500,
                    CacheLevel.COMPONENT: 15000
                }
            })

    def get_version_history(self, limit: Optional[int] = None) -> List[VersionInfo]:
        """Get version history with optional limit"""
        return self.version_manager.list_versions(limit=limit)

    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific version with cache invalidation"""
        try:
            version_data = self.version_manager.get_version(version_id)
            if not version_data:
                return False

            # Invalidate cache before rollback
            cache_key = f"context_{self.session_id}"
            self.cache_manager.invalidate(cache_key, CacheLevel.SYSTEM)

            # Save as new version to maintain history
            return self.save_context(
                version_data["context"],
                f"Rollback to version {version_id}",
                VersionType.MINOR,
                {"rollback_to": version_id}
            )

        except Exception as e:
            self.logger.error(f"Error rolling back to version {version_id}: {e}")
            return False

    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        return self.cache_manager.get_stats()

    def cleanup(self, days_to_keep: int = 30) -> Tuple[int, int]:
        """Clean up old versions and cache entries"""
        versions_cleaned = self.version_manager.cleanup_old_versions(days_to_keep)
        cache_cleaned = self.cache_manager.cleanup()
        return versions_cleaned, cache_cleaned

    def _handle_validation_errors(self, errors: List[ValidationError]) -> None:
        """Handle validation errors with detailed logging"""
        error_msgs = "\n".join(
            f"{e.field}: {e.message} (Severity: {e.severity})"
            for e in errors
        )
        self.logger.error(f"Context validation failed:\n{error_msgs}")

        # Update failure history in context
        if self.current_context:
            self.current_context["component_context"]["failure_history"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "validation_error",
                "errors": [dataclasses.asdict(e) for e in errors]
            })

        raise ValueError(f"Context validation failed: {len(errors)} errors found")

    def _validate_context(self, context: Dict) -> bool:
        """Validate the context using the validation framework"""
        validation_errors = ContextValidator.validate_context(context)
        if validation_errors:
            self._handle_validation_errors(validation_errors)
            return False
        return True

    def get_context_recommendations(self) -> Dict[str, float]:
        """Get recommendations based on learned patterns"""
        return self.cache_manager.get_pattern_recommendations()

    