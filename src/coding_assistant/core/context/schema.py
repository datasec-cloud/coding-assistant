"""Enhanced schema definitions for context management"""
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import timedelta, datetime

class ExecutionMode(str, Enum):
    """Execution modes for context operations"""
    SYNC = "sync"
    ASYNC = "async"
    MIXED = "mixed"

    @classmethod
    def from_str(cls, value: str) -> 'ExecutionMode':
        """Create from string with validation"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid execution mode: {value}")

class CacheMode(str, Enum):
    """Cache operation modes"""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"

    @classmethod
    def from_str(cls, value: str) -> 'CacheMode':
        """Create from string with validation"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid cache mode: {value}")

class ValidationMode(str, Enum):
    """Validation operation modes"""
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"

    @classmethod
    def from_str(cls, value: str) -> 'ValidationMode':
        """Create from string with validation"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid validation mode: {value}")

class ArchitectureState(str, Enum):
    """System architecture states"""
    INITIAL = "initial"
    STABLE = "stable"
    TRANSITIONING = "transitioning"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

@dataclass
class TTLConfig:
    """Time-to-live configuration"""
    system: timedelta
    domain: timedelta
    component: timedelta

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TTLConfig':
        """Create from dictionary with time string parsing"""
        def parse_time(time_str: str) -> timedelta:
            try:
                value = int(time_str[:-1])
                unit = time_str[-1].lower()
                units = {'h': 'hours', 'd': 'days', 'm': 'minutes', 's': 'seconds'}
                if unit not in units:
                    raise ValueError(f"Invalid time unit: {unit}")
                return timedelta(**{units[unit]: value})
            except Exception as e:
                raise ValueError(f"Invalid time format: {time_str}") from e

        return cls(
            system=parse_time(data['system']),
            domain=parse_time(data['domain']),
            component=parse_time(data['component'])
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with formatted time strings"""
        return {
            'system': f"{int(self.system.total_seconds() / 3600)}h",
            'domain': f"{int(self.domain.total_seconds() / 3600)}h",
            'component': f"{int(self.component.total_seconds() / 3600)}h"
        }

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    error_threshold: int = 5
    timeout_seconds: int = 60
    reset_timeout_seconds: int = 300

@dataclass
class FallbackConfig:
    """Fallback strategy configuration"""
    strategy: str = "cache"
    options: List[str] = field(default_factory=lambda: ["cache", "degraded", "manual"])
    max_retries: int = 3
    retry_delay_seconds: int = 5

@dataclass
class ResilienceConfig:
    """Enhanced resilience configuration"""
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    thresholds: Dict[str, int] = field(default_factory=lambda: {
        "error_count": 5,
        "timeout_count": 3,
        "degraded_performance": 1000  # ms
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "circuit_breaker": dataclass.asdict(self.circuit_breaker),
            "fallback": dataclass.asdict(self.fallback),
            "thresholds": self.thresholds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResilienceConfig':
        """Create from dictionary with validation"""
        circuit_breaker = CircuitBreakerConfig(**data.get("circuit_breaker", {}))
        fallback = FallbackConfig(**data.get("fallback", {}))
        thresholds = data.get("thresholds", {})
        return cls(circuit_breaker, fallback, thresholds)

@dataclass
class CachePolicy:
    """Enhanced cache policy configuration"""
    mode: CacheMode
    ttl: TTLConfig
    strategy: str = "write_through"
    prefetch: bool = True
    replication: bool = True
    max_size: Optional[int] = None
    eviction_policy: str = "lru"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mode": self.mode.value,
            "ttl": self.ttl.to_dict(),
            "strategy": self.strategy,
            "prefetch": self.prefetch,
            "replication": self.replication,
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachePolicy':
        """Create from dictionary with validation"""
        mode = CacheMode.from_str(data["mode"])
        ttl = TTLConfig.from_dict(data["ttl"])
        return cls(
            mode=mode,
            ttl=ttl,
            strategy=data.get("strategy", "write_through"),
            prefetch=data.get("prefetch", True),
            replication=data.get("replication", True),
            max_size=data.get("max_size"),
            eviction_policy=data.get("eviction_policy", "lru")
        )

@dataclass
class LearningConfig:
    """Enhanced learning system configuration"""
    pattern_detection: bool = True
    confidence_threshold: float = 0.85
    feedback_collection: bool = True
    pattern_storage: bool = True
    min_pattern_occurrences: int = 3
    max_pattern_age_hours: int = 24
    learning_rate: float = 0.1

    def validate(self) -> Optional[str]:
        """Validate configuration values"""
        if not 0 <= self.confidence_threshold <= 1:
            return "confidence_threshold must be between 0 and 1"
        if self.min_pattern_occurrences < 1:
            return "min_pattern_occurrences must be positive"
        if self.max_pattern_age_hours < 1:
            return "max_pattern_age_hours must be positive"
        if not 0 < self.learning_rate <= 1:
            return "learning_rate must be between 0 and 1"
        return None

class EnrichedContextSchema:
    """Enhanced schema definition with validation"""
    
    @staticmethod
    def create_base_context(session_id: str) -> Dict[str, Any]:
        """Create base context structure with validation"""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Invalid session ID")

        now = datetime.now().isoformat()
        return {
            "metadata": {
                "version": "2.0.0",
                "session_id": session_id,
                "created_at": now,
                "last_modified": now,
                "execution_mode": ExecutionMode.MIXED.value
            },
            "system_context": {
                "architecture_state": ArchitectureState.INITIAL.value,
                "global_constraints": {},
                "cache_policy": CachePolicy(
                    mode=CacheMode.HYBRID,
                    ttl=TTLConfig(
                        system=timedelta(hours=24),
                        domain=timedelta(hours=12),
                        component=timedelta(hours=6)
                    )
                ).to_dict(),
                "resilience_config": ResilienceConfig().to_dict(),
                "learning_state": dataclass.asdict(LearningConfig())
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

    @staticmethod
    def validate_enriched_context(context: Dict) -> List[str]:
        """Validate the enriched context structure with enhanced checks"""
        errors = []
        required_sections = [
            "metadata", "system_context", "domain_context",
            "component_context", "modification_context"
        ]

        for section in required_sections:
            if section not in context:
                errors.append(f"Missing required section: {section}")
                continue

            if section == "system_context":
                if "cache_policy" not in context[section]:
                    errors.append("Missing cache_policy in system_context")
                else:
                    try:
                        CachePolicy.from_dict(context[section]["cache_policy"])
                    except ValueError as e:
                        errors.append(f"Invalid cache_policy: {str(e)}")

                if "resilience_config" not in context[section]:
                    errors.append("Missing resilience_config in system_context")
                else:
                    try:
                        ResilienceConfig.from_dict(context[section]["resilience_config"])
                    except ValueError as e:
                        errors.append(f"Invalid resilience_config: {str(e)}")

                if "learning_state" not in context[section]:
                    errors.append("Missing learning_state in system_context")
                else:
                    try:
                        learning_config = LearningConfig(**context[section]["learning_state"])
                        if error := learning_config.validate():
                            errors.append(f"Invalid learning_state: {error}")
                    except ValueError as e:
                        errors.append(f"Invalid learning_state: {str(e)}")

            elif section == "modification_context":
                required_subsections = ["parallel_execution", "learning_targets"]
                for subsection in required_subsections:
                    if subsection not in context[section]:
                        errors.append(f"Missing {subsection} in modification_context")

        return errors