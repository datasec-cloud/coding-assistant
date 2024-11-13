# src/coding_assistant/core/context/schema.py
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import timedelta


class ExecutionMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    MIXED = "mixed"


class CacheMode(Enum):
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


class ValidationMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"


@dataclass
class ResilienceConfig:
    circuit_breaker: Dict
    fallback: Dict
    thresholds: Dict


@dataclass
class CachePolicy:
    mode: CacheMode
    ttl: Dict[str, timedelta]
    strategy: str
    prefetch: bool
    replication: bool


@dataclass
class LearningConfig:
    pattern_detection: bool
    confidence_threshold: float
    feedback_collection: bool
    pattern_storage: bool


class EnrichedContextSchema:
    """Enhanced schema definition matching V2 architecture"""

    @staticmethod
    def create_base_context(session_id: str) -> Dict:
        return {
            "metadata": {
                "version": "2.0.0",
                "session_id": session_id,
                "created_at": "",
                "last_modified": "",
                "execution_mode": ExecutionMode.MIXED.value
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
                    "strategy": "hierarchical",
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

    @staticmethod
    def validate_enriched_context(context: Dict) -> List[str]:
        """Validate the enriched context structure"""
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
                if "resilience_config" not in context[section]:
                    errors.append("Missing resilience_config in system_context")
                if "learning_state" not in context[section]:
                    errors.append("Missing learning_state in system_context")

            elif section == "modification_context":
                if "parallel_execution" not in context[section]:
                    errors.append("Missing parallel_execution in modification_context")
                if "learning_targets" not in context[section]:
                    errors.append("Missing learning_targets in modification_context")

        return errors
