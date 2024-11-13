from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class AgentType(Enum):
    ARCHITECT = "architect"
    DOMAIN_EXPERT = "domain_expert"
    CODE_ENGINEER = "code_engineer"
    QA_COORDINATOR = "qa_coordinator"

@dataclass
class AgentConfig:
    """Agent configuration following V2 architecture"""
    agent_type: AgentType
    cache_allocation: float
    execution_mode: str
    parallel_execution: bool
    priority: str
    resilience: Dict
    learning: Dict

class AgentConfigManager:
    """Manages agent configurations according to V2 specifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.configs = self._initialize_configs()

    def _initialize_configs(self) -> Dict[AgentType, AgentConfig]:
        """Initialize agent configurations according to V2 specs"""
        return {
            AgentType.ARCHITECT: AgentConfig(
                agent_type=AgentType.ARCHITECT,
                cache_allocation=0.35,
                execution_mode="sync",
                parallel_execution=True,
                priority="high",
                resilience={
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
                learning={
                    "pattern_detection": True,
                    "confidence_threshold": 0.85,
                    "feedback_collection": True
                }
            ),
            AgentType.DOMAIN_EXPERT: AgentConfig(
                agent_type=AgentType.DOMAIN_EXPERT,
                cache_allocation=0.20,
                execution_mode="async",
                parallel_execution=True,
                priority="medium",
                resilience={
                    "circuit_breaker": {
                        "enabled": True,
                        "thresholds": {
                            "errors": 3,
                            "timeouts": 2
                        }
                    },
                    "fallback": {
                        "strategy": "cache"
                    }
                },
                learning={
                    "pattern_detection": True,
                    "confidence_threshold": 0.80,
                    "feedback_collection": True
                }
            ),
            AgentType.CODE_ENGINEER: AgentConfig(
                agent_type=AgentType.CODE_ENGINEER,
                cache_allocation=0.30,
                execution_mode="mixed",
                parallel_execution=True,
                priority="high",
                resilience={
                    "circuit_breaker": {
                        "enabled": True,
                        "thresholds": {
                            "errors": 4,
                            "timeouts": 2
                        }
                    },
                    "fallback": {
                        "strategy": "degraded"
                    }
                },
                learning={
                    "pattern_detection": True,
                    "confidence_threshold": 0.90,
                    "feedback_collection": True
                }
            ),
            AgentType.QA_COORDINATOR: AgentConfig(
                agent_type=AgentType.QA_COORDINATOR,
                cache_allocation=0.15,
                execution_mode="async",
                parallel_execution=True,
                priority="medium",
                resilience={
                    "circuit_breaker": {
                        "enabled": True,
                        "thresholds": {
                            "errors": 3,
                            "timeouts": 2
                        }
                    },
                    "fallback": {
                        "strategy": "cache"
                    }
                },
                learning={
                    "pattern_detection": True,
                    "confidence_threshold": 0.85,
                    "feedback_collection": True
                }
            )
        }

    def get_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
        """Get configuration for a specific agent type"""
        return self.configs.get(agent_type)

    def update_config(self, agent_type: AgentType, updates: Dict) -> bool:
        """Update configuration for a specific agent"""
        try:
            if agent_type not in self.configs:
                return False

            config = self.configs[agent_type]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return True
        except Exception as e:
            self.logger.error(f"Error updating config for {agent_type}: {e}")
            return False

    def get_cache_allocation(self) -> Dict[AgentType, float]:
        """Get cache allocation for all agents"""
        return {
            agent_type: config.cache_allocation 
            for agent_type, config in self.configs.items()
        }