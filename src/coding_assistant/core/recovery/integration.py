# src/coding_assistant/core/integration.py
from typing import Dict, Optional, List, Any, Callable
from pathlib import Path
from datetime import datetime
import logging
import asyncio
import dataclasses
import traceback

from coding_assistant.core.recovery.recovery_manager import RecoveryManager, Alert
from coding_assistant.core.context.manager import ContextManager
from coding_assistant.core.context.cache_manager import CacheManager
from coding_assistant.core.io.file_system import FileSystemHandler

@dataclasses.dataclass
class RecoveryCoordinator:
    """Coordinates recovery across multiple components."""
    base_dir: Path
    logger: logging.Logger
    recovery_manager: RecoveryManager
    context_manager: ContextManager
    cache_manager: CacheManager
    file_system_handler: FileSystemHandler

    def __post_init__(self):
        """Register core components for recovery."""
        self.register_core_components()

    def register_core_components(self):
        """Register core system components for recovery."""
        # Register Context Manager
        self.recovery_manager.integration.register_component(
            component_id="context_manager",
            recovery_handler=self.context_manager.recover_context,
            fallback_handler=self.context_manager.get_fallback_context,
            health_check=self.context_manager.check_context_health
        )
        
        # Register Cache Manager
        self.recovery_manager.integration.register_component(
            component_id="cache_manager",
            recovery_handler=self.cache_manager.recover_cache_level,
            fallback_handler=self.cache_manager.get_fallback_value,
            health_check=self.cache_manager.check_cache_health
        )
        
        # Register File System Handler
        self.recovery_manager.integration.register_component(
            component_id="file_system",
            recovery_handler=self.file_system_handler.recover_file_system,
            fallback_handler=self.file_system_handler.get_file_fallback,
            health_check=self.file_system_handler.check_filesystem_health
        )
        
        self.logger.info("Core components registered for recovery.")

    async def handle_alerts(self, alerts: List[Alert]) -> bool:
        """
        Handle a list of alerts by coordinating recovery operations.
        
        Args:
            alerts (List[Alert]): List of alerts to handle.
        
        Returns:
            bool: True if all recoveries succeeded, False otherwise.
        """
        try:
            # Coordinate recovery for all alerts
            success = await self.recovery_manager.coordinate_recovery(alerts)
            return success
        except Exception as e:
            self.logger.error(f"Error handling alerts: {e}")
            return False

# Utility function to initialize and register components
def register_core_components(coordinator: RecoveryCoordinator):
    """Utility function to register core components for recovery."""
    coordinator.register_core_components()
