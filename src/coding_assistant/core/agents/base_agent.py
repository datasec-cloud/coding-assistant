from typing import Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime
import logging
import asyncio

from coding_assistant.core.context.manager import ContextManager
from coding_assistant.core.io.transaction import TransactionManager
from coding_assistant.core.recovery.recovery_manager import RecoveryManager, AlertSeverity, Alert, RecoveryStatus

class AgentFailure(Exception):
    """Custom exception for agent failures"""
    pass

class BaseAgent:
    """Enhanced base agent with recovery and alert capabilities"""
    
    def __init__(self, agent_type: str, context_manager: ContextManager):
        self.agent_type = agent_type
        self.context_manager = context_manager
        self.base_dir = context_manager.base_dir
        
        # Component managers
        self.transaction_manager = TransactionManager(self.base_dir / "transactions")
        self.recovery_manager = RecoveryManager(self.base_dir / "recovery")
        self.logger = logging.getLogger(f"agent.{agent_type}")
        
        # Register alert handlers
        self._register_alert_handlers()
        
    def _register_alert_handlers(self):
        """Register handlers for different alert severities"""
        self.recovery_manager.register_alert_handler(
            AlertSeverity.CRITICAL,
            self._handle_critical_alert
        )
        self.recovery_manager.register_alert_handler(
            AlertSeverity.HIGH,
            self._handle_high_alert
        )
        self.recovery_manager.register_alert_handler(
            AlertSeverity.MEDIUM,
            self._handle_medium_alert
        )
        
    async def process_request(self, request: Dict) -> Optional[Dict]:
        """Process request with recovery support"""
        try:
            with self.transaction_manager.transaction() as transaction_id:
                # Validate request
                if not self._validate_request(request):
                    raise AgentFailure("Invalid request format")

                # Process with recovery support
                try:
                    response = await self._handle_request_with_recovery(request, transaction_id)
                except Exception as e:
                    # Handle failure with recovery
                    recovery_id = await self.handle_failure(
                        e, AlertSeverity.HIGH,
                        details={"request_id": request.get("id")}
                    )
                    response = await self._wait_for_recovery(recovery_id)
                
                # Validate response
                if response and self.validate_response(response):
                    return response
                return None

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            await self.handle_failure(e, AlertSeverity.CRITICAL)
            return None
            
    async def handle_failure(self, error: Exception, severity: AlertSeverity = AlertSeverity.HIGH,
                           details: Optional[Dict] = None) -> str:
        """Handle agent failure with recovery"""
        try:
            component_id = f"agent_{self.agent_type}"
            
            # Create detailed error info
            error_details = {
                "agent_type": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                **(details or {})
            }
            
            # Trigger recovery process
            return await self.recovery_manager.handle_failure(
                component_id,
                error,
                severity=severity,
                details=error_details
            )
            
        except Exception as e:
            self.logger.error(f"Error in failure handling: {e}")
            raise
            
    async def _wait_for_recovery(self, recovery_id: str, 
                               timeout: float = 300) -> Optional[Dict]:
        """Wait for recovery completion with timeout"""
        try:
            start_time = datetime.now()
            
            while True:
                status = self.recovery_manager.get_recovery_status(recovery_id)
                
                if status == RecoveryStatus.COMPLETED:
                    return await self._handle_recovery_completion(recovery_id)
                elif status == RecoveryStatus.FAILED:
                    raise AgentFailure(f"Recovery {recovery_id} failed")
                
                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout:
                    raise TimeoutError(f"Recovery {recovery_id} timed out")
                    
                await asyncio.sleep(1.0)
                
        except Exception as e:
            self.logger.error(f"Error waiting for recovery: {e}")
            raise
            
    async def _handle_recovery_completion(self, recovery_id: str) -> Dict:
        """Handle successful recovery completion"""
        return {
            "status": "recovered",
            "recovery_id": recovery_id,
            "timestamp": datetime.now().isoformat()
        }
        
    def _handle_critical_alert(self, alert: Alert):
        """Handle critical severity alerts"""
        self.logger.critical(f"Critical alert in {self.agent_type}: {alert.message}")
        self.context_manager.update_context(
            {
                "type": "critical_alert",
                "component": f"agent_{self.agent_type}",
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "recovery_id": alert.recovery_id
            },
            ["component_context", "failure_history"],
            f"Critical alert in {self.agent_type}"
        )
        
    def _handle_high_alert(self, alert: Alert):
        """Handle high severity alerts"""
        self.logger.error(f"High severity alert in {self.agent_type}: {alert.message}")
        # Update agent state and metrics
        
    def _handle_medium_alert(self, alert: Alert):
        """Handle medium severity alerts"""
        self.logger.warning(f"Medium severity alert in {self.agent_type}: {alert.message}")
        # Log and track patterns
        
    def validate_response(self, response: Dict) -> bool:
        """Validate a response before sending"""
        raise NotImplementedError("Subclasses must implement validate_response")