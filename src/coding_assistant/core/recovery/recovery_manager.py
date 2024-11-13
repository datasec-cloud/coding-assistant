from typing import Dict, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import shutil
import threading
from enum import Enum
from dataclasses import dataclass
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import dataclasses



class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class RecoveryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Alert:
    id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    message: str
    details: Optional[Dict] = None
    recovery_id: Optional[str] = None

@dataclass
class RecoveryOperation:
    id: str
    component: str
    operation_type: str
    status: RecoveryStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    backup_path: Optional[Path] = None
    alerts: List[Alert] = None

class RecoveryManager:
    """Manages system recovery operations and alerts"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.recovery_dir = self.base_dir / "recovery"
        self.backup_dir = self.base_dir / "backups"
        self.alert_dir = self.base_dir / "alerts"
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Alert queue and handlers
        self.alert_queue = queue.PriorityQueue()
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }
        
        # Recovery tracking
        self.active_recoveries: Dict[str, RecoveryOperation] = {}
        self._recovery_lock = threading.RLock()
        
        # Alert processing thread
        self._alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._shutdown = threading.Event()
        
        # Recovery executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize storage
        self._initialize_storage()
        self._alert_thread.start()

    def _initialize_storage(self):
        """Initialize required directories"""
        for directory in [self.recovery_dir, self.backup_dir, self.alert_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def handle_failure(self, component: str, error: Exception,
                           severity: AlertSeverity = AlertSeverity.HIGH) -> str:
        """Handle component failure with recovery"""
        try:
            # Generate recovery ID
            recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create alert
            alert = Alert(
                id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity=severity,
                component=component,
                message=f"Component failure detected: {str(error)}",
                details={"error": str(error), "stack_trace": "".join(traceback.format_exc())},
                recovery_id=recovery_id
            )
            
            # Queue alert
            await self._queue_alert(alert)
            
            # Initialize recovery operation
            recovery_op = RecoveryOperation(
                id=recovery_id,
                component=component,
                operation_type="failure_recovery",
                status=RecoveryStatus.PENDING,
                started_at=datetime.now(),
                alerts=[alert]
            )
            
            # Start recovery process
            asyncio.create_task(self._execute_recovery(recovery_op))
            
            return recovery_id
            
        except Exception as e:
            self.logger.error(f"Error handling failure for {component}: {e}")
            raise

    async def _execute_recovery(self, recovery_op: RecoveryOperation):
        """Execute recovery operation"""
        try:
            with self._recovery_lock:
                self.active_recoveries[recovery_op.id] = recovery_op
                recovery_op.status = RecoveryStatus.IN_PROGRESS
            
            # Create backup
            backup_path = await self._create_backup(recovery_op.component)
            recovery_op.backup_path = backup_path
            
            # Execute recovery steps
            try:
                await self._perform_recovery_steps(recovery_op)
                recovery_op.status = RecoveryStatus.COMPLETED
                recovery_op.completed_at = datetime.now()
                
                # Send success alert
                await self._queue_alert(Alert(
                    id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.INFO,
                    component=recovery_op.component,
                    message=f"Recovery completed successfully",
                    recovery_id=recovery_op.id
                ))
                
            except Exception as e:
                recovery_op.status = RecoveryStatus.FAILED
                recovery_op.error = str(e)
                recovery_op.completed_at = datetime.now()
                
                # Send failure alert
                await self._queue_alert(Alert(
                    id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.CRITICAL,
                    component=recovery_op.component,
                    message=f"Recovery failed: {str(e)}",
                    recovery_id=recovery_op.id,
                    details={"error": str(e), "stack_trace": "".join(traceback.format_exc())}
                ))
                
        except Exception as e:
            self.logger.error(f"Error executing recovery operation: {e}")
            raise
        finally:
            # Save recovery operation details
            self._save_recovery_details(recovery_op)

    async def _create_backup(self, component: str) -> Optional[Path]:
        """Create backup of component state"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"{component}_{timestamp}.bak"
            
            # Component-specific backup logic
            if component.startswith("file_"):
                # File system component backup
                source_path = Path(component.replace("file_", ""))
                if source_path.exists():
                    if source_path.is_file():
                        shutil.copy2(source_path, backup_path)
                    else:
                        shutil.copytree(source_path, backup_path)
            elif component.startswith("cache_"):
                # Cache component backup
                cache_data = await self._get_cache_data(component)
                if cache_data:
                    backup_path.write_text(json.dumps(cache_data))
            
            return backup_path if backup_path.exists() else None
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {component}: {e}")
            return None

    async def _perform_recovery_steps(self, recovery_op: RecoveryOperation):
        """Perform recovery steps for the component"""
        component = recovery_op.component
        
        if component.startswith("file_"):
            await self._recover_file_component(recovery_op)
        elif component.startswith("cache_"):
            await self._recover_cache_component(recovery_op)
        elif component.startswith("agent_"):
            await self._recover_agent_component(recovery_op)
        else:
            await self._recover_generic_component(recovery_op)

    async def _queue_alert(self, alert: Alert):
        """Queue alert for processing"""
        # Priority based on severity
        priority = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4
        }[alert.severity]
        
        self.alert_queue.put((priority, alert))
        
        # Save alert to disk
        alert_path = self.alert_dir / f"{alert.id}.json"
        with alert_path.open('w') as f:
            json.dump(dataclasses.asdict(alert), f)

    def _process_alerts(self):
        """Process queued alerts"""
        while not self._shutdown.is_set():
            try:
                # Get alert with timeout
                try:
                    priority, alert = self.alert_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process alert through registered handlers
                self._handle_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")

    def _handle_alert(self, alert: Alert):
        """Process alert through registered handlers"""
        try:
            # Call registered handlers for this severity
            handlers = self.alert_handlers.get(alert.severity, [])
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling alert: {e}")

    def register_alert_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Register a handler for alerts of specific severity"""
        self.alert_handlers[severity].append(handler)

    def get_recovery_status(self, recovery_id: str) -> Optional[RecoveryStatus]:
        """Get status of a recovery operation"""
        with self._recovery_lock:
            recovery_op = self.active_recoveries.get(recovery_id)
            return recovery_op.status if recovery_op else None

    def _save_recovery_details(self, recovery_op: RecoveryOperation):
        """Save recovery operation details to disk"""
        try:
            file_path = self.recovery_dir / f"{recovery_op.id}.json"
            with file_path.open('w') as f:
                json.dump(dataclasses.asdict(recovery_op), f)
        except Exception as e:
            self.logger.error(f"Error saving recovery details: {e}")

    def shutdown(self):
        """Shutdown recovery manager"""
        self._shutdown.set()
        self._alert_thread.join()
        self.executor.shutdown(wait=True)