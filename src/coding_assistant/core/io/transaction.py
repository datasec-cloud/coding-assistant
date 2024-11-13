"""Enhanced transaction management with improved safety and recovery"""
from typing import Dict, List, Optional, Callable, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import shutil
import fcntl
import traceback
import hashlib
from queue import Queue
import signal
import time

class TransactionStatus(Enum):
    """Transaction states with enhanced status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    ABANDONED = "abandoned"

class OperationType(Enum):
    """Operation types with enhanced categorization"""
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    CACHE_UPDATE = "cache_update"
    CACHE_INVALIDATE = "cache_invalidate"
    CONTEXT_UPDATE = "context_update"
    METADATA_UPDATE = "metadata_update"
    BACKUP_CREATE = "backup_create"
    INDEX_UPDATE = "index_update"

@dataclass
class TransactionMetrics:
    """Transaction performance metrics"""
    start_time: datetime
    end_time: Optional[datetime] = None
    operation_count: int = 0
    rollback_count: int = 0
    retry_count: int = 0
    total_time: float = 0.0
    operation_times: Dict[str, float] = None
    
    def __post_init__(self):
        self.operation_times = {}

    def add_operation_time(self, operation_type: str, duration: float):
        """Record operation execution time"""
        if operation_type not in self.operation_times:
            self.operation_times[operation_type] = []
        self.operation_times[operation_type].append(duration)

    def get_average_operation_time(self, operation_type: str) -> float:
        """Get average execution time for operation type"""
        times = self.operation_times.get(operation_type, [])
        return sum(times) / len(times) if times else 0.0

@dataclass
class Operation:
    """Enhanced operation tracking"""
    type: OperationType
    target: str
    data: Dict
    timestamp: datetime
    status: TransactionStatus
    checksum: str
    error: Optional[str] = None
    backup_path: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0

    @classmethod
    def create(cls, type: OperationType, target: str, data: Dict) -> 'Operation':
        """Create new operation with checksum"""
        checksum = hashlib.sha256(
            json.dumps({"type": type.value, "target": target, "data": data}, 
                      sort_keys=True).encode()
        ).hexdigest()
        
        return cls(
            type=type,
            target=target,
            data=data,
            timestamp=datetime.now(),
            status=TransactionStatus.PENDING,
            checksum=checksum
        )

@dataclass
class Transaction:
    """Enhanced transaction tracking"""
    id: str
    operations: List[Operation]
    status: TransactionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: TransactionMetrics = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = TransactionMetrics(start_time=self.started_at)

class TransactionManager:
    """Enhanced transaction manager with improved reliability"""
    
    def __init__(self, base_dir: Path, max_retries: int = 3):
        self.base_dir = Path(base_dir)
        self.transactions_dir = self.base_dir / "transactions"
        self.backup_dir = self.base_dir / "backups"
        self.log_dir = self.base_dir / "logs"
        self.metrics_dir = self.base_dir / "metrics"
        
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.lock = threading.RLock()
        
        # Enhanced state tracking
        self._active_transactions: Dict[str, Transaction] = {}
        self._pending_operations: Queue = Queue()
        self._shutdown_event = threading.Event()
        
        # Initialize storage and recovery
        self._initialize()
        self._start_background_tasks()
        self._register_signals()

    def _initialize(self):
        """Initialize storage and recover pending transactions"""
        try:
            # Create directories
            for dir_path in [self.transactions_dir, self.backup_dir, 
                           self.log_dir, self.metrics_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Recover pending transactions
            self._recover_pending_transactions()
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise RuntimeError("Failed to initialize transaction manager") from e

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True
        )
        self._cleanup_thread.start()

    def _register_signals(self):
        """Register signal handlers"""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    @contextmanager
    def transaction(self):
        """Enhanced transaction context manager"""
        transaction_id = None
        try:
            transaction_id = self._create_transaction_id()
            self.begin_transaction(transaction_id)
            yield transaction_id
            self.commit(transaction_id)
            
        except Exception as e:
            if transaction_id:
                self.logger.error(f"Transaction {transaction_id} failed: {e}")
                self.rollback(transaction_id)
            raise
        
        finally:
            if transaction_id in self._active_transactions:
                if self._active_transactions[transaction_id].status == TransactionStatus.PENDING:
                    self.rollback(transaction_id)

    def begin_transaction(self, transaction_id: str) -> bool:
        """Begin new transaction with enhanced tracking"""
        with self.lock:
            try:
                if transaction_id in self._active_transactions:
                    self.logger.error(f"Transaction {transaction_id} already exists")
                    return False

                transaction = Transaction(
                    id=transaction_id,
                    operations=[],
                    status=TransactionStatus.PENDING,
                    started_at=datetime.now()
                )
                
                self._active_transactions[transaction_id] = transaction
                self._persist_transaction(transaction)
                
                self.logger.debug(f"Transaction {transaction_id} started")
                return True
                
            except Exception as e:
                self.logger.error(f"Error beginning transaction: {e}")
                return False

    def add_operation(self, transaction_id: str, operation_type: OperationType,
                     target: str, data: Dict, backup: bool = True) -> bool:
        """Add operation to transaction with enhanced validation"""
        with self.lock:
            try:
                if transaction_id not in self._active_transactions:
                    self.logger.error(f"No active transaction found: {transaction_id}")
                    return False
                
                transaction = self._active_transactions[transaction_id]
                if transaction.status != TransactionStatus.PENDING:
                    self.logger.error(
                        f"Transaction {transaction_id} is not pending"
                    )
                    return False
                
                # Create and validate operation
                operation = Operation.create(operation_type, target, data)
                
                # Create backup if requested
                if backup and Path(target).exists():
                    backup_path = self._create_backup(target, transaction_id)
                    if backup_path:
                        operation.backup_path = str(backup_path)
                
                # Add operation to transaction
                transaction.operations.append(operation)
                transaction.metrics.operation_count += 1
                
                # Update persistent state
                self._persist_transaction(transaction)
                self.logger.debug(
                    f"Operation {operation_type.value} added to transaction {transaction_id}"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding operation: {e}")
                return False

    def commit(self, transaction_id: str) -> bool:
        """Commit transaction with enhanced safety"""
        with self.lock:
            try:
                if transaction_id not in self._active_transactions:
                    self.logger.error(f"No active transaction found: {transaction_id}")
                    return False
                
                transaction = self._active_transactions[transaction_id]
                if transaction.status != TransactionStatus.PENDING:
                    self.logger.error(
                        f"Transaction {transaction_id} is not pending"
                    )
                    return False
                
                # Begin commit phase
                transaction.status = TransactionStatus.COMMITTING
                self._persist_transaction(transaction)
                
                # Execute operations
                for operation in transaction.operations:
                    operation_start = time.time()
                    success = self._execute_operation(operation)
                    if not success:
                        self.rollback(transaction_id)
                        return False
                    
                    operation.execution_time = time.time() - operation_start
                    transaction.metrics.add_operation_time(
                        operation.type.value,
                        operation.execution_time
                    )
                
                # Complete transaction
                transaction.status = TransactionStatus.COMMITTED
                transaction.completed_at = datetime.now()
                transaction.metrics.end_time = transaction.completed_at
                transaction.metrics.total_time = (
                    transaction.completed_at - transaction.started_at
                ).total_seconds()
                
                # Update persistent state
                self._persist_transaction(transaction)
                self._archive_transaction(transaction_id)
                
                self.logger.info(f"Transaction {transaction_id} committed successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error committing transaction: {e}")
                self.rollback(transaction_id)
                return False

    def rollback(self, transaction_id: str) -> bool:
        """Rollback transaction with enhanced recovery"""
        with self.lock:
            try:
                transaction = self._active_transactions.get(transaction_id)
                if not transaction:
                    self.logger.error(f"No transaction found to rollback: {transaction_id}")
                    return False
                
                # Begin rollback phase
                transaction.status = TransactionStatus.ROLLING_BACK
                self._persist_transaction(transaction)
                
                # Reverse operations
                for operation in reversed(transaction.operations):
                    rollback_start = time.time()
                    try:
                        self._reverse_operation(operation)
                        operation.status = TransactionStatus.ROLLED_BACK
                    except Exception as e:
                        operation.status = TransactionStatus.FAILED
                        operation.error = str(e)
                        self.logger.error(
                            f"Error rolling back operation {operation.type.value}: {e}"
                        )
                    finally:
                        operation.execution_time = time.time() - rollback_start
                
                # Complete rollback
                transaction.status = TransactionStatus.ROLLED_BACK
                transaction.completed_at = datetime.now()
                transaction.metrics.end_time = transaction.completed_at
                transaction.metrics.total_time = (
                    transaction.completed_at - transaction.started_at
                ).total_seconds()
                transaction.metrics.rollback_count += 1
                
                # Update persistent state
                self._persist_transaction(transaction)
                self._archive_transaction(transaction_id)
                
                self.logger.info(f"Transaction {transaction_id} rolled back successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error in rollback: {e}")
                return False

    def _execute_operation(self, operation: Operation) -> bool:
        """Execute operation with retry logic"""
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                if operation.type == OperationType.FILE_WRITE:
                    return self._execute_file_write(operation)
                elif operation.type == OperationType.FILE_DELETE:
                    return self._execute_file_delete(operation)
                elif operation.type == OperationType.CACHE_UPDATE:
                    return self._execute_cache_update(operation)
                elif operation.type == OperationType.CACHE_INVALIDATE:
                    return self._execute_cache_invalidate(operation)
                else:
                    self.logger.error(f"Unsupported operation type: {operation.type}")
                    return False
                    
            except Exception as e:
                retry_count += 1
                operation.retry_count = retry_count
                operation.error = str(e)
                
                if retry_count <= self.max_retries:
                    self.logger.warning(
                        f"Retrying operation {operation.type.value} "
                        f"({retry_count}/{self.max_retries})"
                    )
                    time.sleep(1)  # Basic exponential backoff
                else:
                    self.logger.error(
                        f"Operation {operation.type.value} failed after "
                        f"{self.max_retries} retries: {e}"
                    )
                    return False

    def _reverse_operation(self, operation: Operation):
        """Reverse operation with enhanced safety"""
        if operation.backup_path:
            try:
                backup_path = Path(operation.backup_path)
                target_path = Path(operation.target)
                if backup_path.exists():
                    if operation.type in [
                        OperationType.FILE_DELETE,
                        OperationType.FILE_WRITE
                    ]:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_path, target_path)
                    
            except Exception as e:
                self.logger.error(f"Error restoring from backup: {e}")
                raise
                
        elif operation.type in [
            OperationType.CACHE_UPDATE,
            OperationType.CACHE_INVALIDATE
        ]:
            # Implement cache operation reversal
            pass
        
        elif operation.type == OperationType.CONTEXT_UPDATE:
            # Implement context update reversal
            pass

    def _execute_file_write(self, operation: Operation) -> bool:
        """Execute file write operation"""
        try:
            target_path = Path(operation.target)
            temp_path = Path(operation.data["temp_path"])
            
            if temp_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_path, target_path)
                temp_path.unlink()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing file write: {e}")
            return False

    def _execute_file_delete(self, operation: Operation) -> bool:
        """Execute file delete operation"""
        try:
            target_path = Path(operation.target)
            if target_path.exists():
                target_path.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing file delete: {e}")
            return False

    def _execute_cache_update(self, operation: Operation) -> bool:
        """Execute cache update operation"""
        # Implement cache update logic
        return True

    def _execute_cache_invalidate(self, operation: Operation) -> bool:
        """Execute cache invalidation operation"""
        # Implement cache invalidation logic
        return True

    def _create_backup(self, target: str, transaction_id: str) -> Optional[Path]:
        """Create backup with enhanced metadata"""
        try:
            source_path = Path(target)
            if not source_path.exists():
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / transaction_id / f"{source_path.name}.{timestamp}.bak"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file with metadata preservation
            shutil.copy2(source_path, backup_path)
            
            # Create backup metadata
            metadata = {
                "original_path": str(source_path),
                "backup_time": timestamp,
                "transaction_id": transaction_id,
                "checksum": self._calculate_checksum(source_path)
            }
            
            metadata_path = backup_path.with_suffix('.meta')
            with metadata_path.open('w') as f:
                json.dump(metadata, f, indent=2)
                
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            sha256_hash = hashlib.sha256()
            with file_path.open('rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating checksum: {e}")
            return ""

    def _persist_transaction(self, transaction: Transaction) -> bool:
        """Persist transaction state with safety checks"""
        try:
            file_path = self.transactions_dir / f"{transaction.id}.transaction"
            temp_path = file_path.with_suffix('.tmp')
            
            # Write to temporary file first
            with temp_path.open('w') as f:
                json.dump({
                    'id': transaction.id,
                    'status': transaction.status.value,
                    'started_at': transaction.started_at.isoformat(),
                    'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None,
                    'error': transaction.error,
                    'operations': [
                        {
                            'type': op.type.value,
                            'target': op.target,
                            'data': op.data,
                            'timestamp': op.timestamp.isoformat(),
                            'status': op.status.value,
                            'checksum': op.checksum,
                            'error': op.error,
                            'backup_path': op.backup_path,
                            'retry_count': op.retry_count,
                            'execution_time': op.execution_time
                        }
                        for op in transaction.operations
                    ],
                    'metrics': {
                        'start_time': transaction.metrics.start_time.isoformat(),
                        'end_time': transaction.metrics.end_time.isoformat() if transaction.metrics.end_time else None,
                        'operation_count': transaction.metrics.operation_count,
                        'rollback_count': transaction.metrics.rollback_count,
                        'retry_count': transaction.metrics.retry_count,
                        'total_time': transaction.metrics.total_time,
                        'operation_times': transaction.metrics.operation_times
                    }
                }, f, indent=2)
                
            # Atomic rename
            temp_path.rename(file_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error persisting transaction: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def _recover_pending_transactions(self) -> None:
        """Recover pending transactions after system restart"""
        try:
            for transaction_file in self.transactions_dir.glob("*.transaction"):
                try:
                    with transaction_file.open('r') as f:
                        data = json.load(f)
                        if data['status'] == TransactionStatus.PENDING.value:
                            self.logger.info(f"Recovering transaction: {transaction_file.stem}")
                            self._rollback_transaction(transaction_file.stem)
                            
                except Exception as e:
                    self.logger.error(f"Error recovering transaction {transaction_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in transaction recovery: {e}")

    def _archive_transaction(self, transaction_id: str) -> None:
        """Archive completed transaction"""
        try:
            source = self.transactions_dir / f"{transaction_id}.transaction"
            archive_dir = self.transactions_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            if source.exists():
                # Create archive copy with timestamp
                timestamp = datetime.now().strftime("%Y%m%d")
                archive_path = archive_dir / timestamp / f"{transaction_id}.transaction"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, archive_path)
                
                # Remove original
                source.unlink()
                
            # Remove from active transactions
            self._active_transactions.pop(transaction_id, None)
            
        except Exception as e:
            self.logger.error(f"Error archiving transaction {transaction_id}: {e}")

    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old transactions and backups"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old archives
                self.cleanup_old_transactions(days=30)
                # Clean up old backups
                self.cleanup_old_backups(days=7)
                # Sleep for 24 hours
                time.sleep(86400)
                
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                time.sleep(3600)  # Retry after an hour on error

    def cleanup_old_transactions(self, days: int = 30) -> int:
        """Clean up old archived transactions"""
        try:
            archive_dir = self.transactions_dir / "archive"
            if not archive_dir.exists():
                return 0
                
            cutoff = datetime.now() - timedelta(days=days)
            removed = 0
            
            for date_dir in archive_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                    
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y%m%d")
                    if dir_date < cutoff:
                        shutil.rmtree(date_dir)
                        removed += 1
                except ValueError:
                    continue
                    
            return removed
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old transactions: {e}")
            return 0

    def cleanup_old_backups(self, days: int = 7) -> int:
        """Clean up old backup files"""
        try:
            if not self.backup_dir.exists():
                return 0
                
            cutoff = datetime.now() - timedelta(days=days)
            removed = 0
            
            for backup_file in self.backup_dir.rglob("*.bak"):
                try:
                    if backup_file.stat().st_mtime < cutoff.timestamp():
                        backup_file.unlink()
                        # Remove metadata file if exists
                        meta_file = backup_file.with_suffix('.meta')
                        if meta_file.exists():
                            meta_file.unlink()
                        removed += 1
                except Exception as e:
                    self.logger.error(f"Error removing backup file {backup_file}: {e}")
                    continue
                    
            return removed
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
            return 0

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal"""
        self.logger.info(f"Received shutdown signal {signum}")
        self.shutdown()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Shutdown transaction manager"""
        try:
            self.logger.info("Initiating transaction manager shutdown")
            self._shutdown_event.set()
            
            # Wait for cleanup thread
            if timeout:
                self._cleanup_thread.join(timeout=timeout)
            else:
                self._cleanup_thread.join()
                
            # Rollback any pending transactions
            with self.lock:
                for transaction_id, transaction in self._active_transactions.items():
                    if transaction.status == TransactionStatus.PENDING:
                        self.rollback(transaction_id)
                        
            self.logger.info("Transaction manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionStatus]:
        """Get current transaction status"""
        try:
            # Check active transactions
            if transaction_id in self._active_transactions:
                return self._active_transactions[transaction_id].status
                
            # Check archived transactions
            archive_path = None
            for date_dir in (self.transactions_dir / "archive").iterdir():
                if not date_dir.is_dir():
                    continue
                potential_path = date_dir / f"{transaction_id}.transaction"
                if potential_path.exists():
                    archive_path = potential_path
                    break
                    
            if archive_path and archive_path.exists():
                with archive_path.open('r') as f:
                    data = json.load(f)
                    return TransactionStatus(data['status'])
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving transaction status: {e}")
            return None

    def get_transaction_metrics(self, transaction_id: str) -> Optional[Dict]:
        """Get transaction metrics"""
        try:
            if transaction_id in self._active_transactions:
                transaction = self._active_transactions[transaction_id]
                return {
                    'operation_count': transaction.metrics.operation_count,
                    'rollback_count': transaction.metrics.rollback_count,
                    'retry_count': transaction.metrics.retry_count,
                    'total_time': transaction.metrics.total_time,
                    'operation_times': transaction.metrics.operation_times
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving transaction metrics: {e}")
            return None