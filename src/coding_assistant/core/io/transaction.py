# src/coding_assistant/core/io/transaction.py
from typing import Dict, List, Optional, Callable, Union
from datetime import datetime
import logging
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import shutil


class TransactionStatus(Enum):
    """Represents the possible states of a transaction"""
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class OperationType(Enum):
    """Types of operations that can be performed in a transaction"""
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    CACHE_UPDATE = "cache_update"
    CACHE_INVALIDATE = "cache_invalidate"
    CONTEXT_UPDATE = "context_update"
    METADATA_UPDATE = "metadata_update"


@dataclass
class Operation:
    """Represents a single operation within a transaction"""
    type: OperationType
    target: str
    data: Dict
    timestamp: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    error: Optional[str] = None
    backup_path: Optional[str] = None


@dataclass
class Transaction:
    """Represents a complete transaction"""
    id: str
    operations: List[Operation]
    status: TransactionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class TransactionManager:
    """Manages atomic transactions across file system and cache operations"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.transactions_dir = self.base_dir / "transactions"
        self.backup_dir = self.base_dir / "backups"
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        self._active_transactions: Dict[str, Transaction] = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize the transaction system directories and recover pending transactions"""
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._recover_pending_transactions()
        
    def _recover_pending_transactions(self):
        """Recover any pending transactions after system restart"""
        try:
            for transaction_file in self.transactions_dir.glob("*.transaction"):
                with transaction_file.open('r') as f:
                    data = json.load(f)
                    if data['status'] == TransactionStatus.PENDING.value:
                        # Attempt to rollback pending transaction
                        transaction_id = transaction_file.stem
                        self.logger.info(f"Recovering pending transaction: {transaction_id}")
                        self._rollback_transaction(transaction_id)
        except Exception as e:
            self.logger.error(f"Error recovering transactions: {e}")
            
    @contextmanager
    def transaction(self):
        """Context manager for transaction handling"""
        transaction_id = self._create_transaction_id()
        try:
            self.begin_transaction(transaction_id)
            yield transaction_id
            self.commit(transaction_id)
        except Exception as e:
            self.logger.error(f"Transaction {transaction_id} failed: {e}")
            self.rollback(transaction_id)
            raise
                
    def _create_transaction_id(self) -> str:
        """Create a unique transaction ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"tx_{timestamp}"
        
    def begin_transaction(self, transaction_id: str) -> bool:
        """Start a new transaction"""
        with self.lock:
            if transaction_id in self._active_transactions:
                self.logger.error(f"Transaction {transaction_id} already exists.")
                return False
            transaction = Transaction(
                id=transaction_id,
                operations=[],
                status=TransactionStatus.PENDING,
                started_at=datetime.now()
            )
            self._active_transactions[transaction_id] = transaction
            self._persist_transaction(transaction)
            self.logger.debug(f"Transaction {transaction_id} started.")
            return True
            
    def add_operation(self, transaction_id: str, operation_type: OperationType,
                     target: str, data: Dict, backup: bool = True) -> bool:
        """Add an operation to a transaction"""
        with self.lock:
            if transaction_id not in self._active_transactions:
                self.logger.error(f"No active transaction found: {transaction_id}")
                return False
                
            transaction = self._active_transactions[transaction_id]
            if transaction.status != TransactionStatus.PENDING:
                self.logger.error(f"Transaction {transaction_id} is not pending.")
                return False
                
            # Create backup if requested and applicable
            backup_path = None
            if backup and Path(target).exists():
                backup_path = self._create_backup(target, transaction_id)
                
            operation = Operation(
                type=operation_type,
                target=target,
                data=data,
                timestamp=datetime.now(),
                backup_path=str(backup_path) if backup_path else None
            )
            transaction.operations.append(operation)
            
            # Persist updated transaction state
            self._persist_transaction(transaction)
            self.logger.debug(f"Operation {operation_type.value} added to transaction {transaction_id}.")
            return True
            
    def commit(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        with self.lock:
            if transaction_id not in self._active_transactions:
                self.logger.error(f"No active transaction found: {transaction_id}")
                return False
                
            transaction = self._active_transactions[transaction_id]
            if transaction.status != TransactionStatus.PENDING:
                self.logger.error(f"Transaction {transaction_id} is not pending.")
                return False
                
            try:
                # Mark as committed
                transaction.status = TransactionStatus.COMMITTED
                transaction.completed_at = datetime.now()
                
                # Persist final state
                self._persist_transaction(transaction)
                
                # Archive transaction
                self._archive_transaction(transaction_id)
                self.logger.info(f"Transaction {transaction_id} committed successfully.")
                return True
                
            except Exception as e:
                self.logger.error(f"Error committing transaction {transaction_id}: {e}")
                transaction.status = TransactionStatus.FAILED
                transaction.error = str(e)
                self._persist_transaction(transaction)
                return False
                
    def rollback(self, transaction_id: str) -> bool:
        """Rollback a transaction"""
        with self.lock:
            return self._rollback_transaction(transaction_id)
                
    def _rollback_transaction(self, transaction_id: str) -> bool:
        """Internal method to rollback a transaction"""
        try:
            transaction = self._active_transactions.get(transaction_id)
            if not transaction:
                self.logger.error(f"No transaction found to rollback: {transaction_id}")
                return False
                
            # Reverse operations
            for operation in reversed(transaction.operations):
                try:
                    self._reverse_operation(operation)
                    operation.status = TransactionStatus.ROLLED_BACK
                    self.logger.debug(f"Operation {operation.type.value} rolled back.")
                except Exception as e:
                    self.logger.error(f"Error rolling back operation {operation.type.value}: {e}")
                    operation.status = TransactionStatus.FAILED
                    operation.error = str(e)
                    
            transaction.status = TransactionStatus.ROLLED_BACK
            transaction.completed_at = datetime.now()
            
            # Persist final state
            self._persist_transaction(transaction)
            
            # Archive transaction
            self._archive_transaction(transaction_id)
            self.logger.info(f"Transaction {transaction_id} rolled back successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in rollback of {transaction_id}: {e}")
            return False
            
    def _reverse_operation(self, operation: Operation):
        """Reverse an individual operation"""
        if operation.backup_path:
            # Restore from backup
            try:
                backup_path = Path(operation.backup_path)
                target_path = Path(operation.target)
                if backup_path.exists():
                    if operation.type == OperationType.FILE_DELETE or operation.type == OperationType.FILE_WRITE:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_path, target_path)
            except Exception as e:
                self.logger.error(f"Error restoring from backup: {e}")
                raise
                
        elif operation.type == OperationType.CACHE_UPDATE or operation.type == OperationType.CACHE_INVALIDATE:
            # Implement cache operation reversal if applicable
            pass  # Placeholder for cache reversal logic
                
        elif operation.type == OperationType.CONTEXT_UPDATE:
            # Context updates should be handled by the context manager
            pass  # Placeholder for context reversal logic
                
    def _create_backup(self, target: str, transaction_id: str) -> Optional[Path]:
        """Create a backup of a target file"""
        try:
            source_path = Path(target)
            if not source_path.exists():
                return None
                
            backup_path = self.backup_dir / transaction_id / source_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, backup_path)
            self.logger.debug(f"Backup created at {backup_path} for transaction {transaction_id}.")
            return backup_path
                
        except Exception as e:
            self.logger.error(f"Error creating backup for {target}: {e}")
            return None
                
    def _persist_transaction(self, transaction: Transaction):
        """Persist transaction state to disk"""
        try:
            file_path = self.transactions_dir / f"{transaction.id}.transaction"
            with file_path.open('w') as f:
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
                            'error': op.error,
                            'backup_path': op.backup_path
                        }
                        for op in transaction.operations
                    ]
                }, f, indent=2)
            self.logger.debug(f"Transaction {transaction.id} persisted.")
        except Exception as e:
            self.logger.error(f"Error persisting transaction {transaction.id}: {e}")
            raise
                
    def _cleanup_backups(self, transaction_id: str):
        """Clean up backup files for a committed transaction"""
        try:
            backup_dir = self.backup_dir / transaction_id
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                self.logger.debug(f"Backups for transaction {transaction_id} cleaned up.")
        except Exception as e:
            self.logger.error(f"Error cleaning up backups for {transaction_id}: {e}")
                
    def _archive_transaction(self, transaction_id: str):
        """Archive a completed transaction"""
        try:
            source = self.transactions_dir / f"{transaction_id}.transaction"
            archive_dir = self.transactions_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            if source.exists():
                shutil.move(source, archive_dir / f"{transaction_id}.transaction")
                self.logger.debug(f"Transaction {transaction_id} archived.")
                
            # Remove from active transactions
            self._active_transactions.pop(transaction_id, None)
                
        except Exception as e:
            self.logger.error(f"Error archiving transaction {transaction_id}: {e}")
                
    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionStatus]:
        """Get the current status of a transaction"""
        if transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id].status
                
        # Check archived transactions
        try:
            archive_path = self.transactions_dir / "archive" / f"{transaction_id}.transaction"
            if archive_path.exists():
                with archive_path.open('r') as f:
                    data = json.load(f)
                    return TransactionStatus(data['status'])
        except Exception as e:
            self.logger.error(f"Error retrieving status for transaction {transaction_id}: {e}")
                
        return None
                
    def cleanup_old_transactions(self, days: int = 30) -> int:
        """Clean up old archived transactions"""
        try:
            archive_dir = self.transactions_dir / "archive"
            if not archive_dir.exists():
                return 0
                
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            removed = 0
            
            for transaction_file in archive_dir.glob("*.transaction"):
                if transaction_file.stat().st_mtime < cutoff:
                    transaction_file.unlink()
                    removed += 1
                    self.logger.debug(f"Old transaction {transaction_file.name} removed.")
                    
            return removed
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old transactions: {e}")
            return 0
