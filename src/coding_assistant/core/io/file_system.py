from pathlib import Path
from typing import Optional, List, Dict, Union
import shutil
import json
import yaml
from datetime import datetime
import fcntl
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from coding_assistant.core.io.transaction import TransactionManager, OperationType, TransactionStatus
import dataclasses


class FileOperation(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    

@dataclass
class FileMetadata:
    path: Path
    version: str
    hash: str
    created_at: str
    modified_at: str
    backup_path: Optional[Path] = None


class FileSystemHandler:
    """Handles file operations with versioning, locking, and atomic operations"""
    
    def __init__(self, base_dir: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None, transaction_manager: Optional[TransactionManager] = None):
        self.base_dir = Path(base_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.base_dir / "_backups"
        self.metadata_dir = self.base_dir / "_metadata"
        self.logger = logging.getLogger(__name__)
        self.transaction_manager = transaction_manager or TransactionManager(self.base_dir / "transactions")
        self._initialize_directories()
        
    def _initialize_directories(self):
        """Ensure required directories exist"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_version(self) -> str:
        """Generate a version string based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
        
    def _calculate_hash(self, content: Union[str, bytes]) -> str:
        """Calculate SHA-256 hash of content"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
        
    def _get_metadata_path(self, file_path: Path) -> Path:
        """Get metadata file path for a given file"""
        rel_path = file_path.relative_to(self.base_dir)
        return self.metadata_dir / f"{rel_path}.meta"
        
    def _load_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Load metadata for a file"""
        try:
            meta_path = self._get_metadata_path(file_path)
            if not meta_path.exists():
                return None
            with meta_path.open('r') as f:
                data = json.load(f)
                return FileMetadata(**data)
        except Exception as e:
            self.logger.error(f"Error loading metadata for {file_path}: {e}")
            return None
            
    def _save_metadata(self, file_path: Path, metadata: FileMetadata):
        """Save metadata for a file"""
        try:
            meta_path = self._get_metadata_path(file_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with meta_path.open('w') as f:
                json.dump(dataclasses.asdict(metadata), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata for {file_path}: {e}")
                
    def atomic_write(self, rel_path: str, content: Union[str, bytes], 
                    communication_format: str = "yaml", backup: bool = True) -> Optional[str]:
        """
        Write content atomically with versioning and optional backup.
        
        Args:
            rel_path (str): Relative path to the file.
            content (Union[str, bytes]): Content to write.
            communication_format (str): Format of the file ('yaml' or 'json').
            backup (bool): Whether to create a backup before writing.
        
        Returns:
            Optional[str]: Version identifier if successful, None otherwise.
        """
        try:
            file_path = self.base_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Acquire file lock for atomicity
            with file_path.open('a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                fcntl.flock(f, fcntl.LOCK_UN)
                
            # Create temporary file
            temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
            if communication_format.lower() == "json":
                temp_path.write_text(json.dumps(content, indent=2))
            else:
                temp_path.write_text(yaml.safe_dump(content))
                
            # Calculate hash and create metadata
            content_hash = self._calculate_hash(content)
            version = self._create_version()
            
            # Create backup if requested and file exists
            backup_path = None
            if backup and file_path.exists():
                backup_path = self.create_backup(rel_path)
                
            # Atomic rename
            temp_path.rename(file_path)
            
            # Save metadata
            metadata = FileMetadata(
                path=file_path,
                version=version,
                hash=content_hash,
                created_at=datetime.now().isoformat(),
                modified_at=datetime.now().isoformat(),
                backup_path=backup_path
            )
            self._save_metadata(file_path, metadata)
            
            # Log transaction if TransactionManager is available
            if self.transaction_manager:
                self.transaction_manager.add_operation(
                    transaction_id=self.transaction_manager._create_transaction_id(),
                    operation_type=OperationType.FILE_WRITE,
                    target=str(file_path),
                    data={"version": version, "backup_path": str(backup_path) if backup_path else None},
                    backup=False
                )
                
            self.logger.info(f"File {rel_path} written atomically with version {version}.")
            return version
            
        except Exception as e:
            self.logger.error(f"Error in atomic_write for {rel_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None
            
    def create_backup(self, rel_path: str) -> Optional[Path]:
        """
        Create a backup of a file.
        
        Args:
            rel_path (str): Relative path to the file.
        
        Returns:
            Optional[Path]: Path to the backup file or None if failed.
        """
        try:
            source_path = self.base_dir / rel_path
            if not source_path.exists():
                self.logger.warning(f"Cannot backup non-existent file: {rel_path}")
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{rel_path}.{timestamp}.bak"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, backup_path)
            self.logger.debug(f"Backup created at {backup_path} for file {rel_path}.")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup for {rel_path}: {e}")
            return None
            
    def read_file(self, rel_path: str, version: Optional[str] = None, communication_format: str = "yaml") -> Optional[Union[Dict, str]]:
        """
        Read file content, optionally from a specific version.
        
        Args:
            rel_path (str): Relative path to the file.
            version (Optional[str]): Specific version to load.
            communication_format (str): Format of the file ('yaml' or 'json').
        
        Returns:
            Optional[Union[Dict, str]]: Loaded content or None if an error occurs.
        """
        try:
            if version:
                content = self.load_version(rel_path, version, communication_format)
                if content is not None:
                    self.logger.debug(f"Loaded version {version} of {rel_path} from versions.")
                return content
                
            file_path = self.base_dir / rel_path
            if not file_path.exists():
                self.logger.warning(f"File {rel_path} does not exist.")
                return None
                
            # Acquire file lock for safe reading
            with file_path.open('r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = f.read()
                fcntl.flock(f, fcntl.LOCK_UN)
                
            if communication_format.lower() == "json":
                return json.loads(data)
            else:
                return yaml.safe_load(data)
                
        except Exception as e:
            self.logger.error(f"Error reading file {rel_path}: {e}")
            return None
            
    def load_version(self, rel_path: str, version_timestamp: str, communication_format: str = "yaml") -> Optional[Union[Dict, str]]:
        """
        Load a specific version of a file.
        
        Args:
            rel_path (str): Relative path to the file.
            version_timestamp (str): Timestamp identifier of the version.
            communication_format (str): Format of the file ('yaml' or 'json').
        
        Returns:
            Optional[Union[Dict, str]]: Loaded content or None if an error occurs.
        """
        try:
            version_path = self.base_dir / "versions" / f"{rel_path}.{version_timestamp}"
            if not version_path.exists():
                self.logger.warning(f"Version {version_timestamp} of {rel_path} does not exist.")
                return None
                
            with version_path.open('r') as f:
                data = f.read()
                
            if communication_format.lower() == "json":
                return json.loads(data)
            else:
                return yaml.safe_load(data)
        except Exception as e:
            self.logger.error(f"Error loading version {version_timestamp} of {rel_path}: {e}")
            return None

    def delete_file(self, rel_path: str) -> bool:
        """
        Delete a file with backup creation.
        
        Args:
            rel_path (str): Relative path to the file.
        
        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        try:
            file_path = self.base_dir / rel_path
            if not file_path.exists():
                self.logger.warning(f"File {rel_path} does not exist.")
                return False

            # Create backup before deletion
            backup_path = self.create_backup(rel_path)
            if backup_path is None:
                self.logger.error(f"Failed to create backup for {rel_path}. Deletion aborted.")
                return False

            # Perform atomic deletion within a transaction
            if self.transaction_manager:
                with self.transaction_manager.transaction() as transaction_id:
                    self.transaction_manager.add_operation(
                        transaction_id,
                        OperationType.FILE_DELETE,
                        str(file_path),
                        {"backup_path": str(backup_path)},
                        backup=False
                    )
                    file_path.unlink()
            else:
                file_path.unlink()

            self.logger.info(f"File {rel_path} deleted successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file {rel_path}: {e}")
            return False
