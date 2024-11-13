from pathlib import Path
from typing import Dict, Optional, Union, List, Any
import yaml
import json
import logging
from datetime import datetime
import fcntl
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
import threading
from contextlib import contextmanager
from coding_assistant.core.io.transaction import TransactionManager, OperationType

class FileFormat(str, Enum):
    """Supported file formats"""
    YAML = "yaml"
    JSON = "json"
    
@dataclass
class FileMetadata:
    """File metadata information"""
    path: Path
    format: FileFormat
    version: str
    checksum: str
    created_at: datetime
    modified_at: datetime
    backup_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "path": str(self.path),
            "format": self.format.value,
            "version": self.version,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "backup_path": str(self.backup_path) if self.backup_path else None
        }

class IOHandler:
    """Enhanced I/O handler with improved safety and transaction support"""
    
    def __init__(self, base_dir: Path, enable_transactions: bool = True):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.transaction_manager = TransactionManager(self.base_dir / "transactions") if enable_transactions else None
        self.lock = threading.RLock()
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage directories with error handling"""
        try:
            for dir_name in ["context", "versions", "temp", "backups", "metadata"]:
                directory = self.base_dir / dir_name
                directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize storage directories: {e}")
            raise RuntimeError("Storage initialization failed") from e

    def _calculate_checksum(self, content: Union[str, bytes]) -> str:
        """Calculate SHA-256 checksum of content"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    @contextmanager
    def _file_lock(self, file_path: Path):
        """Context manager for file locking"""
        lock_path = file_path.with_suffix('.lock')
        lock_fd = None
        try:
            lock_fd = open(lock_path, 'w')
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
                lock_path.unlink(missing_ok=True)

    def save_file(self, rel_path: str, content: Union[Dict, List, str],
                format: FileFormat = FileFormat.YAML,
                create_version: bool = True) -> Optional[str]:
        """Save file with enhanced safety and version control"""
        try:
            file_path = self.base_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate checksum before modifications
            content_str = (json.dumps(content) if isinstance(content, (dict, list))
                         else str(content))
            checksum = self._calculate_checksum(content_str)
            
            # Create version identifier
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with self.lock:
                if self.transaction_manager:
                    return self._transactional_save(
                        file_path, content, format, version, checksum, create_version
                    )
                else:
                    return self._direct_save(
                        file_path, content, format, version, checksum, create_version
                    )
                
        except Exception as e:
            self.logger.error(f"Error saving file {rel_path}: {e}")
            return None

    def _transactional_save(self, file_path: Path, content: Any,
                          format: FileFormat, version: str,
                          checksum: str, create_version: bool) -> Optional[str]:
        """Save file with transaction support"""
        try:
            with self.transaction_manager.transaction() as transaction_id:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    dir=self.base_dir / "temp",
                    delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    
                    # Write content
                    if format == FileFormat.JSON:
                        json.dump(content, temp_file, indent=2)
                    else:
                        yaml.safe_dump(content, temp_file)
                
                # Create backup if file exists
                backup_path = None
                if file_path.exists():
                    backup_path = self.create_backup(str(file_path.relative_to(self.base_dir)))
                
                # Add operation to transaction
                self.transaction_manager.add_operation(
                    transaction_id,
                    OperationType.FILE_WRITE,
                    str(file_path),
                    {
                        "temp_path": str(temp_path),
                        "backup_path": str(backup_path) if backup_path else None,
                        "format": format.value,
                        "version": version,
                        "checksum": checksum
                    },
                    backup=True
                )
                
                # Perform atomic rename
                with self._file_lock(file_path):
                    temp_path.rename(file_path)
                
                # Save metadata
                self._save_metadata(file_path, format, version, checksum, backup_path)
                
                # Create version if requested
                if create_version:
                    self._create_version(file_path, content, format, version)
                
                return version
                
        except Exception as e:
            self.logger.error(f"Error in transactional save: {e}")
            if 'temp_path' in locals():
                Path(temp_path).unlink(missing_ok=True)
            return None

    def _direct_save(self, file_path: Path, content: Any,
                    format: FileFormat, version: str,
                    checksum: str, create_version: bool) -> Optional[str]:
        """Save file directly without transaction support"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=self.base_dir / "temp",
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Write content
                if format == FileFormat.JSON:
                    json.dump(content, temp_file, indent=2)
                else:
                    yaml.safe_dump(content, temp_file)
            
            # Create backup if file exists
            backup_path = None
            if file_path.exists():
                backup_path = self.create_backup(str(file_path.relative_to(self.base_dir)))
            
            # Perform atomic rename
            with self._file_lock(file_path):
                temp_path.rename(file_path)
            
            # Save metadata
            self._save_metadata(file_path, format, version, checksum, backup_path)
            
            # Create version if requested
            if create_version:
                self._create_version(file_path, content, format, version)
            
            return version
            
        except Exception as e:
            self.logger.error(f"Error in direct save: {e}")
            if 'temp_path' in locals():
                Path(temp_path).unlink(missing_ok=True)
            return None

    def load_file(self, rel_path: str, format: FileFormat = FileFormat.YAML,
                 version: Optional[str] = None) -> Optional[Union[Dict, List, str]]:
        """Load file with enhanced safety and version support"""
        try:
            if version:
                return self._load_version(rel_path, version, format)
                
            file_path = self.base_dir / rel_path
            if not file_path.exists():
                self.logger.warning(f"File {rel_path} does not exist")
                return None
            
            with self._file_lock(file_path):
                with file_path.open('r') as f:
                    if format == FileFormat.JSON:
                        content = json.load(f)
                    else:
                        content = yaml.safe_load(f)
                    
                # Verify checksum
                metadata = self._load_metadata(file_path)
                if metadata:
                    current_checksum = self._calculate_checksum(
                        json.dumps(content) if isinstance(content, (dict, list))
                        else str(content)
                    )
                    if current_checksum != metadata.checksum:
                        self.logger.error(f"Checksum mismatch for {rel_path}")
                        return None
                    
                return content
                
        except Exception as e:
            self.logger.error(f"Error loading file {rel_path}: {e}")
            return None

    def create_backup(self, rel_path: str) -> Optional[Path]:
        """Create file backup with metadata"""
        try:
            source_path = self.base_dir / rel_path
            if not source_path.exists():
                self.logger.warning(f"Cannot backup non-existent file: {rel_path}")
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.base_dir / "backups" / f"{rel_path}.{timestamp}.bak"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file with metadata preservation
            shutil.copy2(source_path, backup_path)
            
            # Copy metadata if exists
            source_metadata = self._load_metadata(source_path)
            if source_metadata:
                source_metadata.backup_path = backup_path
                self._save_metadata(backup_path, source_metadata.format,
                                 source_metadata.version, source_metadata.checksum)
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {rel_path}: {e}")
            return None

    def _save_metadata(self, file_path: Path, format: FileFormat,
                     version: str, checksum: str,
                     backup_path: Optional[Path] = None) -> None:
        """Save file metadata"""
        try:
            metadata = FileMetadata(
                path=file_path,
                format=format,
                version=version,
                checksum=checksum,
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                modified_at=datetime.now(),
                backup_path=backup_path
            )
            
            metadata_path = self.base_dir / "metadata" / f"{file_path.name}.meta"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with metadata_path.open('w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metadata for {file_path}: {e}")

    def _load_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Load file metadata"""
        try:
            metadata_path = self.base_dir / "metadata" / f"{file_path.name}.meta"
            if not metadata_path.exists():
                return None
            
            with metadata_path.open('r') as f:
                data = json.load(f)
                return FileMetadata(
                    path=Path(data["path"]),
                    format=FileFormat(data["format"]),
                    version=data["version"],
                    checksum=data["checksum"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    modified_at=datetime.fromisoformat(data["modified_at"]),
                    backup_path=Path(data["backup_path"]) if data.get("backup_path") else None
                )
                
        except Exception as e:
            self.logger.error(f"Error loading metadata for {file_path}: {e}")
            return None

    def _create_version(self, file_path: Path, content: Any,
                      format: FileFormat, version: str) -> None:
        """Create version copy of file"""
        try:
            version_path = self.base_dir / "versions" / f"{file_path.name}.{version}"
            version_path.parent.mkdir(parents=True, exist_ok=True)
            
            with version_path.open('w') as f:
                if format == FileFormat.JSON:
                    json.dump(content, f, indent=2)
                else:
                    yaml.safe_dump(content, f)
                    
        except Exception as e:
            self.logger.error(f"Error creating version for {file_path}: {e}")

    def _load_version(self, rel_path: str, version: str,
                    format: FileFormat) -> Optional[Union[Dict, List, str]]:
        """Load specific version of file"""
        try:
            file_path = Path(rel_path)
            version_path = self.base_dir / "versions" / f"{file_path.name}.{version}"
            
            if not version_path.exists():
                self.logger.warning(f"Version {version} of {rel_path} does not exist")
                return None
            
            with version_path.open('r') as f:
                if format == FileFormat.JSON:
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading version {version} of {rel_path}: {e}")
            return None

    def delete_file(self, rel_path: str) -> bool:
        """Delete file with backup and transaction support"""
        try:
            file_path = self.base_dir / rel_path
            if not file_path.exists():
                self.logger.warning(f"File {rel_path} does not exist")
                return False
            
            # Create backup before deletion
            backup_path = self.create_backup(rel_path)
            if not backup_path:
                return False
            
            with self.lock:
                if self.transaction_manager:
                    with self.transaction_manager.transaction() as transaction_id:
                        self.transaction_manager.add_operation(
                            transaction_id,
                            OperationType.FILE_DELETE,
                            str(file_path),
                            {"backup_path": str(backup_path)},
                            backup=True
                        )
                        file_path.unlink()
                else:
                    file_path.unlink()
                    
                # Remove metadata
                metadata_path = self.base_dir / "metadata" / f"{file_path.name}.meta"
                if metadata_path.exists():
                    metadata_path.unlink()
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting file {rel_path}: {e}")
            return False