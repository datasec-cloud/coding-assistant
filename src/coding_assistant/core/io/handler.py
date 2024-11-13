# src/coding_assistant/core/io/handler.py
from pathlib import Path
from typing import Dict, Optional, Union, List
import yaml
import json
import logging
from datetime import datetime
from coding_assistant.core.io.transaction import TransactionManager, OperationType
import shutil

class IOHandler:
    """Handles basic I/O operations with versioning and transaction support"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.transaction_manager = TransactionManager(self.base_dir / "transactions")
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories"""
        (self.base_dir / "context").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "versions").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "temp").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "backups").mkdir(parents=True, exist_ok=True)

    def save_yaml(self, data: Dict, filepath: str, create_version: bool = True, communication_format: str = "yaml") -> bool:
        """
        Save data in YAML or JSON format with optional versioning.
        
        Args:
            data (Dict): Data to be saved.
            filepath (str): Relative path to save the file.
            create_version (bool): Whether to create a version backup.
            communication_format (str): Format to save ('yaml' or 'json').
        
        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            full_path = self.base_dir / filepath
            with self.transaction_manager.transaction() as transaction_id:
                # Add write operation
                self.transaction_manager.add_operation(
                    transaction_id,
                    OperationType.FILE_WRITE,
                    str(full_path),
                    {"content": data, "format": communication_format},
                    backup=create_version
                )
                
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content based on format
                if communication_format.lower() == "json":
                    with full_path.open('w') as f:
                        json.dump(data, f, indent=2)
                else:
                    with full_path.open('w') as f:
                        yaml.safe_dump(data, f)
        
                if create_version:
                    version_path = self.base_dir / "versions" / f"{filepath}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    version_path.parent.mkdir(parents=True, exist_ok=True)
                    if communication_format.lower() == "json":
                        with version_path.open('w') as f:
                            json.dump(data, f, indent=2)
                    else:
                        with version_path.open('w') as f:
                            yaml.safe_dump(data, f)
        
                return True
        except Exception as e:
            self.logger.error(f"Error saving {filepath}: {e}")
            return False

    def load_yaml(self, filepath: str, communication_format: str = "yaml") -> Optional[Dict]:
        """
        Load data from a YAML or JSON file.
        
        Args:
            filepath (str): Relative path to the file.
            communication_format (str): Format of the file ('yaml' or 'json').
        
        Returns:
            Optional[Dict]: Loaded data or None if an error occurs.
        """
        try:
            full_path = self.base_dir / filepath
            if not full_path.exists():
                self.logger.warning(f"File {filepath} does not exist.")
                return None
            
            with full_path.open('r') as f:
                if communication_format.lower() == "json":
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None

    def get_versions(self, filepath: str) -> List[str]:
        """
        Retrieve all versions of a file.
        
        Args:
            filepath (str): Relative path to the file.
        
        Returns:
            List[str]: List of version identifiers.
        """
        try:
            version_dir = self.base_dir / "versions"
            versions = list(version_dir.glob(f"{filepath}.*"))
            return sorted([version.stem.split(f"{filepath}.")[-1] for version in versions])
        except Exception as e:
            self.logger.error(f"Error retrieving versions of {filepath}: {e}")
            return []

    def load_version(self, filepath: str, version_timestamp: str, communication_format: str = "yaml") -> Optional[Dict]:
        """
        Load a specific version of a file.
        
        Args:
            filepath (str): Relative path to the file.
            version_timestamp (str): Timestamp identifier of the version.
            communication_format (str): Format of the file ('yaml' or 'json').
        
        Returns:
            Optional[Dict]: Loaded data or None if an error occurs.
        """
        try:
            version_path = self.base_dir / "versions" / f"{filepath}.{version_timestamp}"
            if not version_path.exists():
                self.logger.warning(f"Version {version_timestamp} of {filepath} does not exist.")
                return None
            
            with version_path.open('r') as f:
                if communication_format.lower() == "json":
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading version {version_timestamp} of {filepath}: {e}")
            return None

    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file with backup creation.
        
        Args:
            filepath (str): Relative path to the file.
        
        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        try:
            full_path = self.base_dir / filepath
            if not full_path.exists():
                self.logger.warning(f"File {filepath} does not exist.")
                return False

            # Create backup before deletion
            backup_path = self.create_backup(filepath)
            if backup_path is None:
                self.logger.error(f"Failed to create backup for {filepath}. Deletion aborted.")
                return False

            with self.transaction_manager.transaction() as transaction_id:
                # Add delete operation
                self.transaction_manager.add_operation(
                    transaction_id,
                    OperationType.FILE_DELETE,
                    str(full_path),
                    {"backup_path": str(backup_path)},
                    backup=False
                )

                # Perform deletion
                full_path.unlink()
                
            return True
        except Exception as e:
            self.logger.error(f"Error deleting {filepath}: {e}")
            return False

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
            backup_path = self.base_dir / "backups" / f"{rel_path}.{timestamp}.bak"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_path)
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup for {rel_path}: {e}")
            return None
