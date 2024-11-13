"""Version management for context data"""
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import logging
import yaml
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib
import json
import fcntl
import shutil

class VersionType(Enum):
    """Types of version changes"""
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, non-breaking
    PATCH = "patch"  # Bug fixes, small changes

@dataclass
class VersionInfo:
    """Information about a specific version"""
    id: str
    type: VersionType
    description: str
    timestamp: datetime
    changes: Dict
    checksum: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "changes": self.changes,
            "checksum": self.checksum,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create instance from dictionary"""
        # Convert string timestamp back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # Convert string version type back to enum
        data['type'] = VersionType(data['type'])
        return cls(**data)

class ContextVersionManager:
    """Manages versioning for context data"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize storage directories"""
        try:
            self.versions_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize version storage: {e}")
            raise
            
    def _calculate_checksum(self, content: Dict) -> str:
        """Calculate checksum for version content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _validate_version_data(self, context: Dict, version_type: VersionType,
                             description: str) -> Optional[str]:
        """Validate version data before creation"""
        if not isinstance(context, dict):
            return "Context must be a dictionary"
        if not isinstance(version_type, VersionType):
            return f"Invalid version type: {version_type}"
        if not description or not isinstance(description, str):
            return "Description must be a non-empty string"
        return None

    def create_version(self, context: Dict, version_type: VersionType,
                      description: str, changes: Dict) -> Optional[str]:
        """Create a new version with improved validation and error handling"""
        try:
            # Validate inputs
            validation_error = self._validate_version_data(context, version_type, description)
            if validation_error:
                self.logger.error(f"Version validation failed: {validation_error}")
                return None
            
            timestamp = datetime.now()
            version_id = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
            checksum = self._calculate_checksum(context)
            
            version_info = VersionInfo(
                id=version_id,
                type=version_type,
                description=description,
                timestamp=timestamp,
                changes=changes,
                checksum=checksum,
                metadata={
                    "created_at": timestamp.isoformat(),
                    "version_type": version_type.value
                }
            )
            
            # Save version data with file locking
            version_path = self.versions_dir / f"{version_id}.yaml"
            temp_path = version_path.with_suffix('.tmp')
            
            try:
                with temp_path.open('w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    yaml.safe_dump({
                        "info": version_info.to_dict(),
                        "context": context,
                        "checksum": checksum
                    }, f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Atomic rename
                temp_path.rename(version_path)
                
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
                
            self.logger.info(f"Created version {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error creating version: {e}")
            return None
            
    def get_version(self, version_id: str) -> Optional[Dict]:
        """Get specific version data with validation"""
        try:
            if not version_id or not isinstance(version_id, str):
                self.logger.error("Invalid version ID provided")
                return None

            version_path = self.versions_dir / f"{version_id}.yaml"
            if not version_path.exists():
                self.logger.warning(f"Version {version_id} does not exist")
                return None
                
            with version_path.open('r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = yaml.safe_load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Verify checksum
                if 'checksum' in data:
                    calculated_checksum = self._calculate_checksum(data['context'])
                    if calculated_checksum != data['checksum']:
                        self.logger.error(f"Checksum mismatch for version {version_id}")
                        return None
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading version {version_id}: {e}")
            return None
            
    def list_versions(self, limit: Optional[int] = None) -> List[VersionInfo]:
        """List available versions with improved error handling"""
        try:
            versions = []
            for version_file in sorted(self.versions_dir.glob("*.yaml"), reverse=True):
                try:
                    with version_file.open('r') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        data = yaml.safe_load(f)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
                        if 'info' not in data:
                            self.logger.warning(f"Invalid version file format: {version_file}")
                            continue
                            
                        info = VersionInfo.from_dict(data['info'])
                        versions.append(info)
                        
                except Exception as e:
                    self.logger.error(f"Error loading version info from {version_file}: {e}")
                    continue
                    
                if limit and len(versions) >= limit:
                    break
                    
            return versions
            
        except Exception as e:
            self.logger.error(f"Error listing versions: {e}")
            return []

    def get_version_diff(self, version_id1: str, version_id2: str) -> Optional[Dict]:
        """Get differences between two versions with validation"""
        try:
            # Validate inputs
            if not all([version_id1, version_id2]):
                self.logger.error("Invalid version IDs provided")
                return None

            v1_data = self.get_version(version_id1)
            v2_data = self.get_version(version_id2)
            
            if not v1_data or not v2_data:
                return None
                
            # Compare contexts
            v1_context = v1_data["context"]
            v2_context = v2_data["context"]
            
            changes = {
                "added": [],
                "removed": [],
                "modified": []
            }
            
            self._compare_dict_changes(v1_context, v2_context, "", changes)
            
            return {
                "version1": version_id1,
                "version2": version_id2,
                "changes": changes,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "v1_info": v1_data["info"],
                    "v2_info": v2_data["info"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting version diff: {e}")
            return None

    def cleanup_old_versions(self, days_to_keep: int) -> int:
        """Clean up old versions with backup support"""
        try:
            if days_to_keep < 1:
                raise ValueError("days_to_keep must be positive")

            cutoff = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            removed = 0
            
            # Create backup directory
            backup_dir = self.base_dir / "backup_versions"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            for version_file in self.versions_dir.glob("*.yaml"):
                try:
                    if version_file.stat().st_mtime < cutoff:
                        # Backup before removal
                        backup_path = backup_dir / version_file.name
                        shutil.copy2(version_file, backup_path)
                        
                        version_file.unlink()
                        removed += 1
                        self.logger.debug(f"Removed and backed up old version {version_file}")
                except Exception as e:
                    self.logger.error(f"Error processing version file {version_file}: {e}")
                    continue
                    
            return removed
            
        except Exception as e:
            self.logger.error(f"Error cleaning up versions: {e}")
            return 0

    def get_latest_version(self) -> Optional[str]:
        """Get the latest version ID with validation"""
        try:
            versions = self.list_versions(limit=1)
            if not versions:
                self.logger.debug("No versions found")
                return None
            return versions[0].id
        except Exception as e:
            self.logger.error(f"Error getting latest version: {e}")
            return None

    def version_exists(self, version_id: str) -> bool:
        """Check if a version exists with validation"""
        try:
            if not version_id or not isinstance(version_id, str):
                return False
            version_path = self.versions_dir / f"{version_id}.yaml"
            return version_path.exists()
        except Exception as e:
            self.logger.error(f"Error checking version existence: {e}")
            return False

    def _compare_dict_changes(self, dict1: Dict, dict2: Dict,
                           path: str, changes: Dict) -> None:
        """Compare two dictionaries and track changes"""
        try:
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in dict1:
                    changes["added"].append(current_path)
                elif key not in dict2:
                    changes["removed"].append(current_path)
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        self._compare_dict_changes(dict1[key], dict2[key],
                                               current_path, changes)
                    else:
                        changes["modified"].append(current_path)
        except Exception as e:
            self.logger.error(f"Error comparing dictionaries at path {path}: {e}")
            raise