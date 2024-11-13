"""Version management for context data"""
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
import yaml
from enum import Enum
from dataclasses import dataclass

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
    metadata: Optional[Dict] = None

class ContextVersionManager:
    """Manages versioning for context data"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_version(self, context: Dict, version_type: VersionType,
                      description: str, changes: Dict) -> Optional[str]:
        """Create a new version"""
        try:
            timestamp = datetime.now()
            version_id = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            version_info = VersionInfo(
                id=version_id,
                type=version_type,
                description=description,
                timestamp=timestamp,
                changes=changes,
                metadata={
                    "created_at": timestamp.isoformat(),
                    "version_type": version_type.value
                }
            )
            
            # Save version data
            version_path = self.versions_dir / f"{version_id}.yaml"
            with version_path.open('w') as f:
                yaml.safe_dump({
                    "info": {
                        "id": version_info.id,
                        "type": version_info.type.value,
                        "description": version_info.description,
                        "timestamp": version_info.timestamp.isoformat(),
                        "changes": version_info.changes,
                        "metadata": version_info.metadata
                    },
                    "context": context
                }, f)
                
            self.logger.info(f"Created version {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error creating version: {e}")
            return None
            
    def get_version(self, version_id: str) -> Optional[Dict]:
        """Get specific version data"""
        try:
            version_path = self.versions_dir / f"{version_id}.yaml"
            if not version_path.exists():
                return None
                
            with version_path.open('r') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading version {version_id}: {e}")
            return None
            
    def list_versions(self, limit: Optional[int] = None) -> List[VersionInfo]:
        """List available versions"""
        try:
            versions = []
            for version_file in sorted(self.versions_dir.glob("*.yaml"), reverse=True):
                try:
                    with version_file.open('r') as f:
                        data = yaml.safe_load(f)
                        info_dict = data["info"]
                        # Convert string timestamp back to datetime
                        info_dict["timestamp"] = datetime.fromisoformat(info_dict["timestamp"])
                        # Convert string version type back to enum
                        info_dict["type"] = VersionType(info_dict["type"])
                        versions.append(VersionInfo(**info_dict))
                        
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
        """Get differences between two versions"""
        try:
            v1_data = self.get_version(version_id1)
            v2_data = self.get_version(version_id2)
            
            if not v1_data or not v2_data:
                return None
                
            # Compare contexts
            v1_context = v1_data["context"]
            v2_context = v2_data["context"]
            
            # Track changes
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
            
    def _compare_dict_changes(self, dict1: Dict, dict2: Dict,
                           path: str, changes: Dict) -> None:
        """Compare two dictionaries and track changes"""
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
                    
    def cleanup_old_versions(self, days_to_keep: int) -> int:
        """Clean up old versions"""
        try:
            cutoff = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            removed = 0
            
            for version_file in self.versions_dir.glob("*.yaml"):
                if version_file.stat().st_mtime < cutoff:
                    version_file.unlink()
                    removed += 1
                    self.logger.debug(f"Removed old version {version_file}")
                    
            return removed
            
        except Exception as e:
            self.logger.error(f"Error cleaning up versions: {e}")
            return 0

    def get_latest_version(self) -> Optional[str]:
        """Get the latest version ID"""
        try:
            versions = self.list_versions(limit=1)
            if versions:
                return versions[0].id
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest version: {e}")
            return None

    def version_exists(self, version_id: str) -> bool:
        """Check if a version exists"""
        try:
            version_path = self.versions_dir / f"{version_id}.yaml"
            return version_path.exists()
        except Exception as e:
            self.logger.error(f"Error checking version existence: {e}")
            return False