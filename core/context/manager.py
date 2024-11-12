import logging
from typing import Dict, Optional
import yaml

class ContextManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
    def load_context(self) -> Optional[Dict]:
        """Load context from file system"""
        try:
            with open(f"data/context/{self.session_id}/context.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading context: {e}")
            return None
            
    def save_context(self, context: Dict) -> bool:
        """Save context to file system"""
        try:
            with open(f"data/context/{self.session_id}/context.yaml", 'w') as f:
                yaml.safe_dump(context, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving context: {e}")
            return False
