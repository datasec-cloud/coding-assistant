from typing import Dict, Optional
from ..context.manager import ContextManager

class BaseAgent:
    def __init__(self, agent_type: str, context_manager: ContextManager):
        self.agent_type = agent_type
        self.context_manager = context_manager
        
    def process_request(self, request: Dict) -> Optional[Dict]:
        """Process a request and return a response"""
        raise NotImplementedError("Subclasses must implement process_request")
        
    def validate_response(self, response: Dict) -> bool:
        """Validate a response before sending"""
        raise NotImplementedError("Subclasses must implement validate_response")
