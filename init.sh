#!/bin/bash

# Script d'initialisation du projet d'assistant de codage
# Usage: ./init_project.sh [target_directory]

set -e  # Exit on error

# Default target directory is current directory
TARGET_DIR=${1:-.}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Initializing Coding Assistant Project...${NC}"

# Create main project structure
create_directory_structure() {
    local base_dir="$1"
    echo -e "${BLUE}Creating directory structure...${NC}"
    
    # Core directories
    mkdir -p "$base_dir"/{core/{context,agents,io},simulation/{templates,scenarios},data/{context,history},tools/bootstrap}
    
    # Create empty __init__.py files for Python modules
    find "$base_dir" -type d -exec touch "{}/__init__.py" \;
    
    echo -e "${GREEN}Directory structure created successfully${NC}"
}

# Create initial configuration files
create_config_files() {
    local base_dir="$1"
    echo -e "${BLUE}Creating configuration files...${NC}"
    
    # Create logging configuration
    cat > "$base_dir/core/logging_config.yaml" << EOL
version: 1
formatters:
  detailed:
    format: '%(asctime)s %(levelname)s [%(name)s] %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: detailed
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    formatter: detailed
    filename: logs/development.log
root:
  level: DEBUG
  handlers: [console, file]
EOL

    # Create base context template
    cat > "$base_dir/simulation/templates/base_context.yaml" << EOL
metadata:
  version: "1.0.0"
  created_at: ""
  session_id: ""
context:
  project:
    name: ""
    path: ""
    current_branch: ""
  interaction:
    request_id: ""
    type: ""
    status: "pending"
  validation:
    required_steps: []
    completed_steps: []
EOL

    # Create agent templates
    cat > "$base_dir/simulation/templates/architect_request.yaml" << EOL
type: architect_request
metadata:
  timestamp: ""
  request_id: ""
payload:
  context_ref: ""
  request_type: ""
  description: ""
  acceptance_criteria: []
validation:
  required: true
  steps: []
EOL

    echo -e "${GREEN}Configuration files created successfully${NC}"
}

# Create basic Python files for core functionality
create_core_files() {
    local base_dir="$1"
    echo -e "${BLUE}Creating core Python files...${NC}"
    
    # Context manager
    cat > "$base_dir/core/context/manager.py" << EOL
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
EOL

    # Base agent
    cat > "$base_dir/core/agents/base_agent.py" << EOL
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
EOL

    echo -e "${GREEN}Core Python files created successfully${NC}"
}

# Create requirements.txt
create_requirements() {
    local base_dir="$1"
    echo -e "${BLUE}Creating requirements.txt...${NC}"
    
    cat > "$base_dir/requirements.txt" << EOL
pyyaml>=6.0
python-dotenv>=0.19.0
gitpython>=3.1.0
pytest>=7.0.0
black>=22.0.0
pylint>=2.8.0
EOL

    echo -e "${GREEN}Requirements file created successfully${NC}"
}

# Create .gitignore
create_gitignore() {
    local base_dir="$1"
    echo -e "${BLUE}Creating .gitignore...${NC}"
    
    cat > "$base_dir/.gitignore" << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Local development
.env
.env.local

# Project specific
data/context/*
!data/context/.gitkeep
data/history/*
!data/history/.gitkeep
EOL

    echo -e "${GREEN}.gitignore created successfully${NC}"
}

# Create README.md
create_readme() {
    local base_dir="$1"
    echo -e "${BLUE}Creating README.md...${NC}"
    
    cat > "$base_dir/README.md" << EOL
# Coding Assistant Project

## Overview
This project implements an interactive coding assistant system capable of handling projects up to 250,000 lines of code, with optimization for projects around 25,000 lines.

## Project Structure
\`\`\`
project/
├── core/               # Core functionality
│   ├── context/       # Context management
│   ├── agents/        # Agent implementations
│   └── io/           # Input/Output handling
├── simulation/        # Simulation components
│   ├── templates/    # Message templates
│   └── scenarios/    # Test scenarios
├── data/             # Data storage
│   ├── context/     # Context data
│   └── history/     # Interaction history
└── tools/           # Development tools
    └── bootstrap/   # Bootstrap utilities
\`\`\`

## Getting Started

1. Create and activate a virtual environment:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Initialize the project:
   \`\`\`bash
   ./tools/bootstrap/init_project.sh
   \`\`\`

## Development

The project follows a bootstrap approach where we progressively use the system for its own development. Initial development focuses on:

1. File-based implementation
2. Context management
3. Interaction chain
4. Manual simulation of components

## Testing

Run tests using pytest:
\`\`\`bash
pytest
\`\`\`

## Contributing

1. Create a feature branch
2. Implement changes
3. Write/update tests
4. Submit pull request

## License

[Your chosen license]
EOL

    echo -e "${GREEN}README.md created successfully${NC}"
}

# Keep empty directories with .gitkeep
create_gitkeep_files() {
    local base_dir="$1"
    echo -e "${BLUE}Creating .gitkeep files...${NC}"
    
    find "$base_dir" -type d -empty -exec touch "{}/.gitkeep" \;
    
    echo -e "${GREEN}.gitkeep files created successfully${NC}"
}

# Main execution
main() {
    local target_dir="$1"
    
    echo -e "${BLUE}Starting project initialization in: ${target_dir}${NC}"
    
    # Create all required components
    create_directory_structure "$target_dir"
    create_config_files "$target_dir"
    create_core_files "$target_dir"
    create_requirements "$target_dir"
    create_gitignore "$target_dir"
    create_readme "$target_dir"
    create_gitkeep_files "$target_dir"
    
    # Create logs directory
    mkdir -p "$target_dir/logs"
    
    echo -e "${GREEN}Project initialization completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Create and activate a virtual environment:"
    echo -e "   python -m venv venv"
    echo -e "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    echo -e "2. Install dependencies:"
    echo -e "   pip install -r requirements.txt"
    echo -e "3. Start developing!"
}

# Execute main function
main "$TARGET_DIR"