# Coding Assistant Project

## Overview
This project implements an interactive coding assistant system capable of handling projects up to 250,000 lines of code, with optimization for projects around 25,000 lines.

## Project Structure
```
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
```

## Getting Started

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the project:
   ```bash
   ./tools/bootstrap/init_project.sh
   ```

## Development

The project follows a bootstrap approach where we progressively use the system for its own development. Initial development focuses on:

1. File-based implementation
2. Context management
3. Interaction chain
4. Manual simulation of components

## Testing

Run tests using pytest:
```bash
pytest
```

## Contributing

1. Create a feature branch
2. Implement changes
3. Write/update tests
4. Submit pull request

## License

[Your chosen license]
