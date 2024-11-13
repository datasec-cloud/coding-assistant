import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = None):
    """Setup logging configuration"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "core" / "logging_config.yaml"
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Logging configuration file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.config.dictConfig(config)