import pytest
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict

@pytest.fixture
def test_session_id():
    """Generate a unique test session ID"""
    return f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@pytest.fixture
def test_base_dir(tmp_path):
    """Create a temporary test directory"""
    base_dir = tmp_path / "test_data"
    base_dir.mkdir(exist_ok=True)
    yield base_dir
    shutil.rmtree(base_dir)

@pytest.fixture
def test_cache_dir(test_base_dir):
    """Create a test cache directory"""
    cache_dir = test_base_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

@pytest.fixture
def test_context_dir(test_base_dir):
    """Create a test context directory"""
    context_dir = test_base_dir / "context"
    context_dir.mkdir(exist_ok=True)
    return context_dir

@pytest.fixture
def sample_context() -> Dict:
    """Provide a sample test context"""
    return {
        "metadata": {
            "version": "2.0.0",
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        },
        "system_context": {
            "architecture_state": "initial",
            "global_constraints": {}
        },
        "domain_context": {
            "business_rules": []
        },
        "component_context": {
            "local_state": {}
        }
    }