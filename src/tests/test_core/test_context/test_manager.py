# tests/test_core/test_context/test_enhanced_manager.py
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import yaml
import shutil

from coding_assistant.core.context.manager import ContextManager
from coding_assistant.core.context.validation import ValidationError
from coding_assistant.core.context.version_manager import VersionType

@pytest.fixture
def test_session_id():
    return "test_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")

@pytest.fixture
def test_base_dir(tmp_path):
    base_dir = tmp_path / "test_data"
    base_dir.mkdir(exist_ok=True)
    yield base_dir
    shutil.rmtree(base_dir)

@pytest.fixture
def context_manager(test_session_id, test_base_dir):
    return ContextManager(test_session_id, test_base_dir)

@pytest.fixture
def sample_context_update():
    return {
        "business_rules": [
            {
                "id": "BR001",
                "name": "Test Rule",
                "description": "Test rule description",
                "priority": "high"
            }
        ]
    }

class TestEnhancedContextManager:
    """Test suite for enhanced context management system"""

    def test_initialization(self, context_manager, test_base_dir):
        """Test context manager initialization"""
        assert context_manager.current_context is not None
        assert context_manager.context_dir.exists()
        assert (test_base_dir / "cache").exists()
        assert context_manager.current_context["metadata"]["session_id"] == context_manager.session_id

    def test_save_and_load_context(self, context_manager):
        """Test basic context saving and loading"""
        # Modify context
        context_manager.current_context["system_context"]["test_key"] = "test_value"
        
        # Save with description
        success = context_manager.save_context(
            context_manager.current_context,
            "Test save",
            VersionType.MINOR
        )
        assert success is True

        # Clear current context
        context_manager.current_context = None

        # Load and verify
        loaded_context = context_manager.load_context()
        assert loaded_context is not None
        assert loaded_context["system_context"]["test_key"] == "test_value"

    def test_context_versioning(self, context_manager):
        """Test context versioning capabilities"""
        # Create multiple versions
        original_context = context_manager.current_context.copy()

        # Make changes and create versions
        changes = [
            ("First change", {"test_key": "value1"}),
            ("Second change", {"test_key": "value2"}),
            ("Major change", {"test_key": "value3"})
        ]

        versions = []
        for desc, change in changes:
            context_manager.current_context["system_context"].update(change)
            version_type = VersionType.MAJOR if "Major" in desc else VersionType.MINOR
            success = context_manager.save_context(
                context_manager.current_context,
                desc,
                version_type
            )
            assert success is True
            versions.append(context_manager.current_version)

        # Get version history
        history = context_manager.get_version_history()
        assert len(history) >= len(changes)

        # Test rollback
        success = context_manager.rollback_to_version(versions[0])
        assert success is True
        assert context_manager.current_context["system_context"]["test_key"] == "value1"

    def test_context_update(self, context_manager, sample_context_update):
        """Test context update functionality"""
        success = context_manager.update_context(
            sample_context_update,
            ["domain_context"],
            "Adding business rules"
        )
        assert success is True

        # Verify update
        assert "business_rules" in context_manager.current_context["domain_context"]
        assert len(context_manager.current_context["domain_context"]["business_rules"]) == 1
        assert context_manager.current_context["domain_context"]["business_rules"][0]["id"] == "BR001"

        # Check history
        assert len(context_manager.current_context["history"]) > 0
        last_entry = context_manager.current_context["history"][-1]
        assert last_entry["description"] == "Adding business rules"

    def test_cache_integration(self, context_manager):
        """Test cache integration"""
        # Make changes
        context_manager.current_context["test_cached_key"] = "test_cached_value"
        
        # Save context (should update cache)
        success = context_manager.save_context(
            context_manager.current_context,
            "Test cache integration",
            VersionType.MINOR
        )
        assert success is True

        # Get cache stats
        stats = context_manager.get_cache_stats()
        assert stats is not None

        # Clear current context and load from cache
        context_manager.current_context = None
        loaded_context = context_manager.load_context()
        assert loaded_context["test_cached_key"] == "test_cached_value"

    def test_validation(self, context_manager):
        """Test context validation"""
        # Try to save invalid context
        invalid_context = {"invalid": "structure"}
        
        with pytest.raises(ValueError) as exc_info:
            context_manager.save_context(
                invalid_context,
                "Invalid context test",
                VersionType.MINOR
            )
        assert "Context validation failed" in str(exc_info.value)

    def test_cleanup(self, context_manager):
        """Test cleanup functionality"""
        # Create multiple versions
        for i in range(3):
            context_manager.current_context["test_key"] = f"value_{i}"
            context_manager.save_context(
                context_manager.current_context,
                f"Change {i}",
                VersionType.MINOR
            )

        # Perform cleanup
        versions_cleaned, cache_cleaned = context_manager.cleanup(days_to_keep=0)
        assert versions_cleaned >= 0
        assert cache_cleaned >= 0

    def test_transaction_handling(self, context_manager):
        """Test transaction handling during updates"""
        # Simulate failed transaction
        try:
            with context_manager.transaction_manager.transaction() as transaction_id:
                context_manager.current_context["test_key"] = "test_value"
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Verify changes were rolled back
        loaded_context = context_manager.load_context()
        assert "test_key" not in loaded_context

    def test_concurrent_access(self, context_manager):
        """Test concurrent access handling"""
        import threading
        import time

        def update_context(delay: float, value: str):
            time.sleep(delay)
            context_manager.update_context(
                {"test_key": value},
                ["system_context"],
                f"Update with {value}"
            )

        # Create two threads that update the context
        thread1 = threading.Thread(target=update_context, args=(0.1, "value1"))
        thread2 = threading.Thread(target=update_context, args=(0.2, "value2"))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Verify final state is consistent
        assert context_manager.current_context["system_context"]["test_key"] in ["value1", "value2"]

    def test_resilience_mechanisms(self, context_manager, test_base_dir):
        """Test resilience mechanisms"""
        # Test cache fallback
        original_context = context_manager.current_context.copy()
        
        # Simulate file system issue by removing the context file
        context_file = context_manager.context_dir / "context.yaml"
        if context_file.exists():
            context_file.unlink()

        # Should still load from cache
        loaded_context = context_manager.load_context()
        assert loaded_context is not None
        assert loaded_context == original_context

    def test_performance_metrics(self, context_manager):
        """Test performance metrics tracking"""
        # Perform multiple operations
        for i in range(5):
            context_manager.update_context(
                {"test_key": f"value_{i}"},
                ["component_context"],
                f"Update {i}"
            )

        # Check metrics
        metrics = context_manager.current_context["component_context"]["performance_metrics"]
        assert "response_time" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics

    def test_version_comparison(self, context_manager):
        """Test version comparison functionality"""
        # Create initial version
        initial_version = context_manager.current_version

        # Make changes
        context_manager.update_context(
            {"test_key": "new_value"},
            ["system_context"],
            "Test change"
        )

        # Get version history and compare
        history = context_manager.get_version_history(limit=2)
        assert len(history) >= 2

        # Compare versions
        diff = context_manager.version_manager.get_version_diff(
            initial_version,
            context_manager.current_version
        )
        assert diff is not None
        assert "changes" in diff
        assert any("test_key" in path for path in diff["changes"]["modified"])