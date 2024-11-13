# src/coding_assistant/core/context/validation.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: Optional[Dict] = None


class ContextValidator:
    """Context validation rules"""

    @staticmethod
    def validate_system_context(context: Dict) -> List[ValidationError]:
        """Validate system context"""
        errors = []
        sys_ctx = context.get("system_context", {})

        required_fields = {
            "architecture_state": str,
            "global_constraints": dict,
            "cache_policy": dict,
            "resilience_config": dict,
            "learning_state": dict
        }

        for field, expected_type in required_fields.items():
            if field not in sys_ctx:
                errors.append(ValidationError(
                    field=f"system_context.{field}",
                    message=f"Missing required field: {field}"
                ))
            elif not isinstance(sys_ctx[field], expected_type):
                errors.append(ValidationError(
                    field=f"system_context.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}"
                ))

        return errors

    @staticmethod
    def validate_domain_context(context: Dict) -> List[ValidationError]:
        """Validate domain context"""
        errors = []
        domain_ctx = context.get("domain_context", {})

        required_fields = {
            "business_rules": list,
            "validation_rules": list,
            "cached_patterns": list,
            "impact_analysis": dict,
            "risk_assessment": dict
        }

        for field, expected_type in required_fields.items():
            if field not in domain_ctx:
                errors.append(ValidationError(
                    field=f"domain_context.{field}",
                    message=f"Missing required field: {field}"
                ))
            elif not isinstance(domain_ctx[field], expected_type):
                errors.append(ValidationError(
                    field=f"domain_context.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}"
                ))

        return errors

    @staticmethod
    def validate_component_context(context: Dict) -> List[ValidationError]:
        """Validate component context"""
        errors = []
        comp_ctx = context.get("component_context", {})

        required_fields = {
            "local_state": dict,
            "dependencies": list,
            "cached_results": dict,
            "failure_history": list,
            "performance_metrics": dict
        }

        for field, expected_type in required_fields.items():
            if field not in comp_ctx:
                errors.append(ValidationError(
                    field=f"component_context.{field}",
                    message=f"Missing required field: {field}"
                ))
            elif not isinstance(comp_ctx[field], expected_type):
                errors.append(ValidationError(
                    field=f"component_context.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}"
                ))

        return errors

    @staticmethod
    def validate_modification_context(context: Dict) -> List[ValidationError]:
        """Validate modification context"""
        errors = []
        mod_ctx = context.get("modification_context", {})

        required_fields = {
            "change_scope": list,
            "validation_rules": dict,
            "cache_strategy": dict,
            "fallback_options": list,
            "learning_targets": dict,
            "parallel_execution": dict
        }

        for field, expected_type in required_fields.items():
            if field not in mod_ctx:
                errors.append(ValidationError(
                    field=f"modification_context.{field}",
                    message=f"Missing required field: {field}"
                ))
            elif not isinstance(mod_ctx[field], expected_type):
                errors.append(ValidationError(
                    field=f"modification_context.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}"
                ))

        return errors

    @staticmethod
    def validate_metadata(context: Dict) -> List[ValidationError]:
        """Validate metadata section"""
        errors = []
        metadata = context.get("metadata", {})

        required_fields = {
            "version": str,
            "created_at": str,
            "session_id": str,
            "last_modified": str,
            "execution_mode": str
        }

        for field, expected_type in required_fields.items():
            if field not in metadata:
                errors.append(ValidationError(
                    field=f"metadata.{field}",
                    message=f"Missing required field: {field}"
                ))
            elif not isinstance(metadata[field], expected_type):
                errors.append(ValidationError(
                    field=f"metadata.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}"
                ))

        # Validate date formats
        for date_field in ["created_at", "last_modified"]:
            if date_field in metadata:
                try:
                    datetime.fromisoformat(metadata[date_field])
                except ValueError:
                    errors.append(ValidationError(
                        field=f"metadata.{date_field}",
                        message=f"Invalid ISO date format in {date_field}"
                    ))

        return errors

    @classmethod
    def validate_context(cls, context: Dict) -> List[ValidationError]:
        """Validate complete context hierarchy"""
        errors = []

        # Basic structure validation
        if not isinstance(context, dict):
            errors.append(ValidationError(
                field="root",
                message="Context must be a dictionary"
            ))
            return errors

        # Validate each section
        errors.extend(cls.validate_metadata(context))
        errors.extend(cls.validate_system_context(context))
        errors.extend(cls.validate_domain_context(context))
        errors.extend(cls.validate_component_context(context))
        errors.extend(cls.validate_modification_context(context))

        return errors
