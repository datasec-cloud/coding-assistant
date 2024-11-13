# src/coding_assistant/core/context/validation.py
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import re
from dateutil.parser import parse as parse_date
from contextlib import contextmanager

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
    details: Optional[Dict] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    validation_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert validation error to dictionary"""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "priority": self.priority.value,
            "validation_time": self.validation_time.isoformat()
        }

class ValidationContext:
    """Context manager for validation operations"""
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.logger = logging.getLogger(__name__)

    def add_error(self, error: ValidationError):
        """Add validation error with logging"""
        self.errors.append(error)
        self.logger.error(f"Validation error: {error.field} - {error.message}")

    def add_warning(self, field: str, message: str, details: Optional[Dict] = None):
        """Add validation warning with logging"""
        error = ValidationError(
            field=field,
            message=message,
            severity=ValidationSeverity.WARNING,
            details=details,
            priority=Priority.LOW
        )
        self.errors.append(error)
        self.logger.warning(f"Validation warning: {field} - {message}")

class ContextValidator:
    """Enhanced context validation with improved error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validation_context = ValidationContext()

    @classmethod
    def validate_context(cls, context: Dict) -> List[ValidationError]:
        """Validate complete context hierarchy with enhanced error handling"""
        validator = cls()
        try:
            if not isinstance(context, dict):
                return [ValidationError(
                    field="root",
                    message="Context must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                )]

            # Validate each section
            errors = []
            errors.extend(validator.validate_metadata(context))
            errors.extend(validator.validate_system_context(context))
            errors.extend(validator.validate_domain_context(context))
            errors.extend(validator.validate_component_context(context))
            errors.extend(validator.validate_modification_context(context))
            
            return errors

        except Exception as e:
            validator.logger.error(f"Error during context validation: {e}")
            return [ValidationError(
                field="validation",
                message=f"Validation process failed: {str(e)}",
                severity=ValidationSeverity.ERROR,
                priority=Priority.HIGH,
                details={"exception": str(e)}
            )]

    def validate_metadata(self, context: Dict) -> List[ValidationError]:
        """Validate metadata section with enhanced date parsing"""
        errors = []
        metadata = context.get("metadata", {})

        required_fields = {
            "version": (str, self._validate_version_format),
            "created_at": (str, self._validate_datetime),
            "session_id": (str, self._validate_session_id),
            "last_modified": (str, self._validate_datetime),
            "execution_mode": (str, self._validate_execution_mode)
        }

        for field, (expected_type, validator) in required_fields.items():
            if field not in metadata:
                errors.append(ValidationError(
                    field=f"metadata.{field}",
                    message=f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))
                continue

            value = metadata[field]
            if not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field=f"metadata.{field}",
                    message=f"Invalid type for {field}, expected: {expected_type.__name__}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))
                continue

            # Run field-specific validation
            validation_error = validator(value)
            if validation_error:
                errors.append(validation_error)

        return errors

    def validate_system_context(self, context: Dict) -> List[ValidationError]:
        """Validate system context with enhanced checks"""
        errors = []
        sys_ctx = context.get("system_context", {})

        required_sections = {
            "architecture_state": self._validate_architecture_state,
            "global_constraints": self._validate_global_constraints,
            "cache_policy": self._validate_cache_policy,
            "resilience_config": self._validate_resilience_config,
            "learning_state": self._validate_learning_state
        }

        for section, validator in required_sections.items():
            if section not in sys_ctx:
                errors.append(ValidationError(
                    field=f"system_context.{section}",
                    message=f"Missing required section: {section}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))
                continue

            section_errors = validator(sys_ctx[section])
            if section_errors:
                errors.extend(section_errors)

        return errors

    def validate_domain_context(self, context: Dict) -> List[ValidationError]:
        """Validate domain context with enhanced validation"""
        errors = []
        domain_ctx = context.get("domain_context", {})

        required_sections = {
            "business_rules": self._validate_business_rules,
            "validation_rules": self._validate_validation_rules,
            "cached_patterns": self._validate_cached_patterns,
            "impact_analysis": self._validate_impact_analysis,
            "risk_assessment": self._validate_risk_assessment
        }

        for section, validator in required_sections.items():
            if section not in domain_ctx:
                errors.append(ValidationError(
                    field=f"domain_context.{section}",
                    message=f"Missing required section: {section}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))
                continue

            section_errors = validator(domain_ctx[section])
            if section_errors:
                errors.extend(section_errors)

        return errors

    def _validate_datetime(self, value: str) -> Optional[ValidationError]:
        """Validate datetime string with enhanced parsing"""
        try:
            parse_date(value)
            return None
        except Exception:
            return ValidationError(
                field="datetime",
                message=f"Invalid datetime format: {value}",
                severity=ValidationSeverity.ERROR,
                priority=Priority.HIGH,
                details={"value": value}
            )

    def _validate_version_format(self, value: str) -> Optional[ValidationError]:
        """Validate version string format"""
        version_pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(version_pattern, value):
            return ValidationError(
                field="version",
                message=f"Invalid version format: {value}. Expected format: X.Y.Z",
                severity=ValidationSeverity.ERROR,
                priority=Priority.HIGH,
                details={"value": value, "expected_pattern": version_pattern}
            )
        return None

    def _validate_session_id(self, value: str) -> Optional[ValidationError]:
        """Validate session ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            return ValidationError(
                field="session_id",
                message="Invalid session ID format",
                severity=ValidationSeverity.ERROR,
                priority=Priority.HIGH,
                details={"value": value}
            )
        return None

    def _validate_execution_mode(self, value: str) -> Optional[ValidationError]:
        """Validate execution mode"""
        valid_modes = ["sync", "async", "mixed"]
        if value not in valid_modes:
            return ValidationError(
                field="execution_mode",
                message=f"Invalid execution mode: {value}",
                severity=ValidationSeverity.ERROR,
                priority=Priority.HIGH,
                details={"value": value, "valid_modes": valid_modes}
            )
        return None

    def _validate_architecture_state(self, value: Dict) -> List[ValidationError]:
        """Validate architecture state"""
        errors = []
        required_fields = ["mode", "version", "status"]
        
        for field in required_fields:
            if field not in value:
                errors.append(ValidationError(
                    field=f"architecture_state.{field}",
                    message=f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))

        return errors

    def _validate_cache_policy(self, value: Dict) -> List[ValidationError]:
        """Validate cache policy configuration"""
        errors = []
        required_fields = {
            "mode": ["memory", "disk", "hybrid"],
            "strategy": ["write_through", "write_back", "write_around"],
            "ttl": dict
        }

        for field, expected in required_fields.items():
            if field not in value:
                errors.append(ValidationError(
                    field=f"cache_policy.{field}",
                    message=f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))
                continue

            if isinstance(expected, list) and value[field] not in expected:
                errors.append(ValidationError(
                    field=f"cache_policy.{field}",
                    message=f"Invalid value: {value[field]}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH,
                    details={"valid_values": expected}
                ))
            elif isinstance(expected, type) and not isinstance(value[field], expected):
                errors.append(ValidationError(
                    field=f"cache_policy.{field}",
                    message=f"Invalid type: expected {expected.__name__}",
                    severity=ValidationSeverity.ERROR,
                    priority=Priority.HIGH
                ))

        return errors

    @contextmanager
    def validation_session(self):
        """Context manager for validation sessions"""
        try:
            yield self._validation_context
        finally:
            self._validation_context.errors.clear()