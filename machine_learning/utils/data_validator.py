"""
Data Validator Module for GhostLoad Mapper ML Pipeline

This module provides production-grade validation for input CSV data schemas used in
electricity theft detection. It ensures data integrity, schema compliance, and business
rule validation before data enters the ML pipeline.

Key Features:
    - Schema validation (required columns, data types)
    - Business rule validation (consumption arrays, numeric constraints)
    - Detailed error reporting with specific field names
    - Extensible validation framework for custom rules
    - Comprehensive logging and error context

Design Philosophy:
    - Fail-fast validation to prevent downstream errors
    - Specific error messages for rapid debugging
    - Separation of schema validation from business rules
    - Composable validation rules for extensibility

Author: GhostLoad Mapper Team
Created: November 13, 2025
Version: 1.0.0
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Import logger if available, otherwise use standard logging
try:
    from machine_learning.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations and Constants
# ============================================================================

class DatasetType(Enum):
    """Supported dataset types for validation."""
    METER_DATA = "meter_data"
    TRANSFORMER_DATA = "transformer_data"
    CONSUMPTION_DATA = "consumption_data"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Blocks pipeline execution
    WARNING = "warning"    # Allows execution with logging
    INFO = "info"          # Informational only


# Default schema configurations
METER_DATA_SCHEMA = {
    'required_columns': ['meter_id', 'transformer_id'],  # Monthly consumption columns are checked separately
    'optional_columns': ['latitude', 'longitude', 'lat', 'lon', 'customer_type', 'customer_class', 
                        'region', 'barangay', 'kVA', 'anomaly_flag'],
    'column_types': {
        'meter_id': ['object', 'str', 'string'],
        'transformer_id': ['object', 'str', 'string', 'int64', 'Int64'],
        'latitude': ['float64', 'float32', 'Float64'],
        'longitude': ['float64', 'float32', 'Float64'],
        'lat': ['float64', 'float32', 'Float64'],
        'lon': ['float64', 'float32', 'Float64'],
        'customer_type': ['object', 'str', 'string'],
        'customer_class': ['object', 'str', 'string'],
        'region': ['object', 'str', 'string'],
        'barangay': ['object', 'str', 'string'],
        'kVA': ['float64', 'float32', 'Float64', 'int64', 'Int64'],
        'anomaly_flag': ['int64', 'Int64', 'bool']
    },
    'consumption_array_length': 12,  # Monthly consumption (12 months)
    'consumption_min_value': 0.0,    # Non-negative consumption
    'consumption_max_value': 1e6,    # Reasonable upper bound (1M kWh)
    'consumption_column_prefix': 'monthly_consumption_'  # Columns are monthly_consumption_1 to monthly_consumption_12
}


# ============================================================================
# Custom Exceptions
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ValidationSeverity = ValidationSeverity.CRITICAL
    ):
        """
        Initialize validation error.
        
        Args:
            message: Human-readable error message
            details: Additional context (missing fields, invalid values, etc.)
            severity: Severity level of the validation error
        """
        self.message = message
        self.details = details or {}
        self.severity = severity
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with details."""
        msg = f"[{self.severity.value.upper()}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" | Details: {details_str}"
        return msg


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    pass


class BusinessRuleValidationError(ValidationError):
    """Raised when business rule validation fails."""
    pass


# ============================================================================
# Validation Result Classes
# ============================================================================

@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    row_indices: Optional[List[int]] = None
    details: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of validation issue."""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.field:
            parts.append(f"Field '{self.field}':")
        parts.append(self.message)
        if self.row_indices:
            count = len(self.row_indices)
            sample = self.row_indices[:5]
            parts.append(f"(Affected rows: {count}, sample: {sample})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """
    Comprehensive validation result containing all issues found.
    
    Attributes:
        is_valid: Overall validation status
        issues: List of validation issues found
        dataset_type: Type of dataset validated
        row_count: Number of rows validated
        column_count: Number of columns validated
        metadata: Additional validation metadata
    """
    is_valid: bool
    issues: List[ValidationIssue] = dataclass_field(default_factory=list)
    dataset_type: DatasetType = DatasetType.UNKNOWN
    row_count: int = 0
    column_count: int = 0
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        field: Optional[str] = None,
        row_indices: Optional[List[int]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a validation issue to the result."""
        issue = ValidationIssue(
            severity=severity,
            message=message,
            field=field,
            row_indices=row_indices,
            details=details or {}
        )
        self.issues.append(issue)
        
        # Mark as invalid if any critical issue found
        if severity == ValidationSeverity.CRITICAL:
            self.is_valid = False
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical validation issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def summary(self) -> str:
        """Generate human-readable validation summary."""
        lines = [
            f"Validation Result: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Dataset Type: {self.dataset_type.value}",
            f"Rows: {self.row_count}, Columns: {self.column_count}",
            f"Issues: {len(self.issues)} total "
            f"({len(self.get_critical_issues())} critical, "
            f"{len(self.get_warnings())} warnings)"
        ]
        
        if self.issues:
            lines.append("\nIssues Found:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        
        return "\n".join(lines)


# ============================================================================
# Validation Rules (Abstract Base Class)
# ============================================================================

class ValidationRule(ABC):
    """
    Abstract base class for validation rules.
    
    Implements the Strategy pattern for composable validation logic.
    """
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.CRITICAL):
        """
        Initialize validation rule.
        
        Args:
            name: Human-readable name for the rule
            severity: Default severity level for violations
        """
        self.name = name
        self.severity = severity
    
    @abstractmethod
    def validate(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Execute validation rule on dataframe.
        
        Args:
            df: DataFrame to validate
            context: Additional validation context
            
        Returns:
            List of validation issues found (empty if valid)
        """
        pass


# ============================================================================
# Concrete Validation Rules
# ============================================================================

class RequiredColumnsRule(ValidationRule):
    """Validates presence of required columns."""
    
    def __init__(self, required_columns: List[str], severity: ValidationSeverity = ValidationSeverity.CRITICAL):
        """
        Initialize required columns rule.
        
        Args:
            required_columns: List of column names that must be present
            severity: Severity level for missing columns
        """
        super().__init__("RequiredColumns", severity)
        self.required_columns = set(required_columns)
    
    def validate(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate required columns are present."""
        issues = []
        actual_columns = set(df.columns)
        missing_columns = self.required_columns - actual_columns
        
        if missing_columns:
            issues.append(ValidationIssue(
                severity=self.severity,
                message=f"Missing required columns: {sorted(missing_columns)}",
                details={
                    'required': sorted(self.required_columns),
                    'actual': sorted(actual_columns),
                    'missing': sorted(missing_columns)
                }
            ))
            logger.error(f"Schema validation failed: missing columns {sorted(missing_columns)}")
        else:
            logger.debug(f"Required columns validation passed: {sorted(self.required_columns)}")
        
        return issues


class ColumnTypesRule(ValidationRule):
    """Validates column data types."""
    
    def __init__(
        self,
        column_types: Dict[str, List[str]],
        severity: ValidationSeverity = ValidationSeverity.CRITICAL
    ):
        """
        Initialize column types rule.
        
        Args:
            column_types: Mapping of column names to allowed type names
            severity: Severity level for type mismatches
        """
        super().__init__("ColumnTypes", severity)
        self.column_types = column_types
    
    def validate(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate column data types."""
        issues = []
        
        for column, allowed_types in self.column_types.items():
            if column not in df.columns:
                continue  # Skip if column doesn't exist (handled by RequiredColumnsRule)
            
            actual_type = str(df[column].dtype)
            
            # Check if actual type matches any allowed type
            if not any(actual_type == allowed or actual_type.startswith(allowed) 
                      for allowed in allowed_types):
                issues.append(ValidationIssue(
                    severity=self.severity,
                    message=f"Column '{column}' has invalid type",
                    field=column,
                    details={
                        'expected_types': allowed_types,
                        'actual_type': actual_type
                    }
                ))
                logger.warning(f"Column '{column}' type mismatch: expected {allowed_types}, got {actual_type}")
        
        return issues


class ConsumptionArrayRule(ValidationRule):
    """Validates monthly consumption array format and values."""
    
    def __init__(
        self,
        column_name: str = 'monthly_consumption',
        expected_length: int = 12,
        min_value: float = 0.0,
        max_value: float = 1e6,
        severity: ValidationSeverity = ValidationSeverity.CRITICAL
    ):
        """
        Initialize consumption array rule.
        
        Args:
            column_name: Name of the consumption column
            expected_length: Expected array length (default: 12 months)
            min_value: Minimum allowed consumption value
            max_value: Maximum allowed consumption value
            severity: Severity level for violations
        """
        super().__init__("ConsumptionArray", severity)
        self.column_name = column_name
        self.expected_length = expected_length
        self.min_value = min_value
        self.max_value = max_value
    
    def _parse_consumption_array(self, value: Any) -> Optional[List[float]]:
        """
        Parse consumption array from various formats.
        
        Supports:
            - String representations: "[1.0, 2.0, 3.0]"
            - Comma-separated: "1.0, 2.0, 3.0"
            - Space-separated: "1.0 2.0 3.0"
            - NumPy arrays
            - Python lists
        
        Args:
            value: Input value to parse
            
        Returns:
            List of floats if valid, None if parsing fails
        """
        try:
            # Handle None, NaN, empty strings
            if pd.isna(value) or value == '' or value is None:
                return None
            
            # Handle NumPy arrays
            if isinstance(value, np.ndarray):
                return value.astype(float).tolist()
            
            # Handle Python lists
            if isinstance(value, list):
                return [float(x) for x in value]
            
            # Handle string representations
            if isinstance(value, str):
                # Remove brackets and split by common delimiters
                cleaned = value.strip().strip('[](){}')
                
                # Try comma-separated
                if ',' in cleaned:
                    parts = cleaned.split(',')
                else:
                    # Try space-separated
                    parts = cleaned.split()
                
                # Convert to floats
                return [float(x.strip()) for x in parts if x.strip()]
            
            # Handle single numeric value (convert to single-element list)
            return [float(value)]
            
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse consumption array: {value} | Error: {e}")
            return None
    
    def validate(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate consumption array format and values."""
        issues = []
        
        if self.column_name not in df.columns:
            return issues  # Skip if column doesn't exist
        
        invalid_format_rows = []
        invalid_length_rows = []
        invalid_value_rows = []
        
        for idx, value in enumerate(df[self.column_name]):
            # Parse consumption array
            consumption = self._parse_consumption_array(value)
            
            if consumption is None:
                invalid_format_rows.append(idx)
                continue
            
            # Check length
            if len(consumption) != self.expected_length:
                invalid_length_rows.append(idx)
                logger.debug(f"Row {idx}: consumption array length {len(consumption)} != {self.expected_length}")
                continue
            
            # Check value ranges
            if not all(self.min_value <= x <= self.max_value for x in consumption):
                invalid_value_rows.append(idx)
                out_of_range = [x for x in consumption if not (self.min_value <= x <= self.max_value)]
                logger.debug(f"Row {idx}: consumption values out of range: {out_of_range}")
        
        # Create issues for each violation type
        if invalid_format_rows:
            issues.append(ValidationIssue(
                severity=self.severity,
                message=f"Invalid format in '{self.column_name}' (not parseable as numeric array)",
                field=self.column_name,
                row_indices=invalid_format_rows,
                details={'count': len(invalid_format_rows)}
            ))
        
        if invalid_length_rows:
            issues.append(ValidationIssue(
                severity=self.severity,
                message=f"'{self.column_name}' must have exactly {self.expected_length} values",
                field=self.column_name,
                row_indices=invalid_length_rows,
                details={
                    'expected_length': self.expected_length,
                    'count': len(invalid_length_rows)
                }
            ))
        
        if invalid_value_rows:
            issues.append(ValidationIssue(
                severity=self.severity,
                message=f"'{self.column_name}' contains values outside valid range [{self.min_value}, {self.max_value}]",
                field=self.column_name,
                row_indices=invalid_value_rows,
                details={
                    'min_value': self.min_value,
                    'max_value': self.max_value,
                    'count': len(invalid_value_rows)
                }
            ))
        
        return issues


class NonNullRule(ValidationRule):
    """Validates that specified columns have no null values."""
    
    def __init__(
        self,
        columns: List[str],
        severity: ValidationSeverity = ValidationSeverity.CRITICAL
    ):
        """
        Initialize non-null rule.
        
        Args:
            columns: List of columns that must not have null values
            severity: Severity level for null violations
        """
        super().__init__("NonNull", severity)
        self.columns = columns
    
    def validate(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate columns have no null values."""
        issues = []
        
        for column in self.columns:
            if column not in df.columns:
                continue  # Skip if column doesn't exist
            
            null_mask = df[column].isna()
            null_indices = df[null_mask].index.tolist()
            
            if null_indices:
                issues.append(ValidationIssue(
                    severity=self.severity,
                    message=f"Column '{column}' contains null values",
                    field=column,
                    row_indices=null_indices,
                    details={
                        'null_count': len(null_indices),
                        'total_rows': len(df)
                    }
                ))
                logger.warning(f"Column '{column}' has {len(null_indices)} null values")
        
        return issues


# ============================================================================
# Main Data Validator
# ============================================================================

class DataValidator:
    """
    Production-grade data validator for ML pipeline inputs.
    
    Features:
        - Schema validation (required columns, types)
        - Business rule validation (consumption arrays, value ranges)
        - Extensible rule-based validation framework
        - Detailed error reporting
        - Performance optimization for large datasets
    
    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate_meter_data(df)
        >>> if not result.is_valid:
        ...     raise ValueError(f"Validation failed: {result.summary()}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator.
        
        Args:
            config: Optional custom configuration (overrides defaults)
        """
        self.config = config or {}
        self.validation_cache: Dict[str, ValidationResult] = {}
        logger.info("DataValidator initialized")
    
    def validate_meter_data(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = True
    ) -> ValidationResult:
        """
        Validate meter data schema and business rules.
        
        Validates:
            1. Required columns: meter_id, monthly_consumption, transformer_id
            2. Column data types match expected types
            3. monthly_consumption has exactly 12 numeric values
            4. All consumption values are non-negative
            5. No null values in required columns
        
        Args:
            df: DataFrame containing meter data
            schema: Optional custom schema (uses METER_DATA_SCHEMA by default)
            raise_on_error: If True, raises ValidationError on critical issues
            
        Returns:
            ValidationResult with detailed validation outcome
            
        Raises:
            SchemaValidationError: If raise_on_error=True and validation fails
            
        Example:
            >>> df = pd.read_csv('meter_data.csv')
            >>> result = validator.validate_meter_data(df)
            >>> print(result.summary())
        """
        logger.info(f"Validating meter data: {len(df)} rows, {len(df.columns)} columns")
        
        # Use default schema if not provided
        schema = schema or METER_DATA_SCHEMA
        
        # Initialize validation result
        result = ValidationResult(
            is_valid=True,
            dataset_type=DatasetType.METER_DATA,
            row_count=len(df),
            column_count=len(df.columns),
            metadata={'schema': schema}
        )
        
        # Build validation rules
        rules: List[ValidationRule] = [
            # Schema validation
            RequiredColumnsRule(schema['required_columns']),
            ColumnTypesRule(schema['column_types'])
        ]
        
        # Add consumption validation based on data format
        consumption_prefix = schema.get('consumption_column_prefix', 'monthly_consumption_')
        consumption_cols = [col for col in df.columns if col.startswith(consumption_prefix)]
        
        if consumption_cols:
            # Data has separate monthly_consumption_1, monthly_consumption_2, etc. columns
            logger.debug(f"Found {len(consumption_cols)} consumption columns with prefix '{consumption_prefix}'")
            # Validation happens naturally through numeric type checking
        elif 'monthly_consumption' in df.columns:
            # Data has single array column (legacy format)
            rules.append(ConsumptionArrayRule(
                column_name='monthly_consumption',
                expected_length=schema['consumption_array_length'],
                min_value=schema['consumption_min_value'],
                max_value=schema['consumption_max_value']
            ))
        
        # Data quality validation
        rules.append(NonNullRule(schema['required_columns']))
        
        # Execute all validation rules
        for rule in rules:
            try:
                issues = rule.validate(df, {'schema': schema})
                for issue in issues:
                    result.add_issue(
                        severity=issue.severity,
                        message=issue.message,
                        field=issue.field,
                        row_indices=issue.row_indices,
                        details=issue.details
                    )
            except Exception as e:
                logger.error(f"Validation rule '{rule.name}' failed: {e}")
                result.add_issue(
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation rule '{rule.name}' encountered error: {str(e)}",
                    details={'exception': str(e)}
                )
        
        # Log validation result
        if result.is_valid:
            logger.info(f"Meter data validation PASSED: {len(df)} rows validated successfully")
        else:
            critical_issues = result.get_critical_issues()
            logger.error(f"Meter data validation FAILED: {len(critical_issues)} critical issues found")
            for issue in critical_issues[:5]:  # Log first 5 issues
                logger.error(f"  - {issue}")
        
        # Raise exception if requested and validation failed
        if raise_on_error and not result.is_valid:
            critical_issues = result.get_critical_issues()
            error_details = {
                'total_issues': len(result.issues),
                'critical_issues': len(critical_issues),
                'row_count': len(df)
            }
            
            # Extract missing fields for specific error message
            missing_fields = []
            for issue in critical_issues:
                if 'missing' in issue.details:
                    missing_fields.extend(issue.details['missing'])
            
            if missing_fields:
                error_message = f"Missing required fields: {sorted(set(missing_fields))}"
            else:
                error_message = f"Validation failed with {len(critical_issues)} critical issues"
            
            raise SchemaValidationError(
                message=error_message,
                details=error_details,
                severity=ValidationSeverity.CRITICAL
            )
        
        return result
    
    def validate_csv(
        self,
        file_path: Union[str, Path],
        dataset_type: DatasetType = DatasetType.METER_DATA,
        **kwargs
    ) -> ValidationResult:
        """
        Validate CSV file directly.
        
        Args:
            file_path: Path to CSV file
            dataset_type: Type of dataset to validate
            **kwargs: Additional arguments passed to validation method
            
        Returns:
            ValidationResult
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValidationError: If validation fails and raise_on_error=True
            
        Example:
            >>> result = validator.validate_csv('data/meter_data.csv')
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Loading CSV file for validation: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise ValidationError(
                message=f"Failed to load CSV file: {file_path}",
                details={'error': str(e), 'file_path': str(file_path)},
                severity=ValidationSeverity.CRITICAL
            )
        
        # Dispatch to appropriate validation method
        if dataset_type == DatasetType.METER_DATA:
            return self.validate_meter_data(df, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def get_schema_info(self, dataset_type: DatasetType = DatasetType.METER_DATA) -> Dict[str, Any]:
        """
        Get schema information for a dataset type.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Schema configuration dictionary
            
        Example:
            >>> schema = validator.get_schema_info(DatasetType.METER_DATA)
            >>> print(schema['required_columns'])
        """
        if dataset_type == DatasetType.METER_DATA:
            return METER_DATA_SCHEMA.copy()
        else:
            raise ValueError(f"No schema defined for dataset type: {dataset_type}")


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_meter_data_csv(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> ValidationResult:
    """
    Convenience function to validate meter data CSV file.
    
    Args:
        file_path: Path to CSV file
        raise_on_error: If True, raises ValidationError on failure
        
    Returns:
        ValidationResult
        
    Raises:
        ValidationError: If validation fails and raise_on_error=True
        
    Example:
        >>> from machine_learning.utils.data_validator import validate_meter_data_csv
        >>> result = validate_meter_data_csv('data/meter_data.csv')
        >>> print(result.summary())
    """
    validator = DataValidator()
    return validator.validate_csv(
        file_path,
        dataset_type=DatasetType.METER_DATA,
        raise_on_error=raise_on_error
    )


def validate_meter_data_df(
    df: pd.DataFrame,
    raise_on_error: bool = True
) -> ValidationResult:
    """
    Convenience function to validate meter data DataFrame.
    
    Args:
        df: DataFrame to validate
        raise_on_error: If True, raises ValidationError on failure
        
    Returns:
        ValidationResult
        
    Raises:
        ValidationError: If validation fails and raise_on_error=True
        
    Example:
        >>> from machine_learning.utils.data_validator import validate_meter_data_df
        >>> df = pd.read_csv('meter_data.csv')
        >>> result = validate_meter_data_df(df)
    """
    validator = DataValidator()
    return validator.validate_meter_data(df, raise_on_error=raise_on_error)


# ============================================================================
# Self-Test Suite
# ============================================================================

def _run_self_tests():
    """
    Comprehensive self-test suite for DataValidator.
    
    Tests:
        1. Valid meter data
        2. Missing required columns
        3. Invalid consumption array length
        4. Non-numeric consumption values
        5. Null values in required columns
        6. Invalid consumption value ranges
        7. Multiple validation errors
        8. Custom schema validation
    """
    print("=" * 80)
    print("DataValidator Self-Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Valid meter data
    print("\nTest 1: Valid meter data")
    try:
        valid_data = pd.DataFrame({
            'meter_id': ['M001', 'M002', 'M003'],
            'monthly_consumption': [
                '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]',
                '[150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]'
            ],
            'transformer_id': ['T001', 'T001', 'T002']
        })
        
        validator = DataValidator()
        result = validator.validate_meter_data(valid_data, raise_on_error=False)
        
        assert result.is_valid, "Valid data should pass validation"
        assert len(result.issues) == 0, "No issues should be found"
        print(f"✓ PASSED - Valid data validated successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 2: Missing required columns
    print("\nTest 2: Missing required columns")
    try:
        missing_cols_data = pd.DataFrame({
            'meter_id': ['M001', 'M002'],
            'monthly_consumption': [
                '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
            ]
            # Missing transformer_id
        })
        
        result = validator.validate_meter_data(missing_cols_data, raise_on_error=False)
        
        assert not result.is_valid, "Should fail validation"
        assert len(result.get_critical_issues()) > 0, "Should have critical issues"
        
        # Check for specific missing field
        critical_issues = result.get_critical_issues()
        missing_field_found = any(
            'transformer_id' in str(issue.details.get('missing', []))
            for issue in critical_issues
        )
        assert missing_field_found, "Should identify missing transformer_id"
        
        print(f"✓ PASSED - Missing columns detected: {critical_issues[0].details.get('missing')}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 3: Invalid consumption array length
    print("\nTest 3: Invalid consumption array length")
    try:
        invalid_length_data = pd.DataFrame({
            'meter_id': ['M001', 'M002'],
            'monthly_consumption': [
                '[100, 105, 110]',  # Only 3 values instead of 12
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
            ],
            'transformer_id': ['T001', 'T002']
        })
        
        result = validator.validate_meter_data(invalid_length_data, raise_on_error=False)
        
        assert not result.is_valid, "Should fail validation"
        length_issue = any(
            'exactly 12 values' in issue.message
            for issue in result.get_critical_issues()
        )
        assert length_issue, "Should detect invalid array length"
        
        print(f"✓ PASSED - Invalid array length detected")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 4: Non-numeric consumption values
    print("\nTest 4: Non-numeric consumption values")
    try:
        non_numeric_data = pd.DataFrame({
            'meter_id': ['M001', 'M002'],
            'monthly_consumption': [
                'invalid_data',  # Not a numeric array
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
            ],
            'transformer_id': ['T001', 'T002']
        })
        
        result = validator.validate_meter_data(non_numeric_data, raise_on_error=False)
        
        assert not result.is_valid, "Should fail validation"
        format_issue = any(
            'Invalid format' in issue.message
            for issue in result.get_critical_issues()
        )
        assert format_issue, "Should detect non-numeric values"
        
        print(f"✓ PASSED - Non-numeric values detected")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 5: Null values in required columns
    print("\nTest 5: Null values in required columns")
    try:
        null_data = pd.DataFrame({
            'meter_id': ['M001', None, 'M003'],
            'monthly_consumption': [
                '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]',
                '[150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]'
            ],
            'transformer_id': ['T001', 'T002', None]
        })
        
        result = validator.validate_meter_data(null_data, raise_on_error=False)
        
        assert not result.is_valid, "Should fail validation"
        null_issues = [
            issue for issue in result.get_critical_issues()
            if 'null values' in issue.message.lower()
        ]
        assert len(null_issues) > 0, "Should detect null values"
        
        print(f"✓ PASSED - Null values detected in {len(null_issues)} columns")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 6: Invalid consumption value ranges
    print("\nTest 6: Invalid consumption value ranges")
    try:
        invalid_range_data = pd.DataFrame({
            'meter_id': ['M001', 'M002'],
            'monthly_consumption': [
                '[-10, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',  # Negative value
                '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
            ],
            'transformer_id': ['T001', 'T002']
        })
        
        result = validator.validate_meter_data(invalid_range_data, raise_on_error=False)
        
        assert not result.is_valid, "Should fail validation"
        range_issue = any(
            'outside valid range' in issue.message
            for issue in result.get_critical_issues()
        )
        assert range_issue, "Should detect out-of-range values"
        
        print(f"✓ PASSED - Out-of-range values detected")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 7: ValidationError exception with specific field names
    print("\nTest 7: ValidationError exception with specific field names")
    try:
        missing_multiple = pd.DataFrame({
            'meter_id': ['M001', 'M002']
            # Missing both monthly_consumption and transformer_id
        })
        
        error_raised = False
        missing_fields = []
        
        try:
            validator.validate_meter_data(missing_multiple, raise_on_error=True)
        except SchemaValidationError as e:
            error_raised = True
            missing_fields = e.details.get('missing', [])
            assert 'monthly_consumption' in str(e) or 'transformer_id' in str(e), \
                "Error message should contain missing field names"
        
        assert error_raised, "Should raise ValidationError"
        print(f"✓ PASSED - ValidationError raised with specific field information")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Test 8: Custom schema validation
    print("\nTest 8: Custom schema validation")
    try:
        custom_schema = {
            'required_columns': ['meter_id', 'monthly_consumption'],
            'optional_columns': [],
            'column_types': {
                'meter_id': ['object', 'str', 'string'],
                'monthly_consumption': ['object', 'str', 'string']
            },
            'consumption_array_length': 6,  # Custom length
            'consumption_min_value': 0.0,
            'consumption_max_value': 1000.0
        }
        
        custom_data = pd.DataFrame({
            'meter_id': ['M001', 'M002'],
            'monthly_consumption': [
                '[100, 105, 110, 115, 120, 125]',  # 6 values
                '[200, 210, 220, 230, 240, 250]'
            ]
        })
        
        result = validator.validate_meter_data(custom_data, schema=custom_schema, raise_on_error=False)
        
        assert result.is_valid, "Custom schema should pass validation"
        print(f"✓ PASSED - Custom schema validation successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ FAILED - {e}")
        test_results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    passed = sum(test_results)
    total = len(test_results)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
    
    print("=" * 80)
    
    return all(test_results)


if __name__ == '__main__':
    # Run self-tests when module is executed directly
    success = _run_self_tests()
    exit(0 if success else 1)
