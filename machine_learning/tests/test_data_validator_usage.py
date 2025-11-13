"""
Integration Example: DataValidator Usage

Demonstrates real-world usage of the DataValidator module for validating
meter data CSV files before processing in the ML pipeline.
"""

import pandas as pd
from machine_learning.utils.data_validator import (
    DataValidator,
    validate_meter_data_csv,
    validate_meter_data_df,
    DatasetType,
    SchemaValidationError
)
from machine_learning.utils.logger import get_logger, LogContext

# Initialize logger
logger = get_logger('example_usage')

def example_1_valid_data():
    """Example 1: Validate valid meter data."""
    print("\n" + "=" * 80)
    print("Example 1: Validating Valid Meter Data")
    print("=" * 80)
    
    # Create valid test data
    valid_data = pd.DataFrame({
        'meter_id': ['M001', 'M002', 'M003', 'M004', 'M005'],
        'monthly_consumption': [
            '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
            '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]',
            '[150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]',
            '[300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410]',
            '[250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]'
        ],
        'transformer_id': ['T001', 'T001', 'T002', 'T002', 'T003'],
        'latitude': [14.5995, 14.6091, 14.5843, 14.6234, 14.5756],
        'longitude': [120.9842, 120.9936, 121.0245, 121.0012, 120.9567]
    })
    
    # Validate using convenience function
    with LogContext(operation='validate_valid_data'):
        logger.info("Validating valid meter data")
        result = validate_meter_data_df(valid_data, raise_on_error=False)
        
        print(f"\nValidation Status: {'✓ PASSED' if result.is_valid else '✗ FAILED'}")
        print(f"Rows Validated: {result.row_count}")
        print(f"Columns Validated: {result.column_count}")
        print(f"Issues Found: {len(result.issues)}")
        
        if result.is_valid:
            logger.info("✓ Valid data passed all validation checks")
            print("\n✓ Data is ready for ML pipeline processing")


def example_2_missing_columns():
    """Example 2: Handle missing required columns."""
    print("\n" + "=" * 80)
    print("Example 2: Handling Missing Required Columns")
    print("=" * 80)
    
    # Create data with missing transformer_id
    invalid_data = pd.DataFrame({
        'meter_id': ['M001', 'M002', 'M003'],
        'monthly_consumption': [
            '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
            '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]',
            '[150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]'
        ]
        # Missing transformer_id column
    })
    
    with LogContext(operation='validate_missing_columns'):
        logger.info("Attempting to validate data with missing columns")
        
        try:
            result = validate_meter_data_df(invalid_data, raise_on_error=True)
        except SchemaValidationError as e:
            logger.error(f"Validation failed: {e.message}")
            print(f"\n✗ Validation Error: {e.message}")
            print(f"Severity: {e.severity.value.upper()}")
            print(f"Details: {e.details}")
            print("\nAction Required: Add missing columns to CSV file")


def example_3_invalid_consumption_arrays():
    """Example 3: Detect invalid consumption arrays."""
    print("\n" + "=" * 80)
    print("Example 3: Detecting Invalid Consumption Arrays")
    print("=" * 80)
    
    # Create data with various consumption array issues
    problematic_data = pd.DataFrame({
        'meter_id': ['M001', 'M002', 'M003', 'M004'],
        'monthly_consumption': [
            '[100, 105, 110]',  # Too short (only 3 values)
            'invalid_data',      # Not parseable
            '[-10, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',  # Negative value
            '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'   # Valid
        ],
        'transformer_id': ['T001', 'T002', 'T003', 'T004']
    })
    
    validator = DataValidator()
    
    with LogContext(operation='validate_consumption_arrays'):
        logger.info("Validating consumption arrays")
        result = validator.validate_meter_data(problematic_data, raise_on_error=False)
        
        print(f"\nValidation Status: {'✓ PASSED' if result.is_valid else '✗ FAILED'}")
        print(f"\nIssues Found: {len(result.issues)}")
        
        for i, issue in enumerate(result.issues, 1):
            print(f"\nIssue #{i}:")
            print(f"  Severity: {issue.severity.value.upper()}")
            print(f"  Field: {issue.field}")
            print(f"  Message: {issue.message}")
            print(f"  Affected Rows: {len(issue.row_indices)} rows")
            print(f"  Sample Indices: {issue.row_indices[:3]}")
        
        logger.warning(f"Found {len(result.get_critical_issues())} critical issues")


def example_4_validation_summary():
    """Example 4: Generate comprehensive validation summary."""
    print("\n" + "=" * 80)
    print("Example 4: Comprehensive Validation Summary")
    print("=" * 80)
    
    # Create mixed quality data
    mixed_data = pd.DataFrame({
        'meter_id': ['M001', 'M002', None, 'M004'],
        'monthly_consumption': [
            '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
            '[200, 210]',  # Too short
            '[300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410]',
            'invalid'      # Not parseable
        ],
        'transformer_id': ['T001', 'T002', 'T003', None]
    })
    
    validator = DataValidator()
    
    with LogContext(operation='validate_summary'):
        logger.info("Generating validation summary")
        result = validator.validate_meter_data(mixed_data, raise_on_error=False)
        
        # Print comprehensive summary
        print("\n" + result.summary())
        
        # Separate critical issues from warnings
        critical = result.get_critical_issues()
        warnings = result.get_warnings()
        
        print(f"\nCritical Issues: {len(critical)}")
        for issue in critical:
            print(f"  - {issue}")
        
        if warnings:
            print(f"\nWarnings: {len(warnings)}")
            for issue in warnings:
                print(f"  - {issue}")


def example_5_custom_schema():
    """Example 5: Use custom validation schema."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Validation Schema")
    print("=" * 80)
    
    # Custom schema for quarterly data (4 values instead of 12)
    custom_schema = {
        'required_columns': ['meter_id', 'quarterly_consumption', 'transformer_id'],
        'optional_columns': [],
        'column_types': {
            'meter_id': ['object', 'str', 'string'],
            'quarterly_consumption': ['object', 'str', 'string'],
            'transformer_id': ['object', 'str', 'string']
        },
        'consumption_array_length': 4,  # Quarterly (Q1, Q2, Q3, Q4)
        'consumption_min_value': 0.0,
        'consumption_max_value': 100000.0
    }
    
    # Data with quarterly consumption
    quarterly_data = pd.DataFrame({
        'meter_id': ['M001', 'M002', 'M003'],
        'quarterly_consumption': [
            '[1200, 1350, 1500, 1400]',  # Q1-Q4
            '[2400, 2600, 2800, 2700]',
            '[1800, 1950, 2100, 2050]'
        ],
        'transformer_id': ['T001', 'T002', 'T003']
    })
    
    validator = DataValidator()
    
    # Update schema to use 'quarterly_consumption' field
    custom_schema_adapted = custom_schema.copy()
    
    # Create custom validation using the ConsumptionArrayRule
    from machine_learning.utils.data_validator import ConsumptionArrayRule
    
    with LogContext(operation='validate_custom_schema'):
        logger.info("Validating with custom quarterly schema")
        
        # Manual validation with custom rule
        rule = ConsumptionArrayRule(
            column_name='quarterly_consumption',
            expected_length=4,
            min_value=0.0,
            max_value=100000.0
        )
        
        issues = rule.validate(quarterly_data, {})
        
        print(f"\nCustom Schema Validation:")
        print(f"  Expected Length: 4 (quarterly)")
        print(f"  Column Name: quarterly_consumption")
        print(f"  Issues Found: {len(issues)}")
        
        if not issues:
            print("\n✓ Quarterly data validated successfully")
            logger.info("Custom schema validation passed")
        else:
            print("\n✗ Validation failed:")
            for issue in issues:
                print(f"  - {issue}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DataValidator Integration Examples")
    print("=" * 80)
    
    logger.info("Starting DataValidator examples")
    
    # Run all examples
    example_1_valid_data()
    example_2_missing_columns()
    example_3_invalid_consumption_arrays()
    example_4_validation_summary()
    example_5_custom_schema()
    
    print("\n" + "=" * 80)
    print("✓ All Examples Completed")
    print("=" * 80)
    
    logger.info("All examples completed successfully")


if __name__ == '__main__':
    main()
