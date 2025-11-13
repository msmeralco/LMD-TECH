"""
Final Verification: DataValidator Module

This script performs comprehensive verification of the data_validator module
to ensure production readiness.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("DataValidator Module - Final Verification")
print("=" * 80)

# Test 1: Module imports
print("\n[1/5] Testing module imports...")
try:
    from machine_learning.utils.data_validator import (
        DataValidator,
        ValidationResult,
        ValidationIssue,
        ValidationRule,
        DatasetType,
        ValidationSeverity,
        SchemaValidationError,
        BusinessRuleValidationError,
        validate_meter_data_csv,
        validate_meter_data_df,
        METER_DATA_SCHEMA
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create validator instance
print("\n[2/5] Creating DataValidator instance...")
try:
    validator = DataValidator()
    print(f"✓ DataValidator instance created: {validator}")
except Exception as e:
    print(f"✗ Failed to create instance: {e}")
    sys.exit(1)

# Test 3: Validate schema info
print("\n[3/5] Checking schema information...")
try:
    schema = validator.get_schema_info(DatasetType.METER_DATA)
    assert 'required_columns' in schema
    assert 'meter_id' in schema['required_columns']
    assert 'monthly_consumption' in schema['required_columns']
    assert 'transformer_id' in schema['required_columns']
    assert schema['consumption_array_length'] == 12
    print(f"✓ Schema validation passed")
    print(f"  Required columns: {schema['required_columns']}")
    print(f"  Consumption length: {schema['consumption_array_length']}")
except Exception as e:
    print(f"✗ Schema check failed: {e}")
    sys.exit(1)

# Test 4: Test validation with sample data
print("\n[4/5] Testing validation with sample data...")
try:
    import pandas as pd
    
    # Valid data
    valid_df = pd.DataFrame({
        'meter_id': ['M001', 'M002'],
        'monthly_consumption': [
            '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
            '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
        ],
        'transformer_id': ['T001', 'T002']
    })
    
    result = validator.validate_meter_data(valid_df, raise_on_error=False)
    assert result.is_valid, "Valid data should pass validation"
    print("✓ Valid data validated successfully")
    
    # Invalid data (missing column)
    invalid_df = pd.DataFrame({
        'meter_id': ['M001', 'M002'],
        'monthly_consumption': [
            '[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]',
            '[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310]'
        ]
        # Missing transformer_id
    })
    
    result = validator.validate_meter_data(invalid_df, raise_on_error=False)
    assert not result.is_valid, "Invalid data should fail validation"
    critical_issues = result.get_critical_issues()
    assert len(critical_issues) > 0, "Should have critical issues"
    
    # Check for missing field in error
    has_missing_transformer = any(
        'transformer_id' in str(issue.details.get('missing', []))
        for issue in critical_issues
    )
    assert has_missing_transformer, "Should detect missing transformer_id"
    print("✓ Invalid data detected correctly")
    print(f"  Critical issues found: {len(critical_issues)}")
    
except Exception as e:
    print(f"✗ Validation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test exception handling
print("\n[5/5] Testing exception handling...")
try:
    invalid_df = pd.DataFrame({
        'meter_id': ['M001'],
        'monthly_consumption': ['[100, 105, 110]']  # Only 3 values
        # Missing transformer_id
    })
    
    exception_raised = False
    missing_fields = []
    
    try:
        validator.validate_meter_data(invalid_df, raise_on_error=True)
    except SchemaValidationError as e:
        exception_raised = True
        error_message = str(e)
        assert 'Missing required fields' in error_message or 'transformer_id' in error_message
        print("✓ SchemaValidationError raised correctly")
        print(f"  Error message: {e.message}")
        print(f"  Error details: {e.details}")
    
    assert exception_raised, "Should raise SchemaValidationError"
    
except Exception as e:
    print(f"✗ Exception test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("✓ ALL VERIFICATION TESTS PASSED")
print("=" * 80)
print("\nModule Status: ✅ PRODUCTION READY")
print("Test Coverage: 5/5 verification tests passing")
print("\nKey Features Verified:")
print("  ✓ Module imports")
print("  ✓ Instance creation")
print("  ✓ Schema validation")
print("  ✓ Data validation (valid & invalid)")
print("  ✓ Exception handling")
print("\nThe data_validator module is ready for production use!")
print("=" * 80)
