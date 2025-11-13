"""
Diagnostic Test: AnomalyScorer API Verification
================================================

This test verifies the actual signature of AnomalyScorer.score() method
to identify the parameter mismatch causing the pipeline failure.

Author: ML Systems Engineer
Date: November 13, 2025
"""

import sys
from pathlib import Path
import inspect
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from machine_learning.evaluation.anomaly_scorer import AnomalyScorer, ScoringConfig

print("=" * 80)
print("ANOMALY SCORER API DIAGNOSTIC TEST")
print("=" * 80)

# 1. Inspect the score() method signature
print("\n1. Inspecting AnomalyScorer.score() signature:")
print("-" * 80)
sig = inspect.signature(AnomalyScorer.score)
print(f"Method signature: {sig}")

print("\nParameters:")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    default = param.default
    default_str = f" = {default}" if default != inspect.Parameter.empty else " (required)"
    annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
    print(f"  - {param_name}: {annotation}{default_str}")

# 2. Check the docstring
print("\n2. Method Documentation:")
print("-" * 80)
if AnomalyScorer.score.__doc__:
    print(AnomalyScorer.score.__doc__)
else:
    print("No docstring available")

# 3. Test actual usage
print("\n3. Testing Actual Usage:")
print("-" * 80)

try:
    # Create scorer
    config = ScoringConfig(isolation_weight=0.7, ratio_weight=0.3)
    scorer = AnomalyScorer(config=config)
    
    # Create dummy data
    n_samples = 10
    isolation_scores = np.random.rand(n_samples)
    consumption_ratios = np.random.rand(n_samples) * 2
    spatial_anomalies = np.zeros(n_samples)
    
    print(f"✓ Created scorer with config")
    print(f"✓ Generated test data: {n_samples} samples")
    
    # Test 1: Try with spatial_anomalies (current pipeline code)
    try:
        result = scorer.score(
            isolation_scores=isolation_scores,
            consumption_ratios=consumption_ratios,
            spatial_anomalies=spatial_anomalies
        )
        print(f"✓ Test 1 PASSED: score() accepts spatial_anomalies parameter")
        print(f"  Result type: {type(result)}")
        
    except TypeError as e:
        print(f"✗ Test 1 FAILED: {e}")
        
        # Test 2: Try without spatial_anomalies
        try:
            result = scorer.score(
                isolation_scores=isolation_scores,
                consumption_ratios=consumption_ratios
            )
            print(f"✓ Test 2 PASSED: score() works WITHOUT spatial_anomalies")
            print(f"  Result type: {type(result)}")
            
            # Check if result has expected attributes
            if hasattr(result, 'composite_score'):
                print(f"  ✓ Result has 'composite_score' attribute")
            if hasattr(result, 'confidence'):
                print(f"  ✓ Result has 'confidence' attribute")
                
        except Exception as e2:
            print(f"✗ Test 2 FAILED: {e2}")
            
except Exception as e:
    print(f"✗ Setup failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
