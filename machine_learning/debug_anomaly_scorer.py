"""
Debug script to investigate AnomalyScorer API
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect
from machine_learning.evaluation.anomaly_scorer import AnomalyScorer

# Get the __init__ signature
print("=" * 80)
print("AnomalyScorer.__init__ Signature Analysis")
print("=" * 80)

sig = inspect.signature(AnomalyScorer.__init__)
print(f"\nMethod signature: {sig}")

print("\nParameters:")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    
    default = param.default
    default_str = f" = {default}" if default != inspect.Parameter.empty else " (required)"
    annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
    
    print(f"  - {param_name}: {annotation}{default_str}")

# Get docstring
print("\n" + "=" * 80)
print("Docstring:")
print("=" * 80)
if AnomalyScorer.__init__.__doc__:
    print(AnomalyScorer.__init__.__doc__)
else:
    print("No docstring available")

# Try to instantiate with minimal params
print("\n" + "=" * 80)
print("Test Instantiation:")
print("=" * 80)

try:
    # Try with no arguments
    scorer = AnomalyScorer()
    print("✓ Successfully created with no arguments")
    print(f"  Scorer object: {scorer}")
except TypeError as e:
    print(f"✗ Failed with no arguments: {e}")
    
    # Try with some common parameters
    try:
        scorer = AnomalyScorer(
            weights={'isolation_forest': 1.0}
        )
        print("✓ Successfully created with weights parameter")
    except TypeError as e2:
        print(f"✗ Failed with weights parameter: {e2}")
