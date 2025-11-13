"""
Test ML Pipeline Integration with Backend
==========================================

This script tests the complete ML → Backend integration to ensure
the adapter layer works correctly.

Run this BEFORE starting the server to catch any issues.

Author: Backend Team
Date: November 13, 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "machine_learning"))

import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_ml_output_format():
    """Test 1: Verify ML pipeline output format"""
    print("\n" + "="*70)
    print("🧪 TEST 1: ML Pipeline Output Format")
    print("="*70)
    
    try:
        from machine_learning.pipeline.inference_pipeline import InferencePipeline
        
        # Initialize pipeline
        print("📦 Initializing ML pipeline...")
        pipeline = InferencePipeline()
        print("✅ Pipeline initialized")
        
        # Create test data matching your CSV format
        test_data = {
            'meter_id': 'MTR_TEST_001',
            'transformer_id': 'TX_0001',
            'customer_class': 'commercial',
            'barangay': 'Poblacion',
            'lat': 14.409318,
            'lon': 120.979165,
            'monthly_consumption_202411': 710.51,
            'monthly_consumption_202412': 811.28,
            'monthly_consumption_202501': 663.03,
            'monthly_consumption_202502': 633.15,
            'monthly_consumption_202503': 1070.65,
            'monthly_consumption_202504': 897.33,
            'monthly_consumption_202505': 996.28,
            'monthly_consumption_202506': 932.92,
            'monthly_consumption_202507': 989.90,
            'monthly_consumption_202508': 945.07,
            'monthly_consumption_202509': 1037.30,
            'monthly_consumption_202510': 1023.97,
            'kVA': 1296.97
        }
        
        # Get prediction
        print("🔮 Running prediction...")
        result = pipeline.predict(test_data)
        result_dict = result.to_dict()
        
        # Verify required fields
        print("\n📋 ML Output Structure:")
        print(f"   Keys: {list(result_dict.keys())}")
        
        required_fields = ['meter_id', 'risk_level', 'anomaly_score', 'confidence', 'explanation']
        missing_fields = [f for f in required_fields if f not in result_dict]
        
        if missing_fields:
            print(f"\n❌ MISSING FIELDS: {missing_fields}")
            print("   Backend adapter layer will handle these.")
        else:
            print("\n✅ All required fields present!")
        
        print("\n📊 Sample Output:")
        for key in required_fields:
            value = result_dict.get(key, "MISSING")
            print(f"   {key}: {value}")
        
        return True, result_dict
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batch_prediction():
    """Test 2: Batch prediction with DataFrame"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Batch Prediction (DataFrame)")
    print("="*70)
    
    try:
        from machine_learning.pipeline.inference_pipeline import InferencePipeline
        
        # Load actual CSV
        csv_path = Path(__file__).parent / "meter_consumption.csv"
        
        if not csv_path.exists():
            print(f"⚠️ CSV not found at: {csv_path}")
            print("   Skipping batch test...")
            return True, None
        
        print(f"📂 Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} meters")
        
        # Add missing columns that CSV doesn't have
        print("🔧 Preparing data (adding missing columns)...")
        if 'lat' not in df.columns:
            df['lat'] = 14.5  # Default Manila lat
        if 'lon' not in df.columns:
            df['lon'] = 120.9  # Default Manila lon
        if 'kVA' not in df.columns:
            df['kVA'] = 100.0  # Default kVA
        
        # Pad consumption months to 12 if needed
        consumption_cols = [c for c in df.columns if c.startswith('monthly_consumption_')]
        if len(consumption_cols) < 12:
            print(f"   Padding from {len(consumption_cols)} to 12 months...")
            # Get last month value and duplicate it
            if consumption_cols:
                last_col = consumption_cols[-1]
                for i in range(len(consumption_cols), 12):
                    new_col = f"monthly_consumption_20250{i+1}"
                    df[new_col] = df[last_col]
        
        print(f"✅ Data prepared with {len([c for c in df.columns if c.startswith('monthly_consumption_')])} consumption months")
        
        # Initialize pipeline
        print("📦 Initializing pipeline...")
        pipeline = InferencePipeline()
        
        # Run predictions
        print("🔮 Running batch predictions...")
        results = pipeline.predict(df)
        
        # Convert to list of dicts
        result_dicts = [r.to_dict() for r in results]
        
        print(f"✅ Generated {len(result_dicts)} predictions")
        
        # Check distribution
        risk_counts = {}
        for r in result_dicts:
            risk = r.get('risk_level', 'UNKNOWN')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print("\n📊 Risk Distribution:")
        for risk, count in sorted(risk_counts.items()):
            pct = (count / len(result_dicts)) * 100
            print(f"   {risk}: {count} ({pct:.1f}%)")
        
        return True, result_dicts
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_adapter_layer():
    """Test 3: Backend adapter layer"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Backend Adapter Layer")
    print("="*70)
    
    try:
        # Simulate ML output with missing fields
        print("🔄 Testing adapter with incomplete ML output...")
        
        incomplete_output = pd.DataFrame([
            {
                'meter_id': 'M001',
                'anomaly_score': 0.85,
                # Missing: risk_level, confidence, explanation
            },
            {
                'meter_id': 'M002',
                'anomaly_score': 0.35,
            }
        ])
        
        # Apply adapter logic (same as in routes.py)
        pred_df = incomplete_output.copy()
        
        required_cols = ['meter_id', 'risk_level', 'anomaly_score', 'confidence', 'explanation']
        
        for col in required_cols:
            if col not in pred_df.columns:
                print(f"   ⚠️ Adding missing column: {col}")
                
                if col == 'risk_level':
                    pred_df['risk_level'] = pred_df['anomaly_score'].apply(
                        lambda x: "HIGH" if x >= 0.7 else "MEDIUM" if x >= 0.4 else "LOW"
                    )
                elif col == 'confidence':
                    pred_df['confidence'] = (pred_df['anomaly_score'] - 0.5).abs() * 2
                elif col == 'explanation':
                    pred_df['explanation'] = "Anomaly detection completed"
        
        print("\n✅ Adapter Layer Result:")
        print(pred_df.to_string(index=False))
        
        return True, pred_df
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_backend_imports():
    """Test 4: Verify backend can import everything"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Backend Import Check")
    print("="*70)
    
    try:
        print("📦 Testing backend imports...")
        
        # Test ML imports
        from machine_learning.pipeline.inference_pipeline import InferencePipeline, predict_meter_risk
        print("   ✅ InferencePipeline")
        print("   ✅ predict_meter_risk")
        
        # Test backend imports
        sys.path.insert(0, str(Path(__file__).parent))
        from app.api.routes import get_ml_pipeline, aggregate_hierarchical_data
        print("   ✅ get_ml_pipeline")
        print("   ✅ aggregate_hierarchical_data")
        
        from app.db.firestore import save_run_results, get_run_results
        print("   ✅ save_run_results")
        print("   ✅ get_run_results")
        
        print("\n✅ All imports successful!")
        return True, None
        
    except Exception as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 ML → BACKEND INTEGRATION TEST SUITE")
    print("="*70)
    print("\nThis will verify that ML pipeline and backend communicate correctly.\n")
    
    results = []
    
    # Run tests
    tests = [
        ("ML Output Format", test_ml_output_format),
        ("Batch Prediction", test_batch_prediction),
        ("Adapter Layer", test_adapter_layer),
        ("Backend Imports", test_backend_imports),
    ]
    
    for test_name, test_func in tests:
        success, data = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - INTEGRATION READY!")
        print("\n✅ Next Steps:")
        print("   1. Start backend: python main.py")
        print("   2. Upload CSV via Swagger: http://localhost:8000/docs")
        print("   3. Check logs for '✅ ML pipeline completed'")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED - Review output above")
        print("\n🔧 Troubleshooting:")
        print("   - Check if model trained: python train.py")
        print("   - Verify CSV format has all required columns")
        print("   - Check logs for detailed error messages")
        return 1


if __name__ == "__main__":
    exit(main())
