"""
Backend Integration Test Script
================================

Tests the complete workflow:
1. Upload CSV
2. ML pipeline processes data
3. Results saved to Firestore
4. Frontend can fetch and display results

Run this after starting the FastAPI server (python main.py)
"""

import requests
import json
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health"""
    print("\n" + "="*70)
    print("🏥 Testing API Health...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"   ML Pipeline: {data['ml_pipeline']}")
        print(f"   Firestore: {data['firestore']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_upload_and_run():
    """Test CSV upload and ML pipeline execution"""
    print("\n" + "="*70)
    print("📤 Testing CSV Upload + ML Pipeline...")
    print("="*70)
    
    # Find sample CSV
    csv_path = Path(__file__).parent / "meter_consumption.csv"
    
    if not csv_path.exists():
        print(f"❌ Sample CSV not found: {csv_path}")
        return None
    
    print(f"📁 Using CSV: {csv_path.name}")
    
    # Upload CSV
    with open(csv_path, 'rb') as f:
        files = {'file': ('meter_consumption.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/run", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Analysis completed!")
        print(f"   Run ID: {data['run_id']}")
        print(f"   Total Meters: {data['total_meters']}")
        print(f"   Total Transformers: {data['total_transformers']}")
        print(f"   High Risk Count: {data['high_risk_count']}")
        print(f"   Processing Time: {data['processing_time_seconds']}s")
        return data['run_id']
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_get_results(run_id):
    """Test fetching results"""
    print("\n" + "="*70)
    print("📊 Testing Results Retrieval...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/results/{run_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Results retrieved!")
        print(f"   Cities: {len(data['cities'])}")
        print(f"   Barangays: {len(data['barangays'])}")
        print(f"   Transformers: {len(data['transformers'])}")
        print(f"   Meters: {len(data['meters'])}")
        print(f"\n📍 High-Risk Summary:")
        summary = data['high_risk_summary']
        print(f"   Most Anomalous City: {summary['most_anomalous_city']}")
        print(f"   Most Anomalous Barangay: {summary['most_anomalous_barangay']}")
        print(f"   Most Anomalous Transformer: {summary['most_anomalous_transformer']}")
        print(f"   Top Transformers: {len(summary['top_10_transformers'])}")
        return True
    else:
        print(f"❌ Failed to get results: {response.status_code}")
        return False

def test_geojson(run_id):
    """Test GeoJSON generation"""
    print("\n" + "="*70)
    print("🗺️  Testing GeoJSON Generation...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/transformers.geojson?run_id={run_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ GeoJSON generated!")
        print(f"   Features: {len(data['features'])}")
        if data['features']:
            sample = data['features'][0]
            print(f"\n   Sample Feature:")
            print(f"   - Transformer: {sample['properties']['transformer_id']}")
            print(f"   - Barangay: {sample['properties']['barangay']}")
            print(f"   - Risk Level: {sample['properties']['risk_level']}")
            print(f"   - Coordinates: {sample['geometry']['coordinates']}")
        return True
    else:
        print(f"❌ GeoJSON generation failed: {response.status_code}")
        return False

def test_export(run_id):
    """Test CSV export"""
    print("\n" + "="*70)
    print("💾 Testing CSV Export...")
    print("="*70)
    
    for level in ['transformer', 'district', 'meter']:
        response = requests.get(f"{BASE_URL}/api/export/{run_id}?level={level}")
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            print(f"✅ {level.capitalize()} export successful ({len(lines)-1} rows)")
        else:
            print(f"❌ {level.capitalize()} export failed: {response.status_code}")
            return False
    
    return True

def test_list_runs():
    """Test listing recent runs"""
    print("\n" + "="*70)
    print("📋 Testing Run List...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/runs")
    
    if response.status_code == 200:
        data = response.json()
        runs = data['runs']
        print(f"✅ Retrieved {len(runs)} recent runs")
        for i, run in enumerate(runs[:3], 1):
            print(f"   {i}. Run ID: {run['run_id'][:8]}... ({run['total_meters']} meters)")
        return True
    else:
        print(f"❌ Failed to list runs: {response.status_code}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 GHOSTLOAD MAPPER - BACKEND INTEGRATION TESTS")
    print("="*70)
    print("\nMake sure FastAPI server is running (python main.py)")
    print("Press Enter to start testing...")
    input()
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        return
    
    time.sleep(1)
    
    # Test 2: Upload and run analysis
    run_id = test_upload_and_run()
    if not run_id:
        print("\n❌ Upload/analysis failed. Check server logs.")
        return
    
    time.sleep(2)
    
    # Test 3: Get results
    if not test_get_results(run_id):
        print("\n❌ Failed to retrieve results.")
        return
    
    time.sleep(1)
    
    # Test 4: GeoJSON
    if not test_geojson(run_id):
        print("\n❌ GeoJSON generation failed.")
        return
    
    time.sleep(1)
    
    # Test 5: Export
    if not test_export(run_id):
        print("\n❌ Export failed.")
        return
    
    time.sleep(1)
    
    # Test 6: List runs
    if not test_list_runs():
        print("\n❌ List runs failed.")
        return
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📝 NEXT STEPS:")
    print("   1. Frontend can now call these endpoints")
    print("   2. Test with Swagger UI: http://localhost:8000/docs")
    print("   3. Check Firestore console for saved data")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
