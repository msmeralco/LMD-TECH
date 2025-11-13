"""
Quick Test - Upload CSV to Backend
===================================

This script tests uploading meter_consumption.csv to the running backend.

Run this while the backend server is running:
    python quick_test.py
"""

import requests
import json
from pathlib import Path
import time

# Configuration
API_URL = "http://localhost:8000"
CSV_FILE = Path(__file__).parent / "meter_consumption.csv"

def test_upload():
    """Test 1: Upload CSV and run analysis"""
    print("\n" + "="*70)
    print("🧪 TEST: Upload CSV and Run Analysis")
    print("="*70)
    
    if not CSV_FILE.exists():
        print(f"❌ CSV file not found: {CSV_FILE}")
        return None
    
    print(f"📂 Uploading: {CSV_FILE.name}")
    print(f"🌐 Endpoint: {API_URL}/api/run")
    
    try:
        with open(CSV_FILE, 'rb') as f:
            files = {'file': (CSV_FILE.name, f, 'text/csv')}
            
            print("⏳ Sending request...")
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/api/run",
                files=files,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS! ({elapsed:.2f}s)")
            print(f"\n📊 Results:")
            print(f"   Run ID: {data.get('run_id')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Total Meters: {data.get('total_meters')}")
            print(f"   Total Transformers: {data.get('total_transformers')}")
            print(f"   High Risk Count: {data.get('high_risk_count')}")
            print(f"   Processing Time: {data.get('processing_time_seconds')}s")
            
            return data.get('run_id')
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR")
        print("   Is the backend server running?")
        print("   Start it with: python start_server.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


def test_get_results(run_id):
    """Test 2: Fetch results"""
    print("\n" + "="*70)
    print("🧪 TEST: Fetch Results")
    print("="*70)
    
    if not run_id:
        print("⚠️ Skipping (no run_id)")
        return
    
    print(f"🔍 Fetching run_id: {run_id}")
    
    try:
        response = requests.get(f"{API_URL}/api/results/{run_id}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS!")
            print(f"\n📊 Data Structure:")
            print(f"   Cities: {len(data.get('cities', []))}")
            print(f"   Barangays: {len(data.get('barangays', []))}")
            print(f"   Transformers: {len(data.get('transformers', []))}")
            print(f"   Meters: {len(data.get('meters', []))}")
            
            # Show high-risk summary
            summary = data.get('high_risk_summary', {})
            if summary:
                print(f"\n🚨 High-Risk Summary:")
                print(f"   Most Anomalous City: {summary.get('most_anomalous_city')}")
                print(f"   Most Anomalous Barangay: {summary.get('most_anomalous_barangay')}")
                print(f"   Most Anomalous Transformer: {summary.get('most_anomalous_transformer')}")
                
                top_10 = summary.get('top_10_transformers', [])
                if top_10:
                    print(f"\n   🔝 Top 3 High-Risk Transformers:")
                    for i, t in enumerate(top_10[:3], 1):
                        print(f"      {i}. {t.get('transformer_id')} - "
                              f"Score: {t.get('avg_anomaly_score', 0):.3f}, "
                              f"Suspicious: {t.get('suspicious_meter_count', 0)} meters")
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_geojson(run_id):
    """Test 3: Get GeoJSON"""
    print("\n" + "="*70)
    print("🧪 TEST: GeoJSON Generation")
    print("="*70)
    
    try:
        url = f"{API_URL}/api/transformers.geojson"
        if run_id:
            url += f"?run_id={run_id}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            print(f"\n✅ SUCCESS!")
            print(f"   GeoJSON Features: {len(features)}")
            
            if features:
                print(f"\n   Sample Feature:")
                sample = features[0]
                props = sample.get('properties', {})
                coords = sample.get('geometry', {}).get('coordinates', [])
                print(f"      Transformer: {props.get('transformer_id')}")
                print(f"      Location: [{coords[1]:.4f}, {coords[0]:.4f}]")
                print(f"      Risk Level: {props.get('risk_level')}")
                print(f"      Suspicious Meters: {props.get('suspicious_meter_count')}")
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_health():
    """Test 4: Health check"""
    print("\n" + "="*70)
    print("🧪 TEST: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/api/health")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ System Status:")
            print(f"   API: {data.get('api')}")
            print(f"   ML Pipeline: {data.get('ml_pipeline')}")
            print(f"   Firestore: {data.get('firestore')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 GHOSTLOAD MAPPER - BACKEND TEST SUITE")
    print("="*70)
    print(f"\n🌐 Testing backend at: {API_URL}")
    print(f"📂 CSV file: {CSV_FILE.name}\n")
    
    # Test health first
    test_health()
    
    # Upload and analyze
    run_id = test_upload()
    
    # Fetch results
    if run_id:
        test_get_results(run_id)
        test_geojson(run_id)
    
    # Summary
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("   1. Check Swagger UI: http://localhost:8000/docs")
    print("   2. View run results in browser")
    print("   3. Test frontend integration")
    print("   4. Export CSV reports")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
