# 🔌 GhostLoad Mapper - Backend Integration Guide

**Version:** 1.0.0  
**Date:** November 13, 2025  
**Status:** Production Ready  
**ML System:** Optimized & Tested

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [API Integration](#api-integration)
4. [Data Schemas](#data-schemas)
5. [Error Handling](#error-handling)
6. [Performance & Monitoring](#performance--monitoring)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### **Single-Line Integration**

```python
from machine_learning.pipeline.inference_pipeline import predict_meter_risk

# In your FastAPI endpoint
result = predict_meter_risk(
    meter_id="MTR_000001",
    consumption_data={
        'monthly_consumption_202411': 710.51,
        'monthly_consumption_202412': 811.28,
        # ... 12 months total (YYYYMM format)
        'monthly_consumption_202510': 1023.97
    },
    transformer_id="TX_0001",
    customer_class="commercial",
    barangay="Poblacion",
    lat=14.409318,
    lon=120.979165,
    kVA=1296.97
)

# Returns JSON-ready dict:
# {
#   "meter_id": "MTR_000001",
#   "risk_level": "HIGH",
#   "anomaly_score": 0.85,
#   "confidence": 0.82,
#   "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
#   "timestamp": "2025-11-13T18:47:36.423429"
# }
```

---

## 🏗️ System Overview

### **ML Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│  POST /upload → POST /run-detection → GET /alerts          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ML INFERENCE PIPELINE                          │
│                                                             │
│  1. Feature Engineering (FeatureEngineer)                  │
│     ├─ Transformer baselines (median/variance)            │
│     ├─ Consumption trends (Theil-Sen regression)          │
│     └─ Relative consumption ratios                         │
│                                                             │
│  2. Anomaly Detection (IsolationForest)                    │
│     ├─ 19 engineered features                             │
│     ├─ contamination=12% (realistic theft rate)           │
│     └─ Decision threshold: 0.0621                          │
│                                                             │
│  3. Risk Assessment (RiskAssessor)                         │
│     ├─ HIGH risk: score ≥ 0.65 (5.2% of meters)          │
│     ├─ MEDIUM risk: score ≥ 0.45 (10.6% of meters)       │
│     └─ LOW risk: score < 0.45 (84.2% of meters)          │
│                                                             │
│  4. Output Generation                                       │
│     └─ JSON response with risk level + explanation        │
└─────────────────────────────────────────────────────────────┘
```

### **Trained Model Performance**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Training Time** | 1.78s | Fast retraining if needed |
| **Model Type** | IsolationForest | Unsupervised anomaly detection |
| **Features** | 19 | 12 consumption months + 4 engineered + 3 metadata |
| **Detection Rate** | 91.8% | Flags majority as anomalies |
| **High Risk Count** | 52 (5.2%) | **Urgent inspection** |
| **Medium Risk Count** | 106 (10.6%) | **Follow-up investigation** |
| **Low Risk Count** | 842 (84.2%) | Monitor only |
| **System Confidence** | 0.74 | High confidence |

**⚠️ IMPORTANT:** The 91.8% detection rate is due to synthetic data uniformity. **Filter for HIGH + MEDIUM risk only** (158 meters) for actionable inspection lists.

---

## 🔌 API Integration

### **Option 1: Single Meter Prediction (Recommended)**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from machine_learning.pipeline.inference_pipeline import predict_meter_risk

app = FastAPI()

class MeterPredictionRequest(BaseModel):
    meter_id: str
    consumption_data: dict  # monthly_consumption_YYYYMM keys
    transformer_id: str
    customer_class: str
    barangay: str
    lat: float
    lon: float
    kVA: float

@app.post("/api/predict-theft")
async def predict_theft_risk(request: MeterPredictionRequest):
    """
    Predict electricity theft risk for a single meter
    
    Returns:
        - risk_level: HIGH, MEDIUM, or LOW
        - anomaly_score: 0.0-1.0 (higher = more suspicious)
        - confidence: 0.0-1.0 (model confidence)
        - explanation: Human-readable risk description
    """
    try:
        result = predict_meter_risk(
            meter_id=request.meter_id,
            consumption_data=request.consumption_data,
            transformer_id=request.transformer_id,
            customer_class=request.customer_class,
            barangay=request.barangay,
            lat=request.lat,
            lon=request.lon,
            kVA=request.kVA
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### **Option 2: Batch Processing (CSV Upload)**

```python
import pandas as pd
from machine_learning.pipeline.inference_pipeline import InferencePipeline

@app.post("/api/upload-and-analyze")
async def analyze_meters_batch(file: UploadFile):
    """
    Process entire CSV of meter data
    
    Expected CSV columns:
        meter_id, transformer_id, customer_class, barangay,
        lat, lon, kVA,
        monthly_consumption_202411, ..., monthly_consumption_202510
    """
    # Read CSV
    df = pd.read_csv(file.file)
    
    # Initialize pipeline
    pipeline = InferencePipeline()
    
    # Get predictions for all meters
    results = []
    for _, row in df.iterrows():
        prediction = pipeline.predict(row.to_dict())
        results.append(prediction.to_dict())
    
    # Filter for actionable meters only
    actionable = [r for r in results if r['risk_level'] in ['HIGH', 'MEDIUM']]
    
    return {
        "total_meters": len(results),
        "high_risk": len([r for r in results if r['risk_level'] == 'HIGH']),
        "medium_risk": len([r for r in results if r['risk_level'] == 'MEDIUM']),
        "actionable_meters": actionable
    }
```

---

## 📊 Data Schemas

### **Input Schema (Required Fields)**

```json
{
  "meter_id": "MTR_000001",
  "transformer_id": "TX_0001",
  "customer_class": "commercial",
  "barangay": "Poblacion",
  "lat": 14.409318,
  "lon": 120.979165,
  "kVA": 1296.97,
  "monthly_consumption_202411": 710.51,
  "monthly_consumption_202412": 811.28,
  "monthly_consumption_202501": 663.03,
  "monthly_consumption_202502": 633.15,
  "monthly_consumption_202503": 1070.65,
  "monthly_consumption_202504": 897.33,
  "monthly_consumption_202505": 996.28,
  "monthly_consumption_202506": 932.92,
  "monthly_consumption_202507": 989.90,
  "monthly_consumption_202508": 945.07,
  "monthly_consumption_202509": 1037.30,
  "monthly_consumption_202510": 1023.97
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `meter_id` | string | ✅ | Unique meter identifier |
| `transformer_id` | string | ✅ | Transformer ID (for baseline comparison) |
| `customer_class` | string | ✅ | "residential", "commercial", "industrial" |
| `barangay` | string | ✅ | Geographic location |
| `lat` | float | ✅ | Latitude coordinate |
| `lon` | float | ✅ | Longitude coordinate |
| `kVA` | float | ✅ | Installed capacity (kilovolt-amperes) |
| `monthly_consumption_YYYYMM` | float | ✅ | 12 months of consumption (kWh) |

**⚠️ CRITICAL:** 
- Consumption columns **must** use `monthly_consumption_YYYYMM` format
- Requires **exactly 12 consecutive months**
- Month format: YYYYMM (e.g., 202411 = November 2024)

### **Output Schema**

```json
{
  "meter_id": "MTR_000001",
  "risk_level": "HIGH",
  "anomaly_score": 0.85,
  "confidence": 0.82,
  "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
  "timestamp": "2025-11-13T18:47:36.423429",
  "consumption_pattern": "NORMAL",
  "spatial_risk": null,
  "temporal_risk": null
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `meter_id` | string | Echo of input meter ID |
| `risk_level` | string | **"HIGH"**, **"MEDIUM"**, or **"LOW"** |
| `anomaly_score` | float | 0.0-1.0 (higher = more suspicious) |
| `confidence` | float | 0.0-1.0 (model confidence in prediction) |
| `explanation` | string | Human-readable risk description |
| `timestamp` | string | ISO 8601 timestamp of prediction |
| `consumption_pattern` | string | "NORMAL", "DECLINING", or "ERRATIC" |
| `spatial_risk` | null/float | Reserved for DBSCAN spatial clustering |
| `temporal_risk` | null/float | Reserved for time-series analysis |

---

## 🛡️ Error Handling

### **Common Errors & Solutions**

| Error | Cause | Solution |
|-------|-------|----------|
| `Model file not found` | Trained model missing | Run `python train.py` first |
| `No consumption columns found` | Wrong column format | Use `monthly_consumption_YYYYMM` |
| `X has 17 features, model trained with 19` | Missing lat/lon/kVA | Include all required fields |
| `Import error: No module named 'machine_learning'` | Path issue | Add ML directory to sys.path |

### **Robust Error Handling Example**

```python
from fastapi import HTTPException

@app.post("/api/predict-theft")
async def predict_theft_risk(request: MeterPredictionRequest):
    try:
        # Validate consumption data
        if not request.consumption_data:
            raise ValueError("consumption_data cannot be empty")
        
        # Check for 12 months
        consumption_keys = [k for k in request.consumption_data.keys() 
                           if k.startswith('monthly_consumption_')]
        if len(consumption_keys) != 12:
            raise ValueError(f"Expected 12 months, got {len(consumption_keys)}")
        
        # Predict
        result = predict_meter_risk(
            meter_id=request.meter_id,
            consumption_data=request.consumption_data,
            transformer_id=request.transformer_id,
            customer_class=request.customer_class,
            barangay=request.barangay,
            lat=request.lat,
            lon=request.lon,
            kVA=request.kVA
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="ML model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

---

## 📈 Performance & Monitoring

### **Expected Performance**

| Operation | Time | Notes |
|-----------|------|-------|
| Model Loading | ~0.5s | First request only (cached) |
| Single Prediction | ~10ms | Feature engineering + inference |
| Batch (100 meters) | ~1s | Parallelizable |
| Batch (1000 meters) | ~10s | Consider async processing |

### **Monitoring Recommendations**

```python
import time
import logging

logger = logging.getLogger("ghostload_api")

@app.post("/api/predict-theft")
async def predict_theft_risk(request: MeterPredictionRequest):
    start_time = time.time()
    
    try:
        result = predict_meter_risk(...)
        
        # Log successful prediction
        elapsed = time.time() - start_time
        logger.info(f"Prediction for {request.meter_id}: "
                   f"risk={result['risk_level']}, "
                   f"score={result['anomaly_score']:.3f}, "
                   f"time={elapsed:.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed for {request.meter_id}: {str(e)}")
        raise
```

### **Key Metrics to Track**

1. **Prediction latency** (p50, p95, p99)
2. **Error rate** (by error type)
3. **Risk level distribution** (HIGH/MEDIUM/LOW ratio)
4. **Average anomaly score** (detect model drift)
5. **Requests per minute** (capacity planning)

---

## 🔧 Troubleshooting

### **Issue: Import Errors**

```python
# Fix: Add project root to sys.path
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "machine_learning"))
```

### **Issue: Model Not Found**

```bash
# Solution 1: Check if model exists
ls output/latest/trained_model.pkl

# Solution 2: Retrain model
cd machine_learning
python train.py
```

### **Issue: Wrong Feature Count**

```python
# Problem: Missing fields
# Solution: Ensure ALL required fields are present
required_fields = [
    'meter_id', 'transformer_id', 'customer_class', 'barangay',
    'lat', 'lon', 'kVA',
    'monthly_consumption_202411', 'monthly_consumption_202412',
    # ... all 12 months
]

# Validate before prediction
for field in required_fields:
    if field not in data:
        raise ValueError(f"Missing required field: {field}")
```

---

## 🎯 Recommended Workflow

### **For Backend Team**

1. **Initial Setup** (5 min)
   ```bash
   cd machine_learning
   python train.py  # Train model (1.8s)
   python pipeline/inference_pipeline.py  # Test (all tests pass)
   ```

2. **FastAPI Integration** (15 min)
   - Copy single meter endpoint from above
   - Test with sample data
   - Add error handling

3. **Frontend Data Flow** (10 min)
   - Create GET /alerts endpoint
   - Filter: `WHERE risk_level IN ('HIGH', 'MEDIUM')`
   - Return: `{meters: [...], total_high: 52, total_medium: 106}`

4. **Testing** (10 min)
   ```bash
   curl -X POST "http://localhost:8000/api/predict-theft" \
     -H "Content-Type: application/json" \
     -d @test_meter.json
   ```

### **Production Checklist**

- [ ] Model trained and saved to `output/latest/`
- [ ] All inference tests passing
- [ ] FastAPI endpoint returning correct JSON
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Frontend receiving HIGH/MEDIUM risk meters
- [ ] Map showing color-coded dots
- [ ] CSV export working

---

## 📞 Support

**ML Team Contact:**  
- Training issues: Check `machine_learning/logs/`
- Inference errors: Review API logs
- Model performance: See `output/latest/metrics.json`

**Critical Files:**
- Model: `output/latest/trained_model.pkl`
- Config: `machine_learning/config/config.yaml`
- Metrics: `output/latest/metrics.json`
- Results: `output/latest/risk_assessment.csv`

---

## 🎉 Success Criteria

Your backend integration is **COMPLETE** when:

✅ Single meter prediction returns JSON in <50ms  
✅ Batch processing handles 1000 meters  
✅ Frontend receives 52 HIGH + 106 MEDIUM risk meters  
✅ Map displays color-coded markers  
✅ Error handling prevents crashes  
✅ **Demo works smoothly for judges!** 🏆

---

**Version:** 1.0.0  
**Last Updated:** November 13, 2025  
**Next Review:** Post-Hackathon
