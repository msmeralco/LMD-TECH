# 🎯 BEGINNER'S GUIDE - Training & Deploying ML Models
## For GhostLoad Mapper Hackathon

**Created**: November 13, 2025  
**For**: ML Team Member (Beginner Friendly!)  
**Status**: Complete Production Guide

---

## 📋 WHAT YOU HAVE NOW

✅ **Complete ML System** (9 modules, all done!)
- Data loading and preprocessing
- Feature engineering
- Model training (IsolationForest)
- Risk assessment and scoring
- Training pipeline (orchestrator)
- **Inference pipeline (NEW - just created!)** ← This is what backend needs!

---

## 🚀 STEP-BY-STEP GUIDE

### **STEP 1: Train Your First Model** (5 minutes)

This creates the trained model that your backend team needs.

```bash
# Navigate to your project
cd C:\Users\Ken Ira Talingting\Desktop\GhostLoadMapper-IDOL_Hackathon-

# Activate virtual environment
machine_learning\venv\Scripts\activate

# Run training script (this trains the model!)
python machine_learning/train.py
```

**What This Does:**
- Loads meter consumption data from `machine_learning/datasets/demo/`
- Engineers 7 statistical features from consumption patterns
- Trains IsolationForest model (100 estimators, 10% contamination)
- Saves trained model to `output/latest/trained_model.pkl`
- Creates versioned backup in `output/v_TIMESTAMP/`
- **Takes 2-3 minutes on demo data**

**Expected Output:**
```
================================================================================
                      GHOSTLOAD MAPPER - ML MODEL TRAINING
================================================================================

📂 Loading data from machine_learning/datasets/demo/meter_consumption.csv...
   ✓ Loaded 200 meters
🔧 Engineering features...
   ✓ Found 6 consumption columns
   ✓ Created 7 features
📊 Normalizing features...
   ✓ Features normalized
🤖 Training Isolation Forest model...
   ✓ Model trained
💾 Saving model...
   ✓ Model saved to: output/latest/trained_model.pkl

================================================================================
                              ✅ TRAINING COMPLETE!
================================================================================

📊 Model Summary:
   • Training samples: 200
   • Features: 7
   • Algorithm: Isolation Forest
   • Estimators: 100
   • Contamination: 10%
```

---

### **STEP 2: Test the Trained Model** (2 minutes)

Verify the model works correctly.

```bash
# Run inference pipeline self-test
python machine_learning/pipeline/inference_pipeline.py
```

**Expected Output:**
```
🧪 INFERENCE PIPELINE - SELF TEST
======================================================================

[1/4] Checking for trained model...
    ✅ Found model at: output/latest/trained_model.pkl

[2/4] Initializing inference pipeline...
    ✅ Pipeline initialized successfully

[3/4] Testing single meter prediction...
    ✅ Prediction successful!
    📊 Result:
       - Meter ID: TEST_001
       - Risk Level: MEDIUM
       - Anomaly Score: 0.652
       - Confidence: 0.700

[4/4] Testing convenience function (for backend)...
    ✅ Convenience function works!

✅ ALL TESTS PASSED - INFERENCE PIPELINE READY FOR BACKEND!
```

---

### **STEP 3: Give Backend Team What They Need** (1 minute)

Your backend team needs **ONE simple function**. Send them this code:

```python
# backend/api/predict.py (or similar)

from machine_learning.pipeline.inference_pipeline import predict_meter_risk

@app.post("/api/predict")
async def predict_risk(meter_id: str, consumption: List[float]):
    """
    Predict risk level for a meter.
    
    Example request:
    POST /api/predict
    {
        "meter_id": "M12345",
        "consumption": [100, 120, 115, 140, 110, 125]
    }
    """
    result = predict_meter_risk(meter_id, consumption)
    return result  # Already JSON-ready!
```

**That's it!** The backend just calls `predict_meter_risk()` and gets results!

---

## 📊 WHAT THE BACKEND GETS

When backend calls `predict_meter_risk()`, they get this JSON:

```json
{
  "meter_id": "M12345",
  "risk_level": "HIGH",
  "anomaly_score": 0.85,
  "confidence": 0.85,
  "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
  "consumption_pattern": "ANOMALOUS",
  "timestamp": "2025-11-13T15:30:00"
}
```

**Risk Levels:**
- 🔴 **HIGH** (score ≥ 0.7): Immediate field inspection needed
- 🟡 **MEDIUM** (score 0.4-0.7): Monitor closely
- 🟢 **LOW** (score < 0.4): Normal consumption

---

## 🎓 UNDERSTANDING WHAT YOU BUILT

### **What is the ML doing?**

1. **Training Phase** (you just did this!):
   - Model learns what "normal" consumption looks like
   - Identifies patterns in regular meter behavior
   - Creates a trained model file

2. **Inference Phase** (backend uses this):
   - Loads the trained model
   - Analyzes new meter data
   - Detects anomalies (unusual patterns)
   - Returns risk level + score

### **How does it detect theft?**

The model looks for:
- 📉 **Sudden drops** in consumption (meter tampering)
- 📊 **Unusual patterns** (different from neighbors)
- 🔄 **Erratic behavior** (inconsistent readings)
- 📍 **Spatial anomalies** (different from transformer area)

---

## 🔧 TROUBLESHOOTING

### **Problem 1: "Model not found"**

**Solution:** Train the model first!
```bash
python machine_learning/pipeline/training_pipeline.py
```

---

### **Problem 2: "Module not found"**

**Solution:** Make sure virtual environment is activated!
```bash
machine_learning\venv\Scripts\activate
```

---

### **Problem 3: "No data found"**

**Solution:** Check datasets folder:
```bash
dir machine_learning\datasets\demo
```
Should see: `meter_consumption.csv`, `transformers.csv`

---

### **Problem 4: Backend can't import**

**Solution:** Add to Python path in backend code:
```python
import sys
sys.path.append('C:/Users/Ken Ira Talingting/Desktop/GhostLoadMapper-IDOL_Hackathon-')

from machine_learning.pipeline.inference_pipeline import predict_meter_risk
```

---

## 📝 COMPLETE WORKFLOW FOR HACKATHON

### **Morning Setup** (You - ML Team)

1. ✅ Train model (5 min):
   ```bash
   python machine_learning/pipeline/training_pipeline.py
   ```

2. ✅ Test inference (2 min):
   ```bash
   python machine_learning/pipeline/inference_pipeline.py
   ```

3. ✅ Share with backend team:
   - Send them `inference_pipeline.py` location
   - Send them the one function: `predict_meter_risk()`

### **Afternoon Integration** (Backend Team)

4. Backend adds to FastAPI (5 min):
   ```python
   from machine_learning.pipeline.inference_pipeline import predict_meter_risk
   
   @app.post("/predict")
   async def predict(meter_id: str, consumption: List[float]):
       return predict_meter_risk(meter_id, consumption)
   ```

5. Frontend calls API (already done by frontend team):
   ```javascript
   fetch('/api/predict', {
       method: 'POST',
       body: JSON.stringify({
           meter_id: 'M001',
           consumption: [100, 120, 115]
       })
   })
   ```

### **Demo Time** (Everyone)

6. Show live predictions on map! 🎉

---

## 🎯 YOUR NEXT ACTIONS (In Order)

### **RIGHT NOW** (Next 10 minutes):

1. **Train the model:**
   ```bash
   cd C:\Users\Ken Ira Talingting\Desktop\GhostLoadMapper-IDOL_Hackathon-
   machine_learning\venv\Scripts\activate
   python machine_learning/pipeline/training_pipeline.py
   ```
   ⏱️ Wait 2-3 minutes for training to complete

2. **Test inference:**
   ```bash
   python machine_learning/pipeline/inference_pipeline.py
   ```
   ✅ Should see "ALL TESTS PASSED"

3. **Tell your team:**
   - Message backend team: "ML inference ready! Use `predict_meter_risk()` function"
   - Show them this file location: `machine_learning/pipeline/inference_pipeline.py`

---

### **AFTER TRAINING** (When backend integrates):

4. **Help backend test:**
   ```python
   # Test in Python console
   from machine_learning.pipeline.inference_pipeline import predict_meter_risk
   
   result = predict_meter_risk('M001', [100, 120, 115, 140])
   print(result)
   ```

5. **Monitor results:**
   - Check if predictions make sense
   - HIGH risk should be unusual consumption
   - LOW risk should be normal consumption

6. **Celebrate!** 🎉
   - Your ML system is working!
   - Backend can now show predictions on map!

---

## 📞 QUICK HELP

### **If stuck, check these:**

✅ **Virtual environment activated?**
```bash
# Should see (venv) at start of prompt
(venv) PS C:\...>
```

✅ **In correct directory?**
```bash
# Should be in project root
pwd
# Should show: ...\GhostLoadMapper-IDOL_Hackathon-
```

✅ **Model trained?**
```bash
# Check if model exists
Test-Path output\latest\trained_model.pkl
# Should return: True
```

✅ **Packages installed?**
```bash
pip list | findstr scikit-learn
# Should show: scikit-learn 1.x.x
```

---

## 🎓 LEARNING RESOURCES (Optional)

### **Want to understand more?**

- **IsolationForest**: Detects anomalies by isolating unusual data points
- **Feature Engineering**: Creating useful data from raw consumption readings
- **Risk Scoring**: Converting model output to risk levels (HIGH/MEDIUM/LOW)

### **Files to explore:**
1. `training_pipeline.py` - How model is trained
2. `inference_pipeline.py` - How predictions are made
3. `config.yaml` - All settings and thresholds

---

## ✅ SUCCESS CHECKLIST

Before telling backend team you're ready:

- [ ] Model trained successfully (`output/latest/trained_model.pkl` exists)
- [ ] Inference self-test passes (all 4 tests ✅)
- [ ] Can run `predict_meter_risk()` in Python console
- [ ] Understand what JSON backend receives
- [ ] Know where `inference_pipeline.py` file is located

**All checked?** You're ready! 🚀

---

## 🎉 CONGRATULATIONS!

You've built a **complete production ML system** for the hackathon!

**What you accomplished:**
✅ Trained an anomaly detection model  
✅ Created inference pipeline for real-time predictions  
✅ Enabled backend integration with one simple function  
✅ Ready to detect electricity theft and save millions!  

**Your backend team can now:**
- Get predictions for any meter
- Display risk levels on the map
- Prioritize field inspections
- Demo live during hackathon!

---

## 📬 COMMUNICATION TEMPLATES

### **Message to Backend Team:**

```
Hey Backend Team! 🎉

ML inference is ready! Here's what you need:

📍 File: machine_learning/pipeline/inference_pipeline.py

🔧 Usage (add to your FastAPI):
```python
from machine_learning.pipeline.inference_pipeline import predict_meter_risk

@app.post("/api/predict")
async def predict(meter_id: str, consumption: List[float]):
    return predict_meter_risk(meter_id, consumption)
```

📊 Returns JSON with:
- risk_level: "HIGH" / "MEDIUM" / "LOW"
- anomaly_score: 0.0 to 1.0
- explanation: Why it's flagged

Tested and working! Let me know if you need help integrating.
```

---

### **Message to Frontend Team:**

```
Hey Frontend Team! 🎨

Backend will have /api/predict endpoint ready soon.

📝 Request format:
{
  "meter_id": "M12345",
  "consumption": [100, 120, 115, 140]
}

📊 Response format:
{
  "risk_level": "HIGH",  ← Use for color coding!
  "anomaly_score": 0.85,
  "explanation": "High anomaly detected"
}

Color suggestion:
- HIGH = Red (🔴)
- MEDIUM = Yellow (🟡)
- LOW = Green (🟢)
```

---

**Good luck with your hackathon! You've got this! 💪**

---

**Questions?** Check:
1. This guide
2. `inference_pipeline.py` self-test
3. `COMPLETE_SYSTEM_SUMMARY.md` in machine_learning/docs/
