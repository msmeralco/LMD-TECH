# 📊 Dataset Selection Guide for Hackathon

## Quick Reference

### **RECOMMENDED: `development`** ⭐
```python
dataset_dir = machine_learning_dir / "datasets" / "development"
```
- **Size**: 1,000 meters, 30 transformers
- **Anomalies**: 76 theft cases (7.6% rate)
- **Risk Distribution**: High: 41, Medium: 4, Low: 31
- **Training Time**: ~3 minutes
- **Best For**: 1-day hackathon - balanced, realistic, convincing

### Alternative Options:

#### For Dramatic Demo: `scenarios/high_anomaly` 🎯
```python
dataset_dir = machine_learning_dir / "datasets" / "scenarios" / "high_anomaly"
```
- **Size**: 500 meters, 20 transformers
- **Anomalies**: 74 theft cases (14.8% rate!)
- **Risk Distribution**: High: 55, Medium: 1, Low: 18
- **Training Time**: ~2 minutes
- **Best For**: Impressing judges with detection rate

#### For Quick Testing: `demo` ⚡
```python
dataset_dir = machine_learning_dir / "datasets" / "demo"
```
- **Size**: 200 meters, 10 transformers
- **Anomalies**: 19 cases (9.5% rate)
- **Training Time**: <1 minute
- **Best For**: Testing changes quickly

#### For Realistic Scenario: `production` 🏭
```python
dataset_dir = machine_learning_dir / "datasets" / "production"
```
- **Size**: 2,000 meters, 50 transformers
- **Anomalies**: 151 cases (7.6% rate)
- **Training Time**: ~5 minutes
- **Best For**: Showing scalability (but slower)

## How to Change Dataset

Edit `machine_learning/train.py` line ~58:

```python
# Change this line:
dataset_dir = machine_learning_dir / "datasets" / "development"

# To one of:
dataset_dir = machine_learning_dir / "datasets" / "demo"
dataset_dir = machine_learning_dir / "datasets" / "development"  # RECOMMENDED
dataset_dir = machine_learning_dir / "datasets" / "production"
dataset_dir = machine_learning_dir / "datasets" / "validation"
dataset_dir = machine_learning_dir / "datasets" / "scenarios" / "high_anomaly"
dataset_dir = machine_learning_dir / "datasets" / "scenarios" / "low_anomaly"
dataset_dir = machine_learning_dir / "datasets" / "scenarios" / "large_scale"
```

## Recommendation for 1-Day Hackathon

**Use `development`** because:
1. ✅ Enough data to be convincing (1,000 meters)
2. ✅ Realistic theft rate (7.6% matches real-world scenarios)
3. ✅ Fast enough for demos (<3 min training)
4. ✅ Good variety of risk levels to show ML capabilities
5. ✅ Not too large - leaves time for frontend/backend integration

**Switch to `scenarios/high_anomaly`** if:
- Judges want to see dramatic results
- You need to show "lots of detections"
- Time is very limited (faster training)
