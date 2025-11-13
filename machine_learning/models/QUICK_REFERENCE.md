# Base Model Quick Reference Card

## 🚀 Quick Start (30 seconds)

```python
from base_model import BaseAnomalyDetector, ModelConfig
import numpy as np

# 1. Create custom detector
class MyDetector(BaseAnomalyDetector):
    def _fit_implementation(self, X, y=None):
        self._mean = X.mean(axis=0)
    
    def _predict_implementation(self, X):
        return np.linalg.norm(X - self._mean, axis=1)

# 2. Use it
detector = MyDetector()
detector.fit(X_train)
scores = detector.predict(X_test)
```

---

## 📋 Essential API

### Training
```python
detector.fit(X)                    # Train on data
detector.fit(X, feature_names=...)  # With feature names
detector.fit(df[cols])             # Auto-extract names from DataFrame
```

### Prediction
```python
scores = detector.predict(X)                               # Anomaly scores
result = detector.predict(X, return_probabilities=True)   # Full result
labels = detector.predict_binary(X)                       # Binary labels
```

### Persistence
```python
detector.save('model.pkl')            # Save model
loaded = MyDetector.load('model.pkl')  # Load model
```

---

## 🎛️ Configuration

```python
config = ModelConfig(
    contamination=0.1,      # Expected anomaly rate
    normalize_scores=True,  # Scale to [0, 1]
    random_state=42,       # Reproducibility
    verbose=True           # Enable logging
)
detector = MyDetector(config=config)
```

---

## 📦 Output Formats

### Simple Scores
```python
scores = detector.predict(X)
# NDArray[np.float64] shape (n_samples,)
# Higher = more anomalous
```

### Full Result
```python
result = detector.predict(X, return_probabilities=True)
# PredictionResult with:
#   - anomaly_scores: continuous scores
#   - predictions: binary labels (0/1)
#   - probabilities: sigmoid probabilities
#   - metadata: timing, statistics
```

---

## ⚡ Common Patterns

### Train-Test Split
```python
detector.fit(X_train)
scores = detector.predict(X_test)
```

### Fit-Predict
```python
scores = detector.fit_predict(X)  # Train and predict in one call
```

### Custom Threshold
```python
labels = detector.predict_binary(X, threshold=0.8)
```

### With DataFrame
```python
detector.fit(df[feature_cols])
print(detector.feature_names)  # Auto-tracked
```

---

## 🔧 Creating Custom Detectors

### Minimal Template
```python
class MyDetector(BaseAnomalyDetector):
    def _fit_implementation(self, X, y=None):
        # Train model (X is validated)
        pass
    
    def _predict_implementation(self, X):
        # Return 1D scores (n_samples,)
        return scores
```

### With Hyperparameters
```python
class MyDetector(BaseAnomalyDetector):
    def __init__(self, config=None, my_param=1.0):
        super().__init__(config)
        self.my_param = my_param
    
    def _fit_implementation(self, X, y=None):
        # Use self.my_param
        pass
    
    def _predict_implementation(self, X):
        return scores
```

---

## ✅ Validation Checklist

### Input Validation (Automatic)
- ✅ 2D array required
- ✅ No NaN/Inf values
- ✅ Feature count consistency
- ✅ Type conversion (DataFrame → array)

### Output Validation (Your Responsibility)
- ✅ Return 1D array from `_predict_implementation`
- ✅ Shape must be (n_samples,)
- ✅ No NaN/Inf in scores

---

## 🎯 Properties & Methods

### Properties
```python
detector.is_fitted           # bool: Model trained?
detector.n_features_in       # int: Feature count
detector.feature_names       # List[str]: Feature names
detector.threshold           # float: Decision threshold
```

### Methods
```python
detector.fit(X)              # Train
detector.predict(X)          # Predict scores
detector.fit_predict(X)      # Train + predict
detector.predict_binary(X)   # Binary labels
detector.save(path)          # Save model
detector.load(path)          # Load model [classmethod]
detector.get_model_info()    # Get metadata
```

---

## 🐛 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Model must be fitted" | Predict before fit | Call `fit()` first |
| "Feature count mismatch" | Wrong # features | Use same features |
| "X contains NaN" | Missing values | Preprocess data |
| "must return 1D array" | Wrong output shape | Return `scores.flatten()` |

---

## 📊 Metadata Access

```python
info = detector.get_model_info()
print(info['is_fitted'])              # bool
print(info['n_features'])             # int
print(info['threshold'])              # float
print(info['metadata']['training_samples'])  # int
print(info['metadata']['training_time_seconds'])  # float
```

---

## 🎓 Key Concepts

### Contamination
```python
contamination = 0.1  # Expect 10% anomalies
# → Threshold at 90th percentile
# → Top 10% scores classified as anomalies
```

### Normalization
```python
normalize_scores = True   # Scores in [0, 1]
normalize_scores = False  # Raw scores (algorithm-dependent)
```

### Threshold Computation
```python
# Auto-computed during fit():
threshold = np.percentile(train_scores, (1 - contamination) * 100)
```

---

## 📁 File Structure

```
models/
├── base_model.py                    # Core implementation (1,306 LOC)
├── BASE_MODEL_README.md             # User guide (850+ LOC)
├── BASE_MODEL_DELIVERY_SUMMARY.md   # Technical details (600+ LOC)
├── custom_detector_example.py       # Working example (350+ LOC)
├── PACKAGE_README.md                # Package overview
└── QUICK_REFERENCE.md               # This file
```

---

## 🔗 Resources

- **Full Documentation**: `BASE_MODEL_README.md`
- **Technical Details**: `BASE_MODEL_DELIVERY_SUMMARY.md`
- **Working Example**: `custom_detector_example.py`
- **Self-Test**: Run `python base_model.py`

---

## 💡 Tips

1. **Always set random_state** for reproducibility
2. **Use feature names** (DataFrame) for trackability
3. **Save metadata** alongside model (`include_metadata=True`)
4. **Monitor threshold** - may need tuning based on contamination
5. **Validate output shape** - must be (n_samples,)

---

## 🎯 Next Steps

1. Review: `BASE_MODEL_README.md`
2. Run: `python base_model.py` (self-test)
3. Study: `custom_detector_example.py`
4. Create: Your own detector
5. Integrate: With feature_engineer output

---

**Version**: 1.0.0  
**Updated**: November 13, 2025  
**Status**: ✅ Production Ready
