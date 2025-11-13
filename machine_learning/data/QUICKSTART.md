# Quick Start Guide: Synthetic Data Generator

## 5-Minute Setup

### Step 1: Install Dependencies

```powershell
# Navigate to the data directory
cd machine_learning\data

# Install required packages
pip install numpy pandas scipy
```

### Step 2: Generate Your First Dataset

```powershell
# Run with default settings
python synthetic_data_generator.py

# Check the output
dir generated_data
```

**Expected Output**:
```
generated_data/
├── transformers.csv           (50 transformers)
├── meter_consumption.csv      (2000 meters × 12 months)
├── anomaly_labels.csv         (150 anomalies)
├── transformers.geojson       (Map visualization)
└── generation_report.txt      (Summary statistics)
```

### Step 3: Verify Results

```powershell
# View summary report
type generated_data\generation_report.txt
```

You should see:
```
Synthetic Data Generation Report
================================================================================

Generated: 2025-11-13 14:23:04
Random Seed: 42

Dataset Statistics:
--------------------------------------------------------------------------------
Transformers: 50
Meters: 2000
Anomalies: 152
Anomaly Rate: 7.60%

Customer Class Distribution:
--------------------------------------------------------------------------------
  residential: 1400 (70.0%)
  commercial: 400 (20.0%)
  industrial: 200 (10.0%)
...
```

## Common Use Cases

### Use Case 1: Generate Demo Data for Presentation

```powershell
# Small, fast dataset for demos
python synthetic_data_generator.py --num-transformers 10 --num-meters 200 --num-months 6 --output-dir demo_data
```

**Result**: <1 second generation, ~500 KB output

### Use Case 2: Generate Full Training Dataset

```powershell
# Larger dataset for ML training
python synthetic_data_generator.py --num-transformers 100 --num-meters 5000 --num-months 24 --output-dir training_data
```

**Result**: ~8 seconds generation, ~30 MB output

### Use Case 3: Generate Multiple Scenarios

```powershell
# Low anomaly scenario
python synthetic_data_generator.py --anomaly-rate 0.05 --seed 100 --output-dir low_anomaly

# High anomaly scenario
python synthetic_data_generator.py --anomaly-rate 0.15 --seed 200 --output-dir high_anomaly

# Normal scenario
python synthetic_data_generator.py --anomaly-rate 0.075 --seed 300 --output-dir normal
```

**Result**: Three datasets with different anomaly distributions

## Python Script Usage

### Basic Generation

```python
from pathlib import Path
from synthetic_data_generator import GeneratorConfig, SyntheticDataPipeline

# Generate with default config
pipeline = SyntheticDataPipeline()
outputs = pipeline.generate_all()
pipeline.save_outputs(outputs)

print(f"Generated {len(outputs['meters_df'])} meters")
```

### Custom Configuration

```python
# Create custom config
config = GeneratorConfig(
    random_seed=42,
    num_transformers=30,
    num_meters=1000,
    num_months=12,
    anomaly_rate=0.08,
    output_dir=Path("custom_data"),
    customer_classes={
        'residential': 0.60,
        'commercial': 0.30,
        'industrial': 0.10
    }
)

# Generate
pipeline = SyntheticDataPipeline(config)
outputs = pipeline.generate_all()
pipeline.save_outputs(outputs)
```

### Access Data Directly

```python
# Generate without saving
pipeline = SyntheticDataPipeline()
outputs = pipeline.generate_all()

# Access DataFrames
transformers_df = outputs['transformers_df']
meters_df = outputs['meters_df']
anomaly_labels_df = outputs['anomaly_labels_df']
geojson = outputs['geojson']

# Do analysis
print(transformers_df.head())
print(f"Anomaly rate: {len(anomaly_labels_df) / len(meters_df) * 100:.2f}%")
```

## Integration Examples

### Load Generated Data for ML

```python
import pandas as pd
import numpy as np

# Load meter consumption
meters_df = pd.read_csv('generated_data/meter_consumption.csv')

# Extract features
consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_')]
X = meters_df[consumption_cols].values

# Compute statistics
medians = np.median(X, axis=1)
variances = np.var(X, axis=1)

print(f"Feature matrix shape: {X.shape}")
print(f"Mean consumption: {X.mean():.2f} kWh")
```

### Upload to Backend API

```python
import requests

# Generate data
pipeline = SyntheticDataPipeline()
outputs = pipeline.generate_all()

# Convert to JSON for upload
meters_json = outputs['meters_df'].to_dict(orient='records')

# Upload to API
response = requests.post(
    'http://localhost:8000/upload',
    json={'meters': meters_json}
)

print(f"Upload status: {response.status_code}")
```

### Visualize on Map

```python
import json

# Load GeoJSON
with open('generated_data/transformers.geojson', 'r') as f:
    geojson = json.load(f)

# Extract coordinates for plotting
coordinates = [
    feature['geometry']['coordinates']
    for feature in geojson['features']
]

print(f"Found {len(coordinates)} transformer locations")
```

## Troubleshooting

### Problem: ModuleNotFoundError: No module named 'numpy'

**Solution**:
```powershell
pip install numpy pandas scipy
```

### Problem: "customer_classes probabilities must sum to 1.0"

**Solution**: Ensure probabilities sum exactly to 1.0:
```python
config = GeneratorConfig(
    customer_classes={
        'residential': 0.70,
        'commercial': 0.20,
        'industrial': 0.10  # Sum = 1.0
    }
)
```

### Problem: Permission denied when saving files

**Solution**: Run with elevated privileges or choose different output directory:
```python
config = GeneratorConfig(
    output_dir=Path("C:/Users/YourName/Documents/generated_data")
)
```

### Problem: Generation is slow

**Solution**: Reduce dataset size for faster iteration:
```powershell
python synthetic_data_generator.py --num-transformers 10 --num-meters 200 --num-months 6
```

## Next Steps

1. **Run Examples**: `python examples.py` to see all usage patterns
2. **Run Tests**: `pytest test_generator.py -v` to verify installation
3. **Integrate with ML**: Load data into your training pipeline
4. **Customize**: Modify `GeneratorConfig` for your specific needs

## Reference

- **Full Documentation**: See `README.md`
- **Design Details**: See `DESIGN.md`
- **Examples**: See `examples.py`
- **Tests**: See `test_generator.py`

## CLI Options Reference

```
--num-transformers INT    Number of transformers (default: 50)
--num-meters INT          Number of meters (default: 2000)
--num-months INT          Months of history (default: 12)
--anomaly-rate FLOAT      Anomaly rate 0.0-1.0 (default: 0.075)
--seed INT                Random seed (default: 42)
--output-dir PATH         Output directory (default: generated_data)
```

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `DESIGN.md` for architecture details
3. Run `python examples.py` to see working examples
4. Contact the ML team

---

**Quick Start Version**: 1.0.0  
**Last Updated**: 2025-11-13
