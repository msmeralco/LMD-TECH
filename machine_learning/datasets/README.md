# Generated Datasets for GhostLoad Mapper ML Development

**Generated**: 2025-11-13 10:26:39
**Generation Time**: 2.03 seconds

## Datasets

### Demo
- **Location**: `../datasets/demo`
- **Description**: 200 meters, 6 months
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### Development
- **Location**: `../datasets/development`
- **Description**: 1,000 meters, 12 months
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### Validation
- **Location**: `../datasets/validation`
- **Description**: 1,000 meters, 12 months, 10% anomalies
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### Production
- **Location**: `../datasets/production`
- **Description**: 2,000 meters, 12 months
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### Low Anomaly
- **Location**: `../datasets/scenarios/low_anomaly`
- **Description**: 500 meters, 5% anomalies
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### High Anomaly
- **Location**: `../datasets/scenarios/high_anomaly`
- **Description**: 500 meters, 15% anomalies
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

### Large Scale
- **Location**: `../datasets/scenarios/large_scale`
- **Description**: 5,000 meters
- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson

## Usage

```python
import pandas as pd

# Load demo dataset
meters = pd.read_csv('demo/meter_consumption.csv')
anomalies = pd.read_csv('demo/anomaly_labels.csv')
```
