# Workflow & Integration Guide

## 🔄 Complete Data Generation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                  │
│  CLI args or GeneratorConfig parameters                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               CONFIGURATION VALIDATION                          │
│  - Anomaly rate in [0, 1]                                       │
│  - Customer classes sum to 1.0                                  │
│  - Output directory created                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            TRANSFORMER GENERATION                               │
│  TransformerGenerator                                           │
│  ├─ Assign to barangays                                        │
│  ├─ Generate clustered coordinates (lat/lon)                   │
│  ├─ Assign capacities (log-normal distribution)                │
│  └─ Output: transformers_df                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              METER ALLOCATION                                   │
│  MeterGenerator._allocate_meters_to_transformers()              │
│  ├─ Allocate proportional to capacity                          │
│  ├─ Ensure min 10 meters per transformer                       │
│  └─ Balance to exact total                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           CONSUMPTION GENERATION                                │
│  MeterGenerator._generate_consumption_series()                  │
│  For each meter:                                                │
│  ├─ Sample customer class (residential/commercial/industrial)  │
│  ├─ Determine if anomaly (random < anomaly_rate)               │
│  └─ Generate monthly consumption:                              │
│      Consumption = base + seasonal + trend + noise              │
│      if anomaly: Consumption *= anomaly_factor                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            ANOMALY LABELING                                     │
│  - Flag anomalous meters                                        │
│  - Assign risk bands (High/Medium/Low)                          │
│  - Output: anomaly_labels_df                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            GEOJSON GENERATION                                   │
│  GeoJSONGenerator.generate()                                    │
│  ├─ Create FeatureCollection                                   │
│  ├─ One feature per transformer                                │
│  └─ Include associated meter IDs                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              VALIDATION                                         │
│  SyntheticDataPipeline._validate_outputs()                      │
│  ✓ Record counts match config                                  │
│  ✓ Anomaly rate within ±2%                                     │
│  ✓ Foreign keys valid (meter → transformer)                    │
│  ✓ No null values                                              │
│  ✓ Consumption columns present                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              FILE PERSISTENCE                                   │
│  SyntheticDataPipeline.save_outputs()                           │
│  ├─ transformers.csv                                           │
│  ├─ meter_consumption.csv                                      │
│  ├─ anomaly_labels.csv                                         │
│  ├─ transformers.geojson                                       │
│  └─ generation_report.txt                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT FILES                                    │
│  Ready for ML pipeline, backend upload, visualization          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Integration Points

### 1. Backend API Integration

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Frontend   │  POST   │   FastAPI    │ Trigger │  Synthetic   │
│   (React)    │ ──────> │   Backend    │ ──────> │  Generator   │
│              │         │              │         │              │
└──────────────┘         └──────────────┘         └──────────────┘
                                │                         │
                                │                         │ generates
                                │                         ▼
                                │                  ┌──────────────┐
                                │                  │  CSV Files   │
                                │                  │  + GeoJSON   │
                                │                  └──────────────┘
                                │                         │
                                │ <───────────────────────┘
                                │ returns metadata
                                ▼
                         ┌──────────────┐
                         │   Supabase   │
                         │   Database   │
                         └──────────────┘
```

**Implementation**:
```python
# backend/api/generate_endpoint.py

from synthetic_data_generator import GeneratorConfig, SyntheticDataPipeline

@app.post("/api/generate-synthetic-data")
async def generate_data(params: GenerationParams):
    config = GeneratorConfig(
        num_transformers=params.num_transformers,
        num_meters=params.num_meters,
        anomaly_rate=params.anomaly_rate
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    
    # Upload to database
    await upload_to_supabase(outputs['meters_df'])
    
    return {
        "status": "success",
        "meters_generated": len(outputs['meters_df'])
    }
```

---

### 2. ML Pipeline Integration

```
┌──────────────────┐         ┌──────────────────┐
│  Synthetic Data  │ feeds   │  Preprocessing   │
│  Generator       │ ──────> │  Module          │
│                  │         │  - Normalize     │
└──────────────────┘         │  - Feature eng   │
                             └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  Isolation       │
                             │  Forest          │
                             │  Training        │
                             └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  Model           │
                             │  Evaluation      │
                             │  - Precision     │
                             │  - Recall        │
                             └──────────────────┘
```

**Implementation**:
```python
# training/train_model.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load generated data
meters_df = pd.read_csv('generated_data/meter_consumption.csv')
anomaly_labels = pd.read_csv('generated_data/anomaly_labels.csv')

# Extract features
consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_')]
X = meters_df[consumption_cols].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = IsolationForest(contamination=0.075, random_state=42)
model.fit(X_scaled)

# Evaluate
predictions = model.predict(X_scaled)
true_labels = meters_df['meter_id'].isin(anomaly_labels['meter_id']).astype(int)
```

---

### 3. Map Visualization Integration

```
┌──────────────────┐         ┌──────────────────┐
│  transformers    │ loads   │  React Leaflet   │
│  .geojson        │ ──────> │  Map Component   │
│                  │         │                  │
└──────────────────┘         └──────────────────┘
                                      │
                                      │ renders
                                      ▼
                             ┌──────────────────┐
                             │  Map with        │
                             │  Transformer     │
                             │  Markers         │
                             │  - Cluster view  │
                             │  - Click details │
                             └──────────────────┘
```

**Implementation**:
```typescript
// frontend/src/components/Map.tsx

import { MapContainer, GeoJSON } from 'react-leaflet';
import transformersData from './transformers.geojson';

const TransformerMap = () => {
  const onFeatureClick = (feature: any) => {
    console.log(`Transformer: ${feature.properties.transformer_id}`);
    console.log(`Meters: ${feature.properties.num_meters}`);
  };
  
  return (
    <MapContainer center={[14.55, 121.05]} zoom={12}>
      <GeoJSON 
        data={transformersData}
        onEachFeature={(feature, layer) => {
          layer.on('click', () => onFeatureClick(feature));
        }}
      />
    </MapContainer>
  );
};
```

---

## 📊 Data Flow Diagram

### Consumption Pattern Generation

```
Customer Class
    │
    ├─ Residential  ──> Base: μ=150,  σ=450  kWh/month
    ├─ Commercial   ──> Base: μ=800,  σ=300  kWh/month
    └─ Industrial   ──> Base: μ=2500, σ=800  kWh/month
          │
          ▼
    ┌─────────────────┐
    │  Base           │
    │  Consumption    │ ← Sampled from Normal(μ, σ)
    └─────────────────┘
          │
          ├──> + Seasonal Component
          │        └─> 15% × sin(2π(month-4)/12)
          │
          ├──> + Trend Component
          │        └─> -2% to +5% monthly growth
          │
          ├──> + Noise Component
          │        └─> Normal(0, 0.3σ)
          │
          └──> × Anomaly Factor (if flagged)
                   └─> 0.3 - (0.2 × month/total)
                       (gradual decline)
          │
          ▼
    ┌─────────────────┐
    │  Final Monthly  │
    │  Consumption    │ ← Clipped to [0, ∞)
    └─────────────────┘
```

---

## 🎯 Hackathon Day Workflow

### Morning (Setup Phase)

```
09:00 - Install dependencies
        └─> pip install numpy pandas scipy

09:15 - Generate demo dataset
        └─> python synthetic_data_generator.py --num-meters 500 --output-dir demo_data

09:30 - Verify outputs
        └─> Check demo_data/ directory
            ├─ transformers.csv
            ├─ meter_consumption.csv
            └─ transformers.geojson

09:45 - Test backend upload
        └─> POST demo_data/meter_consumption.csv to API
```

### Afternoon (Integration Phase)

```
13:00 - Integrate with ML pipeline
        └─> Load CSV into Isolation Forest training

14:00 - Test map visualization
        └─> Load GeoJSON into React map component

15:00 - End-to-end validation
        └─> CSV upload → Anomaly detection → Map display → Drilldown
```

### Evening (Presentation Phase)

```
17:00 - Generate final dataset
        └─> python synthetic_data_generator.py --num-meters 2000 --seed 42

17:30 - Prepare demo flow
        └─> Demo CSV ready for live upload demo

18:00 - Practice presentation
        └─> Show data generation → anomaly detection → visualization
```

---

## 🔧 Troubleshooting Workflow

### Issue: Generation Too Slow

```
Problem: Taking >30 seconds for 2000 meters
    │
    ├─> Check: Are you using --num-months 24?
    │   └─> Solution: Reduce to 12 for faster iteration
    │
    ├─> Check: Is dataset very large (>10K meters)?
    │   └─> Solution: Generate in batches
    │
    └─> Check: Is disk I/O slow?
        └─> Solution: Use SSD or faster storage
```

### Issue: Anomaly Rate Wrong

```
Problem: Getting 5% instead of 7.5%
    │
    ├─> Check: Is num_meters small (<100)?
    │   └─> Solution: Increase to 500+ for statistical stability
    │
    ├─> Check: Is random seed different?
    │   └─> Solution: Use same seed for reproducibility
    │
    └─> Check: Validation tolerance
        └─> Expected range: 7.5% ± 2% = [5.5%, 9.5%]
```

### Issue: Map Markers Not Showing

```
Problem: GeoJSON not rendering
    │
    ├─> Check: Is GeoJSON valid?
    │   └─> Validate at geojson.io
    │
    ├─> Check: Are coordinates in bounds?
    │   └─> Default: Philippines region (14.4-14.7°N, 120.9-121.2°E)
    │
    └─> Check: Is FeatureCollection structure correct?
        └─> Must have type: "FeatureCollection" and features: []
```

---

## 📈 Scaling Workflow

### From Demo (500 meters) to Production (100K meters)

```
Stage 1: Demo Dataset (500 meters)
    └─> Generation time: <1 second
        Use for: Rapid iteration, testing

Stage 2: MVP Dataset (2,000 meters)
    └─> Generation time: 3 seconds
        Use for: Hackathon demo, initial ML training

Stage 3: Full Dataset (10,000 meters)
    └─> Generation time: 25 seconds
        Use for: Comprehensive ML validation

Stage 4: Production Dataset (100,000 meters)
    └─> Generation time: ~5 minutes (estimated)
        Use for: Production-scale testing
        Note: Consider batch processing

Stage 5: Enterprise Dataset (1,000,000 meters)
    └─> Generation time: ~50 minutes (estimated)
        Use for: Large utility simulation
        Note: Require distributed generation (Spark/Dask)
```

---

## 🚀 Deployment Workflow

### Local Development → Cloud Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│  Local Development                                              │
│  - Generate test datasets                                       │
│  - Validate with unit tests                                     │
│  - Integrate with local backend                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Docker Containerization                                        │
│  - Package generator + dependencies                             │
│  - Create Dockerfile                                            │
│  - Test container locally                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cloud Deployment (AWS/Azure/GCP)                               │
│  - Deploy as Lambda/Cloud Function                              │
│  - Trigger via API Gateway                                      │
│  - Store outputs in S3/Blob/GCS                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Summary

This workflow guide provides:

✅ **Complete data generation pipeline** from config to output  
✅ **Integration patterns** for backend, ML, and frontend  
✅ **Hackathon day schedule** with time-boxed activities  
✅ **Troubleshooting decision trees** for common issues  
✅ **Scaling strategy** from demo to production datasets  
✅ **Deployment pathway** from local to cloud  

Use this guide as a reference during development, hackathon day, and future scaling activities.

---

**Version**: 1.0.0  
**Last Updated**: November 13, 2025
