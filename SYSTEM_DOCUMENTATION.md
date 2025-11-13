# 🎯 GhostLoad Mapper — Complete System Documentation

## System Overview

**GhostLoad Mapper** is an AI-powered anomaly detection system that identifies low-consumption meters (potential electricity theft) by combining spatial clustering (transformer grouping) with behavioral outlier detection.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + TypeScript)                │
│  - Interactive Leaflet map (NCR → City → Barangay → Transformer)│
│  - High-Risk Locations sidebar                                  │
│  - Transformer analytics modal                                  │
│  - CSV export for field inspections                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP REST API
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI + Python)                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ POST /api/run                                                ││
│  │  1. Receive CSV upload                                       ││
│  │  2. Validate columns                                         ││
│  │  3. Pass to ML pipeline                                      ││
│  │  4. Aggregate results (City→Barangay→Transformer→Meter)      ││
│  │  5. Save to Firestore                                        ││
│  │  6. Return run_id                                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ GET /api/results/{run_id}                                    ││
│  │  - Fetch hierarchical data from Firestore                    ││
│  │  - Return cities, barangays, transformers, meters            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ GET /api/transformers.geojson                                ││
│  │  - Convert transformer data to GeoJSON                       ││
│  │  - Return for Leaflet map rendering                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ GET /api/export/{run_id}?level=transformer                   ││
│  │  - Generate CSV report for field teams                       ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────┬────────────────────────────────────────┘
                         │ Import & Call
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING PIPELINE (scikit-learn)            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ InferencePipeline.predict(dataframe)                         ││
│  │                                                               ││
│  │  1. Feature Engineering (FeatureEngineer)                    ││
│  │     - Transformer baseline computation (median/MAD)          ││
│  │     - Consumption trend analysis (Theil-Sen)                 ││
│  │     - Relative consumption ratios                            ││
│  │     - Statistical features (23 features total)               ││
│  │                                                               ││
│  │  2. Anomaly Detection (Isolation Forest)                     ││
│  │     - Trained model from output/latest/trained_model.pkl     ││
│  │     - Anomaly score (0-1)                                    ││
│  │                                                               ││
│  │  3. Risk Classification                                      ││
│  │     - HIGH: score ≥ 0.7                                      ││
│  │     - MEDIUM: 0.4 ≤ score < 0.7                              ││
│  │     - LOW: score < 0.4                                       ││
│  │                                                               ││
│  │  4. Return Predictions                                       ││
│  │     - List of RiskPrediction objects                         ││
│  │     - meter_id, risk_level, anomaly_score, confidence,       ││
│  │       explanation                                            ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FIRESTORE (Firebase Database)                   │
│  runs/                                                            │
│    └─ {run_id}/                                                  │
│        ├─ timestamp: 2025-11-13T19:02:23Z                        │
│        ├─ status: "completed"                                    │
│        ├─ total_meters: 1500                                     │
│        ├─ total_transformers: 120                                │
│        ├─ high_risk_count: 75                                    │
│        │                                                          │
│        ├─ cities: [                                              │
│        │     {city_id, city_name, lat, lon, total_transformers,  │
│        │      high_risk_count}                                   │
│        │   ]                                                      │
│        │                                                          │
│        ├─ barangays: [                                           │
│        │     {barangay, city_id, lat, lon, total_transformers,   │
│        │      high_risk_count}                                   │
│        │   ]                                                      │
│        │                                                          │
│        ├─ transformers: [                                        │
│        │     {transformer_id, barangay, lat, lon, total_meters,  │
│        │      suspicious_meter_count, risk_level, avg_consumption}│
│        │   ]                                                      │
│        │                                                          │
│        ├─ meters: [                                              │
│        │     {meter_id, transformer_id, barangay, lat, lon,      │
│        │      risk_level, anomaly_score, confidence, explanation}│
│        │   ]                                                      │
│        │                                                          │
│        └─ high_risk_summary: {                                   │
│              most_anomalous_city, most_anomalous_barangay,       │
│              most_anomalous_transformer, top_10_transformers     │
│            }                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 User Flow

### 1. Initial View: NCR Map

- User sees Manila Metro area with city markers
- Each city shows:
  - Total transformers
  - Number of high-risk transformers

### 2. Zoom to City

- User clicks city marker (e.g., "Manila")
- Map zooms to city bounds
- Barangay markers appear
- Each barangay shows:
  - Total transformers in that barangay
  - Number of high-risk transformers

### 3. Zoom to Barangay

- User clicks barangay marker (e.g., "Tondo")
- Map zooms to barangay bounds
- Transformer markers appear with color coding:
  - 🔴 Red: HIGH risk
  - 🟠 Orange: MEDIUM risk
  - 🟢 Green: LOW risk
- Heat zones (orange clusters) show concentration of anomalies

### 4. View Transformer Details

- User clicks transformer marker
- Modal popup shows:
  - Transformer ID
  - Risk level and score
  - Total meters connected
  - Number of suspicious meters
  - Average consumption statistics
  - **Meter-level visualization**: Individual meters as small dots around transformer
  - **Consumption sparkline** (last 12 months)

### 5. Sidebar: "High-Risk Locations"

- Always visible on right side
- Lists top 10 most anomalous transformers
- Each entry shows:
  - Transformer ID
  - Barangay name
  - Risk badge (HIGH/MEDIUM/LOW)
  - Number of suspicious meters
- Click to zoom directly to that transformer

### 6. Export Report

- Click "CSV Export" button
- Choose level:
  - **Transformer report**: One row per transformer with aggregated stats
  - **District report**: One row per barangay with summary
  - **Meter report**: One row per meter with individual predictions
- Download CSV for field inspection teams

---

## 📊 Data Flow

### Step 1: CSV Upload

```
User uploads: meter_consumption.csv
Contains:
  - meter_id, transformer_id, customer_class, barangay, lat, lon, kVA
  - monthly_consumption_202411, monthly_consumption_202412, ...
  - (12 months of consumption data)
```

### Step 2: ML Pipeline Processing

```
Frontend → POST /api/run → Backend

Backend:
  1. Read CSV (pandas)
  2. Validate columns
  3. Call ML pipeline:
     pipeline.predict(dataframe)

ML Pipeline:
  1. Feature Engineering
     - Compute transformer baselines (median consumption per transformer)
     - Calculate consumption trends (Theil-Sen regression)
     - Compute ratios (meter consumption / transformer median)
     - Generate 23 statistical features

  2. Anomaly Detection
     - Load trained Isolation Forest model
     - Predict anomaly scores (0-1)

  3. Risk Classification
     - HIGH: score ≥ 0.7 (⚠️ Prioritize for inspection)
     - MEDIUM: 0.4 ≤ score < 0.7 (⚡ Monitor closely)
     - LOW: score < 0.4 (✅ Normal pattern)

  4. Return predictions
     - List of RiskPrediction objects

Backend receives predictions:
  [{
    meter_id: "MTR_000001",
    risk_level: "HIGH",
    anomaly_score: 0.92,
    confidence: 0.85,
    explanation: "⚠️ High anomaly detected - Prioritize for inspection"
  }, ...]
```

### Step 3: Data Aggregation

```
Backend aggregates predictions hierarchically:

Meter Level (raw predictions):
  - All 1,500 meters with individual scores

Transformer Level:
  - Group by transformer_id
  - Calculate:
    * total_meters
    * suspicious_meter_count (risk_level == HIGH)
    * avg_anomaly_score
    * avg_consumption, median_consumption
    * risk_level (based on avg_anomaly_score)

Barangay Level:
  - Group by barangay
  - Calculate:
    * total_transformers
    * high_risk_count (risk_level == HIGH transformers)
    * avg_risk_score
    * centroid coordinates (mean lat/lon)

City Level:
  - Group by city_id
  - Calculate:
    * total_transformers (sum from all barangays)
    * high_risk_count (sum from all barangays)
    * avg_risk_score
    * centroid coordinates

High-Risk Summary:
  - Top 10 transformers (sorted by anomaly score)
  - Most anomalous city/barangay/transformer
  - Total high-risk counts
```

### Step 4: Save to Firestore

```
Backend saves to Firestore:

Document ID: {run_id}
Path: runs/{run_id}

Data:
  {
    run_id: "abc-123-def",
    timestamp: "2025-11-13T19:02:23Z",
    status: "completed",
    total_meters: 1500,
    total_transformers: 120,
    high_risk_count: 75,

    cities: [...],
    barangays: [...],
    transformers: [...],
    meters: [...],
    high_risk_summary: {...}
  }

Backend returns:
  {
    run_id: "abc-123-def",
    status: "completed",
    processing_time_seconds: 5.2
  }
```

### Step 5: Frontend Display

```
Frontend receives run_id → calls GET /api/results/{run_id}

Backend fetches from Firestore → returns complete data

Frontend renders:
  1. City markers on initial map
  2. Barangay markers when city clicked
  3. Transformer markers when barangay clicked
  4. Meter dots when transformer clicked
  5. "High-Risk Locations" sidebar
  6. Consumption charts in modal
```

---

## 🗂️ Dataset Structure

### Input CSV Requirements

**meter_consumption.csv**

```csv
meter_id,transformer_id,customer_class,barangay,lat,lon,kVA,monthly_consumption_202411,monthly_consumption_202412,...
MTR_000001,TX_MAIN_001,residential,Tondo,14.601,120.961,7.91,322.07,342.21,...
MTR_000002,TX_MAIN_001,residential,Tondo,14.598,120.974,8.66,275.12,297.79,...
```

**Required Columns:**

- `meter_id` — Unique meter identifier
- `transformer_id` — Transformer this meter belongs to
- `customer_class` — residential, commercial, or industrial
- `barangay` — Barangay name
- `lat`, `lon` — GPS coordinates (decimal degrees)
- `kVA` — Meter capacity
- `monthly_consumption_YYYYMM` — **At least 6 months** (12 recommended)

**Optional (but recommended):**

- `city_id` — City identifier (manila, quezon_city, etc.)
- `feeder_id` — Electrical feeder ID

---

## 🎯 ML Pipeline Details

### Feature Engineering (23 features)

The ML pipeline creates 23 features from raw consumption data:

**1. Transformer Baseline Features (4)**

- `transformer_median` — Median consumption of all meters on transformer
- `transformer_mad` — Median Absolute Deviation (robust variance)
- `consumption_ratio` — meter_consumption / transformer_median
- `deviation_from_baseline` — (meter - median) / MAD

**2. Temporal Trend Features (6)**

- `consumption_trend` — Theil-Sen regression slope (robust to outliers)
- `trend_significance` — Is trend statistically significant?
- `trend_direction` — Increasing, decreasing, or stable
- `recent_drop` — Drop in last 3 months?
- `seasonal_variance` — Variance across months
- `cv` — Coefficient of variation

**3. Statistical Features (8)**

- `mean_consumption` — Average over all months
- `median_consumption` — Median over all months
- `std_consumption` — Standard deviation
- `min_consumption`, `max_consumption` — Range
- `q25_consumption`, `q75_consumption` — Quartiles
- `iqr_consumption` — Interquartile range

**4. Relative Features (5)**

- `percentile_in_transformer` — Where this meter ranks in its transformer
- `z_score` — Standardized score relative to transformer
- `distance_from_median` — Absolute difference from transformer median
- `is_low_consumer` — Boolean: consumption_ratio < 0.5
- `is_high_consumer` — Boolean: consumption_ratio > 1.5

### Anomaly Detection Model

**Model:** Isolation Forest (scikit-learn)

**Why Isolation Forest?**

- **Unsupervised:** No labeled data needed
- **Fast:** Efficient for large datasets
- **Robust:** Handles outliers and noise well
- **Explainable:** Anomaly score directly interpretable

**Hyperparameters:**

- `contamination=0.05` — Assume 5% of meters are anomalous
- `n_estimators=100` — 100 decision trees
- `random_state=42` — Reproducible results

**Training:**

```python
# Train on historical data
from machine_learning.pipeline.training_pipeline import train_model
train_model(
    input_csv="datasets/demo/meter_consumption.csv",
    output_dir="output/run_20251113"
)
```

**Inference:**

```python
# Predict on new data
from machine_learning.pipeline.inference_pipeline import InferencePipeline
pipeline = InferencePipeline()
predictions = pipeline.predict(new_dataframe)
```

---

## 🎨 Frontend Components

### 1. Map Component (Leaflet)

```typescript
import L from "leaflet";
import "react-leaflet";

const map = L.map("map").setView([14.5995, 120.9842], 11);

// Add base layer
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);

// Add city markers
data.cities.forEach((city) => {
  L.marker([city.lat, city.lon])
    .bindPopup(`<b>${city.city_name}</b>`)
    .on("click", () => zoomToCity(city))
    .addTo(map);
});
```

### 2. High-Risk Sidebar

```typescript
<aside className="sidebar">
  <h2>High-Risk Locations</h2>
  {data.high_risk_summary.top_10_transformers.map((t) => (
    <div key={t.transformer_id} className="risk-item">
      <span className={`badge ${t.risk_level}`}>{t.risk_level}</span>
      <strong>{t.transformer_id}</strong>
      <small>{t.barangay}</small>
      <span>{t.suspicious_meter_count} suspicious meters</span>
    </div>
  ))}
</aside>
```

### 3. Transformer Modal

```typescript
<Modal show={showModal} onHide={closeModal}>
  <Modal.Header>
    <h3>{selectedTransformer.transformer_id}</h3>
    <span className={`badge ${selectedTransformer.risk_level}`}>
      {selectedTransformer.risk_level}
    </span>
  </Modal.Header>
  <Modal.Body>
    <p>Barangay: {selectedTransformer.barangay}</p>
    <p>Total Meters: {selectedTransformer.total_meters}</p>
    <p>Suspicious: {selectedTransformer.suspicious_meter_count}</p>

    {/* Consumption chart */}
    <Chart data={transformerConsumptionData} />

    {/* Meter list */}
    <table>
      {transformerMeters.map((meter) => (
        <tr key={meter.meter_id}>
          <td>{meter.meter_id}</td>
          <td className={`badge ${meter.risk_level}`}>{meter.risk_level}</td>
          <td>{meter.anomaly_score.toFixed(2)}</td>
        </tr>
      ))}
    </table>
  </Modal.Body>
</Modal>
```

### 4. Export Button

```typescript
<button onClick={() => downloadReport('transformer')}>
  <Icon name="download" />
  Export Transformer Report
</button>

<button onClick={() => downloadReport('meter')}>
  <Icon name="download" />
  Export Meter Report
</button>
```

---

## 🚀 Deployment

### Backend (Railway / Render)

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment Variables:**

```
FIREBASE_PROJECT_ID=ghost-load-mapper
FIREBASE_CREDENTIALS_PATH=./credentials.json
```

### Frontend (Firebase Hosting / Netlify)

```bash
# Build React app
npm run build

# Deploy to Firebase
firebase deploy --only hosting

# Or deploy to Netlify
netlify deploy --prod --dir=build
```

---

## 📈 Performance Metrics

| Dataset Size  | Processing Time | Memory Usage |
| ------------- | --------------- | ------------ |
| 100 meters    | ~1 second       | ~50 MB       |
| 1,000 meters  | ~5 seconds      | ~200 MB      |
| 10,000 meters | ~30 seconds     | ~1 GB        |

**Optimization Tips:**

- Use batch processing for large datasets
- Cache ML model in memory
- Use Firestore batch writes
- Implement pagination for frontend

---

## 🎓 Key Technologies

| Component | Technology         | Purpose             |
| --------- | ------------------ | ------------------- |
| Frontend  | React + TypeScript | UI framework        |
| Map       | Leaflet            | Interactive mapping |
| Backend   | FastAPI + Python   | REST API server     |
| ML        | scikit-learn       | Anomaly detection   |
| Database  | Firestore          | NoSQL data storage  |
| Features  | pandas + NumPy     | Data processing     |

---

## ✅ Testing Checklist

### Backend

- [ ] Firebase connection test (`python test_firebase.py`)
- [ ] ML pipeline loads successfully
- [ ] CSV upload accepts valid files
- [ ] Predictions returned correctly
- [ ] Data aggregation works
- [ ] Firestore saves complete
- [ ] GeoJSON generation works
- [ ] CSV export downloads
- [ ] Health check returns healthy

### Frontend

- [ ] Map loads with city markers
- [ ] City click zooms to barangays
- [ ] Barangay click shows transformers
- [ ] Transformer colors match risk
- [ ] Modal displays details
- [ ] Sidebar shows top 10
- [ ] Export buttons download CSV
- [ ] Error handling works

### Integration

- [ ] End-to-end flow works
- [ ] Data matches between backend and frontend
- [ ] Performance acceptable (<10s for 1k meters)
- [ ] Multiple concurrent uploads work
- [ ] Firestore data structure correct

---

**System Status:** ✅ Ready for Hackathon!

**Last Updated:** November 13, 2025
