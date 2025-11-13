# GhostLoad Mapper - Backend Documentation

Complete FastAPI backend for transformer-aware anomaly detection system.

## 🏗️ Architecture Overview

```
Frontend (React + Leaflet)
    ↓ HTTP Requests
Backend (FastAPI)
    ↓ DataFrame processing
ML Pipeline (Isolation Forest)
    ↓ Predictions
Backend (Data Aggregation)
    ↓ Save results
Firestore (Database)
    ↑ Fetch data
Frontend (Map Visualization)
```

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization
│   ├── config.py            # Environment settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # All API endpoints
│   └── db/
│       ├── __init__.py
│       └── firestore.py     # Firestore helpers
│
├── main.py                  # Server runner (python main.py)
├── requirements.txt         # Dependencies
├── .env                     # Environment variables
├── .gitignore               # Git ignore rules
│
├── test_firebase.py         # Test Firebase connection
├── test_integration.py      # Test complete workflow
│
├── meter_consumption.csv    # Sample meter data
└── transformers.csv         # Sample transformer data
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Firebase

1. Download Firebase credentials JSON from Firebase Console
2. Place in `backend/` folder
3. Update `.env`:

```bash
DEBUG=True
FIREBASE_PROJECT_ID=ghost-load-mapper
FIREBASE_CREDENTIALS_PATH=./ghost-load-mapper-firebase-adminsdk-*.json
```

### 3. Test Firebase Connection

```bash
python test_firebase.py
```

Expected output:

```
✅ Firebase connected successfully!
Data: {'message': 'Firebase connected!', 'timestamp': ...}
```

### 4. Start Server

```bash
python main.py
```

Server starts on: **http://localhost:8000**

### 5. Test Complete Workflow

```bash
# In another terminal
python test_integration.py
```

---

## 📡 API Endpoints

### 1. POST /api/run

**Purpose:** Upload CSV and run complete analysis pipeline

**Request:**

```bash
curl -X POST "http://localhost:8000/api/run" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@meter_consumption.csv"
```

**Response:**

```json
{
  "run_id": "abc-123-def-456",
  "status": "completed",
  "total_meters": 52,
  "total_transformers": 5,
  "total_barangays": 5,
  "total_cities": 1,
  "high_risk_count": 8,
  "processing_time_seconds": 3.42
}
```

**CSV Requirements:**

- `meter_id`, `transformer_id`, `customer_class`, `barangay`, `lat`, `lon`, `kVA`
- `monthly_consumption_YYYYMM` (at least 6 months)

---

### 2. GET /api/results/{run_id}

**Purpose:** Fetch complete analysis results

**Request:**

```bash
curl "http://localhost:8000/api/results/abc-123-def-456"
```

**Response:**

```json
{
  "run_id": "abc-123-def-456",
  "timestamp": "2025-11-13T19:02:23Z",
  "cities": [
    {
      "city_id": "manila",
      "city_name": "Manila",
      "lat": 14.599,
      "lon": 120.968,
      "total_transformers": 5,
      "high_risk_count": 3
    }
  ],
  "barangays": [
    {
      "barangay": "Tondo",
      "city_id": "manila",
      "lat": 14.599,
      "lon": 120.968,
      "total_transformers": 1,
      "high_risk_count": 1
    }
  ],
  "transformers": [
    {
      "transformer_id": "TX_MAIN_001",
      "barangay": "Tondo",
      "lat": 14.599,
      "lon": 120.968,
      "total_meters": 10,
      "suspicious_meter_count": 1,
      "risk_level": "MEDIUM",
      "avg_consumption": 1856.45
    }
  ],
  "meters": [
    {
      "meter_id": "MTR_000001",
      "transformer_id": "TX_MAIN_001",
      "barangay": "Tondo",
      "risk_level": "LOW",
      "anomaly_score": 0.32,
      "confidence": 0.68,
      "explanation": "✅ Normal consumption pattern"
    }
  ],
  "high_risk_summary": {
    "total_high_risk_meters": 8,
    "most_anomalous_city": "Manila",
    "most_anomalous_barangay": "Tondo",
    "most_anomalous_transformer": "TX_MAIN_001",
    "top_10_transformers": [...]
  }
}
```

---

### 3. GET /api/transformers.geojson

**Purpose:** Get transformer locations in GeoJSON format

**Request:**

```bash
curl "http://localhost:8000/api/transformers.geojson?run_id=abc-123"
```

**Response:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [120.968, 14.599]
      },
      "properties": {
        "transformer_id": "TX_MAIN_001",
        "barangay": "Tondo",
        "total_meters": 10,
        "suspicious_meter_count": 1,
        "risk_level": "MEDIUM"
      }
    }
  ]
}
```

---

### 4. GET /api/export/{run_id}

**Purpose:** Download CSV report for field inspections

**Request:**

```bash
curl "http://localhost:8000/api/export/abc-123?level=transformer" \
  --output report.csv
```

**Query Parameters:**

- `level`: `"transformer"`, `"district"` (barangay), or `"meter"`

---

### 5. GET /api/runs

**Purpose:** List recent analysis runs

**Request:**

```bash
curl "http://localhost:8000/api/runs?limit=10"
```

**Response:**

```json
{
  "runs": [
    {
      "run_id": "abc-123-def-456",
      "timestamp": "2025-11-13T19:02:23Z",
      "total_meters": 52,
      "high_risk_count": 8,
      "status": "completed"
    }
  ]
}
```

---

### 6. GET /api/health

**Purpose:** Check API health

**Request:**

```bash
curl "http://localhost:8000/api/health"
```

**Response:**

```json
{
  "status": "healthy",
  "api": "running",
  "ml_pipeline": "healthy",
  "firestore": "healthy",
  "timestamp": "2025-11-13T19:02:23"
}
```

---

## 🗺️ Frontend Integration Guide

### 1. Upload CSV and Run Analysis

```typescript
// Upload CSV
const formData = new FormData();
formData.append("file", csvFile);

const response = await fetch("http://localhost:8000/api/run", {
  method: "POST",
  body: formData,
});

const { run_id } = await response.json();
// run_id = "abc-123-def-456"
```

### 2. Fetch Results

```typescript
const resultsResponse = await fetch(
  `http://localhost:8000/api/results/${run_id}`
);
const data = await resultsResponse.json();
```

### 3. Display Cities on Map

```typescript
// Initial map load - show city markers
data.cities.forEach((city) => {
  L.marker([city.lat, city.lon])
    .bindPopup(
      `
      <b>${city.city_name}</b><br>
      Transformers: ${city.total_transformers}<br>
      High Risk: ${city.high_risk_count}
    `
    )
    .addTo(map);
});
```

### 4. Zoom to Barangay (User Clicks City)

```typescript
// Filter barangays for selected city
const cityBarangays = data.barangays.filter(
  (b) => b.city_id === selectedCity.city_id
);

// Zoom map to city bounds
map.flyTo([selectedCity.lat, selectedCity.lon], 12);

// Show barangay markers
cityBarangays.forEach((barangay) => {
  L.marker([barangay.lat, barangay.lon])
    .bindPopup(
      `
      <b>${barangay.barangay}</b><br>
      Transformers: ${barangay.total_transformers}<br>
      High Risk: ${barangay.high_risk_count}
    `
    )
    .addTo(map);
});
```

### 5. Show Transformers (User Clicks Barangay)

```typescript
// Filter transformers for selected barangay
const barangayTransformers = data.transformers.filter(
  (t) => t.barangay === selectedBarangay.barangay
);

// Zoom to barangay
map.flyTo([selectedBarangay.lat, selectedBarangay.lon], 14);

// Show transformer markers with color by risk
barangayTransformers.forEach((transformer) => {
  const color =
    transformer.risk_level === "HIGH"
      ? "red"
      : transformer.risk_level === "MEDIUM"
      ? "orange"
      : "green";

  L.circleMarker([transformer.lat, transformer.lon], {
    color: color,
    radius: 10,
  })
    .bindPopup(
      `
    <b>${transformer.transformer_id}</b><br>
    Risk: ${transformer.risk_level}<br>
    Suspicious Meters: ${transformer.suspicious_meter_count}/${transformer.total_meters}
  `
    )
    .addTo(map);
});
```

### 6. Show Meter Details (User Clicks Transformer)

```typescript
// Filter meters for selected transformer
const transformerMeters = data.meters.filter(
  (m) => m.transformer_id === selectedTransformer.transformer_id
);

// Show modal with analytics
showModal({
  transformer: selectedTransformer,
  meters: transformerMeters,
  charts: {
    avgConsumption: selectedTransformer.avg_consumption,
    suspiciousCount: selectedTransformer.suspicious_meter_count,
  },
});
```

### 7. Populate "High-Risk Locations" Sidebar

```typescript
const topTransformers = data.high_risk_summary.top_10_transformers;

const sidebarHTML = topTransformers
  .map(
    (t) => `
  <div class="risk-item" onclick="zoomToTransformer('${t.transformer_id}')">
    <span class="risk-badge ${t.risk_level}">${t.risk_level}</span>
    <strong>${t.transformer_id}</strong>
    <small>${t.barangay}</small>
    <span>${t.suspicious_meter_count} suspicious meters</span>
  </div>
`
  )
  .join("");

document.getElementById("high-risk-list").innerHTML = sidebarHTML;
```

### 8. Export Report

```typescript
const downloadReport = async (level: "transformer" | "district" | "meter") => {
  const response = await fetch(
    `http://localhost:8000/api/export/${run_id}?level=${level}`
  );

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ghostload_report_${level}.csv`;
  a.click();
};
```

---

## 🔥 Firestore Data Structure

```
runs/
  └─ {run_id}/
      ├─ timestamp: 2025-11-13T19:02:23Z
      ├─ status: "completed"
      ├─ total_meters: 52
      ├─ total_transformers: 5
      ├─ high_risk_count: 8
      │
      ├─ cities: [...]
      ├─ barangays: [...]
      ├─ transformers: [...]
      ├─ meters: [...]
      └─ high_risk_summary: {...}
```

---

## 🧪 Testing

### Test Firebase Connection

```bash
python test_firebase.py
```

### Test Complete Workflow

```bash
python test_integration.py
```

### Test with Swagger UI

Open browser: **http://localhost:8000/docs**

### Test with curl

```bash
# Health check
curl http://localhost:8000/api/health

# Upload CSV
curl -X POST http://localhost:8000/api/run \
  -F "file=@meter_consumption.csv"

# Get results
curl http://localhost:8000/api/results/{run_id}

# Export CSV
curl "http://localhost:8000/api/export/{run_id}?level=transformer" \
  --output report.csv
```

---

## 🐛 Troubleshooting

### ML Pipeline Not Loading

**Error:** `ModuleNotFoundError: No module named 'machine_learning'`

**Fix:**

- Ensure `/machine_learning` folder exists at project root
- Check `sys.path` includes machine_learning folder
- Run from correct directory (`/backend`)

### Firebase Connection Failed

**Error:** `FileNotFoundError: firebase-credentials.json not found`

**Fix:**

- Download credentials from Firebase Console
- Place in `/backend` folder
- Update `.env` with correct path

### Port Already in Use

**Error:** `Address already in use`

**Fix:**

```bash
# Kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

---

## 📊 Performance Notes

- **CSV Upload:** ~1-2 seconds for 1,000 meters
- **ML Pipeline:** ~3-5 seconds for 1,000 meters
- **Data Aggregation:** ~0.5 seconds
- **Firestore Save:** ~1 second
- **Total Processing:** ~5-8 seconds for 1,000 meters

---

## 🔒 Security Notes

**For Production:**

- Add authentication (JWT tokens)
- Implement rate limiting
- Validate file sizes
- Sanitize CSV inputs
- Use HTTPS
- Implement RBAC for Firestore

**For Hackathon:**

- Basic CORS enabled
- No authentication (demo only)
- File size limit: 10MB
- Local development only

---

## 📝 Next Steps

1. ✅ Backend setup complete
2. ✅ ML pipeline integrated
3. ✅ Firestore configured
4. ⏸️ **Connect frontend** (React + Leaflet)
5. ⏸️ Deploy to Railway/Render
6. ⏸️ Add authentication

---

## 🤝 Team Collaboration

### Backend Team

- API endpoints: `/api/routes.py`
- Database: `/app/db/firestore.py`
- Configuration: `/app/config.py`

### ML Team

- Pipeline: `/machine_learning/pipeline/inference_pipeline.py`
- Models: `/machine_learning/models/`
- Data: `/machine_learning/datasets/`

### Frontend Team

- API documentation: This README
- Sample responses: `/test_integration.py`
- GeoJSON format: `/api/transformers.geojson`

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Leaflet Documentation](https://leafletjs.com/)

---

**Happy Coding! 🚀**
