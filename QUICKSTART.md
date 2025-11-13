# 🚀 GhostLoad Mapper — Quick Start Guide

## Prerequisites Setup (5 minutes)

### 1. Install Dependencies

**Backend:**

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Frontend:**

```bash
cd frontend
npm install
```

---

## Firebase Setup (3 minutes)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `ghost-load-mapper`
3. **Enable Firestore:**
   - Click "Firestore Database" → "Create database"
   - Choose "Start in test mode"
   - Select region: `asia-southeast1`
4. Your Firebase credentials are already in `backend/ghost-load-mapper-firebase-adminsdk-*.json` ✅

---

## Test Backend (2 minutes)

```bash
cd backend

# Test Firebase connection
python test_firebase.py
# Should see: ✅ Firebase connected successfully!

# Start server
python main.py
# Server starts on http://localhost:8000
```

**Open in browser:** http://localhost:8000/docs

- You'll see Swagger UI with all endpoints ✅

---

## Test ML Pipeline Integration (1 minute)

**In another terminal:**

```bash
cd backend
python test_integration.py
```

**Expected output:**

```
🏥 Testing API Health...
✅ API Status: healthy
   ML Pipeline: healthy
   Firestore: healthy

📤 Testing CSV Upload + ML Pipeline...
✅ Analysis completed!
   Run ID: abc-123-def-456
   Total Meters: 52
   Total Transformers: 5
   High Risk Count: 8
   Processing Time: 3.42s

✅ ALL TESTS PASSED!
```

---

## Start Frontend (2 minutes)

```bash
cd frontend
npm start
# Opens http://localhost:3000
```

---

## Complete Workflow Test (3 minutes)

### 1. Upload CSV via Frontend

**Sample CSV location:** `backend/meter_consumption.csv`

```typescript
// In your React component
const handleUpload = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/api/run", {
    method: "POST",
    body: formData,
  });

  const { run_id } = await response.json();
  console.log("Analysis complete! Run ID:", run_id);

  // Fetch results
  fetchResults(run_id);
};
```

### 2. Display Results on Map

```typescript
const fetchResults = async (run_id: string) => {
  const response = await fetch(`http://localhost:8000/api/results/${run_id}`);
  const data = await response.json();

  // Display cities
  data.cities.forEach((city) => {
    L.marker([city.lat, city.lon])
      .bindPopup(`<b>${city.city_name}</b>`)
      .addTo(map);
  });
};
```

---

## API Endpoints Cheat Sheet

```bash
# Health check
GET http://localhost:8000/api/health

# Upload CSV and run analysis
POST http://localhost:8000/api/run
Body: multipart/form-data with 'file' field

# Get results
GET http://localhost:8000/api/results/{run_id}

# Get GeoJSON for map
GET http://localhost:8000/api/transformers.geojson?run_id={run_id}

# Export CSV report
GET http://localhost:8000/api/export/{run_id}?level=transformer

# List recent runs
GET http://localhost:8000/api/runs?limit=10
```

---

## Expected Data Structure

### Frontend receives from `/api/results/{run_id}`:

```json
{
  "cities": [
    {
      "city_id": "manila",
      "city_name": "Manila",
      "lat": 14.599,
      "lon": 120.968,
      "total_transformers": 5,
      "high_risk_count": 2
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
      "avg_anomaly_score": 0.55,
      "avg_consumption": 1856.45
    }
  ],
  "meters": [
    {
      "meter_id": "MTR_000001",
      "transformer_id": "TX_MAIN_001",
      "barangay": "Tondo",
      "lat": 14.601,
      "lon": 120.961,
      "risk_level": "HIGH",
      "anomaly_score": 0.92,
      "confidence": 0.85,
      "explanation": "⚠️ High anomaly detected"
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

## Troubleshooting

### Backend won't start

**Error:** `ModuleNotFoundError: No module named 'firebase_admin'`

**Fix:**

```bash
cd backend
venv\Scripts\activate
pip install firebase-admin fastapi uvicorn pandas scikit-learn python-multipart python-dotenv
```

### ML Pipeline not loading

**Error:** `ModuleNotFoundError: No module named 'machine_learning'`

**Fix:**

- Ensure `/machine_learning` folder exists at project root
- The backend automatically adds it to `sys.path`
- Run from `/backend` directory

### Firebase connection failed

**Error:** `FileNotFoundError: firebase-credentials.json`

**Fix:**

- Check `.env` file has correct path
- Your credentials file: `ghost-load-mapper-firebase-adminsdk-*.json`
- Update `.env`:
  ```
  FIREBASE_CREDENTIALS_PATH=./ghost-load-mapper-firebase-adminsdk-fbsvc-57b15bb690.json
  ```

### CORS errors in frontend

**Error:** `Access-Control-Allow-Origin`

**Fix:**

- Backend already has CORS enabled for `localhost:3000` and `localhost:5173`
- If using different port, update `backend/app/config.py`:
  ```python
  ALLOWED_ORIGINS: list = [
      "http://localhost:3000",
      "http://localhost:5173",
      "http://localhost:YOUR_PORT",  # Add your port
  ]
  ```

---

## Sample Frontend Implementation

### Map Component

```typescript
import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
} from "react-leaflet";
import L from "leaflet";

interface AnalysisResults {
  cities: City[];
  barangays: Barangay[];
  transformers: Transformer[];
  meters: Meter[];
  high_risk_summary: HighRiskSummary;
}

export function GhostLoadMap({ runId }: { runId: string }) {
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [zoomLevel, setZoomLevel] = useState<
    "city" | "barangay" | "transformer"
  >("city");

  useEffect(() => {
    fetchResults(runId);
  }, [runId]);

  const fetchResults = async (runId: string) => {
    const response = await fetch(`http://localhost:8000/api/results/${runId}`);
    const data = await response.json();
    setResults(data);
  };

  if (!results) return <div>Loading...</div>;

  return (
    <div className="map-container">
      <MapContainer center={[14.5995, 120.9842]} zoom={11}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {/* City markers */}
        {zoomLevel === "city" &&
          results.cities.map((city) => (
            <Marker
              key={city.city_id}
              position={[city.lat, city.lon]}
              eventHandlers={{
                click: () => {
                  setZoomLevel("barangay");
                  // Zoom to city
                },
              }}
            >
              <Popup>
                <strong>{city.city_name}</strong>
                <br />
                Transformers: {city.total_transformers}
                <br />
                High Risk: {city.high_risk_count}
              </Popup>
            </Marker>
          ))}

        {/* Transformer markers with color by risk */}
        {zoomLevel === "transformer" &&
          results.transformers.map((transformer) => (
            <CircleMarker
              key={transformer.transformer_id}
              center={[transformer.lat, transformer.lon]}
              radius={10}
              pathOptions={{
                color:
                  transformer.risk_level === "HIGH"
                    ? "red"
                    : transformer.risk_level === "MEDIUM"
                    ? "orange"
                    : "green",
              }}
            >
              <Popup>
                <strong>{transformer.transformer_id}</strong>
                <br />
                Risk: {transformer.risk_level}
                <br />
                Suspicious: {transformer.suspicious_meter_count}/
                {transformer.total_meters}
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>

      {/* High-Risk Sidebar */}
      <aside className="sidebar">
        <h2>High-Risk Locations</h2>
        {results.high_risk_summary.top_10_transformers.map((t) => (
          <div key={t.transformer_id} className="risk-item">
            <span className={`badge ${t.risk_level}`}>{t.risk_level}</span>
            <strong>{t.transformer_id}</strong>
            <small>{t.barangay}</small>
          </div>
        ))}
      </aside>
    </div>
  );
}
```

---

## CSV Upload Component

```typescript
import { useState } from "react";

export function CSVUpload() {
  const [uploading, setUploading] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/run", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setRunId(data.run_id);
      alert(`Analysis complete! Processed ${data.total_meters} meters.`);
    } catch (error) {
      alert("Upload failed: " + error.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleUpload(file);
        }}
        disabled={uploading}
      />
      {uploading && <p>Processing... This may take a few seconds.</p>}
      {runId && <GhostLoadMap runId={runId} />}
    </div>
  );
}
```

---

## Next Steps

1. ✅ **Backend is ready** — All endpoints working
2. ✅ **ML pipeline integrated** — Predictions working
3. ✅ **Firestore configured** — Data saved successfully
4. ⏸️ **Wire up frontend** — Connect React components to API
5. ⏸️ **Add visualizations** — Charts, heatmaps, sparklines
6. ⏸️ **Polish UI** — Styling, animations, error handling

---

## Resources

- **Backend README:** `backend/README.md`
- **System Documentation:** `SYSTEM_DOCUMENTATION.md`
- **API Docs (Swagger):** http://localhost:8000/docs
- **ML Pipeline Docs:** `machine_learning/BACKEND_INTEGRATION_GUIDE.md`

---

**Ready to hack! 🚀**

**Questions?** Check `SYSTEM_DOCUMENTATION.md` for complete details.
