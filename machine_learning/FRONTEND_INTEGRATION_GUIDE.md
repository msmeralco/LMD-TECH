# 🎨 GhostLoad Mapper - Frontend Integration Guide

**Version:** 1.0.0  
**Date:** November 13, 2025  
**Tech Stack:** React + Tailwind CSS + Leaflet  
**Backend API:** FastAPI

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Flow Overview](#data-flow-overview)
3. [API Endpoints](#api-endpoints)
4. [Component Integration](#component-integration)
5. [Map Visualization](#map-visualization)
6. [UI/UX Recommendations](#uiux-recommendations)
7. [Example Code](#example-code)

---

## 🚀 Quick Start

### **What Frontend Needs from Backend**

```javascript
// GET /api/alerts - Returns all suspicious meters
{
  "total_meters": 1000,
  "high_risk_count": 52,
  "medium_risk_count": 106,
  "low_risk_count": 842,
  "actionable_meters": [
    {
      "meter_id": "MTR_000001",
      "risk_level": "HIGH",           // ← Use for color coding
      "anomaly_score": 0.85,          // ← Display in drilldown
      "confidence": 0.82,
      "lat": 14.409318,               // ← Map coordinates
      "lon": 120.979165,
      "transformer_id": "TX_0001",
      "barangay": "Poblacion",
      "customer_class": "commercial",
      "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
      "consumption_history": [        // ← For sparkline chart
        710.51, 811.28, 663.03, 633.15, 1070.65, 897.33,
        996.28, 932.92, 989.90, 945.07, 1037.30, 1023.97
      ]
    },
    // ... 157 more suspicious meters
  ]
}
```

---

## 🔄 Data Flow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                        │
│                                                             │
│  1. CSV Upload Component                                    │
│     └─> POST /api/upload (meter_consumption.csv)          │
│         └─> Returns: upload_id                              │
│                                                             │
│  2. Trigger Analysis                                        │
│     └─> POST /api/run-detection?upload_id={id}            │
│         └─> Returns: job_id                                 │
│                                                             │
│  3. Fetch Results                                           │
│     └─> GET /api/alerts                                    │
│         └─> Returns: suspicious meters + metadata          │
│                                                             │
│  4. Display Results                                         │
│     ├─> Map: Leaflet with color-coded markers             │
│     ├─> Table: Sorted list with filters                    │
│     ├─> Modal: Drilldown with consumption chart           │
│     └─> Export: Download CSV of high-risk meters          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints

### **1. Upload Meter Data**

```http
POST /api/upload
Content-Type: multipart/form-data

File: meter_consumption.csv
```

**Response:**
```json
{
  "status": "success",
  "upload_id": "upload_20251113_184556",
  "meters_uploaded": 1000,
  "message": "CSV uploaded successfully"
}
```

### **2. Run ML Detection**

```http
POST /api/run-detection?upload_id=upload_20251113_184556
```

**Response:**
```json
{
  "status": "processing",
  "job_id": "job_12345",
  "estimated_time": 10,
  "message": "Analysis started. Poll /api/status/{job_id} for updates"
}
```

### **3. Get Alerts (Main Data Source)**

```http
GET /api/alerts?risk_level=HIGH,MEDIUM&limit=200
```

**Response:**
```json
{
  "total_meters": 1000,
  "high_risk_count": 52,
  "medium_risk_count": 106,
  "low_risk_count": 842,
  "actionable_meters": [
    {
      "meter_id": "MTR_000001",
      "risk_level": "HIGH",
      "anomaly_score": 0.85,
      "confidence": 0.82,
      "lat": 14.409318,
      "lon": 120.979165,
      "transformer_id": "TX_0001",
      "barangay": "Poblacion",
      "customer_class": "commercial",
      "kVA": 1296.97,
      "explanation": "⚠️ High anomaly detected",
      "consumption_history": [710.51, 811.28, ...]
    }
  ],
  "transformers": [
    {
      "transformer_id": "TX_0001",
      "lat": 14.41,
      "lon": 120.98,
      "high_risk_meters": 5,
      "medium_risk_meters": 12,
      "total_meters": 35
    }
  ]
}
```

### **4. Get Meter Details (Drilldown)**

```http
GET /api/meter/{meter_id}
```

**Response:**
```json
{
  "meter_id": "MTR_000001",
  "risk_level": "HIGH",
  "anomaly_score": 0.85,
  "confidence": 0.82,
  "transformer_id": "TX_0001",
  "customer_class": "commercial",
  "barangay": "Poblacion",
  "lat": 14.409318,
  "lon": 120.979165,
  "kVA": 1296.97,
  "consumption_trend": "declining",
  "consumption_history": [
    { "month": "2024-11", "kwh": 710.51 },
    { "month": "2024-12", "kwh": 811.28 },
    { "month": "2025-01", "kwh": 663.03 },
    // ... 12 months
  ],
  "transformer_median": 1250.0,
  "consumption_ratio": 0.62,
  "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
  "recommended_action": "Field inspection within 7 days"
}
```

---

## 🗺️ Map Visualization (Leaflet Integration)

### **Color Coding Schema**

```javascript
const getRiskColor = (riskLevel) => {
  switch (riskLevel) {
    case 'HIGH':
      return '#EF4444';   // Red - Urgent inspection
    case 'MEDIUM':
      return '#F59E0B';   // Orange - Follow-up needed
    case 'LOW':
      return '#10B981';   // Green - Monitor only
    default:
      return '#6B7280';   // Gray - Unknown
  }
};

const getRiskIcon = (riskLevel) => {
  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        background-color: ${getRiskColor(riskLevel)};
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      "></div>
    `
  });
};
```

### **Leaflet Map Component**

```jsx
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function SuspiciousMeterMap() {
  const [meters, setMeters] = useState([]);
  const [transformers, setTransformers] = useState([]);

  useEffect(() => {
    // Fetch alerts from backend
    fetch('/api/alerts?risk_level=HIGH,MEDIUM')
      .then(res => res.json())
      .then(data => {
        setMeters(data.actionable_meters);
        setTransformers(data.transformers);
      });
  }, []);

  return (
    <MapContainer 
      center={[14.41, 120.98]} 
      zoom={13} 
      style={{ height: '600px', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      
      {/* Render transformers (larger circles) */}
      {transformers.map(tx => (
        <CircleMarker
          key={tx.transformer_id}
          center={[tx.lat, tx.lon]}
          radius={15}
          fillColor="#3B82F6"
          fillOpacity={0.3}
          color="#3B82F6"
          weight={2}
        >
          <Popup>
            <strong>{tx.transformer_id}</strong><br/>
            High Risk: {tx.high_risk_meters}<br/>
            Medium Risk: {tx.medium_risk_meters}<br/>
            Total Meters: {tx.total_meters}
          </Popup>
        </CircleMarker>
      ))}

      {/* Render suspicious meters */}
      {meters.map(meter => (
        <Marker
          key={meter.meter_id}
          position={[meter.lat, meter.lon]}
          icon={getRiskIcon(meter.risk_level)}
        >
          <Popup>
            <div style={{ minWidth: '200px' }}>
              <h3 className="font-bold">{meter.meter_id}</h3>
              <div className="mt-2">
                <span className={`
                  px-2 py-1 rounded text-xs font-semibold
                  ${meter.risk_level === 'HIGH' ? 'bg-red-500 text-white' : 'bg-orange-500 text-white'}
                `}>
                  {meter.risk_level} RISK
                </span>
              </div>
              <div className="mt-2 text-sm">
                <p><strong>Score:</strong> {(meter.anomaly_score * 100).toFixed(0)}%</p>
                <p><strong>Confidence:</strong> {(meter.confidence * 100).toFixed(0)}%</p>
                <p><strong>Class:</strong> {meter.customer_class}</p>
                <p><strong>Barangay:</strong> {meter.barangay}</p>
              </div>
              <button 
                onClick={() => openMeterModal(meter.meter_id)}
                className="mt-2 w-full bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
              >
                View Details
              </button>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
```

---

## 📊 Component Integration

### **1. Suspicious Meter List (Table)**

```jsx
import React, { useState, useMemo } from 'react';

function SuspiciousMeterTable({ meters }) {
  const [filter, setFilter] = useState('ALL');
  const [barangayFilter, setBarangayFilter] = useState('');

  const filteredMeters = useMemo(() => {
    return meters
      .filter(m => filter === 'ALL' || m.risk_level === filter)
      .filter(m => !barangayFilter || m.barangay === barangayFilter)
      .sort((a, b) => b.anomaly_score - a.anomaly_score); // Highest risk first
  }, [meters, filter, barangayFilter]);

  return (
    <div>
      {/* Filters */}
      <div className="mb-4 flex gap-4">
        <select 
          value={filter} 
          onChange={(e) => setFilter(e.target.value)}
          className="border rounded px-3 py-2"
        >
          <option value="ALL">All Risks</option>
          <option value="HIGH">High Risk Only</option>
          <option value="MEDIUM">Medium Risk Only</option>
        </select>

        <input
          type="text"
          placeholder="Filter by barangay..."
          value={barangayFilter}
          onChange={(e) => setBarangayFilter(e.target.value)}
          className="border rounded px-3 py-2"
        />

        <button 
          onClick={() => exportCSV(filteredMeters)}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
        >
          Export CSV ({filteredMeters.length} meters)
        </button>
      </div>

      {/* Table */}
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="border p-2">Meter ID</th>
            <th className="border p-2">Risk</th>
            <th className="border p-2">Score</th>
            <th className="border p-2">Transformer</th>
            <th className="border p-2">Barangay</th>
            <th className="border p-2">Class</th>
            <th className="border p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {filteredMeters.map(meter => (
            <tr key={meter.meter_id} className="hover:bg-gray-50">
              <td className="border p-2 font-mono">{meter.meter_id}</td>
              <td className="border p-2">
                <span className={`
                  px-2 py-1 rounded text-xs font-semibold
                  ${meter.risk_level === 'HIGH' ? 'bg-red-500 text-white' : ''}
                  ${meter.risk_level === 'MEDIUM' ? 'bg-orange-500 text-white' : ''}
                  ${meter.risk_level === 'LOW' ? 'bg-green-500 text-white' : ''}
                `}>
                  {meter.risk_level}
                </span>
              </td>
              <td className="border p-2">{(meter.anomaly_score * 100).toFixed(0)}%</td>
              <td className="border p-2 font-mono">{meter.transformer_id}</td>
              <td className="border p-2">{meter.barangay}</td>
              <td className="border p-2 capitalize">{meter.customer_class}</td>
              <td className="border p-2">
                <button 
                  onClick={() => openMeterModal(meter.meter_id)}
                  className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
                >
                  Details
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### **2. Meter Drilldown Modal**

```jsx
import React from 'react';
import { Line } from 'react-chartjs-2';

function MeterDrilldownModal({ meter, onClose }) {
  if (!meter) return null;

  const chartData = {
    labels: ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
    datasets: [
      {
        label: 'Monthly Consumption (kWh)',
        data: meter.consumption_history,
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4
      },
      {
        label: 'Transformer Median',
        data: Array(12).fill(meter.transformer_median),
        borderColor: '#10B981',
        borderDash: [5, 5],
        pointRadius: 0
      }
    ]
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold">{meter.meter_id}</h2>
            <span className={`
              inline-block mt-2 px-3 py-1 rounded text-sm font-semibold
              ${meter.risk_level === 'HIGH' ? 'bg-red-500 text-white' : ''}
              ${meter.risk_level === 'MEDIUM' ? 'bg-orange-500 text-white' : ''}
            `}>
              {meter.risk_level} RISK
            </span>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 text-2xl">
            ×
          </button>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Anomaly Score</div>
            <div className="text-2xl font-bold">{(meter.anomaly_score * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Confidence</div>
            <div className="text-2xl font-bold">{(meter.confidence * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Class</div>
            <div className="text-xl font-semibold capitalize">{meter.customer_class}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">kVA</div>
            <div className="text-2xl font-bold">{meter.kVA}</div>
          </div>
        </div>

        {/* Consumption Chart */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Consumption Trend (Last 12 Months)</h3>
          <Line data={chartData} options={{
            responsive: true,
            plugins: {
              legend: { display: true, position: 'top' }
            }
          }} />
        </div>

        {/* Explanation */}
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
          <p className="font-semibold text-yellow-800">AI Explanation:</p>
          <p className="text-yellow-700 mt-1">{meter.explanation}</p>
        </div>

        {/* Additional Details */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <p className="text-sm text-gray-600">Transformer ID</p>
            <p className="font-mono">{meter.transformer_id}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Barangay</p>
            <p>{meter.barangay}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Consumption Ratio</p>
            <p>{(meter.consumption_ratio * 100).toFixed(0)}% of transformer median</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Coordinates</p>
            <p className="font-mono text-sm">{meter.lat.toFixed(6)}, {meter.lon.toFixed(6)}</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button className="flex-1 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600">
            Flag for Inspection
          </button>
          <button className="flex-1 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
            View on Map
          </button>
          <button onClick={onClose} className="px-4 py-2 border rounded hover:bg-gray-50">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
```

### **3. CSV Export Function**

```javascript
function exportCSV(meters) {
  // Prepare CSV data
  const headers = ['Meter ID', 'Risk Level', 'Anomaly Score', 'Confidence', 'Transformer', 'Barangay', 'Class', 'Lat', 'Lon'];
  const rows = meters.map(m => [
    m.meter_id,
    m.risk_level,
    m.anomaly_score.toFixed(3),
    m.confidence.toFixed(3),
    m.transformer_id,
    m.barangay,
    m.customer_class,
    m.lat.toFixed(6),
    m.lon.toFixed(6)
  ]);

  // Convert to CSV string
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n');

  // Download
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `suspicious_meters_${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
}
```

---

## 🎨 UI/UX Recommendations

### **Dashboard Layout**

```
┌─────────────────────────────────────────────────────────────┐
│  GhostLoad Mapper - Electricity Theft Detection             │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │ 52 HIGH     │ │ 106 MEDIUM   │ │ 842 LOW      │         │
│  │ RISK        │ │ RISK         │ │ RISK         │         │
│  └─────────────┘ └──────────────┘ └──────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐  ┌────────────────────────────┐  │
│  │                     │  │ Suspicious Meter List      │  │
│  │                     │  │                            │  │
│  │   Interactive Map   │  │ [Filters] [Export CSV]    │  │
│  │   (Leaflet)         │  │                            │  │
│  │                     │  │ MTR_001  HIGH  85%  [View] │  │
│  │   • = HIGH RISK     │  │ MTR_052  HIGH  72%  [View] │  │
│  │   • = MEDIUM RISK   │  │ MTR_103 MEDIUM 68%  [View] │  │
│  │   ○ = Transformer   │  │ MTR_089 MEDIUM 55%  [View] │  │
│  │                     │  │ ...                        │  │
│  └─────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### **Color Palette**

```css
/* Risk Levels */
--risk-high: #EF4444;      /* Red */
--risk-medium: #F59E0B;    /* Orange */
--risk-low: #10B981;       /* Green */

/* Backgrounds */
--bg-primary: #FFFFFF;
--bg-secondary: #F3F4F6;
--bg-dark: #1F2937;

/* Borders */
--border-light: #E5E7EB;
--border-dark: #9CA3AF;

/* Text */
--text-primary: #111827;
--text-secondary: #6B7280;
```

### **Responsive Breakpoints**

```css
/* Mobile: Stack map above table */
@media (max-width: 768px) {
  .layout { flex-direction: column; }
  .map-container { height: 400px; }
}

/* Tablet: Side by side */
@media (min-width: 769px) and (max-width: 1024px) {
  .map-container { width: 50%; }
  .table-container { width: 50%; }
}

/* Desktop: Optimal layout */
@media (min-width: 1025px) {
  .map-container { width: 60%; }
  .table-container { width: 40%; }
}
```

---

## ✅ Testing Checklist

### **Frontend Integration Tests**

- [ ] CSV upload shows success message
- [ ] Map loads with correct center coordinates
- [ ] HIGH risk meters show as red markers
- [ ] MEDIUM risk meters show as orange markers
- [ ] Clicking marker opens popup with meter info
- [ ] Table filters by risk level correctly
- [ ] Table filters by barangay correctly
- [ ] Sorting by anomaly score works (descending)
- [ ] Modal opens when clicking "View Details"
- [ ] Consumption chart displays 12 months correctly
- [ ] CSV export downloads file with correct data
- [ ] Mobile view stacks components vertically
- [ ] Error handling shows user-friendly messages

### **Performance Tests**

- [ ] Map renders 158 markers in <2 seconds
- [ ] Table handles 1000 rows without lag
- [ ] Modal opens in <100ms
- [ ] CSV export completes in <1 second
- [ ] API calls have loading indicators

---

## 🎯 Demo Flow for Hackathon

### **Recommended Presentation Sequence**

1. **Upload CSV** (10 seconds)
   - "We uploaded 1,000 meters from Meralco's dataset"

2. **Show Summary** (15 seconds)
   - "Our AI detected 52 high-risk and 106 medium-risk meters"
   - Display stats cards

3. **Interactive Map** (30 seconds)
   - Zoom to cluster of red markers
   - "Red dots are urgent cases, orange need follow-up"
   - Click on high-risk meter → show popup

4. **Detailed View** (45 seconds)
   - Click "View Details" → open modal
   - "This meter shows 85% anomaly score"
   - Point to consumption chart
   - "Notice the declining trend - classic theft pattern"

5. **Export Inspection List** (15 seconds)
   - Filter to HIGH risk only
   - Click "Export CSV"
   - "Field teams can now inspect these 52 priority cases"

6. **Business Impact** (15 seconds)
   - "Prioritizing 52 urgent cases instead of checking all 1,000"
   - "Reduces inspection time by 95%"
   - "Estimated revenue recovery: ₱XXX million"

**Total Time: 2 minutes**

---

## 📞 Support & Resources

**API Documentation:** See `BACKEND_INTEGRATION_GUIDE.md`  
**Sample Data:** `machine_learning/output/latest/risk_assessment.csv`  
**Test Endpoint:** `http://localhost:8000/api/alerts`

**Quick Troubleshooting:**
- **Map not loading:** Check Leaflet CSS import
- **No markers showing:** Verify lat/lon are numbers, not strings
- **Colors wrong:** Check risk_level case sensitivity ("HIGH" not "high")
- **Chart broken:** Ensure consumption_history is array of 12 numbers

---

## 🎉 Success Criteria

Your frontend is **DEMO-READY** when:

✅ Map displays color-coded meter markers  
✅ Clicking markers shows meter details  
✅ Table filters and sorts correctly  
✅ Modal displays consumption chart  
✅ CSV export works  
✅ Mobile responsive  
✅ **Judges are impressed!** 🏆

---

**Version:** 1.0.0  
**Last Updated:** November 13, 2025  
**Frontend Stack:** React + Tailwind + Leaflet + Chart.js
