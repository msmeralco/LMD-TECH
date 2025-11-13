# UI Enhancement Quick Reference

## 🎯 What Changed

### 1. Backend - Filter Endpoints
```
NEW: GET /api/filters/{run_id}
→ Returns unique barangays, transformers, risk_levels for dropdowns

ENHANCED: GET /api/export/{run_id}
→ Added query params: ?barangay=...&transformer=...&risk_level=...
→ Sorted by anomaly_score DESC
→ Dynamic filename with filter indicators
```

### 2. Frontend - New Components

#### FloatingNavbar (Replaces old header when results loaded)
```
┌─────────────────────────────────────────────────────────┐
│ [⚡] GhostLoad Mapper  [📊 50] [⚠️ 18 (36%)] [▣ Show]  │
└─────────────────────────────────────────────────────────┘
```
- Glass morphism design
- Shows total meters + high-risk count
- Toggle button for sidebar

#### RankingSidebar (Overlays map from right)
```
                                    ┌──────────────────────┐
                                    │ 📊 Meter Rankings    │
                                    │ Showing 7 of 50      │
                                    ├──────────────────────┤
                                    │ 📍 [All Barangays ▼] │
                                    │ ⚡ [All Feeders ▼]   │
                                    │ 🎯 [HIGH ▼]         │
                                    │ [📥 Export CSV]      │
                                    ├──────────────────────┤
                                    │ #1 [HIGH] METER_001  │
                                    │    Score: 89.3%      │
                                    ├──────────────────────┤
                                    │ #2 [HIGH] METER_042  │
                                    │    Score: 87.1%      │
                                    └──────────────────────┘
```
- 450px width, slides in from right
- 3 filter dropdowns (barangay, transformer, risk)
- Ranked meter list (highest risk first)
- Export button with applied filters
- Click meter → Opens DrilldownModal

### 3. Layout Changes

**Before:**
```
┌─────────────────────────────────────────────┐
│ Header with stats                           │
├────────────────┬────────────────────────────┤
│                │                            │
│   Map (70%)    │   MeterList Sidebar (30%) │
│                │                            │
└────────────────┴────────────────────────────┘
```

**After:**
```
┌─────────────────────────────────────────────┐
│ [FloatingNavbar centered at top]           │
├─────────────────────────────────────────────┤
│                                             │
│         Map (Full Width 100%)       [Ranking│
│                                      Sidebar│
│                                      Overlay]│
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Run Backend
```powershell
cd backend
python start_server.py
# → http://localhost:8000
```

### Run Frontend
```powershell
cd frontend
npm start
# → http://localhost:3000
```

### Test Workflow
1. Upload `meter_consumption.csv`
2. Wait for processing (2-3 seconds)
3. FloatingNavbar appears with stats
4. Click "Show Rankings" button
5. Apply filters in sidebar
6. Click "Export Filtered CSV"
7. Verify downloaded file has filters applied

## 📊 API Endpoints

### Get Filter Options
```bash
GET http://localhost:8000/api/filters/{run_id}

Response:
{
  "barangays": ["Ermita", "Malate", ...],
  "transformers": ["TX_MAIN_001", ...],
  "risk_levels": ["HIGH", "MEDIUM", "LOW"]
}
```

### Export with Filters
```bash
GET http://localhost:8000/api/export/{run_id}?level=meter&barangay=Ermita&risk_level=HIGH

Response: CSV file download
Filename: ghostload_meters_Ermita_HIGH_{run_id}.csv
```

## 🎨 Component Files

| File | Lines | Purpose |
|------|-------|---------|
| `FloatingNavbar.tsx` | 135 | Stats display + sidebar toggle |
| `RankingSidebar.tsx` | 236 | Filters + ranked list + export |
| `AnomalyDashboard.tsx` | Modified | Integration of new components |
| `api.ts` | Modified | Added filter endpoints |
| `routes.py` | Modified | Backend filter logic |

## 🧪 Testing Commands

```powershell
# Test filter endpoint
curl http://localhost:8000/api/filters/YOUR_RUN_ID

# Test filtered export
curl "http://localhost:8000/api/export/YOUR_RUN_ID?level=meter&barangay=Ermita&risk_level=HIGH" -o test.csv

# Verify CSV sorted by risk
cat test.csv | head -5
```

## 📝 Next Steps (Pending)

1. **Heatmap Clustering** - Only show when high-risk meters within 500m
2. **Circle Markers** - Replace pins with circles
3. **City Icons** - Add city-level navigation layer
4. **Max Zoom** - Constrain to NCR region

## 🎉 Success Criteria

- ✅ FloatingNavbar shows stats correctly
- ✅ Sidebar slides in smoothly
- ✅ Filters update list instantly
- ✅ Export includes filter parameters
- ✅ CSV sorted by anomaly_score DESC
- ✅ Meter click opens modal
- ✅ Sidebar closes on backdrop click

## 💡 Tips

- **Clear Filters**: Select "All" in each dropdown
- **Export All**: Leave all filters on "All" before exporting
- **View Details**: Click any meter card in sidebar
- **Close Sidebar**: Click backdrop or "Hide Rankings" button

---

**Ready for IDOL Hackathon 2025! 🚀**
