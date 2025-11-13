# Meralco Anomaly Detection Dashboard

## Overview
A comprehensive React + TailwindCSS + Leaflet.js dashboard for visualizing electricity distribution anomalies with interactive maps, consumption analytics, and meter drilldown capabilities.

## Component Architecture

### 1. **AnomalyDashboard.tsx** (Main Component)
**Purpose**: Orchestrates all dashboard components and manages global state.

**Key Features**:
- Manages selected district and meter state
- Handles "Ready Graph Data" button functionality
- Coordinates between map, overlay, list, and modal components
- Filters anomalies and meters by selected district

**State Management**:
- `selectedDistrict`: Currently selected district ID
- `selectedMeter`: Currently selected meter for drilldown
- `isModalOpen`: Modal visibility state
- `isDataReady`: Data loading state (triggered by "Ready Graph Data" button)

### 2. **DistrictMapView.tsx** (Map Component)
**Purpose**: Displays clickable district boundaries on the map.

**Key Features**:
- Renders districts as clickable circular markers
- Smooth zoom animation when district is clicked
- District selection panel overlay
- Uses Leaflet's `flyTo()` for smooth transitions

**Interactions**:
- Click district marker → zooms to district center
- Click district in sidebar → selects and zooms to district
- Selected district highlighted with orange color

### 3. **AnomalyMapper.tsx** (Overlay Component)
**Purpose**: Displays high-risk anomalies as orange pins on the map.

**Key Features**:
- Custom SVG markers styled like Meralco logo
- Color-coded by risk level (high/medium/low)
- Clickable markers that open meter drilldown
- Popup shows anomaly details

**Marker Styling**:
- High risk: Orange (#FF6F00)
- Medium risk: Light orange (#FFA500)
- Low risk: Gold (#FFD700)

### 4. **MeterList.tsx** (Sidebar Component)
**Purpose**: Displays ranked list of suspicious meters with filtering.

**Key Features**:
- Three filter dropdowns: Barangay, Feeder, Risk Band
- Meters sorted by anomaly score (descending)
- Click meter to open drilldown modal
- Real-time filtering with React useMemo

**Filtering Logic**:
- All filters work independently
- "All" option shows all meters
- Results sorted by anomaly score

### 5. **DrilldownModal.tsx** (Modal Component)
**Purpose**: Displays detailed meter information with consumption trends.

**Key Features**:
- Line chart: 12-month consumption trend (kWh and kVA)
- Bar chart: Monthly comparison with anomaly highlighting
- Anomaly score display with risk band
- Statistics summary (average, peak, total period)
- Chart.js integration for data visualization

**Chart Types**:
- **Line Chart**: Dual-axis showing kWh (left) and kVA (right)
- **Bar Chart**: Monthly consumption with color-coded anomalies

### 6. **mockData.ts** (Data Layer)
**Purpose**: Provides mock data structure for development.

**Data Structures**:
- `District`: Geographic district information
- `Anomaly`: Anomaly detection results
- `Meter`: Complete meter data with consumption history
- `ConsumptionData`: Monthly consumption records

## User Flow

1. **Initial View**: Map shows all districts as clickable markers
2. **District Selection**: Click district → map zooms in, anomaly overlay appears
3. **Anomaly View**: Orange pins show high-risk anomalies
4. **Meter List**: Sidebar shows filtered list of suspicious meters
5. **Filtering**: Use dropdowns to filter by Barangay, Feeder, Risk Band
6. **Meter Drilldown**: Click meter → modal opens with consumption charts
7. **Graph Data**: Click "Ready Graph Data" to load/refresh consumption data
8. **Reset View**: Click "Reset View" to return to initial map state

## Technical Implementation

### Dependencies
- **React**: ^19.2.0
- **react-leaflet**: ^5.0.0 (Leaflet.js wrapper)
- **leaflet**: ^1.9.4 (Mapping library)
- **chart.js**: Latest (Charting library)
- **react-chartjs-2**: Latest (React wrapper for Chart.js)
- **tailwindcss**: Latest (Utility-first CSS framework)

### Key React Patterns
- **Functional Components**: All components use React hooks
- **Custom Hooks**: `useMap()` from react-leaflet for map access
- **Refs**: `useRef` and `useImperativeHandle` for map control
- **Memoization**: `useMemo` for expensive filtering operations
- **State Management**: Local state with useState hooks

### Styling Approach
- **TailwindCSS**: Utility-first classes for layout and styling
- **Meralco Colors**: Custom color palette defined in tailwind.config.js
- **Responsive Design**: Mobile-first approach with Tailwind breakpoints

### Map Interactions
- **Smooth Animations**: Leaflet's `flyTo()` for zoom transitions
- **Event Handlers**: Click handlers on markers and districts
- **Custom Icons**: SVG-based markers styled with Meralco branding

## Color Palette

- **Primary Orange**: #FF6F00 (Meralco brand color)
- **Black**: #000000
- **Gray**: #666666
- **Light Gray**: #E5E5E5
- **White**: #FFFFFF
- **Dark Gray**: #333333

## File Structure

```
src/
├── components/
│   ├── AnomalyDashboard.tsx    # Main dashboard component
│   ├── DistrictMapView.tsx      # Map with districts
│   ├── AnomalyMapper.tsx       # Anomaly pins overlay
│   ├── MeterList.tsx            # Filterable meter list
│   └── DrilldownModal.tsx       # Meter detail modal
├── data/
│   └── mockData.ts              # Mock data structure
├── styles/
│   ├── index.css               # Global styles + Tailwind
│   └── MeralcoMap.css          # Custom marker styles
└── App.tsx                      # Root component
```

## Usage

1. **Start Development Server**:
   ```bash
   npm start
   ```

2. **Interact with Dashboard**:
   - Click districts on map to zoom in
   - Click "Ready Graph Data" to load consumption data
   - Filter meters using sidebar dropdowns
   - Click meters to view detailed consumption trends
   - Click "Reset View" to return to overview

## Future Enhancements

- Real API integration for data fetching
- Real-time anomaly detection updates
- Export functionality for reports
- Advanced filtering and search
- District boundary polygons (GeoJSON)
- Historical trend analysis
- Alert notifications for critical anomalies

