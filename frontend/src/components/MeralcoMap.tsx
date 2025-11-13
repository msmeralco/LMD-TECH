import React, { useRef, useImperativeHandle, forwardRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import '../styles/MeralcoMap.css';

// Metro Manila coordinates (center)
const METRO_MANILA_CENTER: [number, number] = [14.6042, 121.0408];
const INITIAL_ZOOM = 12;

// Component to handle map reset functionality
const MapController = forwardRef<{ resetView: () => void }>((props, ref) => {
  const map = useMap();

  useImperativeHandle(ref, () => ({
    resetView: () => {
      map.flyTo(METRO_MANILA_CENTER, INITIAL_ZOOM, {
        animate: true,
        duration: 1.0,
      });
    },
  }));

  return null;
});

MapController.displayName = 'MapController';

// Component to handle marker click and zoom
interface MarkerWithZoomProps {
  position: [number, number];
  icon: L.DivIcon;
  children: React.ReactNode;
  zoomLevel?: number;
}

const MarkerWithZoom: React.FC<MarkerWithZoomProps> = ({ 
  position, 
  icon, 
  children,
  zoomLevel = 16 
}) => {
  const map = useMap();

  const handleMarkerClick = () => {
    // Smooth zoom and pan to the marker position
    map.flyTo(position, zoomLevel, {
      animate: true,
      duration: 1.0, // Animation duration in seconds
    });
  };

  return (
    <Marker
      position={position}
      icon={icon}
      eventHandlers={{
        click: handleMarkerClick,
      }}
    >
      {children}
    </Marker>
  );
};

// Custom orange marker icon
const createOrangeMarkerIcon = () => {
  return L.divIcon({
    className: 'custom-orange-marker',
    html: `
      <svg width="30" height="42" viewBox="0 0 30 42" xmlns="http://www.w3.org/2000/svg">
        <path d="M15 0 C6.716 0 0 6.716 0 15 C0 22.5 15 42 15 42 C15 42 30 22.5 30 15 C30 6.716 23.284 0 15 0 Z" 
              fill="#FF6F00" stroke="#FFFFFF" stroke-width="2"/>
        <circle cx="15" cy="15" r="6" fill="#FFFFFF"/>
      </svg>
    `,
    iconSize: [30, 42],
    iconAnchor: [15, 42],
    popupAnchor: [0, -42],
  });
};

// Sample locations in Metro Manila for demonstration
const sampleLocations: Array<{ id: number; name: string; position: [number, number]; description: string }> = [
  { id: 1, name: 'Makati City', position: [14.5547, 121.0244], description: 'Business District' },
  { id: 2, name: 'Quezon City', position: [14.6760, 121.0437], description: 'Government Center' },
  { id: 3, name: 'Manila City', position: [14.5995, 120.9842], description: 'Capital District' },
  { id: 4, name: 'Pasig City', position: [14.5764, 121.0851], description: 'Commercial Area' },
  { id: 5, name: 'Taguig City', position: [14.5176, 121.0509], description: 'Financial Hub' },
];

const MeralcoMap: React.FC = () => {
  const orangeMarkerIcon = createOrangeMarkerIcon();
  const mapControllerRef = useRef<{ resetView: () => void }>(null);

  const handleResetView = () => {
    if (mapControllerRef.current) {
      mapControllerRef.current.resetView();
    }
  };

  return (
    <div className="meralco-app">
      {/* Header */}
      <header className="meralco-header">
        <div className="header-content">
          <div className="logo-placeholder">
            <span className="logo-text">MERALCO</span>
          </div>
          <nav className="header-nav">
            <a href="#dashboard" className="nav-link">Dashboard</a>
            <a href="#map" className="nav-link active">Map</a>
            <a href="#reports" className="nav-link">Reports</a>
            <a href="#settings" className="nav-link">Settings</a>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <div className="main-container">
        {/* Sidebar */}
        <aside className="meralco-sidebar">
          <div className="sidebar-header">
            <h2>Map Controls</h2>
          </div>
          <div className="sidebar-content">
            <div className="control-group">
              <label className="control-label">Layer</label>
              <select className="control-select">
                <option value="street">Street Map</option>
                <option value="satellite">Satellite</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Zoom Level</label>
              <input type="range" min="8" max="18" defaultValue="12" className="control-slider" />
            </div>
            <div className="control-group">
              <label className="control-label">
                <input type="checkbox" defaultChecked className="control-checkbox" />
                Show Markers
              </label>
            </div>
            <div className="control-group">
              <label className="control-label">
                <input type="checkbox" className="control-checkbox" />
                Show Grid
              </label>
            </div>
            <div className="control-group">
              <button className="control-button" onClick={handleResetView}>
                Reset View
              </button>
            </div>
            <div className="control-group">
              <button className="control-button secondary">Export Map</button>
            </div>
          </div>
        </aside>

        {/* Map Container */}
        <main className="map-container">
          <MapContainer
            center={METRO_MANILA_CENTER}
            zoom={INITIAL_ZOOM}
            style={{ height: '100%', width: '100%' }}
            zoomControl={true}
            scrollWheelZoom={true}
          >
            <MapController ref={mapControllerRef} />
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {sampleLocations.map((location) => (
              <MarkerWithZoom
                key={location.id}
                position={location.position}
                icon={orangeMarkerIcon}
                zoomLevel={16}
              >
                <Popup>
                  <div className="marker-popup">
                    <h3>{location.name}</h3>
                    <p>{location.description}</p>
                  </div>
                </Popup>
              </MarkerWithZoom>
            ))}
          </MapContainer>
        </main>
      </div>
    </div>
  );
};

export default MeralcoMap;

