import React, { useRef, useImperativeHandle, forwardRef } from 'react';
import { MapContainer, TileLayer, useMap, Marker } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import DistrictPopup from './DistrictPopup';

interface District {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
  totalTransformers: number;
  riskLevel: 'high' | 'medium' | 'low';
  totalMeters: number;
}

interface DistrictMapViewProps {
  districts: District[];
  onDistrictClick: (districtId: string) => void;
  selectedDistrict: string | null;
  children?: React.ReactNode;
}

const METRO_MANILA_CENTER: [number, number] = [14.6042, 121.0408];
const INITIAL_ZOOM = 11;

// Map Controller Component - handles zoom and pan
const MapController = forwardRef<{ resetView: () => void }, { 
  center: [number, number];
  zoom: number;
}>(({ center, zoom }, ref) => {
  const map = useMap();

  React.useEffect(() => {
    map.flyTo(center, zoom, {
      animate: true,
      duration: 1.0,
    });
  }, [map, center, zoom]);

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

// District Marker Component - clickable district boundaries
interface DistrictMarkerProps {
  district: District;
  isSelected: boolean;
  onClick: () => void;
}

// District Marker Component - popup only opens on click
const DistrictMarker: React.FC<DistrictMarkerProps> = ({ district, isSelected, onClick }) => {
  const map = useMap();
  const markerRef = useRef<L.Marker | null>(null);

  const handleClick = () => {
    // Smooth zoom to district
    map.flyTo(district.center, district.zoom, {
      animate: true,
      duration: 1.0,
    });
    onClick();
    // Open popup after zoom animation completes
    setTimeout(() => {
      if (markerRef.current) {
        markerRef.current.openPopup();
      }
    }, 1000);
  };

  // Create custom icon for district
  const districtIcon = L.divIcon({
    className: 'district-marker',
    html: `
      <div style="
        width: ${isSelected ? '50px' : '40px'};
        height: ${isSelected ? '50px' : '40px'};
        border-radius: 50%;
        background-color: ${isSelected ? '#FF6F00' : '#666666'};
        opacity: ${isSelected ? '0.3' : '0.2'};
        border: ${isSelected ? '3px' : '2px'} solid ${isSelected ? '#FF6F00' : '#666666'};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
      ">
        <div style="
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background-color: ${isSelected ? '#FF6F00' : '#666666'};
        "></div>
      </div>
    `,
    iconSize: [isSelected ? 50 : 40, isSelected ? 50 : 40],
    iconAnchor: [isSelected ? 25 : 20, isSelected ? 25 : 20],
  });

  return (
    <Marker
      position={district.center}
      icon={districtIcon}
      ref={markerRef}
      eventHandlers={{
        click: handleClick,
      }}
    >
      <DistrictPopup district={district} />
    </Marker>
  );
};

/**
 * District Map View Component
 * Displays clickable district boundaries that zoom in when clicked
 */
const DistrictMapView: React.FC<DistrictMapViewProps> = ({
  districts,
  onDistrictClick,
  selectedDistrict,
  children,
}) => {
  const mapControllerRef = useRef<{ resetView: () => void }>(null);
  const [mapCenter, setMapCenter] = React.useState<[number, number]>(METRO_MANILA_CENTER);
  const [mapZoom, setMapZoom] = React.useState(INITIAL_ZOOM);

  // Update map view when district is selected
  React.useEffect(() => {
    if (selectedDistrict) {
      const district = districts.find(d => d.id === selectedDistrict);
      if (district) {
        setMapCenter(district.center);
        setMapZoom(district.zoom);
      }
    } else {
      setMapCenter(METRO_MANILA_CENTER);
      setMapZoom(INITIAL_ZOOM);
    }
  }, [selectedDistrict, districts]);

  return (
    <div className="h-full w-full relative">
      <MapContainer
        center={mapCenter}
        zoom={mapZoom}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
        scrollWheelZoom={true}
        className="z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapController ref={mapControllerRef} center={mapCenter} zoom={mapZoom} />
        
        {/* Render district markers */}
        {districts.map((district) => (
          <DistrictMarker
            key={district.id}
            district={district}
            isSelected={selectedDistrict === district.id}
            onClick={() => onDistrictClick(district.id)}
          />
        ))}
        
        {/* Render children components (like LocationPins) */}
        {children}
      </MapContainer>

      {/* District Labels Overlay */}
      <div className="absolute top-4 left-4 z-[1000] bg-white bg-opacity-90 rounded-lg shadow-lg p-3">
        <h3 className="text-sm font-semibold text-meralco-black mb-2">Districts</h3>
        <div className="space-y-1">
          {districts.map((district) => (
            <button
              key={district.id}
              onClick={() => onDistrictClick(district.id)}
              className={`block w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                selectedDistrict === district.id
                  ? 'bg-meralco-orange text-white'
                  : 'hover:bg-meralco-light-gray text-meralco-black'
              }`}
            >
              {district.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DistrictMapView;

