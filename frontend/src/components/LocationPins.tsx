import React, { useState, useRef } from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { Meter, mockDistricts } from '../data/mockData';

interface LocationPinsProps {
  meters: Meter[];
  onPinClick?: (meter: Meter) => void;
}

/**
 * Determine if a meter belongs to Makati, Manila, Pasig, or Taguig districts
 * Based on proximity to district centers
 */
const isTargetDistrict = (meterPosition: [number, number]): boolean => {
  const targetDistrictIds = ['makati', 'manila', 'pasig', 'taguig'];
  const targetDistricts = mockDistricts.filter(d => targetDistrictIds.includes(d.id));
  
  // Check if meter position is within proximity of any target district
  for (const district of targetDistricts) {
    const [lat1, lon1] = meterPosition;
    const [lat2, lon2] = district.center;
    
    // Simple distance calculation (approximate)
    const latDiff = Math.abs(lat1 - lat2);
    const lonDiff = Math.abs(lon1 - lon2);
    const distance = Math.sqrt(latDiff * latDiff + lonDiff * lonDiff);
    
    // If within ~0.05 degrees (roughly 5km), consider it part of that district
    if (distance < 0.05) {
      return true;
    }
  }
  
  return false;
};

/**
 * Custom Location Pin Icon - Traditional pin shape with brighter orange for target districts
 */
const createLocationPinIcon = (
  riskLevel: 'high' | 'medium' | 'low',
  meterPosition: [number, number]
): L.DivIcon => {
  // Use brighter orange for Makati, Manila, Pasig, Taguig
  const isTarget = isTargetDistrict(meterPosition);
  const brightOrange = '#FF8C00'; // Brighter orange (DarkOrange)
  
  let color: string;
  if (isTarget) {
    color = brightOrange; // Always use bright orange for target districts
  } else {
    color = riskLevel === 'high' ? '#FF6F00' : riskLevel === 'medium' ? '#FFA500' : '#90EE90';
  }
  
  // Traditional pin shape (classic map marker: rounded top, pointed bottom)
  return L.divIcon({
    className: 'location-pin-marker',
    html: `
      <svg width="30" height="44" viewBox="0 0 30 44" xmlns="http://www.w3.org/2000/svg">
        <!-- Traditional pin shape: rounded top with sharp point at bottom -->
        <path d="M15 0 C6.716 0 0 6.716 0 15 C0 22.5 15 44 15 44 C15 44 30 22.5 30 15 C30 6.716 23.284 0 15 0 Z" 
              fill="${color}" stroke="#FFFFFF" stroke-width="2.5" stroke-linejoin="round"/>
        <!-- Pin head circle with highlight -->
        <circle cx="15" cy="15" r="7" fill="#FFFFFF" stroke="${color}" stroke-width="2"/>
        <!-- Inner circle for depth -->
        <circle cx="15" cy="15" r="4" fill="${color}"/>
        <!-- Highlight on pin head -->
        <circle cx="13" cy="13" r="2" fill="rgba(255,255,255,0.6)"/>
        <!-- Shadow at bottom for 3D effect -->
        <ellipse cx="15" cy="42" rx="4" ry="2" fill="rgba(0,0,0,0.25)"/>
      </svg>
    `,
    iconSize: [30, 44],
    iconAnchor: [15, 44],
    popupAnchor: [0, -44],
  });
};

// Get risk level badge styling
const getRiskBadgeStyle = (riskLevel: string) => {
  switch (riskLevel) {
    case 'high':
      return 'bg-red-100 text-red-800 border-red-300';
    case 'medium':
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'low':
      return 'bg-green-100 text-green-800 border-green-300';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

// Individual Location Pin Marker Component
interface LocationPinMarkerProps {
  meter: Meter;
  icon: L.DivIcon;
  isSelected: boolean;
  onPinClick: (meter: Meter) => void;
}

const LocationPinMarker: React.FC<LocationPinMarkerProps> = ({
  meter,
  icon,
  isSelected,
  onPinClick,
}) => {
  const markerRef = useRef<L.Marker | null>(null);

  const handleClick = () => {
    // Open popup when pin is clicked
    if (markerRef.current) {
      markerRef.current.openPopup();
    }
    onPinClick(meter);
  };

  return (
    <Marker
      position={meter.position}
      icon={icon}
      ref={markerRef}
      eventHandlers={{
        click: handleClick,
      }}
    >
      <Popup
        className="location-pin-popup"
        autoPan={true}
        closeButton={true}
        autoClose={true}
        closeOnClick={true}
      >
        <div className={`p-4 min-w-[280px] ${isSelected ? 'bg-meralco-orange bg-opacity-5' : ''}`}>
          {/* Meter Header */}
          <div className="mb-3 pb-3 border-b-2 border-meralco-orange">
            <h3 className="text-lg font-bold text-meralco-black mb-1">
              {meter.meterNumber}
            </h3>
            <div className={`inline-block px-2 py-1 rounded text-xs font-medium border ${getRiskBadgeStyle(meter.riskLevel)}`}>
              {meter.riskBand}
            </div>
          </div>

          {/* Location Information */}
          <div className="space-y-2 mb-3">
            <div className="flex items-start justify-between">
              <span className="text-sm text-meralco-gray font-medium">Barangay:</span>
              <span className="text-sm font-semibold text-meralco-black">{meter.barangay}</span>
            </div>
            <div className="flex items-start justify-between">
              <span className="text-sm text-meralco-gray font-medium">Feeder:</span>
              <span className="text-sm font-semibold text-meralco-black">{meter.feeder}</span>
            </div>
            <div className="flex items-start justify-between">
              <span className="text-sm text-meralco-gray font-medium">Transformer:</span>
              <span className="text-sm font-semibold text-meralco-black">{meter.transformerId}</span>
            </div>
          </div>

          {/* Anomaly Score */}
          <div className="bg-meralco-light-gray rounded-lg p-3 mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-meralco-gray font-medium">Anomaly Score:</span>
              <span className="text-lg font-bold text-meralco-orange">
                {(meter.anomalyScore * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div
                className="bg-meralco-orange h-2 rounded-full transition-all duration-300"
                style={{ width: `${meter.anomalyScore * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Consumption Summary */}
          {meter.consumptionData && meter.consumptionData.length > 0 && (
            <div className="mb-3">
              <p className="text-xs text-meralco-gray font-medium mb-1">Recent Consumption:</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-meralco-gray">Last Month:</span>
                <span className="font-semibold text-meralco-black">
                  {meter.consumptionData[meter.consumptionData.length - 1].kwh} kWh
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-meralco-gray">Average:</span>
                <span className="font-semibold text-meralco-black">
                  {Math.round(
                    meter.consumptionData.reduce((sum, d) => sum + d.kwh, 0) /
                    meter.consumptionData.length
                  )}{' '}
                  kWh
                </span>
              </div>
            </div>
          )}

          {/* Anomaly Notes */}
          {meter.anomalyNotes && (
            <div className="pt-3 border-t border-meralco-light-gray">
              <p className="text-xs text-meralco-gray font-medium mb-1">Notes:</p>
              <p className="text-xs text-meralco-black leading-relaxed">
                {meter.anomalyNotes}
              </p>
            </div>
          )}

          {/* Action Hint */}
          <div className="mt-3 pt-3 border-t border-meralco-light-gray">
            <p className="text-xs text-meralco-orange font-medium text-center">
              Click for detailed consumption trends →
            </p>
          </div>
        </div>
      </Popup>
    </Marker>
  );
};

/**
 * Location Pins Component
 * Displays location pins for meters on the map
 * Shows detailed information only when clicked
 */
const LocationPins: React.FC<LocationPinsProps> = ({ meters, onPinClick }) => {
  const [selectedPin, setSelectedPin] = useState<string | null>(null);

  const handlePinClick = (meter: Meter) => {
    setSelectedPin(meter.id);
    if (onPinClick) {
      onPinClick(meter);
    }
  };

  return (
    <>
      {meters.map((meter) => {
        const icon = createLocationPinIcon(meter.riskLevel, meter.position);
        const isSelected = selectedPin === meter.id;

        return (
          <LocationPinMarker
            key={meter.id}
            meter={meter}
            icon={icon}
            isSelected={isSelected}
            onPinClick={handlePinClick}
          />
        );
      })}
    </>
  );
};

export default LocationPins;
