import React, { useRef, useEffect } from 'react';
import { Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { Anomaly } from '../data/types';

interface AnomalyMapperProps {
  anomalies: Anomaly[];
  onAnomalyClick: (anomaly: Anomaly) => void;
}

/**
 * Custom Orange Pin Icon - styled like Meralco logo
 */
const createAnomalyIcon = (riskLevel: 'high' | 'medium' | 'low'): L.DivIcon => {
  const color = riskLevel === 'high' ? '#FF6F00' : riskLevel === 'medium' ? '#FFA500' : '#FFD700';
  
  return L.divIcon({
    className: 'custom-anomaly-marker',
    html: `
      <svg width="32" height="44" viewBox="0 0 32 44" xmlns="http://www.w3.org/2000/svg">
        <path d="M16 0 C7.163 0 0 7.163 0 16 C0 24 16 44 16 44 C16 44 32 24 32 16 C32 7.163 24.837 0 16 0 Z" 
              fill="${color}" stroke="#FFFFFF" stroke-width="2"/>
        <circle cx="16" cy="16" r="7" fill="#FFFFFF"/>
        <text x="16" y="20" font-size="12" font-weight="bold" fill="${color}" text-anchor="middle">!</text>
      </svg>
    `,
    iconSize: [32, 44],
    iconAnchor: [16, 44],
    popupAnchor: [0, -44],
  });
};

// Individual Anomaly Pin Component with zoom and modal trigger
interface AnomalyPinProps {
  anomaly: Anomaly;
  onAnomalyClick: (anomaly: Anomaly) => void;
}

const AnomalyPin: React.FC<AnomalyPinProps> = ({ anomaly, onAnomalyClick }) => {
  const map = useMap();
  const markerRef = useRef<L.Marker | null>(null);

  const handleClick = () => {
    // Smooth zoom to pin location using map.flyTo()
    map.flyTo(anomaly.position, 16, {
      animate: true,
      duration: 1.0,
    });

    // Auto-open modal after zoom animation completes
    setTimeout(() => {
      onAnomalyClick(anomaly);
    }, 1100); // Slightly longer than zoom duration for smooth transition
  };

  const icon = createAnomalyIcon(anomaly.riskLevel);

  return (
    <Marker
      position={anomaly.position}
      icon={icon}
      ref={markerRef}
      eventHandlers={{
        click: handleClick,
      }}
    >
      <Popup>
        <div className="p-2 min-w-[200px]">
          <h3 className="font-semibold text-meralco-black mb-1">{anomaly.meterId}</h3>
          <div className="space-y-1 text-xs">
            <p className="text-meralco-gray">
              <span className="font-medium">Risk:</span> {anomaly.riskBand}
            </p>
            <p className="text-meralco-gray">
              <span className="font-medium">Score:</span> {(anomaly.anomalyScore * 100).toFixed(0)}%
            </p>
            <p className="text-meralco-gray">
              <span className="font-medium">Barangay:</span> {anomaly.barangay}
            </p>
            <p className="text-meralco-gray">
              <span className="font-medium">Feeder:</span> {anomaly.feeder}
            </p>
            <p className="text-meralco-gray mt-2">{anomaly.description}</p>
            <p className="text-meralco-orange text-xs font-medium mt-2">
              Click to view detailed consumption data →
            </p>
          </div>
        </div>
      </Popup>
    </Marker>
  );
};

/**
 * Anomaly Mapper Component
 * Displays high-risk anomalies as orange pins on the map
 * When clicked, zooms in and automatically opens the drilldown modal
 */
const AnomalyMapper: React.FC<AnomalyMapperProps> = ({ anomalies, onAnomalyClick }) => {
  return (
    <>
      {anomalies.map((anomaly) => (
        <AnomalyPin
          key={anomaly.id}
          anomaly={anomaly}
          onAnomalyClick={onAnomalyClick}
        />
      ))}
    </>
  );
};

export default AnomalyMapper;

