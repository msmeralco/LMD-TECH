import React, { useRef } from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { Meter } from '../data/types';

interface LocationPinsProps {
  meters: Meter[];
  onPinClick: (meter: Meter) => void;
}

// ✅ Hardcode target district coordinates (no mockData needed)
const TARGET_DISTRICTS = [
  { id: 'makati', center: [14.5547, 121.0244] as [number, number] },
  { id: 'manila', center: [14.5995, 121.0374] as [number, number] },
  { id: 'pasig', center: [14.5764, 121.0851] as [number, number] },
  { id: 'taguig', center: [14.5176, 121.0509] as [number, number] },
];

const isTargetDistrict = (meterPosition: [number, number]): boolean => {
  for (const district of TARGET_DISTRICTS) {
    const distance = Math.sqrt(
      Math.pow(meterPosition[0] - district.center[0], 2) +
      Math.pow(meterPosition[1] - district.center[1], 2)
    );
    if (distance < 0.05) return true;
  }
  return false;
};

/**
 * Custom Location Pin Icon
 */
const createPinIcon = (riskLevel: 'high' | 'medium' | 'low') => {
  const colors = {
    high: '#FF6F00',
    medium: '#FFA500',
    low: '#90EE90',
  };

  const color = colors[riskLevel];

  return L.divIcon({
    className: 'custom-pin-icon',
    html: `
      <div style="position: relative;">
        <svg width="30" height="40" viewBox="0 0 30 40" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M15,2 C8.4,2 3,7.4 3,14 C3,22 15,38 15,38 C15,38 27,22 27,14 C27,7.4 21.6,2 15,2 Z"
            fill="${color}"
            stroke="#FFFFFF"
            stroke-width="2"
          />
          <circle cx="15" cy="14" r="5" fill="#FFFFFF" opacity="0.9"/>
        </svg>
      </div>
    `,
    iconSize: [30, 40],
    iconAnchor: [15, 40],
    popupAnchor: [0, -40],
  });
};

const LocationPins: React.FC<LocationPinsProps> = ({ meters, onPinClick }) => {
  return (
    <>
      {meters.map((meter) => {
        const consumptionData = meter.consumptionData || [];
        
        const avgConsumption = consumptionData.length > 0
          ? Math.round(consumptionData.reduce((sum, d) => sum + d.kwh, 0) / consumptionData.length)
          : 0;

        const lastMonthConsumption = consumptionData.length > 0
          ? consumptionData[consumptionData.length - 1]?.kwh || 0
          : 0;

        return (
          <Marker 
            key={meter.id} 
            position={meter.position} 
            icon={createPinIcon(meter.riskLevel)}
          >
            <Popup maxWidth={300} minWidth={250}>
              <div className="p-3">
                {/* Header */}
                <div className="mb-3 pb-3 border-b border-gray-200">
                  <h3 className="font-bold text-lg text-meralco-black mb-1">
                    {meter.meterNumber}
                  </h3>
                  <span
                    className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                      meter.riskLevel === 'high'
                        ? 'bg-red-100 text-red-700'
                        : meter.riskLevel === 'medium'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {meter.riskBand}
                  </span>
                </div>

                {/* Location Info */}
                <div className="space-y-2 text-sm mb-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Barangay:</span>
                    <span className="font-medium text-meralco-black">{meter.barangay}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Feeder:</span>
                    <span className="font-medium text-meralco-black">{meter.feeder}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Transformer:</span>
                    <span className="font-medium text-meralco-black">{meter.transformerId}</span>
                  </div>
                </div>

                {/* Anomaly Score */}
                <div className="mb-3 p-3 bg-orange-50 rounded-lg border border-orange-200">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-orange-700">Anomaly Score:</span>
                    <span className="text-lg font-bold text-orange-600">
                      {(meter.anomalyScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-orange-500 h-2 rounded-full transition-all"
                      style={{ width: `${meter.anomalyScore * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Consumption Summary */}
                {consumptionData.length > 0 && (
                  <div className="mb-3 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Last Month:</span>
                      <span className="font-semibold text-meralco-black">
                        {lastMonthConsumption} kWh
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Average:</span>
                      <span className="font-semibold text-meralco-black">
                        {avgConsumption} kWh
                      </span>
                    </div>
                  </div>
                )}

                {/* Anomaly Notes */}
                {meter.anomalyNotes && (
                  <div className="pt-3 mb-3 border-t border-gray-200">
                    <p className="text-xs text-gray-600 font-medium mb-1">Notes:</p>
                    <p className="text-xs text-meralco-black leading-relaxed">
                      {meter.anomalyNotes}
                    </p>
                  </div>
                )}

                {/* ✅ FIXED: Simple button with direct call */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🔍 Button clicked for meter:', meter.meterNumber);
                    onPinClick(meter);
                  }}
                  className="w-full mt-3 px-4 py-2 bg-meralco-orange text-white text-sm font-medium rounded-md hover:bg-opacity-90 transition-colors cursor-pointer"
                >
                  📊 View Detailed Consumption Trends →
                </button>
              </div>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
};

export default LocationPins;
