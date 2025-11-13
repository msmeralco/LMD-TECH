import React from "react";
import { CircleMarker, Popup } from "react-leaflet";
import { Meter } from "../data/types";

interface LocationPinsProps {
  meters: Meter[];
  onPinClick: (meter: Meter) => void;
}

/**
 * Get circle properties based on risk level
 * Fixed pixel sizes (won't scale with zoom)
 */
const getCircleStyle = (riskLevel: "high" | "medium" | "low") => {
  const styles = {
    high: {
      radius: 8,
      fillColor: "#EF4444", // Bright red
      color: "#FFFFFF", // White border
      weight: 2,
      fillOpacity: 0.9,
      className: "pulse-marker", // Add pulse animation
    },
    medium: {
      radius: 6,
      fillColor: "#F59E0B", // Orange
      color: "#FFFFFF",
      weight: 2,
      fillOpacity: 0.85,
    },
    low: {
      radius: 5,
      fillColor: "#10B981", // Green
      color: "#FFFFFF",
      weight: 2,
      fillOpacity: 0.8,
    },
  };

  return styles[riskLevel];
};

const LocationPins: React.FC<LocationPinsProps> = ({ meters, onPinClick }) => {
  return (
    <>
      {/* Add CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% {
            r: 8;
            opacity: 0.9;
          }
          50% {
            r: 10;
            opacity: 0.6;
          }
        }
        
        /* Ensure circles don't scale with zoom */
        .leaflet-zoom-anim .leaflet-zoom-animated {
          will-change: transform;
        }
      `}</style>

      {meters.map((meter) => {
        const circleStyle = getCircleStyle(meter.riskLevel);
        const consumptionData = meter.consumptionData || [];

        const avgConsumption =
          consumptionData.length > 0
            ? Math.round(
                consumptionData.reduce((sum, d) => sum + d.kwh, 0) /
                  consumptionData.length
              )
            : 0;

        const lastMonthConsumption =
          consumptionData.length > 0
            ? consumptionData[consumptionData.length - 1]?.kwh || 0
            : 0;

        return (
          <CircleMarker
            key={meter.id}
            center={meter.position}
            pathOptions={{
              fillColor: circleStyle.fillColor,
              fillOpacity: circleStyle.fillOpacity,
              color: circleStyle.color,
              weight: circleStyle.weight,
            }}
            radius={circleStyle.radius}
            eventHandlers={{
              click: () => onPinClick(meter),
            }}
          >
            <Popup maxWidth={320} minWidth={280}>
              <div className="p-4">
                {/* Header */}
                <div className="mb-4 pb-3 border-b-2 border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-bold text-xl text-gray-900">
                      {meter.meterNumber}
                    </h3>
                    <div
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold ${
                        meter.riskLevel === "high"
                          ? "bg-red-100 text-red-700 border-2 border-red-300"
                          : meter.riskLevel === "medium"
                          ? "bg-yellow-100 text-yellow-700 border-2 border-yellow-300"
                          : "bg-green-100 text-green-700 border-2 border-green-300"
                      }`}
                    >
                      <div
                        className={`w-2 h-2 rounded-full ${
                          meter.riskLevel === "high"
                            ? "bg-red-600"
                            : meter.riskLevel === "medium"
                            ? "bg-yellow-600"
                            : "bg-green-600"
                        }`}
                      ></div>
                      {meter.riskBand.toUpperCase()}
                    </div>
                  </div>
                  <p className="text-xs text-gray-500">
                    Click for detailed analytics
                  </p>
                </div>

                {/* Location Info */}
                <div className="space-y-2.5 text-sm mb-4 bg-gray-50 p-3 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center gap-1.5">
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                        />
                      </svg>
                      Barangay
                    </span>
                    <span className="font-semibold text-gray-900">
                      {meter.barangay}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center gap-1.5">
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 10V3L4 14h7v7l9-11h-7z"
                        />
                      </svg>
                      Transformer
                    </span>
                    <span className="font-semibold text-gray-900">
                      {meter.transformerId}
                    </span>
                  </div>
                </div>

                {/* Anomaly Score */}
                <div className="mb-4 p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-xl border-2 border-orange-200 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-bold text-orange-800 flex items-center gap-1.5">
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                      </svg>
                      Anomaly Score
                    </span>
                    <span className="text-2xl font-black text-orange-700">
                      {(meter.anomalyScore * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="relative w-full bg-gray-200 rounded-full h-3 overflow-hidden shadow-inner">
                    <div
                      className="absolute top-0 left-0 h-full bg-gradient-to-r from-orange-400 to-red-500 rounded-full transition-all duration-500 ease-out"
                      style={{ width: `${meter.anomalyScore * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Consumption Summary */}
                {consumptionData.length > 0 && (
                  <div className="mb-4 grid grid-cols-2 gap-3">
                    <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                      <p className="text-xs text-blue-600 font-semibold mb-1">
                        Last Month
                      </p>
                      <p className="text-lg font-bold text-blue-900">
                        {lastMonthConsumption.toLocaleString()}{" "}
                        <span className="text-xs font-normal">kWh</span>
                      </p>
                    </div>
                    <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
                      <p className="text-xs text-purple-600 font-semibold mb-1">
                        Average
                      </p>
                      <p className="text-lg font-bold text-purple-900">
                        {avgConsumption.toLocaleString()}{" "}
                        <span className="text-xs font-normal">kWh</span>
                      </p>
                    </div>
                  </div>
                )}

                {/* Anomaly Notes */}
                {meter.anomalyNotes && (
                  <div className="mb-4 p-3 bg-yellow-50 border-l-4 border-yellow-400 rounded">
                    <p className="text-xs text-yellow-800 font-semibold mb-1 flex items-center gap-1.5">
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      Alert
                    </p>
                    <p className="text-xs text-yellow-900 leading-relaxed">
                      {meter.anomalyNotes}
                    </p>
                  </div>
                )}

                {/* Action Button */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onPinClick(meter);
                  }}
                  className="w-full px-4 py-3 bg-gradient-to-r from-orange-500 to-orange-600 text-white text-sm font-bold rounded-lg hover:from-orange-600 hover:to-orange-700 transition-all shadow-md hover:shadow-lg transform hover:scale-[1.02] flex items-center justify-center gap-2"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  View Full Analytics Report
                </button>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </>
  );
};

export default LocationPins;
