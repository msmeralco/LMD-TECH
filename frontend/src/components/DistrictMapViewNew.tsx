import React, { useState, useMemo, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Tooltip,
  useMap,
} from "react-leaflet";
import { District } from "../data/types";
import LocationPins from "./LocationPins";
import { Meter } from "../data/types";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

declare global {
  interface Window {
    L: any;
  }
}

interface DistrictMapViewProps {
  districts: District[];
  onDistrictClick: (districtId: string) => void;
  selectedDistrict: string | null;
  meters?: Meter[];
  onMeterClick?: (meter: Meter) => void;
}

const NCR_CENTER: [number, number] = [14.5995, 120.9842];
const INITIAL_ZOOM = 11;

const NCR_BOUNDS: L.LatLngBoundsLiteral = [
  [14.35, 120.85],
  [14.85, 121.15],
];

/**
 * Haversine distance calculation in meters
 */
const calculateDistance = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number => {
  const R = 6371e3;
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δφ = ((lat2 - lat1) * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

/**
 * Detect tight clusters of high-risk meters (5+ within 200m radius)
 * Returns array of meter IDs that are part of significant clusters
 */
const getClusteredHighRiskMeters = (
  meters: Meter[],
  maxDistance: number = 200,
  minClusterSize: number = 5
): Set<string> => {
  const highRiskMeters = meters.filter((m) => m.riskLevel === "high");
  const clusteredMeterIds = new Set<string>();

  if (highRiskMeters.length < minClusterSize) {
    return clusteredMeterIds;
  }

  // For each high-risk meter, count neighbors within radius
  for (let i = 0; i < highRiskMeters.length; i++) {
    const meter1 = highRiskMeters[i];
    const [lat1, lon1] = meter1.position;
    const neighbors: Meter[] = [meter1];

    for (let j = 0; j < highRiskMeters.length; j++) {
      if (i === j) continue;

      const meter2 = highRiskMeters[j];
      const [lat2, lon2] = meter2.position;
      const distance = calculateDistance(lat1, lon1, lat2, lon2);

      if (distance <= maxDistance) {
        neighbors.push(meter2);
      }
    }

    // Only mark as cluster if meets minimum size requirement
    if (neighbors.length >= minClusterSize) {
      neighbors.forEach(m => clusteredMeterIds.add(m.id));
    }
  }

  return clusteredMeterIds;
};

/**
 * Heatmap layer - only shows significant clusters (5+ high-risk meters within 200m)
 */
const HeatmapLayer: React.FC<{ meters: Meter[] }> = ({ meters }) => {
  const map = useMap();

  useEffect(() => {
    let heatLayer: any = null;

    const loadHeatmap = async () => {
      if (!window.L.heatLayer) {
        await import("leaflet.heat");
      }

      if (!meters || meters.length === 0) return;

      // Get only clustered high-risk meters (5+ within 200m)
      const clusteredMeterIds = getClusteredHighRiskMeters(meters, 200, 5);
      const clusteredMeters = meters.filter(m => clusteredMeterIds.has(m.id));

      if (clusteredMeters.length === 0) return;

      const heatData = clusteredMeters.map(
        (meter) =>
          [
            meter.position[0],
            meter.position[1],
            meter.anomalyScore, // Use full anomaly score for intensity
          ] as [number, number, number]
      );

      heatLayer = L.heatLayer(heatData, {
        radius: 30,
        blur: 20,
        maxZoom: 17,
        max: 1.0,
        minOpacity: 0.4,
        gradient: {
          0.0: "rgba(0, 0, 0, 0)",
          0.3: "rgba(255, 165, 0, 0.3)",
          0.5: "rgba(255, 100, 0, 0.5)",
          0.7: "rgba(255, 50, 0, 0.7)",
          1.0: "rgba(220, 20, 60, 0.85)",
        },
      }).addTo(map);
    };

    loadHeatmap();

    return () => {
      if (heatLayer && map.hasLayer(heatLayer)) {
        map.removeLayer(heatLayer);
      }
    };
  }, [map, meters]);

  return null;
};

/**
 * Map controller for smooth transitions
 */
const MapController: React.FC<{ center: [number, number]; zoom: number }> = ({
  center,
  zoom,
}) => {
  const map = useMap();

  useEffect(() => {
    map.flyTo(center, zoom, {
      duration: 1,
      easeLinearity: 0.25,
    });
  }, [center, zoom, map]);

  return null;
};

type ViewLevel = "ncr" | "city";

const DistrictMapView: React.FC<DistrictMapViewProps> = ({
  districts,
  onDistrictClick,
  selectedDistrict,
  meters = [],
  onMeterClick,
}) => {
  const [mapCenter, setMapCenter] = useState<[number, number]>(NCR_CENTER);
  const [mapZoom, setMapZoom] = useState(INITIAL_ZOOM);
  const [viewLevel, setViewLevel] = useState<ViewLevel>("ncr");
  const [selectedCity, setSelectedCity] = useState<string | null>(null);

  // Extract unique cities from meters with their data
  const cities = useMemo(() => {
    const cityMap = new Map<
      string,
      { center: [number, number]; meters: Meter[]; stats: any }
    >();

    meters.forEach((meter) => {
      // Extract city from meter (assuming barangay contains city info or you have city field)
      const city = (meter as any).city_id || "Unknown";

      if (!cityMap.has(city)) {
        cityMap.set(city, {
          center: meter.position,
          meters: [],
          stats: { high: 0, medium: 0, low: 0 },
        });
      }

      const cityData = cityMap.get(city)!;
      cityData.meters.push(meter);

      if (meter.riskLevel === "high") cityData.stats.high++;
      else if (meter.riskLevel === "medium") cityData.stats.medium++;
      else cityData.stats.low++;

      // Update center to average position
      const avgLat =
        cityData.meters.reduce((sum, m) => sum + m.position[0], 0) /
        cityData.meters.length;
      const avgLon =
        cityData.meters.reduce((sum, m) => sum + m.position[1], 0) /
        cityData.meters.length;
      cityData.center = [avgLat, avgLon];
    });

    return Array.from(cityMap.entries()).map(([name, data]) => ({
      id: name.toLowerCase(),
      name: name.charAt(0).toUpperCase() + name.slice(1),
      ...data,
    }));
  }, [meters]);

  const handleCityClick = (cityId: string) => {
    const city = cities.find((c) => c.id === cityId);
    if (!city) return;

    setSelectedCity(cityId);
    setMapCenter(city.center);
    setMapZoom(13);
    setViewLevel("city");
    onDistrictClick(cityId);
  };

  const handleBackToNCR = () => {
    setSelectedCity(null);
    setMapCenter(NCR_CENTER);
    setMapZoom(INITIAL_ZOOM);
    setViewLevel("ncr");
    onDistrictClick("");
  };

  const currentCityMeters = useMemo(() => {
    if (!selectedCity) return [];
    const city = cities.find((c) => c.id === selectedCity);
    return city ? city.meters : [];
  }, [selectedCity, cities]);

  const showHeatmap = useMemo(() => {
    if (viewLevel !== "city" || !selectedCity) return false;
    const clusteredMeters = getClusteredHighRiskMeters(currentCityMeters, 200, 5);
    const result = clusteredMeters.size >= 5;
    console.log(
      result ? `✅ Heatmap ON: ${clusteredMeters.size} clustered meters (200m radius)` : "❌ Heatmap OFF: No significant cluster"
    );
    return result;
  }, [viewLevel, selectedCity, currentCityMeters]);

  const getRiskColor = (stats: {
    high: number;
    medium: number;
    low: number;
  }) => {
    const total = stats.high + stats.medium + stats.low;
    if (total === 0) return "#CCCCCC";

    const highPercent = stats.high / total;
    if (highPercent > 0.3) return "#EF4444";
    if (highPercent > 0.1) return "#F59E0B";
    return "#10B981";
  };

  return (
    <div className="relative h-full w-full">
      {/* Back Button */}
      {viewLevel === "city" && (
        <button
          onClick={handleBackToNCR}
          className="absolute top-4 left-4 z-[1000] bg-white px-4 py-2.5 rounded-lg shadow-lg hover:shadow-xl transition-all flex items-center gap-2 border border-gray-200"
        >
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
              d="M10 19l-7-7m0 0l7-7m-7 7h18"
            />
          </svg>
          <span className="text-sm font-medium">Back to NCR</span>
        </button>
      )}

      <MapContainer
        center={NCR_CENTER}
        zoom={INITIAL_ZOOM}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        maxBounds={NCR_BOUNDS}
        maxBoundsViscosity={1.0}
        minZoom={10}
        maxZoom={18}
      >
        <MapController center={mapCenter} zoom={mapZoom} />

        <TileLayer
          attribution="&copy; OpenStreetMap"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* NCR View: Show all city markers */}
        {viewLevel === "ncr" &&
          cities.map((city) => (
            <CircleMarker
              key={city.id}
              center={city.center}
              radius={15}
              pathOptions={{
                fillColor: getRiskColor(city.stats),
                fillOpacity: 0.7,
                color: "#FFFFFF",
                weight: 3,
              }}
              eventHandlers={{
                click: () => handleCityClick(city.id),
              }}
            >
              <Tooltip direction="top" offset={[0, -10]} opacity={0.9}>
                <div className="text-center">
                  <div className="font-bold text-sm">{city.name}</div>
                  <div className="text-xs mt-1">
                    {city.meters.length} meters
                  </div>
                </div>
              </Tooltip>
              <Popup>
                <div className="p-3 min-w-[200px]">
                  <h3 className="font-bold text-lg mb-3">{city.name}</h3>
                  <div className="space-y-2 text-sm mb-3">
                    <div className="flex justify-between">
                      <span>Total Meters:</span>
                      <span className="font-semibold">
                        {city.meters.length}
                      </span>
                    </div>
                    <div className="flex justify-between text-red-600">
                      <span>High Risk:</span>
                      <span className="font-semibold">{city.stats.high}</span>
                    </div>
                    <div className="flex justify-between text-yellow-600">
                      <span>Medium Risk:</span>
                      <span className="font-semibold">{city.stats.medium}</span>
                    </div>
                    <div className="flex justify-between text-green-600">
                      <span>Low Risk:</span>
                      <span className="font-semibold">{city.stats.low}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleCityClick(city.id)}
                    className="w-full bg-orange-500 text-white px-4 py-2 rounded-md hover:bg-orange-600 transition-colors text-sm font-medium"
                  >
                    View Details
                  </button>
                </div>
              </Popup>
            </CircleMarker>
          ))}

        {/* City View: Show meters and heatmap */}
        {viewLevel === "city" && selectedCity && (
          <>
            {showHeatmap && <HeatmapLayer meters={currentCityMeters} />}
            {onMeterClick && (
              <LocationPins
                meters={currentCityMeters}
                onPinClick={onMeterClick}
              />
            )}
          </>
        )}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white p-4 rounded-lg shadow-lg z-[1000] max-w-xs border border-gray-200">
        <h4 className="font-semibold text-sm mb-3">Risk Levels</h4>
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span>High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
            <span>Medium Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span>Low Risk</span>
          </div>
        </div>
        {viewLevel === "city" && showHeatmap && (
          <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-red-600 font-medium">
            ● Cluster Detected (200m)
          </div>
        )}
      </div>
    </div>
  );
};

export default DistrictMapView;
