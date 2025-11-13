import React, { useState, useEffect } from "react";
import DistrictMapView from "./DistrictMapView";
import MeterList from "./MeterList";
import DrilldownModal from "./DrilldownModal";
import UploadModal from "./UploadModal";
import FloatingNavbar from "./FloatingNavbar";
import RankingSidebar from "./RankingSidebar";
import StatsPanel from "./StatsPanel";
import { apiService, ResultsData, Meter as APIMeter } from "../services/api";
import { Meter } from "../data/types";
import { BARANGAY_TO_CITY, CITY_METADATA } from "../data/cityMetadata";

/**
 * Calculate distance between two coordinates (Haversine formula)
 */
const calculateDistance = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number => {
  const R = 6371e3; // Earth radius in meters
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
 * Find the nearest city based on coordinates
 * This handles duplicate barangay names by using actual GPS location
 */
const findNearestCity = (lat: number, lon: number): string => {
  let nearestCity = "unknown";
  let minDistance = Infinity;

  Object.entries(CITY_METADATA).forEach(([cityId, metadata]) => {
    const [cityLat, cityLon] = metadata.center;
    const distance = calculateDistance(lat, lon, cityLat, cityLon);

    if (distance < minDistance) {
      minDistance = distance;
      nearestCity = cityId;
    }
  });

  return nearestCity;
};

const AnomalyDashboard: React.FC = () => {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMeter, setSelectedMeter] = useState<Meter | null>(null);
  const [isMeterModalOpen, setIsMeterModalOpen] = useState(false);
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedCity, setSelectedCity] = useState<string | null>(null);

  useEffect(() => {
    if (currentRunId) {
      loadResults(currentRunId);
    }
  }, [currentRunId]);

  const loadResults = async (runId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiService.getResults(runId);
      setResultsData(data);
    } catch (err: any) {
      setError(err.message || "Failed to load results");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadSuccess = (runId: string) => {
    setCurrentRunId(runId);
  };

  const handleResetView = () => {
    setCurrentRunId(null);
    setResultsData(null);
    setSelectedMeter(null);
    setIsMeterModalOpen(false);
    setSelectedDistrict(null);
    setSelectedCity(null);
    setIsSidebarOpen(false);
  };

  const handleUploadClick = () => {
    setIsUploadModalOpen(true);
  };

  const handleConfirmReset = () => {
    handleResetView();
  };

  // Convert backend API meter to component Meter type
  const convertAPIToMeter = (apiMeter: APIMeter): Meter => {
    let riskLevel: "high" | "medium" | "low" = "low";
    let riskBand = "Low Risk";

    if (apiMeter.risk_level === "HIGH") {
      riskLevel = "high";
      riskBand = "High Risk";
    } else if (apiMeter.risk_level === "MEDIUM") {
      riskLevel = "medium";
      riskBand = "Medium Risk";
    }

    // Convert consumption array to consumptionData format
    const consumptionData =
      apiMeter.monthly_consumptions?.map((kwh, index) => ({
        month: new Date(2024, index).toLocaleString("default", {
          month: "short",
        }),
        year: "2024",
        kwh: kwh,
        kva: kwh / 0.9, // Approximate kVA from kWh (assuming 0.9 power factor)
      })) || [];

    // 🎯 ULTRA-SMART CITY DETECTION
    // Handles: 1) Missing city_id from backend, 2) Duplicate barangay names, 3) Unknown barangays
    // Strategy: Always fallback to GPS if barangay not found
    let cityId = apiMeter.city_id;

    if (!cityId || cityId === "unknown") {
      // Try barangay mapping first
      const barangay = apiMeter.barangay?.trim();

      if (barangay && BARANGAY_TO_CITY[barangay]) {
        cityId = BARANGAY_TO_CITY[barangay].city;
        console.log(`🗺️ Mapped barangay "${barangay}" → city "${cityId}"`);
      } else if (apiMeter.lat && apiMeter.lon) {
        // CRITICAL: If barangay not in mapping, use GPS (don't leave as 'unknown')
        cityId = findNearestCity(apiMeter.lat, apiMeter.lon);
        console.log(
          `📍 Unknown barangay "${barangay}", using GPS → city "${cityId}"`
        );
      }

      // For duplicate barangay names (San Antonio, Poblacion, etc.)
      // Use GPS to verify/correct the mapping
      if (cityId && cityId !== "unknown" && apiMeter.lat && apiMeter.lon) {
        const gpsCity = findNearestCity(apiMeter.lat, apiMeter.lon);

        // Only override if barangay mapping might be wrong (check distance)
        const mappedCityCenter = CITY_METADATA[cityId]?.center;
        const gpsCityCenter = CITY_METADATA[gpsCity]?.center;

        if (mappedCityCenter && gpsCityCenter && cityId !== gpsCity) {
          const distToMapped = calculateDistance(
            apiMeter.lat,
            apiMeter.lon,
            mappedCityCenter[0],
            mappedCityCenter[1]
          );
          const distToGPS = calculateDistance(
            apiMeter.lat,
            apiMeter.lon,
            gpsCityCenter[0],
            gpsCityCenter[1]
          );

          // If GPS city is significantly closer (>2km difference), use GPS
          if (distToGPS < distToMapped - 2000) {
            console.log(
              `🔄 Corrected "${barangay}": ${cityId} → ${gpsCity} (GPS override)`
            );
            cityId = gpsCity;
          }
        }
      }
    }

    // Final safety check - log if still unknown
    if (!cityId || cityId === "unknown") {
      console.warn(
        `⚠️ Could not determine city for meter ${apiMeter.meter_id}, barangay: "${apiMeter.barangay}"`
      );
    }

    return {
      id: apiMeter.meter_id,
      meterNumber: apiMeter.meter_id,
      transformerId: apiMeter.transformer_id,
      barangay: apiMeter.barangay,
      barangay_id: apiMeter.barangay_id, // ✅ Pass through barangay_id from backend
      city_id: cityId, // ✅ ULTRA-SMART: GPS-corrected city detection
      feeder: apiMeter.transformer_id,
      riskLevel: riskLevel,
      riskBand: riskBand,
      anomalyScore: apiMeter.anomaly_score,
      position: [apiMeter.lat, apiMeter.lon],
      consumption: apiMeter.monthly_consumptions || [],
      consumptionData: consumptionData,
      anomalyNotes:
        riskLevel === "high"
          ? `High anomaly score detected (${(
              apiMeter.anomaly_score * 100
            ).toFixed(1)}%). Recommend field inspection.`
          : undefined,
    };
  };

  // Convert cities to districts for DistrictMapView
  const getDistricts = () => {
    if (!resultsData) return [];

    return resultsData.cities.map((city) => ({
      id: city.city_id,
      name: city.city_name,
      center: [city.lat, city.lon] as [number, number],
      zoom: 13,
      totalTransformers: city.total_transformers,
      riskLevel: (city.high_risk_count > 5
        ? "high"
        : city.high_risk_count > 2
        ? "medium"
        : "low") as "high" | "medium" | "low",
      totalMeters: resultsData.meters.filter((m) => m.city_id === city.city_id)
        .length,
    }));
  };

  // Get meters for MeterList - REAL BACKEND DATA
  const getMeters = (): Meter[] => {
    if (!resultsData) return [];
    return resultsData.meters.map(convertAPIToMeter);
  };

  // Handle meter click from map/list
  const handleMeterClick = (meter: Meter) => {
    console.log("🔍 Meter clicked:", meter.meterNumber); // Debug log
    setSelectedMeter(meter);
    setIsMeterModalOpen(true);
  };

  // Handle district/city click
  const handleDistrictClick = (districtId: string) => {
    setSelectedDistrict(districtId);
    setSelectedCity(districtId);
  };

  // Handle modal close
  const handleCloseModal = () => {
    console.log("❌ Closing modal"); // Debug log
    setIsMeterModalOpen(false);
    setSelectedMeter(null);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Floating Navbar - Always visible */}
      <FloatingNavbar
        totalMeters={resultsData?.total_meters || 0}
        highRiskCount={resultsData?.high_risk_count || 0}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        isSidebarOpen={isSidebarOpen}
        onUploadClick={handleUploadClick}
        hasData={!!resultsData}
        selectedCity={selectedCity}
      />

      <div className="flex-1 flex overflow-hidden relative">
        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-[100] flex items-center justify-center">
            <div className="text-center">
              <svg
                className="animate-spin h-12 w-12 text-meralco-orange mx-auto mb-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              <p className="text-lg text-gray-600 font-medium">
                Analyzing meter data...
              </p>
              <p className="text-sm text-gray-500 mt-2">
                ML pipeline processing in progress
              </p>
            </div>
          </div>
        )}

        {/* Error Overlay */}
        {error && (
          <div className="absolute inset-0 bg-white/95 backdrop-blur-sm z-[100] flex items-center justify-center p-6">
            <div className="max-w-md w-full">
              <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 shadow-xl">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <svg
                      className="w-8 h-8 text-red-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <p className="font-bold text-red-900 text-lg mb-2">
                      Analysis Failed
                    </p>
                    <p className="text-sm text-red-700 mb-4">{error}</p>
                    <button
                      onClick={() => setError(null)}
                      className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium text-sm"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Map View - Always visible */}
        <div className="flex-1">
          {resultsData ? (
            <>
              {/* Stats Panel - Show data summary */}
              <StatsPanel resultsData={resultsData} />
              
              <DistrictMapView
                districts={getDistricts()}
                onDistrictClick={handleDistrictClick}
                selectedDistrict={selectedDistrict}
                meters={getMeters()}
                onMeterClick={handleMeterClick}
              />
            </>
          ) : (
            // Empty state map with instruction
            <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
              <div className="text-center max-w-md px-6">
                <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
                  <div className="w-20 h-20 bg-gradient-to-br from-orange-500 to-orange-600 rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg
                      className="w-10 h-10 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                      />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-3">
                    Welcome to GhostLoad Mapper
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Upload your meter consumption CSV to start detecting
                    anomalies and ghost loads across your distribution network.
                  </p>
                  <button
                    onClick={handleUploadClick}
                    className="px-6 py-3 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg hover:from-orange-600 hover:to-orange-700 transition-all font-medium shadow-lg flex items-center justify-center gap-2 mx-auto"
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
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                    Upload CSV File
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
        hasExistingData={!!resultsData}
        onConfirmReset={handleConfirmReset}
      />

      {/* Ranking Sidebar - Only show if city is selected */}
      {resultsData && selectedCity && (
        <RankingSidebar
          meters={getMeters().filter((m) => {
            // Filter meters by selected city (use city_id from converted meter)
            if (selectedCity === "ncr") {
              return true; // Show all meters for NCR view
            }
            return m.city_id === selectedCity;
          })}
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          onMeterClick={handleMeterClick}
          runId={currentRunId}
          selectedCity={selectedCity}
        />
      )}

      {/* Drilldown Modal */}
      {isMeterModalOpen && selectedMeter && (
        <DrilldownModal
          meter={selectedMeter}
          isOpen={isMeterModalOpen}
          onClose={handleCloseModal}
          isDataReady={true}
        />
      )}
    </div>
  );
};

export default AnomalyDashboard;
