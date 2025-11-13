import React, { useState, useEffect } from "react";
import DistrictMapView from "./DistrictMapView";
import MeterList from "./MeterList";
import DrilldownModal from "./DrilldownModal";
import FileUpload from "./FileUpload";
import FloatingNavbar from "./FloatingNavbar";
import RankingSidebar from "./RankingSidebar";
import { apiService, ResultsData, Meter as APIMeter } from "../services/api";
import { Meter } from "../data/types";

const AnomalyDashboard: React.FC = () => {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMeter, setSelectedMeter] = useState<Meter | null>(null);
  const [isMeterModalOpen, setIsMeterModalOpen] = useState(false);
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

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

    return {
      id: apiMeter.meter_id,
      meterNumber: apiMeter.meter_id,
      transformerId: apiMeter.transformer_id,
      barangay: apiMeter.barangay,
      feeder: apiMeter.transformer_id,
      riskLevel: riskLevel,
      riskBand: riskBand,
      anomalyScore: apiMeter.anomaly_score,
      position: [apiMeter.lat, apiMeter.lon],
      consumption: apiMeter.monthly_consumptions || [],
      consumptionData: consumptionData, // ✅ Add this
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

  // Handle district click
  const handleDistrictClick = (districtId: string) => {
    setSelectedDistrict(districtId);
  };

  // Handle modal close
  const handleCloseModal = () => {
    console.log("❌ Closing modal"); // Debug log
    setIsMeterModalOpen(false);
    setSelectedMeter(null);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Floating Navbar (Only show when results loaded) */}
      {resultsData && (
        <FloatingNavbar
          totalMeters={resultsData.total_meters}
          highRiskCount={resultsData.high_risk_count}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          isSidebarOpen={isSidebarOpen}
        />
      )}

      {/* Header (Only show when no results) */}
      {!resultsData && (
        <header className="bg-white border-b-2 border-meralco-orange shadow-sm">
          <div className="px-6 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-meralco-orange">
              🔍 GhostLoad Mapper
            </h1>
          </div>
        </header>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Upload Section */}
        {!resultsData && !isLoading && (
          <div className="flex-1 p-6">
            <FileUpload onUploadSuccess={handleUploadSuccess} />

            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-bold text-blue-900 mb-2">📋 How to use:</h3>
              <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
                <li>
                  Upload your{" "}
                  <code className="bg-blue-100 px-1 rounded">
                    meter_consumption.csv
                  </code>{" "}
                  file
                </li>
                <li>Backend will process data through ML pipeline</li>
                <li>Explore map and filter suspicious meters</li>
                <li>Click meters to see detailed analytics</li>
              </ol>
            </div>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex-1 flex items-center justify-center">
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
              <p className="text-lg text-gray-600">Loading results...</p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex-1 p-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
              <p className="font-semibold">Error loading results</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Main View - REAL BACKEND DATA */}
        {resultsData && !isLoading && (
          <div className="flex-1 flex relative">
            {/* Map - Full width (sidebar overlays it) */}
            <div className="flex-1">
              <DistrictMapView
                districts={getDistricts()}
                onDistrictClick={handleDistrictClick}
                selectedDistrict={selectedDistrict}
                meters={getMeters()}
                onMeterClick={handleMeterClick}
              />
            </div>
          </div>
        )}
      </div>

      {/* Ranking Sidebar */}
      {resultsData && (
        <RankingSidebar
          meters={getMeters()}
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          onMeterClick={handleMeterClick}
          runId={currentRunId}
        />
      )}

      {/* Drilldown Modal */}
      {isMeterModalOpen && selectedMeter && (
        <>
          {console.log("🎨 Rendering modal for:", selectedMeter.meterNumber)}
          <DrilldownModal
            meter={selectedMeter}
            isOpen={isMeterModalOpen}
            onClose={handleCloseModal}
            isDataReady={true}
          />
        </>
      )}
    </div>
  );
};

export default AnomalyDashboard;
