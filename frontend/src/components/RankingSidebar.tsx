import React, { useState, useEffect, useMemo } from "react";
import { Meter } from "../data/types";
import { apiService } from "../services/api";

interface RankingSidebarProps {
  meters: Meter[];
  isOpen: boolean;
  onClose: () => void;
  onMeterClick: (meter: Meter) => void;
  runId: string | null;
  selectedCity: string;
}

interface FilterOptions {
  barangays: string[];
  transformers: string[];
  risk_levels: string[];
}

const RankingSidebar: React.FC<RankingSidebarProps> = ({
  meters,
  isOpen,
  onClose,
  onMeterClick,
  runId,
  selectedCity,
}) => {
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(
    null
  );
  const [selectedBarangay, setSelectedBarangay] = useState<string>("all");
  const [selectedTransformer, setSelectedTransformer] = useState<string>("all");
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>("all");
  const [isExporting, setIsExporting] = useState(false);

  // Load filter options (filtered by selected city)
  useEffect(() => {
    if (runId) {
      // Pass selectedCity to filter barangays by city
      const cityId = selectedCity !== "ncr" ? selectedCity : undefined;
      apiService.getFilterOptions(runId, cityId).then(setFilterOptions);
    }
  }, [runId, selectedCity]);

  // Filter and rank meters
  const rankedMeters = useMemo(() => {
    let filtered = [...meters];

    // Apply filters
    if (selectedBarangay !== "all") {
      filtered = filtered.filter((m) => m.barangay === selectedBarangay);
    }

    if (selectedTransformer !== "all") {
      filtered = filtered.filter(
        (m) => m.transformerId === selectedTransformer
      );
    }

    if (selectedRiskLevel !== "all") {
      filtered = filtered.filter(
        (m) => m.riskLevel === selectedRiskLevel.toLowerCase()
      );
    }

    // Sort by anomaly score (highest first)
    return filtered.sort((a, b) => b.anomalyScore - a.anomalyScore);
  }, [meters, selectedBarangay, selectedTransformer, selectedRiskLevel]);

  // Export CSV with filters
  const handleExport = async () => {
    if (!runId) return;

    setIsExporting(true);
    try {
      await apiService.exportCSV(
        runId,
        "meter",
        selectedBarangay !== "all" ? selectedBarangay : undefined,
        selectedTransformer !== "all" ? selectedTransformer : undefined,
        selectedRiskLevel !== "all" ? selectedRiskLevel : undefined
      );
    } catch (error) {
      console.error("Export failed:", error);
      alert("Failed to export CSV. Please try again.");
    } finally {
      setIsExporting(false);
    }
  };

  const getRiskBadge = (riskLevel: string) => {
    const badges = {
      high: "bg-red-100 text-red-700 border-red-300",
      medium: "bg-yellow-100 text-yellow-700 border-yellow-300",
      low: "bg-green-100 text-green-700 border-green-300",
    };
    return badges[riskLevel as keyof typeof badges] || badges.low;
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/20 z-[1100]" onClick={onClose} />

      {/* Sidebar */}
      <div className="fixed right-0 top-0 bottom-0 w-[450px] bg-white shadow-lg z-[1200] flex flex-col">
        {/* Header */}
        <div className="bg-orange-500 text-white p-5 border-b border-orange-600">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold">Meter Rankings</h2>
            <button
              onClick={onClose}
              className="text-white/80 hover:text-white transition-colors"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <p className="text-white/90 text-sm">
            Showing {rankedMeters.length} of {meters.length} meters
          </p>
        </div>

        {/* Filters */}
        <div className="p-4 bg-gray-50 border-b space-y-3">
          {/* Barangay Filter */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 mb-1">
              📍 Barangay
            </label>
            <select
              value={selectedBarangay}
              onChange={(e) => setSelectedBarangay(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            >
              <option value="all">All Barangays</option>
              {filterOptions?.barangays.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>

          {/* Transformer/Feeder Filter */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 mb-1">
              ⚡ Feeder / Transformer
            </label>
            <select
              value={selectedTransformer}
              onChange={(e) => setSelectedTransformer(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            >
              <option value="all">All Feeders</option>
              {filterOptions?.transformers.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          {/* Risk Level Filter */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 mb-1">
              🎯 Risk Band
            </label>
            <select
              value={selectedRiskLevel}
              onChange={(e) => setSelectedRiskLevel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            >
              <option value="all">All Risk Levels</option>
              <option value="HIGH">High Risk</option>
              <option value="MEDIUM">Medium Risk</option>
              <option value="LOW">Low Risk</option>
            </select>
          </div>

          {/* Export Button */}
          <button
            onClick={handleExport}
            disabled={isExporting || rankedMeters.length === 0}
            className="w-full bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isExporting ? (
              <>
                <svg
                  className="animate-spin h-5 w-5"
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
                Exporting...
              </>
            ) : (
              <>
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
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                Export Filtered CSV
              </>
            )}
          </button>
        </div>

        {/* Meter List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {rankedMeters.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <svg
                className="w-16 h-16 mx-auto mb-4 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
                />
              </svg>
              <p className="font-medium">No meters match filters</p>
              <p className="text-sm mt-1">Try adjusting your filter criteria</p>
            </div>
          ) : (
            rankedMeters.map((meter, index) => (
              <div
                key={meter.id}
                onClick={() => onMeterClick(meter)}
                className="bg-white border border-gray-200 rounded-lg p-3 hover:border-orange-400 hover:shadow-md transition-all cursor-pointer"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="bg-gray-100 text-gray-700 font-bold text-xs px-2 py-1 rounded">
                      #{index + 1}
                    </span>
                    <span className="font-semibold text-sm text-gray-900">
                      {meter.meterNumber}
                    </span>
                  </div>
                  <span
                    className={`text-xs font-semibold px-2 py-1 rounded border ${getRiskBadge(
                      meter.riskLevel
                    )}`}
                  >
                    {meter.riskLevel.toUpperCase()}
                  </span>
                </div>

                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>📍 Barangay:</span>
                    <span className="font-medium text-gray-900">
                      {meter.barangay}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>⚡ Feeder:</span>
                    <span className="font-medium text-gray-900">
                      {meter.transformerId}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>⚠️ Anomaly Score:</span>
                    <span className="font-bold text-orange-600">
                      {(meter.anomalyScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <style>{`
        @keyframes slideInRight {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-slideInRight {
          animation: slideInRight 0.3s ease-out forwards;
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out forwards;
        }
      `}</style>
    </>
  );
};

export default RankingSidebar;
