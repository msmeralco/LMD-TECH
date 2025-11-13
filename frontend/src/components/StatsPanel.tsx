import React from "react";
import { ResultsData } from "../services/api";
import { CITY_METADATA, getBarangayById } from "../data/cityMetadata";

interface StatsPanelProps {
  resultsData: ResultsData;
}

/**
 * Stats Panel Component
 * Displays summary statistics after CSV upload
 * Shows cities and barangays grouped correctly using barangay_id
 */
const StatsPanel: React.FC<StatsPanelProps> = ({ resultsData }) => {
  // Group barangays by city with proper ID handling
  const citiesWithBarangays = React.useMemo(() => {
    const cityMap = new Map<string, Set<string>>();

    resultsData.barangays.forEach((barangay) => {
      const cityId = barangay.city_id;
      if (!cityMap.has(cityId)) {
        cityMap.set(cityId, new Set());
      }
      cityMap.get(cityId)?.add(barangay.barangay);
    });

    return Array.from(cityMap.entries()).map(([cityId, barangays]) => ({
      cityId,
      cityName: CITY_METADATA[cityId]?.name || cityId,
      barangayCount: barangays.size,
      barangays: Array.from(barangays).sort(),
    }));
  }, [resultsData.barangays]);

  return (
    <div className="absolute top-24 left-6 z-[999] bg-white rounded-lg shadow-xl border border-gray-200 max-w-md">
      <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-orange-50 to-orange-100">
        <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
          <svg
            className="w-5 h-5 text-orange-600"
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
          Data Summary
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Analysis ready • {resultsData.total_cities} cities detected
        </p>
      </div>

      <div className="p-4 max-h-96 overflow-y-auto">
        {/* Overall Stats */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-blue-50 rounded-lg p-3">
            <p className="text-xs text-blue-600 font-medium">Total Meters</p>
            <p className="text-2xl font-bold text-blue-900">
              {resultsData.total_meters.toLocaleString()}
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-3">
            <p className="text-xs text-purple-600 font-medium">Transformers</p>
            <p className="text-2xl font-bold text-purple-900">
              {resultsData.total_transformers.toLocaleString()}
            </p>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <p className="text-xs text-green-600 font-medium">Cities</p>
            <p className="text-2xl font-bold text-green-900">
              {resultsData.total_cities}
            </p>
          </div>
          <div className="bg-orange-50 rounded-lg p-3">
            <p className="text-xs text-orange-600 font-medium">Barangays</p>
            <p className="text-2xl font-bold text-orange-900">
              {resultsData.total_barangays}
            </p>
          </div>
        </div>

        {/* Cities & Barangays List */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <svg
              className="w-4 h-4 text-gray-500"
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
            Coverage by City
          </h4>

          {citiesWithBarangays.map(
            ({ cityId, cityName, barangayCount, barangays }) => (
              <div
                key={cityId}
                className="border border-gray-200 rounded-lg p-3 hover:border-orange-300 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-semibold text-gray-900">{cityName}</h5>
                  <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full font-medium">
                    {barangayCount} barangay{barangayCount !== 1 ? "s" : ""}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {barangays.slice(0, 5).map((barangay) => (
                    <span
                      key={barangay}
                      className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded border border-blue-200"
                    >
                      {barangay}
                    </span>
                  ))}
                  {barangayCount > 5 && (
                    <span className="text-xs text-gray-500 px-2 py-0.5">
                      +{barangayCount - 5} more
                    </span>
                  )}
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;
