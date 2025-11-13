import React, { useState } from "react";
import { ResultsData } from "../services/api";
import { CITY_METADATA, getBarangayById } from "../data/cityMetadata";

interface StatsPanelProps {
  resultsData: ResultsData;
}

/**
 * Collapsible Stats Panel Component
 * Displays summary statistics after CSV upload
 * Shows cities and barangays grouped correctly using barangay_id
 */
const StatsPanel: React.FC<StatsPanelProps> = ({ resultsData }) => {
  const [isExpanded, setIsExpanded] = useState(false);

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
    <div className="absolute top-24 left-4 z-[999]">
      {/* Collapsed State - Minimal UI */}
      {!isExpanded && (
        <button
          onClick={() => setIsExpanded(true)}
          className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 hover:shadow-xl transition-all hover:border-orange-300 group"
        >
          <div className="flex items-center gap-3">
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
            <div className="text-left">
              <div className="text-sm font-semibold text-gray-900">Data Summary</div>
              <div className="text-xs text-gray-500">
                {resultsData.total_meters.toLocaleString()} meters • {resultsData.total_cities} cities
              </div>
            </div>
            <svg
              className="w-4 h-4 text-gray-400 group-hover:text-orange-600 transition-colors"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </button>
      )}

      {/* Expanded State - Full Panel */}
      {isExpanded && (
        <div className="bg-white rounded-lg shadow-xl border border-gray-200 max-w-sm">
          <div className="p-3 border-b border-gray-200 bg-gradient-to-r from-orange-50 to-orange-100 flex items-center justify-between">
            <div className="flex items-center gap-2">
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
              <div>
                <h3 className="text-sm font-bold text-gray-900">Data Summary</h3>
                <p className="text-xs text-gray-600">
                  {resultsData.total_cities} cities
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsExpanded(false)}
              className="p-1 hover:bg-orange-200 rounded transition-colors"
            >
              <svg
                className="w-4 h-4 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="p-3 max-h-[70vh] overflow-y-auto">
            {/* Overall Stats */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="bg-blue-50 rounded-lg p-2">
                <p className="text-xs text-blue-600 font-medium">Meters</p>
                <p className="text-xl font-bold text-blue-900">
                  {resultsData.total_meters.toLocaleString()}
                </p>
              </div>
              <div className="bg-purple-50 rounded-lg p-2">
                <p className="text-xs text-purple-600 font-medium">Transformers</p>
                <p className="text-xl font-bold text-purple-900">
                  {resultsData.total_transformers.toLocaleString()}
                </p>
              </div>
              <div className="bg-green-50 rounded-lg p-2">
                <p className="text-xs text-green-600 font-medium">Cities</p>
                <p className="text-xl font-bold text-green-900">
                  {resultsData.total_cities}
                </p>
              </div>
              <div className="bg-orange-50 rounded-lg p-2">
                <p className="text-xs text-orange-600 font-medium">Barangays</p>
                <p className="text-xl font-bold text-orange-900">
                  {resultsData.total_barangays}
                </p>
              </div>
            </div>

            {/* Cities & Barangays List */}
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-gray-700 flex items-center gap-2">
                <svg
                  className="w-3 h-3 text-gray-500"
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
                    className="border border-gray-200 rounded-lg p-2 hover:border-orange-300 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <h5 className="text-sm font-semibold text-gray-900">{cityName}</h5>
                      <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full font-medium">
                        {barangayCount}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {barangays.slice(0, 3).map((barangay) => (
                        <span
                          key={barangay}
                          className="text-xs bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded"
                        >
                          {barangay}
                        </span>
                      ))}
                      {barangayCount > 3 && (
                        <span className="text-xs text-gray-500 px-1.5 py-0.5">
                          +{barangayCount - 3}
                        </span>
                      )}
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatsPanel;
