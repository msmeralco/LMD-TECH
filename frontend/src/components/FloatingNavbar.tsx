import React from "react";

interface FloatingNavbarProps {
  totalMeters: number;
  highRiskCount: number;
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
  onUploadClick: () => void;
  hasData: boolean;
  selectedCity: string | null;
}

const FloatingNavbar: React.FC<FloatingNavbarProps> = ({
  totalMeters,
  highRiskCount,
  onToggleSidebar,
  isSidebarOpen,
  onUploadClick,
  hasData,
  selectedCity,
}) => {
  const highRiskPercentage =
    totalMeters > 0 ? ((highRiskCount / totalMeters) * 100).toFixed(1) : "0.0";

  return (
    <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-[1000]">
      <div className="bg-white backdrop-blur-sm rounded-lg shadow-lg px-6 py-3 flex items-center gap-4 border border-gray-200">
        {/* Logo/Title */}
        <div className="flex items-center gap-2">
          <div className="bg-orange-500 p-2 rounded">
            <svg
              className="w-5 h-5 text-white"
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
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              GhostLoad Mapper
            </h2>
            <p className="text-xs text-gray-500">Anomaly Detection</p>
          </div>
        </div>

        {/* Divider */}
        <div className="h-10 w-px bg-gray-300"></div>

        {/* Upload Button */}
        <button
          onClick={onUploadClick}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors font-medium text-sm"
          title="Upload CSV"
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
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <span>Upload</span>
        </button>

        {/* Stats - Only show if data loaded */}
        {hasData && (
          <>
            {/* Divider */}
            <div className="h-10 w-px bg-gray-300"></div>

            {/* Total Meters */}
            <div className="flex items-center gap-2">
              <svg
                className="w-5 h-5 text-blue-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                />
              </svg>
              <div>
                <p className="text-lg font-semibold text-gray-900">
                  {totalMeters.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500">Total Meters</p>
              </div>
            </div>

            {/* High Risk */}
            <div className="flex items-center gap-2">
              <svg
                className="w-5 h-5 text-red-600"
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
              <div>
                <div className="flex items-baseline gap-2">
                  <p className="text-lg font-semibold text-red-600">
                    {highRiskCount}
                  </p>
                  <span className="text-xs font-medium text-red-500">
                    ({highRiskPercentage}%)
                  </span>
                </div>
                <p className="text-xs text-gray-500">High Risk</p>
              </div>
            </div>

            {/* Sidebar Toggle - Only show if city is selected */}
            {selectedCity && (
              <>
                <div className="h-10 w-px bg-gray-300"></div>
                <button
                  onClick={onToggleSidebar}
                  className={`flex items-center gap-2 px-4 py-2 rounded font-medium transition-colors text-sm ${
                    isSidebarOpen
                      ? "bg-orange-500 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                  title="Toggle Rankings"
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
                      d="M4 6h16M4 12h16M4 18h16"
                    />
                  </svg>
                  <span>Rankings</span>
                </button>
              </>
            )}

            {/* City Selected Indicator */}
            {selectedCity && (
              <div className="bg-gray-50 px-3 py-1 rounded border border-gray-300">
                <p className="text-xs text-gray-700 font-medium">
                  {selectedCity.toUpperCase()}
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default FloatingNavbar;
