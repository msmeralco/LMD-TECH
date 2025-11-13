import React from "react";

interface FloatingNavbarProps {
  totalMeters: number;
  highRiskCount: number;
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

const FloatingNavbar: React.FC<FloatingNavbarProps> = ({
  totalMeters,
  highRiskCount,
  onToggleSidebar,
  isSidebarOpen,
}) => {
  const highRiskPercentage =
    totalMeters > 0 ? ((highRiskCount / totalMeters) * 100).toFixed(1) : "0.0";

  return (
    <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-[1000] animate-slideDown">
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl px-8 py-4 flex items-center gap-8 border border-gray-200">
        {/* Logo/Title */}
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 p-2 rounded-lg">
            <svg
              className="w-6 h-6 text-white"
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
            <h2 className="text-lg font-bold text-gray-900">
              GhostLoad Mapper
            </h2>
            <p className="text-xs text-gray-500">Real-time Anomaly Detection</p>
          </div>
        </div>

        {/* Divider */}
        <div className="h-12 w-px bg-gray-300"></div>

        {/* Stats */}
        <div className="flex items-center gap-6">
          {/* Total Meters */}
          <div className="flex items-center gap-3">
            <div className="bg-blue-50 p-2.5 rounded-lg">
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
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">
                {totalMeters.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500 font-medium">Total Meters</p>
            </div>
          </div>

          {/* High Risk */}
          <div className="flex items-center gap-3">
            <div className="bg-red-50 p-2.5 rounded-lg">
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
            </div>
            <div>
              <div className="flex items-baseline gap-2">
                <p className="text-2xl font-bold text-red-600">
                  {highRiskCount}
                </p>
                <span className="text-xs font-semibold text-red-500 bg-red-50 px-2 py-0.5 rounded-full">
                  {highRiskPercentage}%
                </span>
              </div>
              <p className="text-xs text-gray-500 font-medium">High Risk</p>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-12 w-px bg-gray-300"></div>

        {/* Sidebar Toggle */}
        <button
          onClick={onToggleSidebar}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all duration-200 ${
            isSidebarOpen
              ? "bg-orange-500 text-white shadow-md"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
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
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
          <span>{isSidebarOpen ? "Hide" : "Show"} Rankings</span>
        </button>
      </div>
    </div>
  );
};

export default FloatingNavbar;
