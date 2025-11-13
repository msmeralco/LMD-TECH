import React from 'react';
import { Popup } from 'react-leaflet';
import { District } from '../data/types';

interface DistrictPopupProps {
  district: District;
}

/**
 * District Popup Component
 * Displays key district information in a styled popup
 * Shows district name, total transformers, risk level, and total meters
 */
const DistrictPopup: React.FC<DistrictPopupProps> = ({ district }) => {
  // Get risk level styling
  const getRiskLevelStyle = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  return (
    <Popup
      className="district-popup"
      autoPan={true}
      closeButton={true}
      autoClose={true}
      closeOnClick={true}
    >
      <div className="p-3 min-w-[220px]">
        {/* District Name */}
        <h3 className="text-lg font-bold text-meralco-black mb-3 border-b-2 border-meralco-orange pb-2">
          {district.name}
        </h3>
        
        {/* District Information Grid */}
        <div className="space-y-2">
          {/* Total Transformers */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-meralco-gray font-medium">Total Transformers:</span>
            <span className="text-sm font-bold text-meralco-black">{district.totalTransformers}</span>
          </div>
          
          {/* Total Meters */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-meralco-gray font-medium">Total Meters:</span>
            <span className="text-sm font-bold text-meralco-black">
              {district.totalMeters.toLocaleString()}
            </span>
          </div>
          
          {/* Risk Level */}
          <div className="flex items-center justify-between pt-2 border-t border-meralco-light-gray">
            <span className="text-sm text-meralco-gray font-medium">Risk Level:</span>
            <span className={`px-2 py-1 rounded text-xs font-medium border ${getRiskLevelStyle(district.riskLevel)}`}>
              {district.riskLevel.toUpperCase()}
            </span>
          </div>
        </div>
      </div>
    </Popup>
  );
};

export default DistrictPopup;

