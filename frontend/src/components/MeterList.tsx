import React, { useState, useMemo } from 'react';
import { Meter } from '../data/mockData';
import { getUniqueBarangays, getUniqueFeeders, getUniqueRiskBands } from '../data/mockData';

interface MeterListProps {
  meters: Meter[];
  onMeterClick: (meter: Meter) => void;
  isDataReady: boolean;
}

/**
 * Meter List Component
 * Displays ranked list of suspicious meters with filtering capabilities
 */
const MeterList: React.FC<MeterListProps> = ({ meters, onMeterClick, isDataReady }) => {
  const [selectedBarangay, setSelectedBarangay] = useState<string>('all');
  const [selectedFeeder, setSelectedFeeder] = useState<string>('all');
  const [selectedRiskBand, setSelectedRiskBand] = useState<string>('all');

  // Get unique filter values
  const barangays = useMemo(() => getUniqueBarangays(meters), [meters]);
  const feeders = useMemo(() => getUniqueFeeders(meters), [meters]);
  const riskBands = useMemo(() => getUniqueRiskBands(meters), [meters]);

  // Filter meters based on selected filters
  const filteredMeters = useMemo(() => {
    return meters.filter((meter) => {
      const matchesBarangay = selectedBarangay === 'all' || meter.barangay === selectedBarangay;
      const matchesFeeder = selectedFeeder === 'all' || meter.feeder === selectedFeeder;
      const matchesRiskBand = selectedRiskBand === 'all' || meter.riskBand === selectedRiskBand;
      
      return matchesBarangay && matchesFeeder && matchesRiskBand;
    }).sort((a, b) => b.anomalyScore - a.anomalyScore); // Sort by anomaly score descending
  }, [meters, selectedBarangay, selectedFeeder, selectedRiskBand]);

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
    <div className="flex flex-col h-full">
      {/* Filter Section */}
      <div className="p-4 border-b border-meralco-light-gray bg-meralco-light-gray space-y-3">
        <div>
          <label className="block text-xs font-medium text-meralco-black mb-1">
            Barangay
          </label>
          <select
            value={selectedBarangay}
            onChange={(e) => setSelectedBarangay(e.target.value)}
            className="w-full px-2 py-1 text-xs border border-meralco-light-gray rounded focus:outline-none focus:ring-2 focus:ring-meralco-orange"
          >
            <option value="all">All Barangays</option>
            {barangays.map((barangay) => (
              <option key={barangay} value={barangay}>
                {barangay}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-meralco-black mb-1">
            Feeder
          </label>
          <select
            value={selectedFeeder}
            onChange={(e) => setSelectedFeeder(e.target.value)}
            className="w-full px-2 py-1 text-xs border border-meralco-light-gray rounded focus:outline-none focus:ring-2 focus:ring-meralco-orange"
          >
            <option value="all">All Feeders</option>
            {feeders.map((feeder) => (
              <option key={feeder} value={feeder}>
                {feeder}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-meralco-black mb-1">
            Risk Band
          </label>
          <select
            value={selectedRiskBand}
            onChange={(e) => setSelectedRiskBand(e.target.value)}
            className="w-full px-2 py-1 text-xs border border-meralco-light-gray rounded focus:outline-none focus:ring-2 focus:ring-meralco-orange"
          >
            <option value="all">All Risk Bands</option>
            {riskBands.map((riskBand) => (
              <option key={riskBand} value={riskBand}>
                {riskBand}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Meter List */}
      <div className="flex-1 overflow-y-auto">
        {!isDataReady ? (
          <div className="p-4 text-center text-meralco-gray text-sm">
            Click "Ready Graph Data" to load meter information
          </div>
        ) : filteredMeters.length === 0 ? (
          <div className="p-4 text-center text-meralco-gray text-sm">
            No meters found matching the selected filters
          </div>
        ) : (
          <div className="divide-y divide-meralco-light-gray">
            {filteredMeters.map((meter, index) => (
              <button
                key={meter.id}
                onClick={() => onMeterClick(meter)}
                className="w-full text-left p-4 hover:bg-meralco-light-gray transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-meralco-gray">#{index + 1}</span>
                      <h4 className="font-semibold text-sm text-meralco-black">{meter.meterNumber}</h4>
                    </div>
                    <p className="text-xs text-meralco-gray">
                      Transformer: {meter.transformerId}
                    </p>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs font-medium border ${getRiskLevelStyle(meter.riskLevel)}`}>
                    {meter.riskBand}
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-2">
                  <div className="text-xs text-meralco-gray">
                    <span className="font-medium">Score:</span> {(meter.anomalyScore * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-meralco-gray">
                    {meter.barangay} • {meter.feeder}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Summary Footer */}
      <div className="p-3 border-t border-meralco-light-gray bg-meralco-light-gray">
        <p className="text-xs text-meralco-gray text-center">
          Showing {filteredMeters.length} of {meters.length} meters
        </p>
      </div>
    </div>
  );
};

export default MeterList;

