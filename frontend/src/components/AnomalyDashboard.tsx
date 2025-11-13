<<<<<<< HEAD
import React, { useState, useEffect } from 'react';
=======
import React, { useState, useEffect, useRef} from 'react';
>>>>>>> c1022578e0fdcbaf999805988775970a0e861cfa
import DistrictMapView from './DistrictMapView';
import AnomalyMapper from './AnomalyMapper';
import MeterList from './MeterList';
import DrilldownModal from './DrilldownModal';
import LocationPins from './LocationPins';
import { mockDistricts, mockAnomalies, mockMeters, Meter } from '../data/mockData';

/**
 * Main Anomaly Dashboard Component
 * Orchestrates the district map, anomaly overlay, meter list, and drilldown modal
 */
const AnomalyDashboard: React.FC = () => {
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const [selectedMeter, setSelectedMeter] = useState<Meter | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDataReady, setIsDataReady] = useState(false);
  const [anomalies, setAnomalies] = useState(mockAnomalies);
  const [meters, setMeters] = useState(mockMeters);

  // Handle district selection - triggers zoom and shows anomaly overlay
  const handleDistrictClick = (districtId: string) => {
    setSelectedDistrict(districtId);
  };

  // Handle meter selection - opens drilldown modal with auto data ready
  const handleMeterClick = (meter: Meter) => {
    setSelectedMeter(meter);
    setIsDataReady(true); // Auto-enable data when pin is clicked
    setIsModalOpen(true);
  };

  // Close modal handler
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedMeter(null);
  };

  // Reset view - clears district selection
  const handleResetView = () => {
    setSelectedDistrict(null);
    setSelectedMeter(null);
    setIsModalOpen(false);
  };

  // Graph Ready Button - simulates data fetching
  const handleGraphReady = () => {
    setIsDataReady(true);
    // Simulate data refresh - in real app, this would fetch from API
    setTimeout(() => {
      setAnomalies([...mockAnomalies]);
      setMeters([...mockMeters]);
    }, 500);
  };

  // Filter anomalies and meters by selected district
  const districtAnomalies = selectedDistrict
    ? anomalies.filter(a => {
        const district = mockDistricts.find(d => d.id === selectedDistrict);
        // Simple proximity check - in real app, use proper boundary checking
        return district && a.position;
      })
    : [];

  const districtMeters = selectedDistrict
    ? meters.filter(m => {
        // Filter meters that belong to the selected district
        // In real app, use proper district boundaries
        return true; // For demo, show all meters
      })
    : [];

  return (
    <div className="h-screen w-screen flex flex-col bg-white overflow-hidden">
      {/* Header */}
      <header className="bg-white border-b-2 border-meralco-orange shadow-sm z-50">
        <div className="px-6 py-4 flex justify-between items-center">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-meralco-orange tracking-wide">
              MERALCO
            </h1>
            <span className="ml-4 text-sm text-meralco-gray">Anomaly Detection Dashboard</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={handleGraphReady}
              className="px-4 py-2 bg-meralco-orange text-white rounded-md hover:bg-opacity-90 transition-colors font-medium"
            >
              Ready Graph Data
            </button>
            <button
              onClick={handleResetView}
              className="px-4 py-2 bg-white text-meralco-orange border border-meralco-orange rounded-md hover:bg-meralco-orange hover:bg-opacity-10 transition-colors font-medium"
            >
              Reset View
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Map Section */}
        <div className="flex-1 relative">
          <DistrictMapView
            districts={mockDistricts}
            onDistrictClick={handleDistrictClick}
            selectedDistrict={selectedDistrict}
          >
            {/* Location Pins - Always visible, shows all meter locations */}
            <LocationPins
              meters={meters}
              onPinClick={handleMeterClick}
            />
            
            {/* Anomaly Mapper Overlay - appears when district is selected */}
            {selectedDistrict && (
              <AnomalyMapper
                anomalies={districtAnomalies}
                onAnomalyClick={(anomaly: any) => {
                  const meter = meters.find(m => m.id === anomaly.meterId);
                  if (meter) handleMeterClick(meter);
                }}
              />
            )}
          </DistrictMapView>
        </div>

        {/* Sidebar - Meter List */}
        {selectedDistrict && (
          <div className="w-96 bg-white border-l border-meralco-light-gray shadow-lg overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-meralco-light-gray bg-meralco-light-gray">
              <h2 className="text-lg font-semibold text-meralco-black">Suspicious Meter List</h2>
              <p className="text-xs text-meralco-gray mt-1">
                {mockDistricts.find(d => d.id === selectedDistrict)?.name}
              </p>
            </div>
            <MeterList
              meters={districtMeters}
              onMeterClick={handleMeterClick}
              isDataReady={isDataReady}
            />
          </div>
        )}
      </div>

      {/* Drilldown Modal */}
      {isModalOpen && selectedMeter && (
        <DrilldownModal
          meter={selectedMeter}
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          isDataReady={isDataReady}
        />
      )}
    </div>
  );
};

export default AnomalyDashboard;

