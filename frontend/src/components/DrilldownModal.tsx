import React, { useEffect, useState } from 'react';
import { Meter } from '../data/types';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface DrilldownModalProps {
  meter: Meter;
  isOpen: boolean;
  onClose: () => void;
  isDataReady: boolean;
}

const DrilldownModal: React.FC<DrilldownModalProps> = ({ meter, isOpen, onClose, isDataReady }) => {
  const [isChartReady, setIsChartReady] = useState(false);

  useEffect(() => {
    if (isOpen && meter) {
      console.log('📊 Modal opened for meter:', meter.meterNumber);
      console.log('📈 Consumption data:', meter.consumptionData);
      console.log('📉 Raw consumption:', meter.consumption);
      
      // Delay chart rendering to ensure DOM is ready
      setTimeout(() => {
        setIsChartReady(true);
      }, 200);
    } else {
      setIsChartReady(false);
    }
    
    return () => {
      setIsChartReady(false);
    };
  }, [isOpen, meter]);

  if (!isOpen || !meter) return null;

  // ✅ Create consumptionData with fallback
  const consumptionData = meter.consumptionData && meter.consumptionData.length > 0
    ? meter.consumptionData
    : meter.consumption.map((kwh, index) => ({
        month: new Date(2024, index).toLocaleString('default', { month: 'short' }),
        year: '2024',
        kwh: kwh,
        kva: kwh / 0.9,
      }));

  console.log('✅ Using consumption data:', consumptionData);

  // Calculate statistics
  const avgConsumption = consumptionData.length > 0
    ? Math.round(consumptionData.reduce((sum, d) => sum + d.kwh, 0) / consumptionData.length)
    : 0;

  const peakConsumption = consumptionData.length > 0
    ? Math.max(...consumptionData.map(d => d.kwh))
    : 0;

  // ✅ Prepare Line Chart Data
  const lineChartData = {
    labels: consumptionData.map(d => `${d.month} ${d.year}`),
    datasets: [
      {
        label: 'Consumption (kWh)',
        data: consumptionData.map(d => d.kwh),
        borderColor: '#FF6F00',
        backgroundColor: 'rgba(255, 111, 0, 0.1)',
        borderWidth: 3,
        tension: 0.4,
        fill: true,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBackgroundColor: '#FF6F00',
      },
    ],
  };

  // ✅ Prepare Bar Chart Data
  const barChartData = {
    labels: consumptionData.map(d => d.month),
    datasets: [
      {
        label: 'Monthly Consumption (kWh)',
        data: consumptionData.map(d => d.kwh),
        backgroundColor: consumptionData.map((d) => {
          const avg = consumptionData.reduce((sum, item) => sum + item.kwh, 0) / consumptionData.length;
          return d.kwh > avg * 1.3 ? 'rgba(255, 111, 0, 0.8)' : 'rgba(255, 111, 0, 0.4)';
        }),
        borderColor: '#FF6F00',
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2.5,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 10,
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 12,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          font: {
            size: 11,
          },
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          font: {
            size: 11,
          },
        },
      },
    },
  };

  return (
    <div 
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black bg-opacity-50 p-4" 
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-meralco-orange text-white p-6 flex justify-between items-center z-10">
          <div>
            <h2 className="text-2xl font-bold">{meter.meterNumber}</h2>
            <p className="text-sm opacity-90">
              {meter.barangay} • Transformer: {meter.transformerId}
            </p>
            <p className="text-xs opacity-75 mt-1">
              ✓ Graph data ready - {consumptionData.length} months consumption trends loaded
            </p>
          </div>
          <button 
            onClick={onClose} 
            className="text-white hover:bg-white hover:bg-opacity-20 rounded-full w-10 h-10 flex items-center justify-center text-3xl font-bold transition-colors"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Anomaly Score Alert */}
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <h3 className="text-lg font-bold text-red-800 mb-1">Anomaly Score</h3>
                <p className="text-sm text-red-600">
                  {meter.anomalyNotes || `High anomaly score detected (${(meter.anomalyScore * 100).toFixed(1)}%). Recommend field inspection.`}
                </p>
              </div>
              <div className="text-right ml-4">
                <p className="text-5xl font-bold text-red-600">{(meter.anomalyScore * 100).toFixed(0)}%</p>
                <p className="text-sm text-red-500 font-medium mt-1">{meter.riskBand}</p>
              </div>
            </div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-4 border border-orange-200">
              <p className="text-xs text-orange-700 font-medium mb-1">Average Consumption</p>
              <p className="text-2xl font-bold text-orange-900">{avgConsumption} <span className="text-sm">kWh</span></p>
            </div>
            <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4 border border-red-200">
              <p className="text-xs text-red-700 font-medium mb-1">Peak Consumption</p>
              <p className="text-2xl font-bold text-red-900">{peakConsumption} <span className="text-sm">kWh</span></p>
            </div>
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
              <p className="text-xs text-blue-700 font-medium mb-1">Total Period</p>
              <p className="text-2xl font-bold text-blue-900">{consumptionData.length} <span className="text-sm">months</span></p>
            </div>
          </div>

          {/* Charts */}
          {consumptionData.length > 0 ? (
            <>
              {/* Line Chart */}
              <div className="mb-6 bg-white p-6 rounded-lg border-2 border-gray-200 shadow-sm">
                <h3 className="text-lg font-bold text-meralco-black mb-4 flex items-center">
                  📈 12-Month Consumption Trend
                </h3>
                {isChartReady ? (
                  <div className="w-full" style={{ height: '300px' }}>
                    <Line data={lineChartData} options={chartOptions} />
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-meralco-orange mx-auto mb-3"></div>
                      <p>Loading chart...</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Bar Chart */}
              <div className="bg-white p-6 rounded-lg border-2 border-gray-200 shadow-sm">
                <h3 className="text-lg font-bold text-meralco-black mb-4 flex items-center">
                  📊 Monthly Consumption Comparison
                </h3>
                {isChartReady ? (
                  <div className="w-full" style={{ height: '300px' }}>
                    <Bar data={barChartData} options={chartOptions} />
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-meralco-orange mx-auto mb-3"></div>
                      <p>Loading chart...</p>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-gray-500 text-lg">⚠️ No consumption data available</p>
              <p className="text-gray-400 text-sm mt-2">Upload a CSV file with consumption history to view trends</p>
            </div>
          )}

          {/* Close Button */}
          <div className="mt-6 flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-8 py-3 bg-meralco-orange text-white font-medium rounded-md hover:bg-opacity-90 transition-colors shadow-md"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DrilldownModal;

