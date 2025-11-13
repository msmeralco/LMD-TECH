import React, { useEffect, useRef } from 'react';
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
import { Line, Bar } from 'react-chartjs-2';
import { Meter } from '../data/mockData';

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

/**
 * Drilldown Modal Component
 * Displays detailed meter information with consumption trends using Chart.js
 */
const DrilldownModal: React.FC<DrilldownModalProps> = ({
  meter,
  isOpen,
  onClose,
  isDataReady,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  // Close modal on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  // Prepare chart data
  const chartData = {
    labels: meter.consumptionData.map(d => `${d.month} ${d.year}`),
    datasets: [
      {
        label: 'Consumption (kWh)',
        data: meter.consumptionData.map(d => d.kwh),
        borderColor: '#FF6F00',
        backgroundColor: 'rgba(255, 111, 0, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: '#FF6F00',
        pointBorderColor: '#FFFFFF',
        pointBorderWidth: 2,
      },
      {
        label: 'Apparent Power (kVA)',
        data: meter.consumptionData.map(d => d.kva * 1000), // Convert to same scale
        borderColor: '#666666',
        backgroundColor: 'rgba(102, 102, 102, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6,
        yAxisID: 'y1',
      },
    ],
  };

  const barChartData = {
    labels: meter.consumptionData.map(d => d.month),
    datasets: [
      {
        label: 'Monthly Consumption (kWh)',
        data: meter.consumptionData.map(d => d.kwh),
        backgroundColor: meter.consumptionData.map((d, i) => {
          // Color code based on anomaly detection
          const avg = meter.consumptionData.reduce((sum, item) => sum + item.kwh, 0) / meter.consumptionData.length;
          return d.kwh > avg * 1.3 ? 'rgba(255, 111, 0, 0.8)' : 'rgba(255, 111, 0, 0.4)';
        }),
        borderColor: '#FF6F00',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            size: 12,
          },
          color: '#333333',
        },
      },
      title: {
        display: true,
        text: '12-Month Consumption Trend',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
        color: '#000000',
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
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
        title: {
          display: true,
          text: 'Consumption (kWh)',
          font: {
            size: 12,
          },
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Apparent Power (kVA × 1000)',
          font: {
            size: 12,
          },
        },
        grid: {
          drawOnChartArea: false,
        },
      },
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Monthly Consumption Comparison',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
        color: '#000000',
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Consumption (kWh)',
          font: {
            size: 12,
          },
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-[2000] flex items-center justify-center p-4 animate-fadeIn">
      <div
        ref={modalRef}
        className="bg-white rounded-lg shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col animate-slideUp"
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b-2 border-meralco-orange bg-meralco-light-gray">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-meralco-black">{meter.meterNumber}</h2>
              <p className="text-sm text-meralco-gray mt-1">
                {meter.barangay} • {meter.feeder} • Transformer: {meter.transformerId}
              </p>
              {isDataReady && (
                <p className="text-xs text-meralco-orange mt-1 font-medium">
                  ✓ Graph data ready - 12 months consumption trends loaded
                </p>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-meralco-gray hover:text-meralco-black text-2xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-meralco-light-gray transition-colors"
            >
              ×
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {!isDataReady ? (
            <div className="text-center py-12 text-meralco-gray">
              <p>Click "Ready Graph Data" to load consumption data</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Anomaly Score Section */}
              <div className="bg-meralco-light-gray rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-meralco-black">Anomaly Score</h3>
                  <div className="flex items-center gap-2">
                    <div className="text-2xl font-bold text-meralco-orange">
                      {(meter.anomalyScore * 100).toFixed(0)}%
                    </div>
                    <div className={`px-3 py-1 rounded text-sm font-medium ${
                      meter.riskLevel === 'high' 
                        ? 'bg-red-100 text-red-800' 
                        : meter.riskLevel === 'medium'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {meter.riskBand}
                    </div>
                  </div>
                </div>
                <p className="text-sm text-meralco-gray">{meter.anomalyNotes}</p>
              </div>

              {/* Line Chart - Consumption Trend */}
              <div className="bg-white border border-meralco-light-gray rounded-lg p-4">
                <div style={{ height: '300px' }}>
                  <Line data={chartData} options={chartOptions} />
                </div>
              </div>

              {/* Bar Chart - Monthly Comparison */}
              <div className="bg-white border border-meralco-light-gray rounded-lg p-4">
                <div style={{ height: '250px' }}>
                  <Bar data={barChartData} options={barChartOptions} />
                </div>
              </div>

              {/* Statistics Summary */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-meralco-light-gray rounded-lg p-4">
                  <p className="text-xs text-meralco-gray mb-1">Average Consumption</p>
                  <p className="text-lg font-bold text-meralco-black">
                    {Math.round(
                      meter.consumptionData.reduce((sum, d) => sum + d.kwh, 0) / meter.consumptionData.length
                    )}{' '}
                    kWh
                  </p>
                </div>
                <div className="bg-meralco-light-gray rounded-lg p-4">
                  <p className="text-xs text-meralco-gray mb-1">Peak Consumption</p>
                  <p className="text-lg font-bold text-meralco-black">
                    {Math.max(...meter.consumptionData.map(d => d.kwh))} kWh
                  </p>
                </div>
                <div className="bg-meralco-light-gray rounded-lg p-4">
                  <p className="text-xs text-meralco-gray mb-1">Total Period</p>
                  <p className="text-lg font-bold text-meralco-black">
                    {meter.consumptionData.length} months
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-meralco-light-gray bg-meralco-light-gray">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-meralco-orange text-white rounded-md hover:bg-opacity-90 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DrilldownModal;

