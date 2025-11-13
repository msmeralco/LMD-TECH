/**
 * API Service Layer
 * Connects frontend to FastAPI backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface RunResponse {
  run_id: string;
  status: string;
  processing_time_seconds: number;
}

export interface ResultsData {
  run_id: string;
  timestamp: string;
  status: string;
  total_meters: number;
  total_transformers: number;
  high_risk_count: number;
  cities: City[];
  barangays: Barangay[];
  transformers: Transformer[];
  meters: Meter[];
  high_risk_summary: HighRiskSummary;
}

export interface City {
  city_id: string;
  city_name: string;
  lat: number;
  lon: number;
  total_transformers: number;
  high_risk_count: number;
  avg_risk_score: number;
}

export interface Barangay {
  barangay: string;
  city_id: string;
  lat: number;
  lon: number;
  total_transformers: number;
  high_risk_count: number;
  avg_risk_score: number;
}

export interface Transformer {
  transformer_id: string;
  barangay: string;
  city_id: string;
  lat: number;
  lon: number;
  total_meters: number;
  suspicious_meter_count: number;
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
  avg_anomaly_score: number;
  avg_consumption: number;
  median_consumption: number;
  capacity_kVA: number;
}

export interface Meter {
  meter_id: string;
  transformer_id: string;
  barangay: string;
  city_id: string;
  lat: number;
  lon: number;
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
  anomaly_score: number;
  confidence: number;
  explanation: string;
  customer_class?: string;
  monthly_consumptions?: number[];
}

export interface HighRiskSummary {
  most_anomalous_city: string;
  most_anomalous_barangay: string;
  most_anomalous_transformer: string;
  top_10_transformers: Transformer[];
}

export interface RecentRun {
  run_id: string;
  timestamp: string;
  status: string;
  total_meters: number;
}

class APIService {
  /**
   * Upload CSV and run ML pipeline
   */
  async uploadAndRun(file: File): Promise<RunResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/run`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to process file');
    }

    return response.json();
  }

  /**
   * Get results by run_id
   */
  async getResults(runId: string): Promise<ResultsData> {
    const response = await fetch(`${API_BASE_URL}/api/results/${runId}`);

    if (!response.ok) {
      throw new Error('Failed to fetch results');
    }

    return response.json();
  }

  /**
   * Get transformers as GeoJSON
   */
  async getTransformersGeoJSON(runId?: string): Promise<any> {
    const url = runId 
      ? `${API_BASE_URL}/api/transformers.geojson?run_id=${runId}`
      : `${API_BASE_URL}/api/transformers.geojson`;
    
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error('Failed to fetch GeoJSON');
    }

    return response.json();
  }

  /**
   * Export results as CSV
   */
  async exportCSV(runId: string, level: 'transformer' | 'barangay' | 'meter'): Promise<Blob> {
    const response = await fetch(
      `${API_BASE_URL}/api/export/${runId}?level=${level}`
    );

    if (!response.ok) {
      throw new Error('Failed to export CSV');
    }

    return response.blob();
  }

  /**
   * List recent runs
   */
  async getRecentRuns(limit: number = 10): Promise<RecentRun[]> {
    const response = await fetch(`${API_BASE_URL}/api/runs?limit=${limit}`);

    if (!response.ok) {
      throw new Error('Failed to fetch recent runs');
    }

    return response.json();
  }

  /**
   * Delete a run
   */
  async deleteRun(runId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error('Failed to delete run');
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.json();
  }
}

export const apiService = new APIService();