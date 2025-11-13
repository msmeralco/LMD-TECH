/**
 * Frontend Types - Matches Backend API Response
 * Data comes from backend (GeoJSON format)
 */

// Meter type - matches backend API response
export interface Meter {
  id: string;
  meterNumber: string;
  transformerId: string;
  barangay: string;
  city_id: string; // ✅ CRITICAL: Required for city grouping in DistrictMapView
  feeder: string;
  riskLevel: 'high' | 'medium' | 'low';
  riskBand: string;
  anomalyScore: number;
  position: [number, number]; // [lat, lng]
  consumption: number[];
  consumptionData?: Array<{
    month: string;
    year: string;
    kwh: number;
    kva: number;
  }>;
  anomalyNotes?: string;
}

// District type for map view
export interface District {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
  totalTransformers: number;
  riskLevel: 'high' | 'medium' | 'low';
  totalMeters: number;
}

// Anomaly type - FIXED to match usage in AnomalyMapper
export interface Anomaly {
  id: string;
  meterId: string;
  position: [number, number];
  riskLevel: 'high' | 'medium' | 'low';
  riskBand: string;
  anomalyScore: number;
  barangay: string;
  feeder: string;
  type: string;
  description: string;
}

// Utility functions (keep these - they work with any Meter array)
export const getUniqueBarangays = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.barangay))).sort();
};

export const getUniqueFeeders = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.feeder))).sort();
};

export const getUniqueRiskBands = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.riskBand))).sort();
};