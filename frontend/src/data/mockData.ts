// Mock data structure for Meralco Anomaly Detection Dashboard

export interface District {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
  boundaries?: [number, number][];
  totalTransformers: number;
  riskLevel: 'high' | 'medium' | 'low';
  totalMeters: number;
}

export interface Anomaly {
  id: string;
  meterId: string;
  position: [number, number];
  riskLevel: 'high' | 'medium' | 'low';
  anomalyScore: number;
  transformerId: string;
  barangay: string;
  feeder: string;
  riskBand: string;
  description: string;
}

export interface Meter {
  id: string;
  meterNumber: string;
  transformerId: string;
  barangay: string;
  feeder: string;
  riskBand: string;
  anomalyScore: number;
  riskLevel: 'high' | 'medium' | 'low';
  position: [number, number];
  consumptionData: ConsumptionData[];
  anomalyNotes: string;
}

export interface ConsumptionData {
  month: string;
  year: number;
  kwh: number;
  kva: number;
}

// Mock Districts Data
export const mockDistricts: District[] = [
  {
    id: 'makati',
    name: 'Makati District',
    center: [14.5547, 121.0244],
    zoom: 14,
    totalTransformers: 45,
    riskLevel: 'high',
    totalMeters: 1250,
  },
  {
    id: 'quezon',
    name: 'Quezon City District',
    center: [14.6760, 121.0437],
    zoom: 14,
    totalTransformers: 62,
    riskLevel: 'medium',
    totalMeters: 1890,
  },
  {
    id: 'manila',
    name: 'Manila District',
    center: [14.5995, 120.9842],
    zoom: 14,
    totalTransformers: 38,
    riskLevel: 'high',
    totalMeters: 980,
  },
  {
    id: 'pasig',
    name: 'Pasig District',
    center: [14.5764, 121.0851],
    zoom: 14,
    totalTransformers: 52,
    riskLevel: 'medium',
    totalMeters: 1450,
  },
  {
    id: 'taguig',
    name: 'Taguig District',
    center: [14.5176, 121.0509],
    zoom: 14,
    totalTransformers: 41,
    riskLevel: 'low',
    totalMeters: 1120,
  },
];

// Mock Anomalies Data
export const mockAnomalies: Anomaly[] = [
  {
    id: 'anom-1',
    meterId: 'meter-001',
    position: [14.5550, 121.0250],
    riskLevel: 'high',
    anomalyScore: 0.92,
    transformerId: 'trans-001',
    barangay: 'Bel-Air',
    feeder: 'Feeder-A',
    riskBand: 'Critical',
    description: 'Unusual consumption spike detected',
  },
  {
    id: 'anom-2',
    meterId: 'meter-002',
    position: [14.5560, 121.0260],
    riskLevel: 'high',
    anomalyScore: 0.88,
    transformerId: 'trans-001',
    barangay: 'Bel-Air',
    feeder: 'Feeder-A',
    riskBand: 'High',
    description: 'Irregular load pattern',
  },
  {
    id: 'anom-3',
    meterId: 'meter-003',
    position: [14.5570, 121.0270],
    riskLevel: 'medium',
    anomalyScore: 0.75,
    transformerId: 'trans-002',
    barangay: 'San Antonio',
    feeder: 'Feeder-B',
    riskBand: 'Medium',
    description: 'Minor deviation from baseline',
  },
  {
    id: 'anom-4',
    meterId: 'meter-004',
    position: [14.5580, 121.0280],
    riskLevel: 'high',
    anomalyScore: 0.85,
    transformerId: 'trans-002',
    barangay: 'San Antonio',
    feeder: 'Feeder-B',
    riskBand: 'High',
    description: 'Potential meter tampering',
  },
  {
    id: 'anom-5',
    meterId: 'meter-005',
    position: [14.5590, 121.0290],
    riskLevel: 'low',
    anomalyScore: 0.65,
    transformerId: 'trans-003',
    barangay: 'Poblacion',
    feeder: 'Feeder-C',
    riskBand: 'Low',
    description: 'Slight variation in consumption',
  },
];

// Mock Meters Data with consumption history
export const mockMeters: Meter[] = [
  {
    id: 'meter-001',
    meterNumber: 'MTR-2024-001',
    transformerId: 'trans-001',
    barangay: 'Bel-Air',
    feeder: 'Feeder-A',
    riskBand: 'Critical',
    anomalyScore: 0.92,
    riskLevel: 'high',
    position: [14.5550, 121.0250],
    consumptionData: [
      { month: 'Jan', year: 2024, kwh: 1250, kva: 1.25 },
      { month: 'Feb', year: 2024, kwh: 1180, kva: 1.18 },
      { month: 'Mar', year: 2024, kwh: 1320, kva: 1.32 },
      { month: 'Apr', year: 2024, kwh: 1450, kva: 1.45 },
      { month: 'May', year: 2024, kwh: 2100, kva: 2.10 },
      { month: 'Jun', year: 2024, kwh: 1950, kva: 1.95 },
      { month: 'Jul', year: 2024, kwh: 2200, kva: 2.20 },
      { month: 'Aug', year: 2024, kwh: 2150, kva: 2.15 },
      { month: 'Sep', year: 2024, kwh: 1980, kva: 1.98 },
      { month: 'Oct', year: 2024, kwh: 1850, kva: 1.85 },
      { month: 'Nov', year: 2024, kwh: 1750, kva: 1.75 },
      { month: 'Dec', year: 2024, kwh: 1900, kva: 1.90 },
    ],
    anomalyNotes: 'Significant consumption spike detected in May-July period. Possible meter tampering or unauthorized connection.',
  },
  {
    id: 'meter-002',
    meterNumber: 'MTR-2024-002',
    transformerId: 'trans-001',
    barangay: 'Bel-Air',
    feeder: 'Feeder-A',
    riskBand: 'High',
    anomalyScore: 0.88,
    riskLevel: 'high',
    position: [14.5560, 121.0260],
    consumptionData: [
      { month: 'Jan', year: 2024, kwh: 980, kva: 0.98 },
      { month: 'Feb', year: 2024, kwh: 950, kva: 0.95 },
      { month: 'Mar', year: 2024, kwh: 1020, kva: 1.02 },
      { month: 'Apr', year: 2024, kwh: 1100, kva: 1.10 },
      { month: 'May', year: 2024, kwh: 1050, kva: 1.05 },
      { month: 'Jun', year: 2024, kwh: 1800, kva: 1.80 },
      { month: 'Jul', year: 2024, kwh: 1750, kva: 1.75 },
      { month: 'Aug', year: 2024, kwh: 1700, kva: 1.70 },
      { month: 'Sep', year: 2024, kwh: 1650, kva: 1.65 },
      { month: 'Oct', year: 2024, kwh: 1600, kva: 1.60 },
      { month: 'Nov', year: 2024, kwh: 1550, kva: 1.55 },
      { month: 'Dec', year: 2024, kwh: 1500, kva: 1.50 },
    ],
    anomalyNotes: 'Irregular load pattern starting June. Sudden increase in consumption without clear explanation.',
  },
  {
    id: 'meter-003',
    meterNumber: 'MTR-2024-003',
    transformerId: 'trans-002',
    barangay: 'San Antonio',
    feeder: 'Feeder-B',
    riskBand: 'Medium',
    anomalyScore: 0.75,
    riskLevel: 'medium',
    position: [14.5570, 121.0270],
    consumptionData: [
      { month: 'Jan', year: 2024, kwh: 750, kva: 0.75 },
      { month: 'Feb', year: 2024, kwh: 720, kva: 0.72 },
      { month: 'Mar', year: 2024, kwh: 780, kva: 0.78 },
      { month: 'Apr', year: 2024, kwh: 800, kva: 0.80 },
      { month: 'May', year: 2024, kwh: 850, kva: 0.85 },
      { month: 'Jun', year: 2024, kwh: 820, kva: 0.82 },
      { month: 'Jul', year: 2024, kwh: 1100, kva: 1.10 },
      { month: 'Aug', year: 2024, kwh: 1050, kva: 1.05 },
      { month: 'Sep', year: 2024, kwh: 980, kva: 0.98 },
      { month: 'Oct', year: 2024, kwh: 950, kva: 0.95 },
      { month: 'Nov', year: 2024, kwh: 900, kva: 0.90 },
      { month: 'Dec', year: 2024, kwh: 880, kva: 0.88 },
    ],
    anomalyNotes: 'Minor deviation from baseline in July-August. May require monitoring.',
  },
  {
    id: 'meter-004',
    meterNumber: 'MTR-2024-004',
    transformerId: 'trans-002',
    barangay: 'San Antonio',
    feeder: 'Feeder-B',
    riskBand: 'High',
    anomalyScore: 0.85,
    riskLevel: 'high',
    position: [14.5580, 121.0280],
    consumptionData: [
      { month: 'Jan', year: 2024, kwh: 650, kva: 0.65 },
      { month: 'Feb', year: 2024, kwh: 680, kva: 0.68 },
      { month: 'Mar', year: 2024, kwh: 700, kva: 0.70 },
      { month: 'Apr', year: 2024, kwh: 720, kva: 0.72 },
      { month: 'May', year: 2024, kwh: 750, kva: 0.75 },
      { month: 'Jun', year: 2024, kwh: 1400, kva: 1.40 },
      { month: 'Jul', year: 2024, kwh: 1350, kva: 1.35 },
      { month: 'Aug', year: 2024, kwh: 1300, kva: 1.30 },
      { month: 'Sep', year: 2024, kwh: 1250, kva: 1.25 },
      { month: 'Oct', year: 2024, kwh: 1200, kva: 1.20 },
      { month: 'Nov', year: 2024, kwh: 1150, kva: 1.15 },
      { month: 'Dec', year: 2024, kwh: 1100, kva: 1.10 },
    ],
    anomalyNotes: 'Potential meter tampering detected. Consumption doubled starting June. Field inspection recommended.',
  },
  {
    id: 'meter-005',
    meterNumber: 'MTR-2024-005',
    transformerId: 'trans-003',
    barangay: 'Poblacion',
    feeder: 'Feeder-C',
    riskBand: 'Low',
    anomalyScore: 0.65,
    riskLevel: 'low',
    position: [14.5590, 121.0290],
    consumptionData: [
      { month: 'Jan', year: 2024, kwh: 550, kva: 0.55 },
      { month: 'Feb', year: 2024, kwh: 580, kva: 0.58 },
      { month: 'Mar', year: 2024, kwh: 600, kva: 0.60 },
      { month: 'Apr', year: 2024, kwh: 620, kva: 0.62 },
      { month: 'May', year: 2024, kwh: 650, kva: 0.65 },
      { month: 'Jun', year: 2024, kwh: 680, kva: 0.68 },
      { month: 'Jul', year: 2024, kwh: 700, kva: 0.70 },
      { month: 'Aug', year: 2024, kwh: 720, kva: 0.72 },
      { month: 'Sep', year: 2024, kwh: 750, kva: 0.75 },
      { month: 'Oct', year: 2024, kwh: 780, kva: 0.78 },
      { month: 'Nov', year: 2024, kwh: 800, kva: 0.80 },
      { month: 'Dec', year: 2024, kwh: 820, kva: 0.82 },
    ],
    anomalyNotes: 'Slight variation in consumption pattern. Within acceptable range.',
  },
];

// Helper function to get unique filter values
export const getUniqueBarangays = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.barangay))).sort();
};

export const getUniqueFeeders = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.feeder))).sort();
};

export const getUniqueRiskBands = (meters: Meter[]): string[] => {
  return Array.from(new Set(meters.map(m => m.riskBand))).sort();
};

