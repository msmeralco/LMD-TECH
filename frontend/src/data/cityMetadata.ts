/**
 * NCR City Metadata
 * Contains coordinates, bounds, and comprehensive barangay mappings for all NCR cities
 */

export interface CityMetadata {
  id: string;
  name: string;
  center: [number, number]; // [lat, lng]
  zoom: number; // Optimal zoom level for city view
  bounds: [[number, number], [number, number]]; // SW and NE corners
  barangays: string[]; // All barangays in this city
}

/**
 * Comprehensive barangay-to-city mapping
 * Based on official MERALCO service areas in NCR
 * 
 * Note: Some barangay names exist in multiple cities (e.g., "San Antonio", "Poblacion")
 * This maps barangay names to their most common/likely city.
 * For exact matching, the backend should provide city_id directly.
 */
export const BARANGAY_TO_CITY: Record<string, string> = {
  // MANILA (16 districts, 897 barangays - major ones listed)
  'Tondo': 'manila',
  'Ermita': 'manila',
  'Malate': 'manila',
  'Paco': 'manila',
  'Pandacan': 'manila',
  'Port Area': 'manila',
  'Quiapo': 'manila',
  'Sampaloc': 'manila',
  'San Andres': 'manila',
  'San Miguel': 'manila',
  'San Nicolas': 'manila',
  'Santa Ana': 'manila',
  'Santa Cruz': 'manila',
  'Santa Mesa': 'manila',
  'Binondo': 'manila',
  'Intramuros': 'manila',
  'San Antonio': 'manila',  // Also in Makati, Pasig, Paranaque
  'Singalong': 'manila',
  'Moriones': 'manila',
  'Balic-Balic': 'manila',
  
  // QUEZON CITY (142 barangays - major ones listed)
  'Batasan Hills': 'quezon',
  'Commonwealth': 'quezon',
  'Fairview': 'quezon',
  'Novaliches': 'quezon',
  'Diliman': 'quezon',
  'Cubao': 'quezon',
  'Kamuning': 'quezon',
  'Libis': 'quezon',
  'Project 4': 'quezon',
  'Project 6': 'quezon',
  'Project 8': 'quezon',
  'San Francisco Del Monte': 'quezon',
  'Santa Mesa Heights': 'quezon',
  'Talipapa': 'quezon',
  'Teachers Village': 'quezon',
  'Baesa': 'quezon',
  'Bagong Lipunan': 'quezon',
  'Blue Ridge': 'quezon',
  'Holy Spirit': 'quezon',
  'Payatas': 'quezon',
  'Pasong Tamo': 'quezon',
  'Greater Lagro': 'quezon',
  'Bagong Pag-asa': 'quezon',
  'Maharlika': 'quezon',
  'Old Capitol Site': 'quezon',
  
  // MAKATI (33 barangays)
  'Poblacion': 'makati',  // Also in Mandaluyong, Pateros, Muntinlupa, Valenzuela
  'Bel-Air': 'makati',
  'Forbes Park': 'makati',
  'Dasmariñas': 'makati',
  'Urdaneta': 'makati',
  'San Lorenzo': 'makati',
  'Carmona': 'makati',
  'Olympia': 'makati',
  'Guadalupe Nuevo': 'makati',
  'Guadalupe Viejo': 'makati',
  'Pembo': 'makati',
  'Comembo': 'makati',
  'Rizal': 'makati',
  'Magallanes': 'makati',
  'La Paz': 'makati',
  'Makati San Antonio': 'makati',  // Prefixed to avoid conflict
  'Palanan': 'makati',
  'Pinagkaisahan': 'makati',
  'Tejeros': 'makati',
  'Singkamas': 'makati',
  'West Rembo': 'makati',
  'East Rembo': 'makati',
  'Pitogo': 'makati',
  'Cembo': 'makati',
  'South Cembo': 'makati',
  
  // PASIG (30 barangays)
  'Kapitolyo': 'pasig',
  'Ugong': 'pasig',  // Also in Valenzuela
  'Ortigas': 'pasig',
  'Rosario': 'pasig',
  'Santolan': 'pasig',  // Also in Malabon
  'Malinao': 'pasig',
  'Pasig San Antonio': 'pasig',  // Prefixed to avoid conflict
  'San Joaquin': 'pasig',
  'Pasig San Miguel': 'pasig',  // Prefixed to avoid conflict
  'Santa Lucia': 'pasig',  // Also in San Juan
  'Pineda': 'pasig',
  'Manggahan': 'pasig',
  'Maybunga': 'pasig',
  'Caniogan': 'pasig',
  'Kalawaan': 'pasig',
  'Dela Paz': 'pasig',
  'Sagad': 'pasig',
  'Pinagbuhatan': 'pasig',
  'Bambang': 'pasig',  // Also in Taguig
  'Bagong Ilog': 'pasig',
  'Bagong Katipunan': 'pasig',
  
  // MARIKINA (All 16 barangays)
  'Barangka': 'marikina',
  'Concepcion Uno': 'marikina',
  'Concepcion Dos': 'marikina',
  'Industrial Valley': 'marikina',
  'Jesus Dela Pena': 'marikina',
  'Kalumpang': 'marikina',
  'Malanday': 'marikina',  // Also in Valenzuela
  'Nangka': 'marikina',
  'Parang': 'marikina',
  'San Roque': 'marikina',  // Also in Pateros, Pasay, Navotas
  'Santa Elena': 'marikina',
  'Santo Niño': 'marikina',  // Also in Paranaque
  'Tañong': 'marikina',  // Also in Malabon
  'Tumana': 'marikina',
  'Fortune': 'marikina',
  'Marikina Heights': 'marikina',
  
  // MANDALUYONG (27 barangays)
  'Addition Hills': 'mandaluyong',  // Also in San Juan
  'Barangka Drive': 'mandaluyong',
  'Buayang Bato': 'mandaluyong',
  'Burol': 'mandaluyong',
  'Hulo': 'mandaluyong',
  'Mabini-J. Rizal': 'mandaluyong',
  'Malamig': 'mandaluyong',
  'Namayan': 'mandaluyong',
  'New Zaniga': 'mandaluyong',
  'Pag-asa': 'mandaluyong',
  'Plainview': 'mandaluyong',
  'Pleasant Hills': 'mandaluyong',
  'Mandaluyong Poblacion': 'mandaluyong',  // Prefixed
  'San Jose': 'mandaluyong',  // Also in Navotas
  'Vergara': 'mandaluyong',
  'Wack-Wack Greenhills': 'mandaluyong',
  
  // TAGUIG (28 barangays)
  'Bagumbayan': 'taguig',
  'Taguig Bambang': 'taguig',  // Prefixed to avoid conflict
  'Fort Bonifacio': 'taguig',
  'BGC': 'taguig',
  'Hagonoy': 'taguig',
  'Ibayo-Tipas': 'taguig',
  'Ligid-Tipas': 'taguig',
  'Lower Bicutan': 'taguig',
  'Maharlika Village': 'taguig',
  'Napindan': 'taguig',
  'New Lower Bicutan': 'taguig',
  'North Signal Village': 'taguig',
  'Palingon': 'taguig',
  'Pinagsama': 'taguig',
  'Taguig San Miguel': 'taguig',  // Prefixed
  'Taguig Santa Ana': 'taguig',  // Prefixed
  'Tuktukan': 'taguig',
  'Upper Bicutan': 'taguig',
  'Western Bicutan': 'taguig',
  'Central Bicutan': 'taguig',
  'Central Signal Village': 'taguig',
  'Ususan': 'taguig',
  'Wawa': 'taguig',
  'Signal Village': 'taguig',
  'Katuparan': 'taguig',
  'Calzada': 'taguig',
  
  // PATEROS (All 10 barangays)
  'Aguho': 'pateros',
  'Magtanggol': 'pateros',
  'Martires Del 96': 'pateros',
  'Pateros Poblacion': 'pateros',  // Prefixed
  'San Pedro': 'pateros',
  'Pateros San Roque': 'pateros',  // Prefixed
  'Pateros Santa Ana': 'pateros',  // Prefixed
  'Santo Rosario-Kanluran': 'pateros',
  'Santo Rosario-Silangan': 'pateros',
  'Tabacalera': 'pateros',
  
  // PARAÑAQUE (16 barangays)
  'Baclaran': 'paranaque',  // Also in Pasay
  'Don Bosco': 'paranaque',
  'La Huerta': 'paranaque',
  'Paranaque San Antonio': 'paranaque',  // Prefixed
  'San Dionisio': 'paranaque',
  'Paranaque San Isidro': 'paranaque',  // Prefixed
  'San Martin De Porres': 'paranaque',
  'Paranaque Santo Niño': 'paranaque',  // Prefixed
  'Sun Valley': 'paranaque',
  'Tambo': 'paranaque',
  'Vitalez': 'paranaque',
  'BF Homes': 'paranaque',
  'Marcelo Green': 'paranaque',
  'Merville': 'paranaque',
  'Moonwalk': 'paranaque',
  'Sto. Niño': 'paranaque',
  
  // MUNTINLUPA (All 9 barangays)
  'Alabang': 'muntinlupa',
  'Bayanan': 'muntinlupa',
  'Buli': 'muntinlupa',
  'Cupang': 'muntinlupa',
  'Muntinlupa Poblacion': 'muntinlupa',  // Prefixed
  'Putatan': 'muntinlupa',
  'Sucat': 'muntinlupa',
  'Tunasan': 'muntinlupa',
  'Ayala Alabang': 'muntinlupa',
  
  // LAS PIÑAS (20 barangays)
  'Almanza Uno': 'laspinas',
  'Almanza Dos': 'laspinas',
  'BF International': 'laspinas',
  'Daniel Fajardo': 'laspinas',
  'Elias Aldana': 'laspinas',
  'Ilaya': 'laspinas',
  'Manuyo Uno': 'laspinas',
  'Manuyo Dos': 'laspinas',
  'Pamplona Uno': 'laspinas',
  'Pamplona Dos': 'laspinas',
  'Pamplona Tres': 'laspinas',
  'Pilar': 'laspinas',
  'Pulang Lupa Uno': 'laspinas',
  'Pulang Lupa Dos': 'laspinas',
  'Talon Uno': 'laspinas',
  'Talon Dos': 'laspinas',
  'Talon Tres': 'laspinas',
  'Talon Kuatro': 'laspinas',
  'Talon Singko': 'laspinas',
  'Zapote': 'laspinas',
  
  // PASAY (201 barangays - zone-based, major ones listed)
  'Zone 1': 'pasay',
  'Zone 14': 'pasay',
  'Zone 19': 'pasay',
  'Pasay Baclaran': 'pasay',  // Prefixed
  'Malibay': 'pasay',
  'Pasay San Isidro': 'pasay',  // Prefixed
  'San Rafael': 'pasay',
  'Pasay San Roque': 'pasay',  // Prefixed
  'Libertad': 'pasay',
  
  // CALOOCAN (188 barangays - major ones listed)
  'Bagong Barrio': 'caloocan',
  'Bagong Silang': 'caloocan',
  'Kaybiga': 'caloocan',
  'Camarin': 'caloocan',
  'Grace Park': 'caloocan',
  'Maypajo': 'caloocan',
  'Tala': 'caloocan',
  '10th Avenue': 'caloocan',
  'Sangandaan': 'caloocan',
  
  // MALABON (All 21 barangays)
  'Acacia': 'malabon',
  'Baritan': 'malabon',
  'Bayan-bayanan': 'malabon',
  'Catmon': 'malabon',
  'Concepcion': 'malabon',
  'Dampalit': 'malabon',
  'Flores': 'malabon',
  'Hulong Duhat': 'malabon',
  'Ibaba': 'malabon',
  'Longos': 'malabon',
  'Maysilo': 'malabon',
  'Muzon': 'malabon',
  'Niugan': 'malabon',
  'Panghulo': 'malabon',
  'Potrero': 'malabon',
  'San Agustin': 'malabon',
  'Malabon Santolan': 'malabon',  // Prefixed
  'Malabon Tañong': 'malabon',  // Prefixed
  'Tinajeros': 'malabon',
  'Tonsuya': 'malabon',
  'Tugatog': 'malabon',
  
  // NAVOTAS (All 14 barangays)
  'Bagumbayan North': 'navotas',
  'Bagumbayan South': 'navotas',
  'Bangculasi': 'navotas',
  'Daanghari': 'navotas',
  'Navotas East': 'navotas',
  'Navotas West': 'navotas',
  'North Bay Boulevard North': 'navotas',
  'North Bay Boulevard South': 'navotas',
  'Navotas San Jose': 'navotas',  // Prefixed
  'San Rafael Village': 'navotas',
  'Navotas San Roque': 'navotas',  // Prefixed
  'Sipac-Almacen': 'navotas',
  'Tangos': 'navotas',
  'Tanza': 'navotas',
  
  // VALENZUELA (33 barangays)
  'Arkong Bato': 'valenzuela',
  'Bagbaguin': 'valenzuela',
  'Balangkas': 'valenzuela',
  'Bignay': 'valenzuela',
  'Bisig': 'valenzuela',
  'Canumay East': 'valenzuela',
  'Canumay West': 'valenzuela',
  'Coloong': 'valenzuela',
  'Dalandanan': 'valenzuela',
  'Gen. T. De Leon': 'valenzuela',
  'Isla': 'valenzuela',
  'Karuhatan': 'valenzuela',
  'Lawang Bato': 'valenzuela',
  'Lingunan': 'valenzuela',
  'Mabolo': 'valenzuela',
  'Valenzuela Malanday': 'valenzuela',  // Prefixed
  'Malinta': 'valenzuela',
  'Mapulang Lupa': 'valenzuela',
  'Marulas': 'valenzuela',
  'Maysan': 'valenzuela',
  'Palasan': 'valenzuela',
  'Parada': 'valenzuela',
  'Pariancillo Villa': 'valenzuela',
  'Pasolo': 'valenzuela',
  'Valenzuela Poblacion': 'valenzuela',  // Prefixed
  'Polo': 'valenzuela',
  'Punturin': 'valenzuela',
  'Rincon': 'valenzuela',
  'Tagalag': 'valenzuela',
  'Valenzuela Ugong': 'valenzuela',  // Prefixed
  'Viente Reales': 'valenzuela',
  'Wawang Pulo': 'valenzuela',
  'Veinte Reales': 'valenzuela',
  
  // SAN JUAN (21 barangays)
  'San Juan Addition Hills': 'sanjuan',  // Prefixed
  'Balong-Bato': 'sanjuan',
  'Batis': 'sanjuan',
  'Corazon De Jesus': 'sanjuan',
  'Ermitaño': 'sanjuan',
  'Greenhills': 'sanjuan',
  'Isabelita': 'sanjuan',
  'Kabayanan': 'sanjuan',
  'Little Baguio': 'sanjuan',
  'Maytunas': 'sanjuan',
  'Onse': 'sanjuan',
  'Pasadeña': 'sanjuan',
  'Pedro Cruz': 'sanjuan',
  'Progreso': 'sanjuan',
  'Rivera': 'sanjuan',
  'Salapan': 'sanjuan',
  'San Perfecto': 'sanjuan',
  'San Juan Santa Lucia': 'sanjuan',  // Prefixed
  'Tibagan': 'sanjuan',
  'West Crame': 'sanjuan',
  'St. Joseph': 'sanjuan',
};

/**
 * City metadata with accurate coordinates and bounds
 */
export const CITY_METADATA: Record<string, CityMetadata> = {
  manila: {
    id: 'manila',
    name: 'Manila',
    center: [14.5995, 120.9842], // City Hall
    zoom: 13,
    bounds: [
      [14.53, 120.95],
      [14.67, 121.02],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'manila'),
  },
  quezon: {
    id: 'quezon',
    name: 'Quezon City',
    center: [14.6760, 121.0437], // City Hall
    zoom: 12,
    bounds: [
      [14.60, 120.98],
      [14.78, 121.13],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'quezon'),
  },
  makati: {
    id: 'makati',
    name: 'Makati',
    center: [14.5547, 121.0244], // Ayala Avenue
    zoom: 13,
    bounds: [
      [14.52, 121.00],
      [14.59, 121.06],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'makati'),
  },
  pasig: {
    id: 'pasig',
    name: 'Pasig',
    center: [14.5764, 121.0851], // Ortigas
    zoom: 13,
    bounds: [
      [14.53, 121.04],
      [14.62, 121.13],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'pasig'),
  },
  marikina: {
    id: 'marikina',
    name: 'Marikina',
    center: [14.6507, 121.1029], // City Center
    zoom: 13,
    bounds: [
      [14.60, 121.07],
      [14.70, 121.14],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'marikina'),
  },
  mandaluyong: {
    id: 'mandaluyong',
    name: 'Mandaluyong',
    center: [14.5794, 121.0359], // City Center
    zoom: 14,
    bounds: [
      [14.56, 121.02],
      [14.60, 121.06],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'mandaluyong'),
  },
  taguig: {
    id: 'taguig',
    name: 'Taguig',
    center: [14.5176, 121.0509], // BGC
    zoom: 13,
    bounds: [
      [14.45, 121.00],
      [14.58, 121.10],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'taguig'),
  },
  pateros: {
    id: 'pateros',
    name: 'Pateros',
    center: [14.5438, 121.0687], // Town Center
    zoom: 14,
    bounds: [
      [14.53, 121.06],
      [14.56, 121.08],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'pateros'),
  },
  paranaque: {
    id: 'paranaque',
    name: 'Parañaque',
    center: [14.4793, 121.0198], // City Center
    zoom: 13,
    bounds: [
      [14.43, 120.98],
      [14.53, 121.05],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'paranaque'),
  },
  muntinlupa: {
    id: 'muntinlupa',
    name: 'Muntinlupa',
    center: [14.3814, 121.0445], // Alabang
    zoom: 13,
    bounds: [
      [14.33, 121.00],
      [14.43, 121.09],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'muntinlupa'),
  },
  laspinas: {
    id: 'laspinas',
    name: 'Las Piñas',
    center: [14.4453, 120.9830], // City Center
    zoom: 13,
    bounds: [
      [14.40, 120.95],
      [14.49, 121.02],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'laspinas'),
  },
  pasay: {
    id: 'pasay',
    name: 'Pasay',
    center: [14.5378, 121.0014], // City Center
    zoom: 13,
    bounds: [
      [14.50, 120.98],
      [14.57, 121.03],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'pasay'),
  },
  caloocan: {
    id: 'caloocan',
    name: 'Caloocan',
    center: [14.6574, 120.9832], // City Center
    zoom: 12,
    bounds: [
      [14.60, 120.93],
      [14.75, 121.05],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'caloocan'),
  },
  malabon: {
    id: 'malabon',
    name: 'Malabon',
    center: [14.6621, 120.9570], // City Center
    zoom: 14,
    bounds: [
      [14.64, 120.94],
      [14.68, 120.98],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'malabon'),
  },
  navotas: {
    id: 'navotas',
    name: 'Navotas',
    center: [14.6691, 120.9412], // City Center
    zoom: 14,
    bounds: [
      [14.65, 120.92],
      [14.69, 120.96],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'navotas'),
  },
  valenzuela: {
    id: 'valenzuela',
    name: 'Valenzuela',
    center: [14.7008, 120.9830], // City Center
    zoom: 13,
    bounds: [
      [14.65, 120.95],
      [14.75, 121.02],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'valenzuela'),
  },
  sanjuan: {
    id: 'sanjuan',
    name: 'San Juan',
    center: [14.6019, 121.0355], // City Center
    zoom: 14,
    bounds: [
      [14.59, 121.02],
      [14.61, 121.05],
    ],
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b] === 'sanjuan'),
  },
};

/**
 * Get city ID from barangay name
 */
export const getCityFromBarangay = (barangay: string): string | null => {
  return BARANGAY_TO_CITY[barangay] || null;
};

/**
 * Get city metadata
 */
export const getCityMetadata = (cityId: string): CityMetadata | null => {
  return CITY_METADATA[cityId] || null;
};

/**
 * Get all city IDs
 */
export const getAllCityIds = (): string[] => {
  return Object.keys(CITY_METADATA);
};
