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
 * Barangay information with city and unique ID
 */
export interface BarangayInfo {
  id: number;
  city: string;
  barangay: string;
}


/**
 * Comprehensive barangay-to-city mapping with unique IDs
 * Each barangay has a unique numeric ID to resolve duplicate barangay names across cities
 *
 * Format: { 'Barangay Name': { id: number, city: 'cityId', barangay: 'name' } }
 */
export const BARANGAY_TO_CITY: Record<string, BarangayInfo> = {
  // MANILA (16 districts, 897 barangays - major ones listed)
  'Tondo': { id: 1, city: 'manila', barangay: 'Tondo' },
  'Ermita': { id: 2, city: 'manila', barangay: 'Ermita' },
  'Malate': { id: 3, city: 'manila', barangay: 'Malate' },
  'Paco': { id: 4, city: 'manila', barangay: 'Paco' },
  'Pandacan': { id: 5, city: 'manila', barangay: 'Pandacan' },
  'Port Area': { id: 6, city: 'manila', barangay: 'Port Area' },
  'Quiapo': { id: 7, city: 'manila', barangay: 'Quiapo' },
  'Sampaloc': { id: 8, city: 'manila', barangay: 'Sampaloc' },
  'San Andres': { id: 9, city: 'manila', barangay: 'San Andres' },
  'San Miguel': { id: 10, city: 'manila', barangay: 'San Miguel' },
  'San Nicolas': { id: 11, city: 'manila', barangay: 'San Nicolas' },
  'Santa Ana': { id: 12, city: 'manila', barangay: 'Santa Ana' },
  'Santa Cruz': { id: 13, city: 'manila', barangay: 'Santa Cruz' },
  'Santa Mesa': { id: 14, city: 'manila', barangay: 'Santa Mesa' },
  'Binondo': { id: 15, city: 'manila', barangay: 'Binondo' },
  'Intramuros': { id: 16, city: 'manila', barangay: 'Intramuros' },
  'San Antonio': { id: 17, city: 'manila', barangay: 'San Antonio' },
  'Singalong': { id: 18, city: 'manila', barangay: 'Singalong' },
  'Moriones': { id: 19, city: 'manila', barangay: 'Moriones' },
  'Balic-Balic': { id: 20, city: 'manila', barangay: 'Balic-Balic' },
 
  // QUEZON CITY (142 barangays - major ones listed)
  'Batasan Hills': { id: 21, city: 'quezon', barangay: 'Batasan Hills' },
  'Commonwealth': { id: 22, city: 'quezon', barangay: 'Commonwealth' },
  'Fairview': { id: 23, city: 'quezon', barangay: 'Fairview' },
  'Novaliches': { id: 24, city: 'quezon', barangay: 'Novaliches' },
  'Diliman': { id: 25, city: 'quezon', barangay: 'Diliman' },
  'Cubao': { id: 26, city: 'quezon', barangay: 'Cubao' },
  'Kamuning': { id: 27, city: 'quezon', barangay: 'Kamuning' },
  'Libis': { id: 28, city: 'quezon', barangay: 'Libis' },
  'Project 4': { id: 29, city: 'quezon', barangay: 'Project 4' },
  'Project 6': { id: 30, city: 'quezon', barangay: 'Project 6' },
  'Project 8': { id: 31, city: 'quezon', barangay: 'Project 8' },
  'San Francisco Del Monte': { id: 32, city: 'quezon', barangay: 'San Francisco Del Monte' },
  'Santa Mesa Heights': { id: 33, city: 'quezon', barangay: 'Santa Mesa Heights' },
  'Talipapa': { id: 34, city: 'quezon', barangay: 'Talipapa' },
  'Teachers Village': { id: 35, city: 'quezon', barangay: 'Teachers Village' },
  'Baesa': { id: 36, city: 'quezon', barangay: 'Baesa' },
  'Bagong Lipunan': { id: 37, city: 'quezon', barangay: 'Bagong Lipunan' },
  'Blue Ridge': { id: 38, city: 'quezon', barangay: 'Blue Ridge' },
  'Holy Spirit': { id: 39, city: 'quezon', barangay: 'Holy Spirit' },
  'Payatas': { id: 40, city: 'quezon', barangay: 'Payatas' },
  'Pasong Tamo': { id: 41, city: 'quezon', barangay: 'Pasong Tamo' },
  'Greater Lagro': { id: 42, city: 'quezon', barangay: 'Greater Lagro' },
  'Bagong Pag-asa': { id: 43, city: 'quezon', barangay: 'Bagong Pag-asa' },
  'Maharlika': { id: 44, city: 'quezon', barangay: 'Maharlika' },
  'Old Capitol Site': { id: 45, city: 'quezon', barangay: 'Old Capitol Site' },
 
  // MAKATI (33 barangays)
  'Poblacion': { id: 46, city: 'makati', barangay: 'Poblacion' },
  'Bel-Air': { id: 47, city: 'makati', barangay: 'Bel-Air' },
  'Forbes Park': { id: 48, city: 'makati', barangay: 'Forbes Park' },
  'Dasmariñas': { id: 49, city: 'makati', barangay: 'Dasmariñas' },
  'Urdaneta': { id: 50, city: 'makati', barangay: 'Urdaneta' },
  'San Lorenzo': { id: 51, city: 'makati', barangay: 'San Lorenzo' },
  'Carmona': { id: 52, city: 'makati', barangay: 'Carmona' },
  'Olympia': { id: 53, city: 'makati', barangay: 'Olympia' },
  'Guadalupe Nuevo': { id: 54, city: 'makati', barangay: 'Guadalupe Nuevo' },
  'Guadalupe Viejo': { id: 55, city: 'makati', barangay: 'Guadalupe Viejo' },
  'Pembo': { id: 56, city: 'makati', barangay: 'Pembo' },
  'Comembo': { id: 57, city: 'makati', barangay: 'Comembo' },
  'Rizal': { id: 58, city: 'makati', barangay: 'Rizal' },
  'Magallanes': { id: 59, city: 'makati', barangay: 'Magallanes' },
  'La Paz': { id: 60, city: 'makati', barangay: 'La Paz' },
  'Makati San Antonio': { id: 61, city: 'makati', barangay: 'Makati San Antonio' },
  'Palanan': { id: 62, city: 'makati', barangay: 'Palanan' },
  'Pinagkaisahan': { id: 63, city: 'makati', barangay: 'Pinagkaisahan' },
  'Tejeros': { id: 64, city: 'makati', barangay: 'Tejeros' },
  'Singkamas': { id: 65, city: 'makati', barangay: 'Singkamas' },
  'West Rembo': { id: 66, city: 'makati', barangay: 'West Rembo' },
  'East Rembo': { id: 67, city: 'makati', barangay: 'East Rembo' },
  'Pitogo': { id: 68, city: 'makati', barangay: 'Pitogo' },
  'Cembo': { id: 69, city: 'makati', barangay: 'Cembo' },
  'South Cembo': { id: 70, city: 'makati', barangay: 'South Cembo' },
 
  // PASIG (30 barangays)
  'Kapitolyo': { id: 71, city: 'pasig', barangay: 'Kapitolyo' },
  'Ugong': { id: 72, city: 'pasig', barangay: 'Ugong' },
  'Ortigas': { id: 73, city: 'pasig', barangay: 'Ortigas' },
  'Rosario': { id: 74, city: 'pasig', barangay: 'Rosario' },
  'Santolan': { id: 75, city: 'pasig', barangay: 'Santolan' },
  'Malinao': { id: 76, city: 'pasig', barangay: 'Malinao' },
  'Pasig San Antonio': { id: 77, city: 'pasig', barangay: 'Pasig San Antonio' },
  'San Joaquin': { id: 78, city: 'pasig', barangay: 'San Joaquin' },
  'Pasig San Miguel': { id: 79, city: 'pasig', barangay: 'Pasig San Miguel' },
  'Santa Lucia': { id: 80, city: 'pasig', barangay: 'Santa Lucia' },
  'Pineda': { id: 81, city: 'pasig', barangay: 'Pineda' },
  'Manggahan': { id: 82, city: 'pasig', barangay: 'Manggahan' },
  'Maybunga': { id: 83, city: 'pasig', barangay: 'Maybunga' },
  'Caniogan': { id: 84, city: 'pasig', barangay: 'Caniogan' },
  'Kalawaan': { id: 85, city: 'pasig', barangay: 'Kalawaan' },
  'Dela Paz': { id: 86, city: 'pasig', barangay: 'Dela Paz' },
  'Sagad': { id: 87, city: 'pasig', barangay: 'Sagad' },
  'Pinagbuhatan': { id: 88, city: 'pasig', barangay: 'Pinagbuhatan' },
  'Bambang': { id: 89, city: 'pasig', barangay: 'Bambang' },
  'Bagong Ilog': { id: 90, city: 'pasig', barangay: 'Bagong Ilog' },
  'Bagong Katipunan': { id: 91, city: 'pasig', barangay: 'Bagong Katipunan' },
 
  // MARIKINA (All 16 barangays)
  'Barangka': { id: 92, city: 'marikina', barangay: 'Barangka' },
  'Concepcion Uno': { id: 93, city: 'marikina', barangay: 'Concepcion Uno' },
  'Concepcion Dos': { id: 94, city: 'marikina', barangay: 'Concepcion Dos' },
  'Industrial Valley': { id: 95, city: 'marikina', barangay: 'Industrial Valley' },
  'Jesus Dela Pena': { id: 96, city: 'marikina', barangay: 'Jesus Dela Pena' },
  'Kalumpang': { id: 97, city: 'marikina', barangay: 'Kalumpang' },
  'Malanday': { id: 98, city: 'marikina', barangay: 'Malanday' },
  'Nangka': { id: 99, city: 'marikina', barangay: 'Nangka' },
  'Parang': { id: 100, city: 'marikina', barangay: 'Parang' },
  'San Roque': { id: 101, city: 'marikina', barangay: 'San Roque' },
  'Santa Elena': { id: 102, city: 'marikina', barangay: 'Santa Elena' },
  'Santo Niño': { id: 103, city: 'marikina', barangay: 'Santo Niño' },
  'Tañong': { id: 104, city: 'marikina', barangay: 'Tañong' },
  'Tumana': { id: 105, city: 'marikina', barangay: 'Tumana' },
  'Fortune': { id: 106, city: 'marikina', barangay: 'Fortune' },
  'Marikina Heights': { id: 107, city: 'marikina', barangay: 'Marikina Heights' },
 
  // MANDALUYONG (27 barangays)
  'Addition Hills': { id: 108, city: 'mandaluyong', barangay: 'Addition Hills' },
  'Barangka Drive': { id: 109, city: 'mandaluyong', barangay: 'Barangka Drive' },
  'Buayang Bato': { id: 110, city: 'mandaluyong', barangay: 'Buayang Bato' },
  'Burol': { id: 111, city: 'mandaluyong', barangay: 'Burol' },
  'Hulo': { id: 112, city: 'mandaluyong', barangay: 'Hulo' },
  'Mabini-J. Rizal': { id: 113, city: 'mandaluyong', barangay: 'Mabini-J. Rizal' },
  'Malamig': { id: 114, city: 'mandaluyong', barangay: 'Malamig' },
  'Namayan': { id: 115, city: 'mandaluyong', barangay: 'Namayan' },
  'New Zaniga': { id: 116, city: 'mandaluyong', barangay: 'New Zaniga' },
  'Pag-asa': { id: 117, city: 'mandaluyong', barangay: 'Pag-asa' },
  'Plainview': { id: 118, city: 'mandaluyong', barangay: 'Plainview' },
  'Pleasant Hills': { id: 119, city: 'mandaluyong', barangay: 'Pleasant Hills' },
  'Mandaluyong Poblacion': { id: 120, city: 'mandaluyong', barangay: 'Mandaluyong Poblacion' },
  'San Jose': { id: 121, city: 'mandaluyong', barangay: 'San Jose' },
  'Vergara': { id: 122, city: 'mandaluyong', barangay: 'Vergara' },
  'Wack-Wack Greenhills': { id: 123, city: 'mandaluyong', barangay: 'Wack-Wack Greenhills' },
 
  // TAGUIG (28 barangays)
  'Bagumbayan': { id: 124, city: 'taguig', barangay: 'Bagumbayan' },
  'Taguig Bambang': { id: 125, city: 'taguig', barangay: 'Taguig Bambang' },
  'Fort Bonifacio': { id: 126, city: 'taguig', barangay: 'Fort Bonifacio' },
  'BGC': { id: 127, city: 'taguig', barangay: 'BGC' },
  'Hagonoy': { id: 128, city: 'taguig', barangay: 'Hagonoy' },
  'Ibayo-Tipas': { id: 129, city: 'taguig', barangay: 'Ibayo-Tipas' },
  'Ligid-Tipas': { id: 130, city: 'taguig', barangay: 'Ligid-Tipas' },
  'Lower Bicutan': { id: 131, city: 'taguig', barangay: 'Lower Bicutan' },
  'Maharlika Village': { id: 132, city: 'taguig', barangay: 'Maharlika Village' },
  'Napindan': { id: 133, city: 'taguig', barangay: 'Napindan' },
  'New Lower Bicutan': { id: 134, city: 'taguig', barangay: 'New Lower Bicutan' },
  'North Signal Village': { id: 135, city: 'taguig', barangay: 'North Signal Village' },
  'Palingon': { id: 136, city: 'taguig', barangay: 'Palingon' },
  'Pinagsama': { id: 137, city: 'taguig', barangay: 'Pinagsama' },
  'Taguig San Miguel': { id: 138, city: 'taguig', barangay: 'Taguig San Miguel' },
  'Taguig Santa Ana': { id: 139, city: 'taguig', barangay: 'Taguig Santa Ana' },
  'Tuktukan': { id: 140, city: 'taguig', barangay: 'Tuktukan' },
  'Upper Bicutan': { id: 141, city: 'taguig', barangay: 'Upper Bicutan' },
  'Western Bicutan': { id: 142, city: 'taguig', barangay: 'Western Bicutan' },
  'Central Bicutan': { id: 143, city: 'taguig', barangay: 'Central Bicutan' },
  'Central Signal Village': { id: 144, city: 'taguig', barangay: 'Central Signal Village' },
  'Ususan': { id: 145, city: 'taguig', barangay: 'Ususan' },
  'Wawa': { id: 146, city: 'taguig', barangay: 'Wawa' },
  'Signal Village': { id: 147, city: 'taguig', barangay: 'Signal Village' },
  'Katuparan': { id: 148, city: 'taguig', barangay: 'Katuparan' },
  'Calzada': { id: 149, city: 'taguig', barangay: 'Calzada' },
 
  // PATEROS (All 10 barangays)
  'Aguho': { id: 150, city: 'pateros', barangay: 'Aguho' },
  'Magtanggol': { id: 151, city: 'pateros', barangay: 'Magtanggol' },
  'Martires Del 96': { id: 152, city: 'pateros', barangay: 'Martires Del 96' },
  'Pateros Poblacion': { id: 153, city: 'pateros', barangay: 'Pateros Poblacion' },
  'San Pedro': { id: 154, city: 'pateros', barangay: 'San Pedro' },
  'Pateros San Roque': { id: 155, city: 'pateros', barangay: 'Pateros San Roque' },
  'Pateros Santa Ana': { id: 156, city: 'pateros', barangay: 'Pateros Santa Ana' },
  'Santo Rosario-Kanluran': { id: 157, city: 'pateros', barangay: 'Santo Rosario-Kanluran' },
  'Santo Rosario-Silangan': { id: 158, city: 'pateros', barangay: 'Santo Rosario-Silangan' },
  'Tabacalera': { id: 159, city: 'pateros', barangay: 'Tabacalera' },
 
  // PARAÑAQUE (16 barangays)
  'Baclaran': { id: 160, city: 'paranaque', barangay: 'Baclaran' },
  'Don Bosco': { id: 161, city: 'paranaque', barangay: 'Don Bosco' },
  'La Huerta': { id: 162, city: 'paranaque', barangay: 'La Huerta' },
  'Paranaque San Antonio': { id: 163, city: 'paranaque', barangay: 'Paranaque San Antonio' },
  'San Dionisio': { id: 164, city: 'paranaque', barangay: 'San Dionisio' },
  'Paranaque San Isidro': { id: 165, city: 'paranaque', barangay: 'Paranaque San Isidro' },
  'San Martin De Porres': { id: 166, city: 'paranaque', barangay: 'San Martin De Porres' },
  'Paranaque Santo Niño': { id: 167, city: 'paranaque', barangay: 'Paranaque Santo Niño' },
  'Sun Valley': { id: 168, city: 'paranaque', barangay: 'Sun Valley' },
  'Tambo': { id: 169, city: 'paranaque', barangay: 'Tambo' },
  'Vitalez': { id: 170, city: 'paranaque', barangay: 'Vitalez' },
  'BF Homes': { id: 171, city: 'paranaque', barangay: 'BF Homes' },
  'Marcelo Green': { id: 172, city: 'paranaque', barangay: 'Marcelo Green' },
  'Merville': { id: 173, city: 'paranaque', barangay: 'Merville' },
  'Moonwalk': { id: 174, city: 'paranaque', barangay: 'Moonwalk' },
  'Sto. Niño': { id: 175, city: 'paranaque', barangay: 'Sto. Niño' },
 
  // MUNTINLUPA (All 9 barangays)
  'Alabang': { id: 176, city: 'muntinlupa', barangay: 'Alabang' },
  'Bayanan': { id: 177, city: 'muntinlupa', barangay: 'Bayanan' },
  'Buli': { id: 178, city: 'muntinlupa', barangay: 'Buli' },
  'Cupang': { id: 179, city: 'muntinlupa', barangay: 'Cupang' },
  'Muntinlupa Poblacion': { id: 180, city: 'muntinlupa', barangay: 'Muntinlupa Poblacion' },
  'Putatan': { id: 181, city: 'muntinlupa', barangay: 'Putatan' },
  'Sucat': { id: 182, city: 'muntinlupa', barangay: 'Sucat' },
  'Tunasan': { id: 183, city: 'muntinlupa', barangay: 'Tunasan' },
  'Ayala Alabang': { id: 184, city: 'muntinlupa', barangay: 'Ayala Alabang' },
 
  // LAS PIÑAS (20 barangays)
  'Almanza Uno': { id: 185, city: 'laspinas', barangay: 'Almanza Uno' },
  'Almanza Dos': { id: 186, city: 'laspinas', barangay: 'Almanza Dos' },
  'BF International': { id: 187, city: 'laspinas', barangay: 'BF International' },
  'Daniel Fajardo': { id: 188, city: 'laspinas', barangay: 'Daniel Fajardo' },
  'Elias Aldana': { id: 189, city: 'laspinas', barangay: 'Elias Aldana' },
  'Ilaya': { id: 190, city: 'laspinas', barangay: 'Ilaya' },
  'Manuyo Uno': { id: 191, city: 'laspinas', barangay: 'Manuyo Uno' },
  'Manuyo Dos': { id: 192, city: 'laspinas', barangay: 'Manuyo Dos' },
  'Pamplona Uno': { id: 193, city: 'laspinas', barangay: 'Pamplona Uno' },
  'Pamplona Dos': { id: 194, city: 'laspinas', barangay: 'Pamplona Dos' },
  'Pamplona Tres': { id: 195, city: 'laspinas', barangay: 'Pamplona Tres' },
  'Pilar': { id: 196, city: 'laspinas', barangay: 'Pilar' },
  'Pulang Lupa Uno': { id: 197, city: 'laspinas', barangay: 'Pulang Lupa Uno' },
  'Pulang Lupa Dos': { id: 198, city: 'laspinas', barangay: 'Pulang Lupa Dos' },
  'Talon Uno': { id: 199, city: 'laspinas', barangay: 'Talon Uno' },
  'Talon Dos': { id: 200, city: 'laspinas', barangay: 'Talon Dos' },
  'Talon Tres': { id: 201, city: 'laspinas', barangay: 'Talon Tres' },
  'Talon Kuatro': { id: 202, city: 'laspinas', barangay: 'Talon Kuatro' },
  'Talon Singko': { id: 203, city: 'laspinas', barangay: 'Talon Singko' },
  'Zapote': { id: 204, city: 'laspinas', barangay: 'Zapote' },
 
  // PASAY (201 barangays - zone-based, major ones listed)
  'Zone 1': { id: 205, city: 'pasay', barangay: 'Zone 1' },
  'Zone 14': { id: 206, city: 'pasay', barangay: 'Zone 14' },
  'Zone 19': { id: 207, city: 'pasay', barangay: 'Zone 19' },
  'Pasay Baclaran': { id: 208, city: 'pasay', barangay: 'Pasay Baclaran' },
  'Malibay': { id: 209, city: 'pasay', barangay: 'Malibay' },
  'Pasay San Isidro': { id: 210, city: 'pasay', barangay: 'Pasay San Isidro' },
  'San Rafael': { id: 211, city: 'pasay', barangay: 'San Rafael' },
  'Pasay San Roque': { id: 212, city: 'pasay', barangay: 'Pasay San Roque' },
  'Libertad': { id: 213, city: 'pasay', barangay: 'Libertad' },
 
  // CALOOCAN (188 barangays - major ones listed)
  'Bagong Barrio': { id: 214, city: 'caloocan', barangay: 'Bagong Barrio' },
  'Bagong Silang': { id: 215, city: 'caloocan', barangay: 'Bagong Silang' },
  'Kaybiga': { id: 216, city: 'caloocan', barangay: 'Kaybiga' },
  'Camarin': { id: 217, city: 'caloocan', barangay: 'Camarin' },
  'Grace Park': { id: 218, city: 'caloocan', barangay: 'Grace Park' },
  'Maypajo': { id: 219, city: 'caloocan', barangay: 'Maypajo' },
  'Tala': { id: 220, city: 'caloocan', barangay: 'Tala' },
  '10th Avenue': { id: 221, city: 'caloocan', barangay: '10th Avenue' },
  'Sangandaan': { id: 222, city: 'caloocan', barangay: 'Sangandaan' },
 
  // MALABON (All 21 barangays)
  'Acacia': { id: 223, city: 'malabon', barangay: 'Acacia' },
  'Baritan': { id: 224, city: 'malabon', barangay: 'Baritan' },
  'Bayan-bayanan': { id: 225, city: 'malabon', barangay: 'Bayan-bayanan' },
  'Catmon': { id: 226, city: 'malabon', barangay: 'Catmon' },
  'Concepcion': { id: 227, city: 'malabon', barangay: 'Concepcion' },
  'Dampalit': { id: 228, city: 'malabon', barangay: 'Dampalit' },
  'Flores': { id: 229, city: 'malabon', barangay: 'Flores' },
  'Hulong Duhat': { id: 230, city: 'malabon', barangay: 'Hulong Duhat' },
  'Ibaba': { id: 231, city: 'malabon', barangay: 'Ibaba' },
  'Longos': { id: 232, city: 'malabon', barangay: 'Longos' },
  'Maysilo': { id: 233, city: 'malabon', barangay: 'Maysilo' },
  'Muzon': { id: 234, city: 'malabon', barangay: 'Muzon' },
  'Niugan': { id: 235, city: 'malabon', barangay: 'Niugan' },
  'Panghulo': { id: 236, city: 'malabon', barangay: 'Panghulo' },
  'Potrero': { id: 237, city: 'malabon', barangay: 'Potrero' },
  'San Agustin': { id: 238, city: 'malabon', barangay: 'San Agustin' },
  'Malabon Santolan': { id: 239, city: 'malabon', barangay: 'Malabon Santolan' },
  'Malabon Tañong': { id: 240, city: 'malabon', barangay: 'Malabon Tañong' },
  'Tinajeros': { id: 241, city: 'malabon', barangay: 'Tinajeros' },
  'Tonsuya': { id: 242, city: 'malabon', barangay: 'Tonsuya' },
  'Tugatog': { id: 243, city: 'malabon', barangay: 'Tugatog' },
 
  // NAVOTAS (All 14 barangays)
  'Bagumbayan North': { id: 244, city: 'navotas', barangay: 'Bagumbayan North' },
  'Bagumbayan South': { id: 245, city: 'navotas', barangay: 'Bagumbayan South' },
  'Bangculasi': { id: 246, city: 'navotas', barangay: 'Bangculasi' },
  'Daanghari': { id: 247, city: 'navotas', barangay: 'Daanghari' },
  'Navotas East': { id: 248, city: 'navotas', barangay: 'Navotas East' },
  'Navotas West': { id: 249, city: 'navotas', barangay: 'Navotas West' },
  'North Bay Boulevard North': { id: 250, city: 'navotas', barangay: 'North Bay Boulevard North' },
  'North Bay Boulevard South': { id: 251, city: 'navotas', barangay: 'North Bay Boulevard South' },
  'Navotas San Jose': { id: 252, city: 'navotas', barangay: 'Navotas San Jose' },
  'San Rafael Village': { id: 253, city: 'navotas', barangay: 'San Rafael Village' },
  'Navotas San Roque': { id: 254, city: 'navotas', barangay: 'Navotas San Roque' },
  'Sipac-Almacen': { id: 255, city: 'navotas', barangay: 'Sipac-Almacen' },
  'Tangos': { id: 256, city: 'navotas', barangay: 'Tangos' },
  'Tanza': { id: 257, city: 'navotas', barangay: 'Tanza' },
 
  // VALENZUELA (33 barangays)
  'Arkong Bato': { id: 258, city: 'valenzuela', barangay: 'Arkong Bato' },
  'Bagbaguin': { id: 259, city: 'valenzuela', barangay: 'Bagbaguin' },
  'Balangkas': { id: 260, city: 'valenzuela', barangay: 'Balangkas' },
  'Bignay': { id: 261, city: 'valenzuela', barangay: 'Bignay' },
  'Bisig': { id: 262, city: 'valenzuela', barangay: 'Bisig' },
  'Canumay East': { id: 263, city: 'valenzuela', barangay: 'Canumay East' },
  'Canumay West': { id: 264, city: 'valenzuela', barangay: 'Canumay West' },
  'Coloong': { id: 265, city: 'valenzuela', barangay: 'Coloong' },
  'Dalandanan': { id: 266, city: 'valenzuela', barangay: 'Dalandanan' },
  'Gen. T. De Leon': { id: 267, city: 'valenzuela', barangay: 'Gen. T. De Leon' },
  'Isla': { id: 268, city: 'valenzuela', barangay: 'Isla' },
  'Karuhatan': { id: 269, city: 'valenzuela', barangay: 'Karuhatan' },
  'Lawang Bato': { id: 270, city: 'valenzuela', barangay: 'Lawang Bato' },
  'Lingunan': { id: 271, city: 'valenzuela', barangay: 'Lingunan' },
  'Mabolo': { id: 272, city: 'valenzuela', barangay: 'Mabolo' },
  'Valenzuela Malanday': { id: 273, city: 'valenzuela', barangay: 'Valenzuela Malanday' },
  'Malinta': { id: 274, city: 'valenzuela', barangay: 'Malinta' },
  'Mapulang Lupa': { id: 275, city: 'valenzuela', barangay: 'Mapulang Lupa' },
  'Marulas': { id: 276, city: 'valenzuela', barangay: 'Marulas' },
  'Maysan': { id: 277, city: 'valenzuela', barangay: 'Maysan' },
  'Palasan': { id: 278, city: 'valenzuela', barangay: 'Palasan' },
  'Parada': { id: 279, city: 'valenzuela', barangay: 'Parada' },
  'Pariancillo Villa': { id: 280, city: 'valenzuela', barangay: 'Pariancillo Villa' },
  'Pasolo': { id: 281, city: 'valenzuela', barangay: 'Pasolo' },
  'Valenzuela Poblacion': { id: 282, city: 'valenzuela', barangay: 'Valenzuela Poblacion' },
  'Polo': { id: 283, city: 'valenzuela', barangay: 'Polo' },
  'Punturin': { id: 284, city: 'valenzuela', barangay: 'Punturin' },
  'Rincon': { id: 285, city: 'valenzuela', barangay: 'Rincon' },
  'Tagalag': { id: 286, city: 'valenzuela', barangay: 'Tagalag' },
  'Valenzuela Ugong': { id: 287, city: 'valenzuela', barangay: 'Valenzuela Ugong' },
  'Viente Reales': { id: 288, city: 'valenzuela', barangay: 'Viente Reales' },
  'Wawang Pulo': { id: 289, city: 'valenzuela', barangay: 'Wawang Pulo' },
  'Veinte Reales': { id: 290, city: 'valenzuela', barangay: 'Veinte Reales' },
 
  // SAN JUAN (21 barangays)
  'San Juan Addition Hills': { id: 291, city: 'sanjuan', barangay: 'San Juan Addition Hills' },
  'Balong-Bato': { id: 292, city: 'sanjuan', barangay: 'Balong-Bato' },
  'Batis': { id: 293, city: 'sanjuan', barangay: 'Batis' },
  'Corazon De Jesus': { id: 294, city: 'sanjuan', barangay: 'Corazon De Jesus' },
  'Ermitaño': { id: 295, city: 'sanjuan', barangay: 'Ermitaño' },
  'Greenhills': { id: 296, city: 'sanjuan', barangay: 'Greenhills' },
  'Isabelita': { id: 297, city: 'sanjuan', barangay: 'Isabelita' },
  'Kabayanan': { id: 298, city: 'sanjuan', barangay: 'Kabayanan' },
  'Little Baguio': { id: 299, city: 'sanjuan', barangay: 'Little Baguio' },
  'Maytunas': { id: 300, city: 'sanjuan', barangay: 'Maytunas' },
  'Onse': { id: 301, city: 'sanjuan', barangay: 'Onse' },
  'Pasadeña': { id: 302, city: 'sanjuan', barangay: 'Pasadeña' },
  'Pedro Cruz': { id: 303, city: 'sanjuan', barangay: 'Pedro Cruz' },
  'Progreso': { id: 304, city: 'sanjuan', barangay: 'Progreso' },
  'Rivera': { id: 305, city: 'sanjuan', barangay: 'Rivera' },
  'Salapan': { id: 306, city: 'sanjuan', barangay: 'Salapan' },
  'San Perfecto': { id: 307, city: 'sanjuan', barangay: 'San Perfecto' },
  'San Juan Santa Lucia': { id: 308, city: 'sanjuan', barangay: 'San Juan Santa Lucia' },
  'Tibagan': { id: 309, city: 'sanjuan', barangay: 'Tibagan' },
  'West Crame': { id: 310, city: 'sanjuan', barangay: 'West Crame' },
  'St. Joseph': { id: 311, city: 'sanjuan', barangay: 'St. Joseph' },
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'manila'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'quezon'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'makati'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'pasig'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'marikina'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'mandaluyong'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'taguig'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'pateros'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'paranaque'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'muntinlupa'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'laspinas'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'pasay'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'caloocan'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'malabon'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'navotas'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'valenzuela'),
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
    barangays: Object.keys(BARANGAY_TO_CITY).filter(b => BARANGAY_TO_CITY[b].city === 'sanjuan'),
  },
};


/**
 * Get city ID from barangay name
 */
export const getCityFromBarangay = (barangay: string): string | null => {
  const info = BARANGAY_TO_CITY[barangay];
  return info ? info.city : null;
};


/**
 * Get unique barangay ID from barangay name
 */
export const getBarangayId = (barangay: string): number | null => {
  const info = BARANGAY_TO_CITY[barangay];
  return info ? info.id : null;
};


/**
 * Get barangay info (id, city, barangay) from barangay name
 */
export const getBarangayInfo = (barangay: string): BarangayInfo | null => {
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


/**
 * Create reverse lookup map: ID -> BarangayInfo
 * This resolves duplicate barangay names by using unique IDs
 */
export const BARANGAY_BY_ID: Record<number, BarangayInfo> = Object.values(BARANGAY_TO_CITY).reduce((acc, info) => {
  acc[info.id] = info;
  return acc;
}, {} as Record<number, BarangayInfo>);


/**
 * Get barangay info by ID (handles duplicates)
 */
export const getBarangayById = (id: number): BarangayInfo | null => {
  return BARANGAY_BY_ID[id] || null;
};


/**
 * Find all barangays with the same name across different cities
 */
export const findDuplicateBarangays = (barangayName: string): BarangayInfo[] => {
  return Object.values(BARANGAY_TO_CITY).filter(info => info.barangay === barangayName);
};



