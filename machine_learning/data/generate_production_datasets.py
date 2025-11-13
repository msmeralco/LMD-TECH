"""
Generate Production-Scale Inference Datasets for Hackathon Demo
================================================================

Creates TWO large-scale inference datasets for optimal clustering visualization:

1. Manila City Dataset (3,000 meters) - Dense urban clustering
2. Full Meralco Coverage (5,000 meters) - Region-wide distribution

Usage:
    python generate_production_datasets.py

Output:
    datasets/inference_manila/     - 3,000 meters (Manila City only)
    datasets/inference_meralco/    - 5,000 meters (Full Meralco coverage)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET 1: MANILA CITY (20 Barangays)
# ============================================================================

MANILA_BARANGAYS = [
    {"name": "Ermita (Brgy 659)", "lat": 14.5816, "lon": 120.9810},
    {"name": "Intramuros (Brgy 668)", "lat": 14.5899, "lon": 120.9750},
    {"name": "Binondo (Brgy 287)", "lat": 14.5998, "lon": 120.9752},
    {"name": "Quiapo (Brgy 306)", "lat": 14.5990, "lon": 120.9830},
    {"name": "Malate (Brgy 393)", "lat": 14.5714, "lon": 120.9865},
    {"name": "Paco (Brgy 48)", "lat": 14.5792, "lon": 120.9970},
    {"name": "Sampaloc (Brgy 20)", "lat": 14.6045, "lon": 120.9925},
    {"name": "Pandacan (Brgy 174)", "lat": 14.5930, "lon": 121.0070},
    {"name": "Tondo (Brgy 128)", "lat": 14.6180, "lon": 120.9665},
    {"name": "Santa Cruz (Brgy 293)", "lat": 14.6185, "lon": 120.9820},
    {"name": "San Andres Bukid (Brgy 704)", "lat": 14.5670, "lon": 120.9975},
    {"name": "San Miguel (Brgy 798)", "lat": 14.5995, "lon": 121.0000},
    {"name": "Port Area (Brgy 878)", "lat": 14.5890, "lon": 120.9675},
    {"name": "Santa Mesa (Brgy 823)", "lat": 14.6035, "lon": 121.0150},
    {"name": "Sta. Ana (Brgy 745)", "lat": 14.5800, "lon": 121.0155},
    {"name": "Tondo North (Brgy 151)", "lat": 14.6260, "lon": 120.9580},
    {"name": "Ermita South (Brgy 526)", "lat": 14.5760, "lon": 120.9870},
    {"name": "Quiapo West (Brgy 305)", "lat": 14.6020, "lon": 120.9875},
    {"name": "Malate South (Brgy 642)", "lat": 14.5635, "lon": 120.9920},
    {"name": "Paco East (Brgy 90)", "lat": 14.5725, "lon": 121.0050},
]


# ============================================================================
# DATASET 2: FULL MERALCO COVERAGE (50+ Locations)
# ============================================================================

MERALCO_COVERAGE = [
    # Metro Manila - Quezon City
    {"city": "Quezon City", "barangay": "Bago Bantay", "lat": 14.645200, "lon": 121.028100},
    {"city": "Quezon City", "barangay": "Project 6", "lat": 14.637900, "lon": 121.032800},
    {"city": "Quezon City", "barangay": "Cubao", "lat": 14.621100, "lon": 121.052700},
    {"city": "Quezon City", "barangay": "San Roque", "lat": 14.650800, "lon": 121.031300},
    {"city": "Quezon City", "barangay": "Tatalon", "lat": 14.623500, "lon": 121.020400},
    {"city": "Quezon City", "barangay": "Project 4", "lat": 14.644200, "lon": 121.049600},
    {"city": "Quezon City", "barangay": "Fairview", "lat": 14.708800, "lon": 121.049900},
    
    # Metro Manila - Manila City
    {"city": "Manila City", "barangay": "Sampaloc", "lat": 14.609800, "lon": 120.995400},
    {"city": "Manila City", "barangay": "Tondo Norte", "lat": 14.605200, "lon": 120.970900},
    {"city": "Manila City", "barangay": "Binondo", "lat": 14.594800, "lon": 120.971600},
    {"city": "Manila City", "barangay": "Paco", "lat": 14.567400, "lon": 120.985800},
    
    # Metro Manila - Makati
    {"city": "Makati City", "barangay": "Poblacion", "lat": 14.554700, "lon": 121.023200},
    {"city": "Makati City", "barangay": "San Antonio", "lat": 14.551800, "lon": 121.026100},
    {"city": "Makati City", "barangay": "Bel-Air", "lat": 14.552400, "lon": 121.025900},
    {"city": "Makati City", "barangay": "San Lorenzo", "lat": 14.551200, "lon": 121.025000},
    
    # Metro Manila - Pasig
    {"city": "Pasig City", "barangay": "Rosario", "lat": 14.581000, "lon": 121.071500},
    {"city": "Pasig City", "barangay": "Kapitolyo", "lat": 14.586300, "lon": 121.073900},
    
    # Metro Manila - Taguig
    {"city": "Taguig City", "barangay": "Lower Bicutan", "lat": 14.493200, "lon": 121.052800},
    {"city": "Taguig City", "barangay": "Fort Bonifacio", "lat": 14.554700, "lon": 121.049300},
    {"city": "Taguig City", "barangay": "Signal Village", "lat": 14.529800, "lon": 121.048300},
    
    # Metro Manila - Marikina
    {"city": "Marikina City", "barangay": "Industrial Valley", "lat": 14.649400, "lon": 121.107200},
    {"city": "Marikina City", "barangay": "Sto. Niño", "lat": 14.634900, "lon": 121.100600},
    {"city": "Marikina City", "barangay": "Concepcion Dos", "lat": 14.644200, "lon": 121.096400},
    
    # Metro Manila - Mandaluyong
    {"city": "Mandaluyong City", "barangay": "Hulo", "lat": 14.587600, "lon": 121.026900},
    
    # Metro Manila - San Juan
    {"city": "San Juan City", "barangay": "Greenhills", "lat": 14.600900, "lon": 121.035600},
    
    # Metro Manila - Caloocan
    {"city": "Caloocan City", "barangay": "Grace Park", "lat": 14.651100, "lon": 120.984900},
    {"city": "Caloocan City", "barangay": "Camarin", "lat": 14.711300, "lon": 120.986500},
    
    # Metro Manila - Valenzuela
    {"city": "Valenzuela City", "barangay": "Malinta", "lat": 14.692300, "lon": 120.979700},
    {"city": "Valenzuela City", "barangay": "Gen. T. de Leon", "lat": 14.663200, "lon": 120.993400},
    
    # Metro Manila - Malabon
    {"city": "Malabon City", "barangay": "Potrero", "lat": 14.664500, "lon": 120.960800},
    {"city": "Malabon City", "barangay": "Maysilo", "lat": 14.621800, "lon": 120.958900},
    
    # Metro Manila - Navotas
    {"city": "Navotas City", "barangay": "North Bay", "lat": 14.650900, "lon": 120.941000},
    {"city": "Navotas City", "barangay": "Tangos", "lat": 14.645600, "lon": 120.930700},
    
    # Metro Manila - Pasay
    {"city": "Pasay City", "barangay": "Baclaran", "lat": 14.513100, "lon": 120.998300},
    {"city": "Pasay City", "barangay": "Malibay", "lat": 14.573200, "lon": 121.014900},
    
    # Metro Manila - Parañaque
    {"city": "Parañaque City", "barangay": "San Dionisio", "lat": 14.483600, "lon": 121.012200},
    {"city": "Parañaque City", "barangay": "Tambo", "lat": 14.481100, "lon": 121.001600},
    
    # Metro Manila - Las Piñas
    {"city": "Las Piñas City", "barangay": "Pulang Lupa", "lat": 14.444700, "lon": 120.997500},
    {"city": "Las Piñas City", "barangay": "Talon Uno", "lat": 14.425300, "lon": 120.997900},
    
    # Metro Manila - Muntinlupa
    {"city": "Muntinlupa City", "barangay": "Poblacion", "lat": 14.414800, "lon": 121.045900},
    {"city": "Muntinlupa City", "barangay": "Tunasan", "lat": 14.363400, "lon": 121.030500},
    
    # Rizal Province
    {"city": "Antipolo City", "barangay": "Dela Paz", "lat": 14.612300, "lon": 121.129400},
    {"city": "Antipolo City", "barangay": "San Jose", "lat": 14.593800, "lon": 121.122700},
    {"city": "Antipolo City", "barangay": "San Roque West", "lat": 14.617900, "lon": 121.134200},
    {"city": "Cainta", "barangay": "Sto. Niño", "lat": 14.571200, "lon": 121.119900},
    {"city": "Taytay", "barangay": "Dolores", "lat": 14.549700, "lon": 121.127600},
    
    # Cavite Province
    {"city": "Bacoor", "barangay": "Molino", "lat": 14.428900, "lon": 120.946100},
    {"city": "Bacoor", "barangay": "Zapote", "lat": 14.420400, "lon": 120.960700},
    {"city": "Imus", "barangay": "Caridad", "lat": 14.399800, "lon": 120.935500},
    {"city": "Dasmariñas", "barangay": "Salitran", "lat": 14.323600, "lon": 120.940200},
    
    # Bulacan Province
    {"city": "Meycauayan", "barangay": "Langka", "lat": 14.723400, "lon": 120.956800},
    {"city": "San Jose del Monte", "barangay": "Tungkong Mangga", "lat": 14.825900, "lon": 121.052600},
    {"city": "Malolos", "barangay": "Tabang", "lat": 14.819200, "lon": 120.803100},
    
    # Laguna Province
    {"city": "San Pedro", "barangay": "San Antonio", "lat": 14.333900, "lon": 121.032400},
    {"city": "Biñan", "barangay": "Laguna Heights", "lat": 14.350500, "lon": 121.095700},
]


class ProductionDataGenerator:
    """Generate large-scale production inference datasets"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def generate_meters(self, locations, n_meters, anomaly_rate=0.12):
        """Generate meter data distributed across locations"""
        
        logger.info(f"Generating {n_meters} meters across {len(locations)} locations...")
        
        meters = []
        meter_id = 1000000
        
        # Distribute meters across locations
        meters_per_location = n_meters // len(locations)
        extra_meters = n_meters % len(locations)
        
        for idx, location in enumerate(locations):
            # Calculate how many meters for this location
            count = meters_per_location
            if idx < extra_meters:
                count += 1
                
            for _ in range(count):
                # Add small random offset to coordinates for realistic clustering
                lat_offset = np.random.normal(0, 0.002)  # ~200m variance
                lon_offset = np.random.normal(0, 0.002)
                
                meter = {
                    'meter_id': f"M{meter_id:07d}",
                    'transformer_id': f"TX_{location.get('barangay', location.get('name', 'UNKNOWN'))[:15]}",
                    'customer_class': np.random.choice(['Residential', 'Commercial', 'Industrial'], 
                                                       p=[0.75, 0.20, 0.05]),
                    'barangay': location.get('barangay', location.get('name', 'UNKNOWN')),
                    'lat': location['lat'] + lat_offset,
                    'lon': location['lon'] + lon_offset,
                }
                
                # Add monthly consumption (12 months: Nov 2024 - Oct 2025)
                base_date = datetime(2024, 11, 1)
                for month_offset in range(12):
                    month_date = base_date + relativedelta(months=month_offset)
                    col_name = f"monthly_consumption_{month_date.strftime('%Y%m')}"
                    
                    # Generate consumption based on customer class
                    if meter['customer_class'] == 'Residential':
                        base_consumption = np.random.uniform(150, 600)
                    elif meter['customer_class'] == 'Commercial':
                        base_consumption = np.random.uniform(800, 3000)
                    else:  # Industrial
                        base_consumption = np.random.uniform(5000, 15000)
                    
                    # Add seasonal variation (higher in summer months)
                    seasonal_factor = 1.0
                    if month_date.month in [3, 4, 5]:  # Summer in PH
                        seasonal_factor = 1.25
                    elif month_date.month in [6, 7, 8, 9]:  # Rainy season
                        seasonal_factor = 0.95
                        
                    consumption = base_consumption * seasonal_factor * np.random.uniform(0.85, 1.15)
                    meter[col_name] = round(consumption, 2)
                
                # Add kVA capacity (MUST be last column)
                if meter['customer_class'] == 'Residential':
                    meter['kVA'] = np.random.choice([5, 10, 15])
                elif meter['customer_class'] == 'Commercial':
                    meter['kVA'] = np.random.choice([25, 50, 75, 100])
                else:  # Industrial
                    meter['kVA'] = np.random.choice([150, 200, 300, 500])
                
                meters.append(meter)
                meter_id += 1
        
        df = pd.DataFrame(meters)
        
        # Inject anomalies
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
        
        logger.info(f"Injecting {n_anomalies} anomalies ({anomaly_rate*100:.1f}%)...")
        
        for idx in anomaly_indices:
            # Random anomaly type
            anomaly_type = np.random.choice(['theft', 'bypass', 'tampering'])
            
            if anomaly_type == 'theft':
                # Reduce consumption by 30-60%
                reduction_factor = np.random.uniform(0.4, 0.7)
                for col in df.columns:
                    if col.startswith('monthly_consumption_'):
                        df.at[idx, col] *= reduction_factor
                        
            elif anomaly_type == 'bypass':
                # Erratic consumption pattern
                for col in df.columns:
                    if col.startswith('monthly_consumption_'):
                        df.at[idx, col] *= np.random.uniform(0.3, 1.5)
                        
            elif anomaly_type == 'tampering':
                # Suspiciously low consumption
                for col in df.columns:
                    if col.startswith('monthly_consumption_'):
                        df.at[idx, col] *= np.random.uniform(0.1, 0.4)
        
        logger.info(f"Generated {len(df)} meters with {n_anomalies} anomalies")
        return df
    
    def save_dataset(self, df, output_dir):
        """Save dataset to CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / 'meter_consumption.csv'
        df.to_csv(csv_file, index=False)
        
        # Generate report
        report_file = output_path / 'generation_report.txt'
        with open(report_file, 'w') as f:
            f.write(f"Dataset Generation Report\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Meters: {len(df)}\n")
            f.write(f"Locations: {df['barangay'].nunique()}\n")
            f.write(f"Transformers: {df['transformer_id'].nunique()}\n")
            f.write(f"\nCustomer Class Distribution:\n")
            f.write(df['customer_class'].value_counts().to_string())
            f.write(f"\n\nGeographic Bounds:\n")
            f.write(f"  Latitude: {df['lat'].min():.6f} to {df['lat'].max():.6f}\n")
            f.write(f"  Longitude: {df['lon'].min():.6f} to {df['lon'].max():.6f}\n")
            f.write(f"\nFiles Saved:\n")
            f.write(f"  - {csv_file}\n")
            f.write(f"  - {report_file}\n")
        
        logger.info(f"Dataset saved to {output_path}")
        return csv_file, report_file


def main():
    """Generate both production datasets"""
    
    print("\n" + "="*80)
    print("🚀 PRODUCTION DATASET GENERATOR - HACKATHON DEMO")
    print("="*80)
    print("\nGenerating TWO large-scale datasets for optimal clustering visualization:")
    print("  1️⃣  Manila City (3,000 meters) - Dense urban clusters")
    print("  2️⃣  Full Meralco Coverage (5,000 meters) - Region-wide distribution\n")
    
    generator = ProductionDataGenerator(random_seed=42)
    
    # =========================================================================
    # DATASET 1: MANILA CITY (3,000 meters)
    # =========================================================================
    
    print("━"*80)
    print("1️⃣  GENERATING MANILA CITY DATASET (3,000 meters)")
    print("━"*80)
    
    df_manila = generator.generate_meters(
        locations=MANILA_BARANGAYS,
        n_meters=3000,
        anomaly_rate=0.12
    )
    
    output_manila = Path(__file__).parent / '../datasets/inference_manila'
    csv_manila, report_manila = generator.save_dataset(df_manila, output_manila)
    
    print(f"\n✅ Manila City Dataset Complete:")
    print(f"   📊 {len(df_manila)} meters")
    print(f"   🗺️  {len(MANILA_BARANGAYS)} Manila barangays")
    print(f"   🚨 ~{int(len(df_manila) * 0.12)} expected anomalies (12%)")
    print(f"   📁 {output_manila.absolute()}")
    print(f"   ⚡ Dense clustering for clear heatmap visualization")
    
    # =========================================================================
    # DATASET 2: FULL MERALCO COVERAGE (5,000 meters)
    # =========================================================================
    
    print("\n" + "━"*80)
    print("2️⃣  GENERATING FULL MERALCO COVERAGE (5,000 meters)")
    print("━"*80)
    
    df_meralco = generator.generate_meters(
        locations=MERALCO_COVERAGE,
        n_meters=5000,
        anomaly_rate=0.10
    )
    
    output_meralco = Path(__file__).parent / '../datasets/inference_meralco'
    csv_meralco, report_meralco = generator.save_dataset(df_meralco, output_meralco)
    
    print(f"\n✅ Full Meralco Coverage Complete:")
    print(f"   📊 {len(df_meralco)} meters")
    print(f"   🗺️  {len(MERALCO_COVERAGE)} locations (Metro Manila + Rizal + Cavite + Bulacan + Laguna)")
    print(f"   🚨 ~{int(len(df_meralco) * 0.10)} expected anomalies (10%)")
    print(f"   📁 {output_meralco.absolute()}")
    print(f"   ⚡ Region-wide distribution showing full coverage")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ALL PRODUCTION DATASETS GENERATED!")
    print("="*80)
    
    print("\n📊 DATASET COMPARISON:")
    print("─"*80)
    print(f"{'Dataset':<25} {'Meters':<10} {'Locations':<12} {'Anomalies':<12} {'Coverage'}")
    print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*30}")
    print(f"{'Manila City':<25} {'3,000':<10} {'20':<12} {'~360':<12} {'Manila barangays only'}")
    print(f"{'Full Meralco':<25} {'5,000':<10} {f'{len(MERALCO_COVERAGE)}':<12} {'~500':<12} {'Metro Manila + provinces'}")
    
    print("\n🎯 DEMO USAGE:")
    print("─"*80)
    print("\n🗺️  For Dense Manila Clustering:")
    print("   cd ../pipeline")
    print("   python inference_pipeline.py --input ../datasets/inference_manila/meter_consumption.csv")
    print("   → Shows clear clustering in Manila, perfect for heatmap visualization")
    
    print("\n🌏 For Full Coverage Demo:")
    print("   python inference_pipeline.py --input ../datasets/inference_meralco/meter_consumption.csv")
    print("   → Shows region-wide detection across entire Meralco franchise")
    
    print("\n💡 UI/UX Tips:")
    print("─"*80)
    print("  • Manila dataset: Better for zoomed-in cluster analysis")
    print("  • Meralco dataset: Better for showing scale and geographic reach")
    print("  • Both have 12% anomaly rate for realistic detection scenarios")
    print("  • Coordinates have realistic variance (~200m) for natural clustering")
    
    print("\n" + "="*80)
    print("🎉 READY FOR HACKATHON DEMO!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
