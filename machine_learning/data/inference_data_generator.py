"""
Inference-Ready Synthetic Data Generator for GhostLoad Mapper
==============================================================

Production-grade synthetic data generator for ML inference testing and hackathon demos.
Generates highly realistic electricity consumption patterns with configurable anomaly scenarios.

Author: ML Systems Engineering Team
Date: November 13, 2025
Status: Production Ready

Features:
- Realistic consumption patterns (residential, commercial, industrial)
- Seasonal variations (summer AC peaks, winter heating)
- Theft scenarios (sudden drops, meter tampering, bypass patterns)
- Geographic clustering (barangay-based distribution)
- Transformer load balancing
- Configurable anomaly rates (5-20%)
- Export to CSV for immediate inference testing

Usage:
    python inference_data_generator.py --meters 50 --anomaly-rate 0.15 --output demo_inference.csv
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS & DATA CLASSES
# =============================================================================

class CustomerClass(Enum):
    """Customer classification types"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"


class TheftPattern(Enum):
    """Electricity theft pattern types"""
    NONE = "none"
    SUDDEN_DROP = "sudden_drop"          # Sudden 40-70% consumption decrease
    GRADUAL_DECLINE = "gradual_decline"  # Slow decline over 3-6 months
    METER_BYPASS = "meter_bypass"        # Near-zero consumption
    ERRATIC = "erratic"                  # Random spikes and drops


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation (Hackathon-optimized)"""
    n_meters: int = 100  # Professional sample size for demo
    n_transformers: int = 25  # Covers all 25 Metro Manila barangays
    anomaly_rate: float = 0.15  # 15% theft rate (industry standard)
    months: int = 12  # Required for model compatibility
    start_month: str = "202411"  # YYYYMM format
    
    # Geographic bounds (expanded Metro Manila area - all cities)
    lat_min: float = 14.35  # Mandaluyong (southernmost)
    lat_max: float = 14.74  # Valenzuela (northernmost)
    lon_min: float = 120.95  # Western Manila
    lon_max: float = 121.11  # Eastern Quezon City/Marikina
    
    # Consumption baselines (kWh/month)
    residential_mean: float = 300
    residential_std: float = 100
    commercial_mean: float = 1500
    commercial_std: float = 500
    industrial_mean: float = 5000
    industrial_std: float = 1500
    
    # Seasonal variation
    seasonal_amplitude: float = 0.20  # ±20% seasonal variation
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0 <= self.anomaly_rate <= 1:
            raise ValueError("anomaly_rate must be between 0 and 1")
        if self.n_meters < 1:
            raise ValueError("n_meters must be positive")
        if self.n_transformers < 1:
            raise ValueError("n_transformers must be positive")
        if self.months != 12:
            logger.warning("Model trained on 12 months - using different value may cause issues")


# =============================================================================
# BARANGAY & TRANSFORMER DATA (METRO MANILA COVERAGE)
# =============================================================================

# Expanded Metro Manila coverage with geographic coordinates (25 barangays)
METRO_MANILA_BARANGAYS = [
    # Quezon City (7 barangays)
    {"name": "Bagbag, Quezon City", "lat": 14.695474, "lon": 121.029213},
    {"name": "Damayan, Quezon City", "lat": 14.6386, "lon": 121.0141},
    {"name": "Bagong Silangan, Quezon City", "lat": 14.694, "lon": 121.106},
    {"name": "Alicia, Quezon City", "lat": 14.6516, "lon": 121.0441},
    {"name": "Bagong Pag-asa, Quezon City", "lat": 14.6498, "lon": 121.0422},
    {"name": "Bahay Toro, Quezon City", "lat": 14.6500, "lon": 121.0400},
    
    # Las Piñas City (2 barangays)
    {"name": "Ilaya, Las Piñas", "lat": 14.477915, "lon": 120.980103},
    {"name": "Talon Singko, Las Piñas", "lat": 14.4197, "lon": 120.9962},
    
    # Manila City (4 barangays)
    {"name": "Barangay 696, Malate, Manila", "lat": 14.575558, "lon": 120.989128},
    {"name": "Santa Cruz, Manila", "lat": 14.6176, "lon": 120.9848},
    {"name": "Poblacion, Manila", "lat": 14.5995, "lon": 120.9842},
    {"name": "Ermita, Manila", "lat": 14.5833, "lon": 120.9847},
    
    # Mandaluyong City
    {"name": "Addition Hills, Mandaluyong", "lat": 14.35, "lon": 121.02},
    
    # Parañaque City
    {"name": "Merville, Parañaque", "lat": 14.467, "lon": 121.017},
    
    # Marikina City (2 barangays)
    {"name": "Sto. Niño, Marikina", "lat": 14.640, "lon": 121.085},
    {"name": "Concepcion Uno, Marikina", "lat": 14.6395, "lon": 121.1027},
    
    # Caloocan City (2 barangays)
    {"name": "Bagong Silang, Caloocan", "lat": 14.70, "lon": 121.02},
    {"name": "Grace Park West, Caloocan", "lat": 14.6450, "lon": 120.9850},
    
    # Taguig City
    {"name": "Maharlika Village, Taguig", "lat": 14.5785, "lon": 121.0490},
    
    # San Juan City
    {"name": "Greenhills, San Juan", "lat": 14.5950, "lon": 121.0300},
    
    # Valenzuela City
    {"name": "Punturin, Valenzuela", "lat": 14.7370, "lon": 121.0024},
    
    # Additional Manila (for diversity)
    {"name": "Sampaloc, Manila", "lat": 14.6042, "lon": 120.9933},
    {"name": "Tondo, Manila", "lat": 14.6199, "lon": 120.9686},
    {"name": "Binondo, Manila", "lat": 14.5995, "lon": 120.9770},
]

TRANSFORMER_NAMES = [
    # Quezon City transformers
    "TX_QC_BAGBAG_001", "TX_QC_DAMAYAN_001", "TX_QC_SILANGAN_001",
    "TX_QC_ALICIA_001", "TX_QC_BAGONG_PAGASA_001", "TX_QC_BAHAY_TORO_001",
    
    # Las Piñas transformers
    "TX_LASPINAS_ILAYA_001", "TX_LASPINAS_TALON_001",
    
    # Manila transformers
    "TX_MANILA_MALATE_001", "TX_MANILA_SXCRUZ_001", "TX_MANILA_POBLACION_001",
    "TX_MANILA_ERMITA_001", "TX_MANILA_SAMPALOC_001", "TX_MANILA_TONDO_001",
    "TX_MANILA_BINONDO_001",
    
    # Other cities
    "TX_MANDALUYONG_001", "TX_PARANAQUE_001", 
    "TX_MARIKINA_STONINO_001", "TX_MARIKINA_CONCEPCION_001",
    "TX_CALOOCAN_SILANG_001", "TX_CALOOCAN_GRACEPARK_001",
    "TX_TAGUIG_001", "TX_SANJUAN_001", "TX_VALENZUELA_001",
    
    # Extras for larger datasets
    "TX_METRO_001", "TX_METRO_002"
]


# =============================================================================
# CORE GENERATOR CLASS
# =============================================================================

class InferenceDataGenerator:
    """
    Production-grade synthetic data generator for electricity consumption inference.
    
    Generates realistic meter consumption patterns with configurable anomaly scenarios
    suitable for ML model inference testing and demonstration purposes.
    
    Attributes:
        config: GeneratorConfig instance with generation parameters
        rng: NumPy random number generator for reproducibility
    """
    
    def __init__(self, config: GeneratorConfig):
        """
        Initialize generator with configuration.
        
        Args:
            config: GeneratorConfig instance
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        logger.info(f"Initialized InferenceDataGenerator (seed={config.random_seed})")
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset for inference.
        
        Returns:
            Tuple of (meter_data, transformer_data) DataFrames
        """
        logger.info(f"Generating dataset: {self.config.n_meters} meters, "
                   f"{self.config.n_transformers} transformers, "
                   f"{self.config.anomaly_rate:.1%} anomaly rate")
        
        # Generate transformers first
        transformers_df = self._generate_transformers()
        
        # Generate meters
        meters_df = self._generate_meters(transformers_df)
        
        # Add consumption patterns
        meters_df = self._add_consumption_patterns(meters_df)
        
        # Inject theft scenarios
        meters_df = self._inject_theft_scenarios(meters_df)
        
        logger.info(f"✅ Generated {len(meters_df)} meters, {len(transformers_df)} transformers")
        
        return meters_df, transformers_df
    
    def _generate_transformers(self) -> pd.DataFrame:
        """Generate transformer metadata with Metro Manila barangay coverage"""
        transformers = []
        
        for i in range(self.config.n_transformers):
            transformer_id = TRANSFORMER_NAMES[i] if i < len(TRANSFORMER_NAMES) else f"TX_{i+1:04d}"
            
            # Select barangay from Metro Manila list
            barangay_data = METRO_MANILA_BARANGAYS[i % len(METRO_MANILA_BARANGAYS)]
            
            transformers.append({
                'transformer_id': transformer_id,
                'barangay': barangay_data['name'],
                'lat': barangay_data['lat'] + self.rng.uniform(-0.005, 0.005),  # Small variation
                'lon': barangay_data['lon'] + self.rng.uniform(-0.005, 0.005),
                'capacity_kVA': self.rng.choice([500, 750, 1000, 1500, 2000]),
                'feeder_id': f"FEEDER_{(i % 3) + 1}"
            })
        
        return pd.DataFrame(transformers)
    
    def _generate_meters(self, transformers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate meter metadata"""
        meters = []
        
        # Distribute meters across transformers
        meters_per_tx = self.config.n_meters // self.config.n_transformers
        extra_meters = self.config.n_meters % self.config.n_transformers
        
        meter_idx = 0
        for tx_idx, tx_row in transformers_df.iterrows():
            n_meters_this_tx = meters_per_tx + (1 if tx_idx < extra_meters else 0)
            
            for _ in range(n_meters_this_tx):
                meter_idx += 1
                
                # Customer class distribution: 60% residential, 30% commercial, 10% industrial
                class_choice = self.rng.choice(
                    [CustomerClass.RESIDENTIAL, CustomerClass.COMMERCIAL, CustomerClass.INDUSTRIAL],
                    p=[0.60, 0.30, 0.10]
                )
                
                # Assign kVA based on customer class
                if class_choice == CustomerClass.RESIDENTIAL:
                    kVA = self.rng.uniform(5, 15)
                elif class_choice == CustomerClass.COMMERCIAL:
                    kVA = self.rng.uniform(50, 200)
                else:  # Industrial
                    kVA = self.rng.uniform(200, 1000)
                
                meters.append({
                    'meter_id': f"MTR_{meter_idx:06d}",
                    'transformer_id': tx_row['transformer_id'],
                    'customer_class': class_choice.value,
                    'barangay': tx_row['barangay'],
                    'lat': tx_row['lat'] + self.rng.uniform(-0.01, 0.01),  # Cluster near transformer
                    'lon': tx_row['lon'] + self.rng.uniform(-0.01, 0.01),
                    '_kVA_temp': round(kVA, 2)  # Store temporarily, will move to end later
                })
        
        return pd.DataFrame(meters)
    
    def _add_consumption_patterns(self, meters_df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic monthly consumption patterns"""
        logger.info("Generating consumption patterns with seasonal variation...")
        
        # Generate month columns properly using dateutil for accurate month arithmetic
        from dateutil.relativedelta import relativedelta
        start_date = datetime.strptime(self.config.start_month, "%Y%m")
        month_columns = []
        
        for month_offset in range(self.config.months):
            current_date = start_date + relativedelta(months=month_offset)
            month_col = f"monthly_consumption_{current_date.strftime('%Y%m')}"
            month_columns.append(month_col)
        
        logger.info(f"Generated {len(month_columns)} month columns: {month_columns[0]} to {month_columns[-1]}")
        
        # Generate consumption for each meter
        for idx, row in meters_df.iterrows():
            customer_class = row['customer_class']
            
            # Base consumption by class
            if customer_class == "residential":
                base_consumption = self.rng.normal(
                    self.config.residential_mean,
                    self.config.residential_std
                )
            elif customer_class == "commercial":
                base_consumption = self.rng.normal(
                    self.config.commercial_mean,
                    self.config.commercial_std
                )
            else:  # industrial
                base_consumption = self.rng.normal(
                    self.config.industrial_mean,
                    self.config.industrial_std
                )
            
            base_consumption = max(50, base_consumption)  # Minimum 50 kWh
            
            # Generate monthly values with seasonal variation
            for month_idx, month_col in enumerate(month_columns):
                # Seasonal component (peak in summer months 3-5, dip in winter 11-1)
                seasonal_factor = 1.0 + self.config.seasonal_amplitude * np.sin(
                    2 * np.pi * (month_idx - 3) / 12
                )
                
                # Random noise (±5%)
                noise_factor = 1.0 + self.rng.uniform(-0.05, 0.05)
                
                # Final consumption
                consumption = base_consumption * seasonal_factor * noise_factor
                meters_df.at[idx, month_col] = round(max(10, consumption), 2)
        
        # Move kVA to the end to match training data column order
        if '_kVA_temp' in meters_df.columns:
            kva_values = meters_df['_kVA_temp']
            meters_df = meters_df.drop(columns=['_kVA_temp'])
            meters_df['kVA'] = kva_values
            logger.info("Moved kVA column to end (matches training data format)")
        
        return meters_df
    
    def _inject_theft_scenarios(self, meters_df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic electricity theft patterns"""
        n_anomalies = int(self.config.n_meters * self.config.anomaly_rate)
        logger.info(f"Injecting {n_anomalies} theft scenarios ({self.config.anomaly_rate:.1%})...")
        
        # Randomly select meters for anomalies
        anomaly_indices = self.rng.choice(
            meters_df.index,
            size=n_anomalies,
            replace=False
        )
        
        month_cols = [col for col in meters_df.columns if col.startswith('monthly_consumption_')]
        
        for idx in anomaly_indices:
            # Choose theft pattern
            pattern = self.rng.choice([
                TheftPattern.SUDDEN_DROP,
                TheftPattern.GRADUAL_DECLINE,
                TheftPattern.METER_BYPASS,
                TheftPattern.ERRATIC
            ], p=[0.40, 0.30, 0.20, 0.10])
            
            if pattern == TheftPattern.SUDDEN_DROP:
                # Sudden 40-70% drop after month 6
                drop_month = self.rng.randint(6, 10)
                drop_factor = self.rng.uniform(0.30, 0.60)  # Keep 30-60% of consumption
                
                for month_idx in range(drop_month, len(month_cols)):
                    original = meters_df.at[idx, month_cols[month_idx]]
                    meters_df.at[idx, month_cols[month_idx]] = round(original * drop_factor, 2)
            
            elif pattern == TheftPattern.GRADUAL_DECLINE:
                # Gradual decline over 6 months
                start_month = self.rng.randint(3, 7)
                
                for month_idx in range(start_month, len(month_cols)):
                    decline_factor = 1.0 - 0.08 * (month_idx - start_month)  # 8% per month
                    decline_factor = max(0.40, decline_factor)  # Floor at 40%
                    
                    original = meters_df.at[idx, month_cols[month_idx]]
                    meters_df.at[idx, month_cols[month_idx]] = round(original * decline_factor, 2)
            
            elif pattern == TheftPattern.METER_BYPASS:
                # Near-zero consumption (meter bypassed)
                bypass_month = self.rng.randint(4, 9)
                
                for month_idx in range(bypass_month, len(month_cols)):
                    meters_df.at[idx, month_cols[month_idx]] = round(self.rng.uniform(5, 30), 2)
            
            elif pattern == TheftPattern.ERRATIC:
                # Random erratic pattern
                for month_idx in range(3, len(month_cols)):
                    if self.rng.random() < 0.5:
                        spike_factor = self.rng.uniform(0.30, 2.0)
                        original = meters_df.at[idx, month_cols[month_idx]]
                        meters_df.at[idx, month_cols[month_idx]] = round(original * spike_factor, 2)
        
        return meters_df
    
    def save_dataset(
        self,
        meters_df: pd.DataFrame,
        transformers_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Save generated dataset to CSV files.
        
        Args:
            meters_df: Meter consumption DataFrame
            transformers_df: Transformer metadata DataFrame
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping file types to saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files
        meter_file = output_dir / "meter_consumption.csv"
        transformer_file = output_dir / "transformers.csv"
        
        meters_df.to_csv(meter_file, index=False)
        transformers_df.to_csv(transformer_file, index=False)
        
        logger.info(f"✅ Saved meter data: {meter_file}")
        logger.info(f"✅ Saved transformer data: {transformer_file}")
        
        # Generate summary report
        report_file = output_dir / "generation_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SYNTHETIC DATA GENERATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random Seed: {self.config.random_seed}\n\n")
            f.write(f"Total Meters: {len(meters_df)}\n")
            f.write(f"Total Transformers: {len(transformers_df)}\n")
            f.write(f"Anomaly Rate: {self.config.anomaly_rate:.1%}\n")
            f.write(f"Expected Theft Cases: {int(len(meters_df) * self.config.anomaly_rate)}\n\n")
            
            f.write("Customer Class Distribution:\n")
            for class_type in meters_df['customer_class'].value_counts().items():
                f.write(f"  {class_type[0]}: {class_type[1]} ({class_type[1]/len(meters_df):.1%})\n")
            
            f.write(f"\nConsumption Statistics (kWh/month):\n")
            month_cols = [col for col in meters_df.columns if col.startswith('monthly_consumption_')]
            all_consumption = meters_df[month_cols].values.flatten()
            f.write(f"  Mean: {all_consumption.mean():.2f}\n")
            f.write(f"  Std: {all_consumption.std():.2f}\n")
            f.write(f"  Min: {all_consumption.min():.2f}\n")
            f.write(f"  Max: {all_consumption.max():.2f}\n")
        
        logger.info(f"✅ Saved report: {report_file}")
        
        return {
            'meters': meter_file,
            'transformers': transformer_file,
            'report': report_file
        }


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic data for GhostLoad Mapper inference testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 meters with 15% anomaly rate
  python inference_data_generator.py --meters 50 --anomaly-rate 0.15
  
  # Generate 100 meters for production-like demo
  python inference_data_generator.py --meters 100 --anomaly-rate 0.12 --output ../datasets/inference_demo
  
  # Quick test with 20 meters
  python inference_data_generator.py --meters 20 --transformers 3 --anomaly-rate 0.20
        """
    )
    
    parser.add_argument(
        '--meters',
        type=int,
        default=50,
        help='Number of meters to generate (default: 50)'
    )
    
    parser.add_argument(
        '--transformers',
        type=int,
        default=5,
        help='Number of transformers (default: 5)'
    )
    
    parser.add_argument(
        '--anomaly-rate',
        type=float,
        default=0.15,
        help='Anomaly rate 0.0-1.0 (default: 0.15 = 15%%)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../datasets/inference_test',
        help='Output directory (default: ../datasets/inference_test)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = GeneratorConfig(
            n_meters=args.meters,
            n_transformers=args.transformers,
            anomaly_rate=args.anomaly_rate,
            random_seed=args.seed
        )
        
        # Generate data
        generator = InferenceDataGenerator(config)
        meters_df, transformers_df = generator.generate_dataset()
        
        # Save to disk
        output_dir = Path(__file__).parent / args.output
        saved_files = generator.save_dataset(meters_df, transformers_df, output_dir)
        
        # Success summary
        print("\n" + "="*80)
        print("✅ SYNTHETIC DATA GENERATION COMPLETE")
        print("="*80)
        print(f"\n📊 Generated:")
        print(f"   • {len(meters_df)} meters")
        print(f"   • {len(transformers_df)} transformers")
        print(f"   • ~{int(len(meters_df) * config.anomaly_rate)} expected theft cases ({config.anomaly_rate:.1%})")
        print(f"\n📁 Saved to: {output_dir.absolute()}")
        print(f"\n🚀 Ready for inference testing!")
        print(f"\nTest with:")
        print(f"   cd ../pipeline")
        print(f"   python -c \"")
        print(f"   import pandas as pd")
        print(f"   from inference_pipeline import InferencePipeline")
        print(f"   df = pd.read_csv('{saved_files['meters']}')")
        print(f"   pipeline = InferencePipeline()")
        print(f"   result = pipeline.predict(df.iloc[0].to_dict())")
        print(f"   print(result.to_dict())")
        print(f"   \"")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
