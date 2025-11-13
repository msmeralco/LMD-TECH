"""
Synthetic Data Generator for GhostLoad Mapper
==============================================

Generates production-grade synthetic datasets for electrical distribution anomaly detection:
- Meter consumption data with realistic patterns
- Transformer metadata with spatial clustering
- Controlled anomaly injection for validation
- GeoJSON generation for map visualization

Design Principles:
- Deterministic reproducibility via controlled random seeds
- Configurable anomaly rates and patterns
- Spatial coherence for DBSCAN clustering validation
- Extensible architecture for custom distributions

Author: GhostLoad Mapper ML Team
Version: 1.0.0
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('synthetic_data_generation.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class GeneratorConfig:
    """
    Centralized configuration for synthetic data generation.
    
    Attributes:
        random_seed: Seed for reproducibility across runs
        num_transformers: Number of distribution transformers to simulate
        num_meters: Total number of meters across all transformers
        num_months: Months of historical consumption data
        anomaly_rate: Percentage of meters exhibiting anomalous behavior (0.0-1.0)
        spatial_clustering_strength: Controls geographic clustering (0.0-1.0)
        output_dir: Directory for generated datasets
        customer_classes: Distribution of customer types
        barangays: Geographic regions for transformer placement
    """
    random_seed: int = 42
    num_transformers: int = 50
    num_meters: int = 2000
    num_months: int = 12
    anomaly_rate: float = 0.075  # 7.5% anomalies (5-10% range)
    spatial_clustering_strength: float = 0.7
    output_dir: Path = field(default_factory=lambda: Path("generated_data"))
    
    # Customer distribution
    customer_classes: Dict[str, float] = field(default_factory=lambda: {
        'residential': 0.70,
        'commercial': 0.20,
        'industrial': 0.10
    })
    
    # Geographic regions (Philippines barangay simulation)
    barangays: List[str] = field(default_factory=lambda: [
        'San Miguel', 'Santa Cruz', 'Poblacion', 'Bagong Silang', 'Maligaya',
        'Riverside', 'San Isidro', 'Del Pilar', 'Mabini', 'Rizal'
    ])
    
    # Consumption baselines (kWh per month)
    consumption_baselines: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'residential': (150.0, 450.0),   # Mean, std dev
        'commercial': (800.0, 300.0),
        'industrial': (2500.0, 800.0)
    })
    
    # Transformer capacity ranges (kVA)
    transformer_capacity_range: Tuple[float, float] = (50.0, 500.0)
    
    # Geographic bounding box (sample Philippines coordinates)
    geo_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'lat': (14.4, 14.7),  # Latitude range
        'lon': (120.9, 121.2)  # Longitude range
    })
    
    def __post_init__(self):
        """Validate configuration and create output directory."""
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError(f"anomaly_rate must be in [0, 1], got {self.anomaly_rate}")
        
        if not 0.0 <= self.spatial_clustering_strength <= 1.0:
            raise ValueError(f"spatial_clustering_strength must be in [0, 1]")
        
        # Allow small floating-point tolerance for customer class probabilities
        class_sum = sum(self.customer_classes.values())
        if not (0.999 <= class_sum <= 1.001):
            raise ValueError(f"customer_classes probabilities must sum to 1.0, got {class_sum}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Configuration validated. Output directory: {self.output_dir}")


# ============================================================================
# Core Data Generation Components
# ============================================================================

class TransformerGenerator:
    """Generates transformer metadata with spatial clustering."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        
    def generate(self) -> pd.DataFrame:
        """
        Generate transformer metadata with geographic coordinates.
        
        Returns:
            DataFrame with columns: transformer_id, feeder_id, barangay, 
                                   lat, lon, capacity_kVA
        """
        logger.info(f"Generating {self.config.num_transformers} transformers...")
        
        transformer_ids = [f"TX_{i:04d}" for i in range(1, self.config.num_transformers + 1)]
        feeder_ids = [f"FD_{self.rng.randint(1, 11):02d}" for _ in transformer_ids]
        
        # Assign barangays with spatial clustering
        barangays = self.rng.choice(
            self.config.barangays,
            size=self.config.num_transformers,
            replace=True
        )
        
        # Generate spatially clustered coordinates
        lat, lon = self._generate_clustered_coordinates(barangays)
        
        # Generate transformer capacities with log-normal distribution
        capacities = self._generate_capacities()
        
        df = pd.DataFrame({
            'transformer_id': transformer_ids,
            'feeder_id': feeder_ids,
            'barangay': barangays,
            'lat': lat,
            'lon': lon,
            'capacity_kVA': capacities
        })
        
        logger.info(f"Generated {len(df)} transformers across {len(set(barangays))} barangays")
        return df
    
    def _generate_clustered_coordinates(self, barangays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate lat/lon with spatial clustering per barangay.
        
        Uses Gaussian clustering around barangay centroids for realistic
        spatial distribution suitable for DBSCAN validation.
        """
        unique_barangays = list(set(barangays))
        barangay_centers = {}
        
        # Create cluster centers for each barangay
        lat_min, lat_max = self.config.geo_bounds['lat']
        lon_min, lon_max = self.config.geo_bounds['lon']
        
        for brgy in unique_barangays:
            barangay_centers[brgy] = (
                self.rng.uniform(lat_min, lat_max),
                self.rng.uniform(lon_min, lon_max)
            )
        
        # Generate coordinates with clustering
        lats, lons = [], []
        cluster_std = 0.02 * self.config.spatial_clustering_strength  # ~2km at equator
        
        for brgy in barangays:
            center_lat, center_lon = barangay_centers[brgy]
            lats.append(self.rng.normal(center_lat, cluster_std))
            lons.append(self.rng.normal(center_lon, cluster_std))
        
        return np.array(lats), np.array(lons)
    
    def _generate_capacities(self) -> np.ndarray:
        """Generate transformer capacities with realistic distribution."""
        min_cap, max_cap = self.config.transformer_capacity_range
        
        # Log-normal distribution for realistic capacity spread
        log_mean = np.log(np.sqrt(min_cap * max_cap))
        log_std = (np.log(max_cap) - np.log(min_cap)) / 4
        
        capacities = self.rng.lognormal(log_mean, log_std, self.config.num_transformers)
        capacities = np.clip(capacities, min_cap, max_cap)
        
        # Round to standard kVA ratings
        standard_ratings = [50, 75, 100, 150, 200, 300, 500]
        capacities = np.array([min(standard_ratings, key=lambda x: abs(x - c)) for c in capacities])
        
        return capacities.astype(float)


class MeterGenerator:
    """Generates meter consumption data with realistic temporal patterns."""
    
    def __init__(self, config: GeneratorConfig, transformers_df: pd.DataFrame):
        self.config = config
        self.transformers_df = transformers_df
        self.rng = np.random.RandomState(config.random_seed + 1)  # Different seed
        
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate meter consumption data and anomaly labels.
        
        Returns:
            Tuple of (consumption_df, anomaly_labels_df)
        """
        logger.info(f"Generating {self.config.num_meters} meters with {self.config.num_months} months data...")
        
        # Assign meters to transformers
        meters_per_transformer = self._allocate_meters_to_transformers()
        
        # Generate meter metadata and consumption
        meter_records = []
        anomaly_records = []
        meter_id_counter = 1
        
        for tx_id, num_meters in meters_per_transformer.items():
            tx_info = self.transformers_df[self.transformers_df['transformer_id'] == tx_id].iloc[0]
            
            for _ in range(num_meters):
                meter_id = f"MTR_{meter_id_counter:06d}"
                customer_class = self._sample_customer_class()
                is_anomaly = self.rng.random() < self.config.anomaly_rate
                
                # Generate consumption time series
                consumption_data = self._generate_consumption_series(
                    customer_class, is_anomaly
                )
                
                # Build meter record
                record = {
                    'meter_id': meter_id,
                    'transformer_id': tx_id,
                    'customer_class': customer_class,
                    'barangay': tx_info['barangay'],
                    'lat': tx_info['lat'] + self.rng.normal(0, 0.001),  # Small offset
                    'lon': tx_info['lon'] + self.rng.normal(0, 0.001)
                }
                record.update(consumption_data)
                meter_records.append(record)
                
                # Track anomalies
                if is_anomaly:
                    anomaly_records.append({
                        'meter_id': meter_id,
                        'anomaly_flag': 1,
                        'risk_band': self._assign_risk_band(consumption_data),
                        'anomaly_type': 'low_consumption'
                    })
                
                meter_id_counter += 1
        
        consumption_df = pd.DataFrame(meter_records)
        anomaly_df = pd.DataFrame(anomaly_records)
        
        logger.info(f"Generated {len(consumption_df)} meters with {len(anomaly_df)} anomalies "
                   f"({len(anomaly_df)/len(consumption_df)*100:.1f}%)")
        
        return consumption_df, anomaly_df
    
    def _allocate_meters_to_transformers(self) -> Dict[str, int]:
        """Allocate meters to transformers based on capacity."""
        transformer_ids = self.transformers_df['transformer_id'].values
        capacities = self.transformers_df['capacity_kVA'].values
        
        # Allocate proportional to capacity with randomness
        weights = capacities / capacities.sum()
        base_allocation = (weights * self.config.num_meters).astype(int)
        
        # Distribute remainder randomly
        remainder = self.config.num_meters - base_allocation.sum()
        if remainder > 0:
            bonus_indices = self.rng.choice(len(transformer_ids), remainder, replace=False)
            base_allocation[bonus_indices] += 1
        
        # Ensure each transformer has at least 10 meters
        min_meters = 10
        shortfall = np.maximum(0, min_meters - base_allocation)
        base_allocation += shortfall
        
        # Adjust to maintain total
        excess = base_allocation.sum() - self.config.num_meters
        if excess > 0:
            # Remove excess from largest allocations
            large_indices = np.argsort(base_allocation)[-int(excess):]
            base_allocation[large_indices] -= 1
        
        return dict(zip(transformer_ids, base_allocation))
    
    def _sample_customer_class(self) -> str:
        """Sample customer class from configured distribution."""
        classes = list(self.config.customer_classes.keys())
        probs = list(self.config.customer_classes.values())
        return self.rng.choice(classes, p=probs)
    
    def _generate_consumption_series(self, customer_class: str, is_anomaly: bool) -> Dict[str, float]:
        """
        Generate monthly consumption time series with seasonal patterns.
        
        Incorporates:
        - Baseline consumption per customer class
        - Seasonal variation (summer peaks for AC usage)
        - Weekly/monthly noise
        - Anomaly injection (low consumption patterns)
        """
        base_mean, base_std = self.config.consumption_baselines[customer_class]
        
        # Generate base consumption with slight upward trend
        trend_coefficient = self.rng.uniform(-0.02, 0.05)  # -2% to +5% monthly
        seasonal_amplitude = base_mean * 0.15  # 15% seasonal variation
        
        consumption_series = {}
        start_date = datetime.now() - timedelta(days=30 * self.config.num_months)
        
        for month_offset in range(self.config.num_months):
            date = start_date + timedelta(days=30 * month_offset)
            month_key = f"monthly_consumption_{date.strftime('%Y%m')}"
            
            # Seasonal component (peaks in summer: Apr-May in Philippines)
            seasonal_factor = seasonal_amplitude * np.sin(2 * np.pi * (date.month - 4) / 12)
            
            # Trend component
            trend = base_mean * trend_coefficient * month_offset
            
            # Random noise
            noise = self.rng.normal(0, base_std * 0.3)
            
            # Base consumption
            consumption = base_mean + seasonal_factor + trend + noise
            
            # Anomaly injection: sustained low consumption
            if is_anomaly:
                # Gradual decrease over time
                anomaly_factor = 0.3 - (month_offset / self.config.num_months) * 0.2
                consumption *= anomaly_factor
            
            consumption_series[month_key] = max(0, consumption)  # Non-negative
        
        # Add kVA rating based on peak consumption
        peak_consumption = max(consumption_series.values())
        kVA = peak_consumption * self.rng.uniform(1.2, 1.5)  # Power factor approximation
        consumption_series['kVA'] = round(kVA, 2)
        
        return consumption_series
    
    def _assign_risk_band(self, consumption_data: Dict[str, float]) -> str:
        """Assign risk band based on consumption deviation."""
        consumption_values = [v for k, v in consumption_data.items() if k.startswith('monthly_')]
        median_consumption = np.median(consumption_values)
        
        if median_consumption < 50:
            return 'High'
        elif median_consumption < 100:
            return 'Medium'
        else:
            return 'Low'


class GeoJSONGenerator:
    """Generates GeoJSON for map visualization."""
    
    def __init__(self, transformers_df: pd.DataFrame, meters_df: pd.DataFrame):
        self.transformers_df = transformers_df
        self.meters_df = meters_df
    
    def generate(self) -> Dict:
        """
        Generate GeoJSON FeatureCollection for transformers and meters.
        
        Returns:
            GeoJSON dict with transformer features and associated meters
        """
        logger.info("Generating GeoJSON for map visualization...")
        
        features = []
        
        for _, tx in self.transformers_df.iterrows():
            # Get associated meters
            tx_meters = self.meters_df[
                self.meters_df['transformer_id'] == tx['transformer_id']
            ]
            
            meter_ids = tx_meters['meter_id'].tolist()
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [float(tx['lon']), float(tx['lat'])]
                },
                'properties': {
                    'transformer_id': tx['transformer_id'],
                    'feeder_id': tx['feeder_id'],
                    'barangay': tx['barangay'],
                    'capacity_kVA': float(tx['capacity_kVA']),
                    'num_meters': len(meter_ids),
                    'meter_ids': meter_ids[:10]  # Limit for payload size
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        logger.info(f"Generated GeoJSON with {len(features)} transformer features")
        return geojson


# ============================================================================
# Orchestrator
# ============================================================================

class SyntheticDataPipeline:
    """
    Main orchestrator for synthetic data generation pipeline.
    
    Coordinates all generation components and ensures consistent outputs.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        logger.info("Initialized SyntheticDataPipeline with config:")
        logger.info(f"  - Transformers: {self.config.num_transformers}")
        logger.info(f"  - Meters: {self.config.num_meters}")
        logger.info(f"  - Months: {self.config.num_months}")
        logger.info(f"  - Anomaly Rate: {self.config.anomaly_rate * 100:.1f}%")
        
    def generate_all(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Execute full data generation pipeline.
        
        Returns:
            Dictionary containing:
                - transformers_df
                - meters_df
                - anomaly_labels_df
                - geojson
        """
        logger.info("=" * 80)
        logger.info("Starting Synthetic Data Generation Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Generate transformers
            tx_generator = TransformerGenerator(self.config)
            transformers_df = tx_generator.generate()
            
            # Step 2: Generate meters and consumption
            meter_generator = MeterGenerator(self.config, transformers_df)
            meters_df, anomaly_labels_df = meter_generator.generate()
            
            # Step 3: Generate GeoJSON
            geojson_generator = GeoJSONGenerator(transformers_df, meters_df)
            geojson = geojson_generator.generate()
            
            # Validation
            self._validate_outputs(transformers_df, meters_df, anomaly_labels_df)
            
            logger.info("=" * 80)
            logger.info("Data Generation Complete")
            logger.info("=" * 80)
            
            return {
                'transformers_df': transformers_df,
                'meters_df': meters_df,
                'anomaly_labels_df': anomaly_labels_df,
                'geojson': geojson
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def save_outputs(self, outputs: Dict[str, Union[pd.DataFrame, Dict]]) -> None:
        """Save generated datasets to disk."""
        logger.info(f"Saving outputs to {self.config.output_dir}...")
        
        # Save DataFrames as CSV
        outputs['transformers_df'].to_csv(
            self.config.output_dir / 'transformers.csv',
            index=False
        )
        
        outputs['meters_df'].to_csv(
            self.config.output_dir / 'meter_consumption.csv',
            index=False
        )
        
        outputs['anomaly_labels_df'].to_csv(
            self.config.output_dir / 'anomaly_labels.csv',
            index=False
        )
        
        # Save GeoJSON
        import json
        with open(self.config.output_dir / 'transformers.geojson', 'w') as f:
            json.dump(outputs['geojson'], f, indent=2)
        
        # Generate summary report
        self._save_summary_report(outputs)
        
        logger.info("All outputs saved successfully")
    
    def _validate_outputs(self, transformers_df: pd.DataFrame, 
                         meters_df: pd.DataFrame,
                         anomaly_labels_df: pd.DataFrame) -> None:
        """Validate generated data integrity."""
        logger.info("Validating generated data...")
        
        # Check record counts
        assert len(transformers_df) == self.config.num_transformers, \
            f"Expected {self.config.num_transformers} transformers, got {len(transformers_df)}"
        
        assert len(meters_df) == self.config.num_meters, \
            f"Expected {self.config.num_meters} meters, got {len(meters_df)}"
        
        # Check anomaly rate
        actual_anomaly_rate = len(anomaly_labels_df) / len(meters_df)
        expected_range = (self.config.anomaly_rate - 0.02, self.config.anomaly_rate + 0.02)
        assert expected_range[0] <= actual_anomaly_rate <= expected_range[1], \
            f"Anomaly rate {actual_anomaly_rate:.3f} outside expected range {expected_range}"
        
        # Check foreign key integrity
        tx_ids_in_meters = set(meters_df['transformer_id'])
        tx_ids = set(transformers_df['transformer_id'])
        assert tx_ids_in_meters.issubset(tx_ids), "Invalid transformer IDs in meters"
        
        # Check consumption columns
        consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_consumption_')]
        assert len(consumption_cols) == self.config.num_months, \
            f"Expected {self.config.num_months} consumption columns, got {len(consumption_cols)}"
        
        # Check for nulls
        assert not transformers_df.isnull().any().any(), "Nulls found in transformers_df"
        assert not meters_df.isnull().any().any(), "Nulls found in meters_df"
        
        logger.info("✓ All validation checks passed")
    
    def _save_summary_report(self, outputs: Dict) -> None:
        """Generate and save summary statistics report."""
        report_path = self.config.output_dir / 'generation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Synthetic Data Generation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random Seed: {self.config.random_seed}\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Transformers: {len(outputs['transformers_df'])}\n")
            f.write(f"Meters: {len(outputs['meters_df'])}\n")
            f.write(f"Anomalies: {len(outputs['anomaly_labels_df'])}\n")
            f.write(f"Anomaly Rate: {len(outputs['anomaly_labels_df']) / len(outputs['meters_df']) * 100:.2f}%\n\n")
            
            f.write("Customer Class Distribution:\n")
            f.write("-" * 80 + "\n")
            class_dist = outputs['meters_df']['customer_class'].value_counts()
            for cls, count in class_dist.items():
                f.write(f"  {cls}: {count} ({count/len(outputs['meters_df'])*100:.1f}%)\n")
            
            f.write("\nRisk Band Distribution:\n")
            f.write("-" * 80 + "\n")
            if len(outputs['anomaly_labels_df']) > 0:
                risk_dist = outputs['anomaly_labels_df']['risk_band'].value_counts()
                for band, count in risk_dist.items():
                    f.write(f"  {band}: {count}\n")
            
            f.write("\nConsumption Statistics (kWh):\n")
            f.write("-" * 80 + "\n")
            consumption_cols = [c for c in outputs['meters_df'].columns if c.startswith('monthly_')]
            all_consumption = outputs['meters_df'][consumption_cols].values.flatten()
            f.write(f"  Mean: {np.mean(all_consumption):.2f}\n")
            f.write(f"  Median: {np.median(all_consumption):.2f}\n")
            f.write(f"  Std Dev: {np.std(all_consumption):.2f}\n")
            f.write(f"  Min: {np.min(all_consumption):.2f}\n")
            f.write(f"  Max: {np.max(all_consumption):.2f}\n")
        
        logger.info(f"Summary report saved to {report_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic meter consumption data for GhostLoad Mapper'
    )
    parser.add_argument(
        '--num-transformers', type=int, default=50,
        help='Number of transformers to generate (default: 50)'
    )
    parser.add_argument(
        '--num-meters', type=int, default=2000,
        help='Number of meters to generate (default: 2000)'
    )
    parser.add_argument(
        '--num-months', type=int, default=12,
        help='Months of consumption history (default: 12)'
    )
    parser.add_argument(
        '--anomaly-rate', type=float, default=0.075,
        help='Anomaly rate 0.0-1.0 (default: 0.075 = 7.5%%)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='generated_data',
        help='Output directory (default: generated_data)'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = GeneratorConfig(
        random_seed=args.seed,
        num_transformers=args.num_transformers,
        num_meters=args.num_meters,
        num_months=args.num_months,
        anomaly_rate=args.anomaly_rate,
        output_dir=Path(args.output_dir)
    )
    
    # Run pipeline
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    logger.info("✓ Data generation pipeline completed successfully")


if __name__ == '__main__':
    main()
