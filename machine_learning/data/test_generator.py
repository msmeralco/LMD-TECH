"""
Unit and Integration Tests for Synthetic Data Generator
=======================================================

Comprehensive test suite validating:
- Configuration validation
- Data generation correctness
- Anomaly injection accuracy
- Spatial clustering properties
- Output integrity

Run with: pytest test_generator.py -v
"""

import unittest
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

from synthetic_data_generator import (
    GeneratorConfig,
    TransformerGenerator,
    MeterGenerator,
    GeoJSONGenerator,
    SyntheticDataPipeline
)


class TestGeneratorConfig(unittest.TestCase):
    """Test configuration validation and initialization."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = GeneratorConfig()
        self.assertEqual(config.num_transformers, 50)
        self.assertEqual(config.num_meters, 2000)
        self.assertEqual(config.num_months, 12)
        self.assertAlmostEqual(config.anomaly_rate, 0.075)
    
    def test_anomaly_rate_validation(self):
        """Test anomaly rate must be in valid range."""
        with self.assertRaises(ValueError):
            GeneratorConfig(anomaly_rate=-0.1)
        
        with self.assertRaises(ValueError):
            GeneratorConfig(anomaly_rate=1.5)
    
    def test_customer_class_probabilities(self):
        """Test customer class probabilities sum to 1.0."""
        with self.assertRaises(ValueError):
            GeneratorConfig(customer_classes={
                'residential': 0.5,
                'commercial': 0.3
                # Sum != 1.0
            })
    
    def test_output_directory_creation(self):
        """Test output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_output'
            config = GeneratorConfig(output_dir=output_path)
            self.assertTrue(output_path.exists())


class TestTransformerGenerator(unittest.TestCase):
    """Test transformer metadata generation."""
    
    def setUp(self):
        """Create test configuration."""
        self.config = GeneratorConfig(
            random_seed=42,
            num_transformers=10,
            output_dir=Path(tempfile.mkdtemp())
        )
        self.generator = TransformerGenerator(self.config)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.config.output_dir, ignore_errors=True)
    
    def test_transformer_count(self):
        """Test correct number of transformers generated."""
        df = self.generator.generate()
        self.assertEqual(len(df), self.config.num_transformers)
    
    def test_transformer_columns(self):
        """Test all required columns present."""
        df = self.generator.generate()
        required_cols = ['transformer_id', 'feeder_id', 'barangay', 'lat', 'lon', 'capacity_kVA']
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_unique_transformer_ids(self):
        """Test transformer IDs are unique."""
        df = self.generator.generate()
        self.assertEqual(len(df['transformer_id'].unique()), len(df))
    
    def test_geographic_bounds(self):
        """Test coordinates within specified bounds."""
        df = self.generator.generate()
        lat_min, lat_max = self.config.geo_bounds['lat']
        lon_min, lon_max = self.config.geo_bounds['lon']
        
        # Allow small tolerance for clustering
        tolerance = 0.1
        self.assertTrue(df['lat'].min() >= lat_min - tolerance)
        self.assertTrue(df['lat'].max() <= lat_max + tolerance)
        self.assertTrue(df['lon'].min() >= lon_min - tolerance)
        self.assertTrue(df['lon'].max() <= lon_max + tolerance)
    
    def test_capacity_values(self):
        """Test transformer capacities are valid."""
        df = self.generator.generate()
        min_cap, max_cap = self.config.transformer_capacity_range
        self.assertTrue(df['capacity_kVA'].min() >= min_cap)
        self.assertTrue(df['capacity_kVA'].max() <= max_cap)
    
    def test_reproducibility(self):
        """Test same seed produces identical results."""
        df1 = self.generator.generate()
        
        # Create new generator with same seed
        generator2 = TransformerGenerator(self.config)
        df2 = generator2.generate()
        
        pd.testing.assert_frame_equal(df1, df2)


class TestMeterGenerator(unittest.TestCase):
    """Test meter consumption data generation."""
    
    def setUp(self):
        """Create test configuration and transformers."""
        self.config = GeneratorConfig(
            random_seed=42,
            num_transformers=5,
            num_meters=100,
            num_months=6,
            anomaly_rate=0.1,
            output_dir=Path(tempfile.mkdtemp())
        )
        
        tx_generator = TransformerGenerator(self.config)
        self.transformers_df = tx_generator.generate()
        self.generator = MeterGenerator(self.config, self.transformers_df)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.config.output_dir, ignore_errors=True)
    
    def test_meter_count(self):
        """Test correct number of meters generated."""
        meters_df, _ = self.generator.generate()
        self.assertEqual(len(meters_df), self.config.num_meters)
    
    def test_anomaly_rate(self):
        """Test anomaly rate within expected range."""
        _, anomaly_df = self.generator.generate()
        actual_rate = len(anomaly_df) / self.config.num_meters
        
        # Allow 2% tolerance
        self.assertAlmostEqual(actual_rate, self.config.anomaly_rate, delta=0.02)
    
    def test_consumption_columns(self):
        """Test correct number of monthly consumption columns."""
        meters_df, _ = self.generator.generate()
        consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_consumption_')]
        self.assertEqual(len(consumption_cols), self.config.num_months)
    
    def test_transformer_assignment(self):
        """Test all meters assigned to valid transformers."""
        meters_df, _ = self.generator.generate()
        tx_ids = set(self.transformers_df['transformer_id'])
        meter_tx_ids = set(meters_df['transformer_id'])
        self.assertTrue(meter_tx_ids.issubset(tx_ids))
    
    def test_customer_class_distribution(self):
        """Test customer class distribution matches config."""
        meters_df, _ = self.generator.generate()
        class_counts = meters_df['customer_class'].value_counts(normalize=True)
        
        for cls, expected_prob in self.config.customer_classes.items():
            actual_prob = class_counts.get(cls, 0)
            # Allow 10% tolerance for small sample size
            self.assertAlmostEqual(actual_prob, expected_prob, delta=0.1)
    
    def test_consumption_values_positive(self):
        """Test all consumption values are non-negative."""
        meters_df, _ = self.generator.generate()
        consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_consumption_')]
        
        for col in consumption_cols:
            self.assertTrue((meters_df[col] >= 0).all())
    
    def test_anomaly_labels_structure(self):
        """Test anomaly labels have correct structure."""
        _, anomaly_df = self.generator.generate()
        
        if len(anomaly_df) > 0:
            required_cols = ['meter_id', 'anomaly_flag', 'risk_band', 'anomaly_type']
            for col in required_cols:
                self.assertIn(col, anomaly_df.columns)
            
            # All flags should be 1
            self.assertTrue((anomaly_df['anomaly_flag'] == 1).all())
            
            # Risk bands should be valid
            valid_bands = {'High', 'Medium', 'Low'}
            self.assertTrue(set(anomaly_df['risk_band']).issubset(valid_bands))


class TestGeoJSONGenerator(unittest.TestCase):
    """Test GeoJSON generation."""
    
    def setUp(self):
        """Create test data."""
        self.config = GeneratorConfig(
            random_seed=42,
            num_transformers=3,
            num_meters=30,
            output_dir=Path(tempfile.mkdtemp())
        )
        
        tx_generator = TransformerGenerator(self.config)
        self.transformers_df = tx_generator.generate()
        
        meter_generator = MeterGenerator(self.config, self.transformers_df)
        self.meters_df, _ = meter_generator.generate()
        
        self.generator = GeoJSONGenerator(self.transformers_df, self.meters_df)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.config.output_dir, ignore_errors=True)
    
    def test_geojson_structure(self):
        """Test GeoJSON has correct structure."""
        geojson = self.generator.generate()
        
        self.assertEqual(geojson['type'], 'FeatureCollection')
        self.assertIn('features', geojson)
        self.assertIsInstance(geojson['features'], list)
    
    def test_feature_count(self):
        """Test one feature per transformer."""
        geojson = self.generator.generate()
        self.assertEqual(len(geojson['features']), len(self.transformers_df))
    
    def test_feature_properties(self):
        """Test features have required properties."""
        geojson = self.generator.generate()
        
        for feature in geojson['features']:
            self.assertEqual(feature['type'], 'Feature')
            self.assertIn('geometry', feature)
            self.assertIn('properties', feature)
            
            # Check geometry
            self.assertEqual(feature['geometry']['type'], 'Point')
            self.assertIn('coordinates', feature['geometry'])
            self.assertEqual(len(feature['geometry']['coordinates']), 2)
            
            # Check properties
            props = feature['properties']
            required_props = ['transformer_id', 'feeder_id', 'barangay', 
                            'capacity_kVA', 'num_meters', 'meter_ids']
            for prop in required_props:
                self.assertIn(prop, props)


class TestSyntheticDataPipeline(unittest.TestCase):
    """Test end-to-end pipeline integration."""
    
    def setUp(self):
        """Create test pipeline."""
        self.config = GeneratorConfig(
            random_seed=42,
            num_transformers=5,
            num_meters=50,
            num_months=6,
            anomaly_rate=0.1,
            output_dir=Path(tempfile.mkdtemp())
        )
        self.pipeline = SyntheticDataPipeline(self.config)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.config.output_dir, ignore_errors=True)
    
    def test_pipeline_execution(self):
        """Test full pipeline executes without errors."""
        outputs = self.pipeline.generate_all()
        
        self.assertIn('transformers_df', outputs)
        self.assertIn('meters_df', outputs)
        self.assertIn('anomaly_labels_df', outputs)
        self.assertIn('geojson', outputs)
    
    def test_output_validation(self):
        """Test pipeline validates outputs correctly."""
        outputs = self.pipeline.generate_all()
        
        # Should not raise any exceptions
        self.pipeline._validate_outputs(
            outputs['transformers_df'],
            outputs['meters_df'],
            outputs['anomaly_labels_df']
        )
    
    def test_file_saving(self):
        """Test outputs are saved to disk."""
        outputs = self.pipeline.generate_all()
        self.pipeline.save_outputs(outputs)
        
        # Check files exist
        expected_files = [
            'transformers.csv',
            'meter_consumption.csv',
            'anomaly_labels.csv',
            'transformers.geojson',
            'generation_report.txt'
        ]
        
        for filename in expected_files:
            filepath = self.config.output_dir / filename
            self.assertTrue(filepath.exists(), f"Missing file: {filename}")
    
    def test_reproducibility_across_runs(self):
        """Test multiple runs with same seed produce identical results."""
        outputs1 = self.pipeline.generate_all()
        
        # Create new pipeline with same config
        pipeline2 = SyntheticDataPipeline(self.config)
        outputs2 = pipeline2.generate_all()
        
        # Compare DataFrames
        pd.testing.assert_frame_equal(
            outputs1['transformers_df'],
            outputs2['transformers_df']
        )
        pd.testing.assert_frame_equal(
            outputs1['meters_df'],
            outputs2['meters_df']
        )
        pd.testing.assert_frame_equal(
            outputs1['anomaly_labels_df'],
            outputs2['anomaly_labels_df']
        )


if __name__ == '__main__':
    unittest.main()
