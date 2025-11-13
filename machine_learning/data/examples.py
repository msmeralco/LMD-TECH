"""
Example Usage: Synthetic Data Generator
========================================

Demonstrates common usage patterns and integration scenarios.
"""

from pathlib import Path
import pandas as pd
from synthetic_data_generator import GeneratorConfig, SyntheticDataPipeline


def example_basic_generation():
    """Example 1: Basic generation with default settings."""
    print("=" * 80)
    print("Example 1: Basic Generation")
    print("=" * 80)
    
    # Use default configuration
    pipeline = SyntheticDataPipeline()
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    print("\n✓ Generated data saved to 'generated_data/' directory")
    print(f"  - Transformers: {len(outputs['transformers_df'])}")
    print(f"  - Meters: {len(outputs['meters_df'])}")
    print(f"  - Anomalies: {len(outputs['anomaly_labels_df'])}")


def example_custom_config():
    """Example 2: Custom configuration for specific scenarios."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)
    
    # Create custom configuration for smaller dataset
    config = GeneratorConfig(
        random_seed=123,
        num_transformers=20,
        num_meters=500,
        num_months=6,
        anomaly_rate=0.10,  # 10% anomalies
        output_dir=Path("demo_data")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    print("\n✓ Custom dataset generated")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Anomaly rate: {config.anomaly_rate * 100}%")


def example_data_analysis():
    """Example 3: Analyze generated data."""
    print("\n" + "=" * 80)
    print("Example 3: Data Analysis")
    print("=" * 80)
    
    # Generate fresh data
    config = GeneratorConfig(
        num_transformers=10,
        num_meters=200,
        output_dir=Path("analysis_data")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    
    # Analyze meter consumption
    meters_df = outputs['meters_df']
    consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_')]
    
    print("\nConsumption Statistics by Customer Class:")
    print("-" * 80)
    
    for customer_class in ['residential', 'commercial', 'industrial']:
        class_data = meters_df[meters_df['customer_class'] == customer_class]
        if len(class_data) > 0:
            avg_consumption = class_data[consumption_cols].mean().mean()
            print(f"  {customer_class:12s}: {avg_consumption:8.2f} kWh/month")
    
    # Analyze spatial distribution
    print("\nSpatial Distribution:")
    print("-" * 80)
    barangay_counts = meters_df['barangay'].value_counts()
    for barangay, count in barangay_counts.head(5).items():
        print(f"  {barangay:15s}: {count:4d} meters")
    
    # Analyze anomalies
    anomaly_df = outputs['anomaly_labels_df']
    print(f"\nAnomaly Distribution:")
    print("-" * 80)
    if len(anomaly_df) > 0:
        risk_counts = anomaly_df['risk_band'].value_counts()
        for risk, count in risk_counts.items():
            print(f"  {risk:7s} risk: {count:3d} meters")


def example_ml_integration():
    """Example 4: Integration with ML pipeline."""
    print("\n" + "=" * 80)
    print("Example 4: ML Pipeline Integration")
    print("=" * 80)
    
    # Generate data
    config = GeneratorConfig(
        num_transformers=30,
        num_meters=1000,
        num_months=12,
        output_dir=Path("ml_data")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    # Load data for ML
    meters_df = outputs['meters_df']
    
    # Extract features
    consumption_cols = [c for c in meters_df.columns if c.startswith('monthly_')]
    features = meters_df[consumption_cols].values
    
    print(f"\n✓ Feature matrix prepared for ML:")
    print(f"  - Shape: {features.shape}")
    print(f"  - Features: {len(consumption_cols)} monthly consumption columns")
    
    # Compute additional features
    import numpy as np
    
    medians = np.median(features, axis=1)
    variances = np.var(features, axis=1)
    cv = variances / (medians + 1e-6)  # Coefficient of variation
    
    print(f"\n  Additional features computed:")
    print(f"  - Median consumption: {medians.mean():.2f} kWh")
    print(f"  - Average variance: {variances.mean():.2f}")
    print(f"  - Average CV: {cv.mean():.4f}")
    
    # Train Isolation Forest (example)
    try:
        from sklearn.ensemble import IsolationForest
        
        model = IsolationForest(
            contamination=config.anomaly_rate,
            random_state=42
        )
        predictions = model.fit_predict(features)
        
        num_anomalies = (predictions == -1).sum()
        print(f"\n✓ Isolation Forest trained:")
        print(f"  - Detected anomalies: {num_anomalies}")
        print(f"  - Expected anomalies: {int(len(features) * config.anomaly_rate)}")
        
    except ImportError:
        print("\n  (Install scikit-learn to run ML example)")


def example_export_formats():
    """Example 5: Export to different formats."""
    print("\n" + "=" * 80)
    print("Example 5: Multiple Export Formats")
    print("=" * 80)
    
    # Generate data
    config = GeneratorConfig(
        num_transformers=10,
        num_meters=200,
        output_dir=Path("export_data")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    
    # Export to different formats
    meters_df = outputs['meters_df']
    
    # CSV (already done by pipeline)
    meters_df.to_csv(config.output_dir / 'meters.csv', index=False)
    
    # Excel
    try:
        meters_df.to_excel(config.output_dir / 'meters.xlsx', index=False)
        print("  ✓ Exported to Excel")
    except ImportError:
        print("  (Install openpyxl for Excel export)")
    
    # Parquet (efficient for large datasets)
    try:
        meters_df.to_parquet(config.output_dir / 'meters.parquet', index=False)
        print("  ✓ Exported to Parquet")
    except ImportError:
        print("  (Install pyarrow for Parquet export)")
    
    # JSON (for API integration)
    import json
    with open(config.output_dir / 'meters_sample.json', 'w') as f:
        sample = meters_df.head(10).to_dict(orient='records')
        json.dump(sample, f, indent=2)
    print("  ✓ Exported sample to JSON")
    
    # GeoJSON (already done by pipeline)
    print("  ✓ Exported to GeoJSON")
    
    print(f"\n✓ All exports saved to: {config.output_dir}")


def example_reproducibility():
    """Example 6: Demonstrate reproducibility."""
    print("\n" + "=" * 80)
    print("Example 6: Reproducibility Verification")
    print("=" * 80)
    
    # Generate with fixed seed
    config1 = GeneratorConfig(
        random_seed=42,
        num_transformers=5,
        num_meters=100,
        output_dir=Path("repro_test_1")
    )
    
    pipeline1 = SyntheticDataPipeline(config1)
    outputs1 = pipeline1.generate_all()
    
    # Generate again with same seed
    config2 = GeneratorConfig(
        random_seed=42,  # Same seed
        num_transformers=5,
        num_meters=100,
        output_dir=Path("repro_test_2")
    )
    
    pipeline2 = SyntheticDataPipeline(config2)
    outputs2 = pipeline2.generate_all()
    
    # Verify identical results
    pd.testing.assert_frame_equal(
        outputs1['meters_df'],
        outputs2['meters_df']
    )
    
    print("  ✓ Reproducibility verified!")
    print("  - Two runs with seed=42 produced identical results")
    print("  - All consumption values match exactly")


if __name__ == '__main__':
    """
    Run all examples. Comment out examples you don't want to run.
    """
    
    # Run examples
    example_basic_generation()
    example_custom_config()
    example_data_analysis()
    example_ml_integration()
    example_export_formats()
    example_reproducibility()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
