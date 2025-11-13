"""
Comprehensive Usage Examples for GhostLoad Mapper Data Loader
==============================================================

This script demonstrates real-world usage patterns for the data_loader module,
including common ML workflows, error handling, and best practices.

Run this script to see the data loader in action:
    python examples_data_loader.py

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    GhostLoadDataLoader,
    load_dataset,
    validate_dataset,
    DataSchema,
    LoadedData
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def example_1_basic_loading():
    """Example 1: Basic data loading with validation."""
    print_section("Example 1: Basic Data Loading")
    
    # Check if demo dataset exists
    dataset_path = Path('../datasets/demo')
    if not dataset_path.exists():
        print(f"⚠ Dataset not found: {dataset_path}")
        print("Please generate datasets first using generate_all_datasets.py")
        return None
    
    print(f"Loading dataset from: {dataset_path}\n")
    
    # Load dataset with validation
    data = load_dataset(
        dataset_dir=dataset_path,
        validate=True,
        compute_features=True
    )
    
    # Display loaded data
    print(f"✓ Successfully loaded dataset:")
    print(f"  - Meters: {len(data.meters):,} rows")
    print(f"  - Transformers: {len(data.transformers):,} rows")
    print(f"  - Anomalies: {len(data.anomalies) if data.anomalies is not None else 'N/A'} labels")
    print(f"  - Consumption matrix: {data.consumption_matrix.shape}")
    print(f"  - Feature matrix: {data.feature_matrix.shape if data.feature_matrix is not None else 'N/A'}")
    print(f"\nLoad time: {data.metadata['load_time_seconds']:.3f} seconds")
    
    return data


def example_2_consumption_analysis(data: LoadedData):
    """Example 2: Analyze consumption patterns."""
    print_section("Example 2: Consumption Pattern Analysis")
    
    if data is None:
        print("⚠ No data available. Skipping this example.")
        return
    
    consumption = data.consumption_matrix
    
    # Compute statistics
    print("Consumption Statistics:")
    print(f"  - Mean consumption: {np.mean(consumption):.2f} kWh")
    print(f"  - Median consumption: {np.median(consumption):.2f} kWh")
    print(f"  - Std deviation: {np.std(consumption):.2f} kWh")
    print(f"  - Min consumption: {np.min(consumption):.2f} kWh")
    print(f"  - Max consumption: {np.max(consumption):.2f} kWh")
    
    # Find meters with unusual patterns
    row_means = np.mean(consumption, axis=1)
    row_stds = np.std(consumption, axis=1)
    
    # High variability meters
    high_var_idx = np.argsort(row_stds)[-5:]
    print(f"\n📊 Top 5 meters with highest consumption variability:")
    for idx in high_var_idx:
        meter_id = data.meters.iloc[idx]['meter_id']
        mean_val = row_means[idx]
        std_val = row_stds[idx]
        cv = std_val / (mean_val + 1e-6)
        print(f"  - {meter_id}: mean={mean_val:.1f} kWh, std={std_val:.1f} kWh, CV={cv:.2f}")
    
    # Zero consumption detection
    zero_consumption = np.sum(consumption == 0, axis=1)
    zero_meters = np.where(zero_consumption > consumption.shape[1] * 0.3)[0]
    print(f"\n⚠ Found {len(zero_meters)} meters with >30% zero consumption months")


def example_3_anomaly_detection(data: LoadedData):
    """Example 3: Anomaly detection with Isolation Forest."""
    print_section("Example 3: Anomaly Detection with Isolation Forest")
    
    if data is None or data.consumption_matrix.shape[0] < 10:
        print("⚠ Insufficient data for ML training. Skipping this example.")
        return
    
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Prepare data
    X = data.consumption_matrix
    print(f"Training Isolation Forest on {X.shape[0]} meters with {X.shape[1]} months\n")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    contamination = 0.095  # Expected anomaly rate
    if data.anomalies is not None:
        contamination = len(data.anomalies) / len(data.meters)
        print(f"Using contamination={contamination:.3f} based on ground truth\n")
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)
    
    # Predict
    predictions = model.predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)
    
    # Count detections
    n_anomalies_detected = np.sum(predictions == -1)
    print(f"✓ Model trained successfully")
    print(f"  - Detected {n_anomalies_detected} anomalies ({n_anomalies_detected/len(X):.1%})")
    
    # Evaluate if ground truth available
    if data.anomalies is not None:
        y_true = data.meters['meter_id'].isin(data.anomalies['meter_id']).astype(int)
        y_pred = (predictions == -1).astype(int)
        
        print(f"\n📈 Model Performance (vs. ground truth):")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    # Top anomalies
    top_anomalies_idx = np.argsort(anomaly_scores)[:5]
    print(f"\n🚨 Top 5 most anomalous meters:")
    for idx in top_anomalies_idx:
        meter_id = data.meters.iloc[idx]['meter_id']
        score = anomaly_scores[idx]
        customer_class = data.meters.iloc[idx]['customer_class']
        mean_consumption = np.mean(X[idx])
        print(f"  - {meter_id} ({customer_class}): score={score:.3f}, mean={mean_consumption:.1f} kWh")


def example_4_feature_engineering(data: LoadedData):
    """Example 4: Statistical feature engineering."""
    print_section("Example 4: Statistical Feature Engineering")
    
    if data is None or data.feature_matrix is None:
        print("⚠ No feature matrix available. Skipping this example.")
        return
    
    features = data.feature_matrix
    
    print(f"Computed {len(features.columns)} statistical features:")
    print(f"  {', '.join(features.columns.tolist())}\n")
    
    # Feature correlations
    print("Feature Statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])
    
    # Find interesting patterns
    print(f"\n📊 Feature Insights:")
    
    # High variance meters
    high_cv_idx = features['consumption_cv'].nlargest(3).index
    print(f"\nMeters with highest coefficient of variation:")
    for idx in high_cv_idx:
        meter_id = data.meters.iloc[idx]['meter_id']
        cv = features.loc[idx, 'consumption_cv']
        print(f"  - {meter_id}: CV={cv:.2f}")
    
    # Trending meters
    high_trend_idx = features['consumption_trend'].nlargest(3).index
    print(f"\nMeters with strongest upward trend:")
    for idx in high_trend_idx:
        meter_id = data.meters.iloc[idx]['meter_id']
        trend = features.loc[idx, 'consumption_trend']
        print(f"  - {meter_id}: trend={trend:.2f} kWh/month")
    
    # Inactive meters
    inactive_idx = features['zero_consumption_ratio'].nlargest(3).index
    print(f"\nMeters with most inactive months:")
    for idx in inactive_idx:
        meter_id = data.meters.iloc[idx]['meter_id']
        zero_ratio = features.loc[idx, 'zero_consumption_ratio']
        print(f"  - {meter_id}: {zero_ratio:.1%} zero months")


def example_5_custom_validation():
    """Example 5: Custom validation constraints."""
    print_section("Example 5: Custom Validation Constraints")
    
    dataset_path = Path('../datasets/demo')
    if not dataset_path.exists():
        print(f"⚠ Dataset not found: {dataset_path}")
        return
    
    # Define stricter validation constraints
    custom_constraints = {
        'min_consumption': 0.0,
        'max_consumption': 3000.0,  # Stricter than default (10,000)
        'min_capacity': 50.0,  # Higher minimum capacity
        'max_capacity': 500.0,  # Lower maximum capacity
        'max_null_ratio': 0.05,  # Only 5% nulls allowed
    }
    
    print("Using custom validation constraints:")
    for key, value in custom_constraints.items():
        print(f"  - {key}: {value}")
    
    # Initialize with custom constraints
    loader = GhostLoadDataLoader(
        dataset_dir=dataset_path,
        validation_constraints=custom_constraints
    )
    
    try:
        # This might fail if data doesn't meet stricter constraints
        data = loader.load_all(validate=True, compute_features=False)
        print(f"\n✓ Validation passed with custom constraints")
        print(f"  Loaded {len(data.meters)} meters successfully")
    except ValueError as e:
        print(f"\n⚠ Validation failed (expected with stricter constraints):")
        print(f"  {str(e)[:200]}...")


def example_6_data_quality_report():
    """Example 6: Generate data quality report."""
    print_section("Example 6: Data Quality Report")
    
    dataset_path = Path('../datasets/demo')
    if not dataset_path.exists():
        print(f"⚠ Dataset not found: {dataset_path}")
        return
    
    loader = GhostLoadDataLoader(dataset_path)
    report = loader.get_data_quality_report()
    
    print("Data Quality Metrics:\n")
    
    # Meters quality
    print("📋 Meter Data:")
    print(f"  - Total records: {report['meters']['count']:,}")
    print(f"  - Columns: {report['meters']['columns']}")
    print(f"  - Null cells: {report['meters']['null_cells']:,}")
    print(f"  - Null ratio: {report['meters']['null_ratio']:.2%}")
    print(f"  - Duplicate IDs: {report['meters']['duplicate_ids']}")
    
    # Transformer quality
    print(f"\n🔌 Transformer Data:")
    print(f"  - Total records: {report['transformers']['count']:,}")
    print(f"  - Columns: {report['transformers']['columns']}")
    print(f"  - Null ratio: {report['transformers']['null_ratio']:.2%}")
    print(f"  - Duplicate IDs: {report['transformers']['duplicate_ids']}")
    
    # Anomaly distribution
    if 'anomalies' in report:
        print(f"\n🚨 Anomaly Labels:")
        print(f"  - Total anomalies: {report['anomalies']['count']:,}")
        print(f"  - Anomaly rate: {report['anomalies']['anomaly_rate']:.2%}")
        print(f"  - Risk band distribution:")
        for band, count in report['anomalies']['risk_band_distribution'].items():
            print(f"      {band}: {count}")
        print(f"  - Anomaly type distribution:")
        for atype, count in report['anomalies']['anomaly_type_distribution'].items():
            print(f"      {atype}: {count}")


def example_7_geospatial_analysis(data: LoadedData):
    """Example 7: Geospatial clustering analysis."""
    print_section("Example 7: Geospatial Analysis")
    
    if data is None:
        print("⚠ No data available. Skipping this example.")
        return
    
    # Extract coordinates
    coords = data.meters[['lat', 'lon']].values
    
    print(f"Geographic Distribution:")
    print(f"  - Latitude range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
    print(f"  - Longitude range: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
    
    # Count meters per barangay
    barangay_counts = data.meters['barangay'].value_counts()
    print(f"\n📍 Meters per Barangay:")
    for barangay, count in barangay_counts.head(5).items():
        print(f"  - {barangay}: {count} meters")
    
    # Count meters per transformer
    transformer_counts = data.meters['transformer_id'].value_counts()
    print(f"\n🔌 Meters per Transformer (Top 5):")
    for tx_id, count in transformer_counts.head(5).items():
        print(f"  - {tx_id}: {count} meters")
    
    # Customer class distribution
    class_dist = data.meters['customer_class'].value_counts()
    print(f"\n👥 Customer Class Distribution:")
    for customer_class, count in class_dist.items():
        print(f"  - {customer_class}: {count} ({count/len(data.meters):.1%})")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("  GHOSTLOAD MAPPER DATA LOADER - USAGE EXAMPLES")
    print("="*80)
    
    # Example 1: Basic loading
    data = example_1_basic_loading()
    
    # Example 2: Consumption analysis
    example_2_consumption_analysis(data)
    
    # Example 3: Anomaly detection
    example_3_anomaly_detection(data)
    
    # Example 4: Feature engineering
    example_4_feature_engineering(data)
    
    # Example 5: Custom validation
    example_5_custom_validation()
    
    # Example 6: Quality report
    example_6_data_quality_report()
    
    # Example 7: Geospatial analysis
    example_7_geospatial_analysis(data)
    
    # Summary
    print_section("Summary")
    print("* Completed all 7 examples successfully!")
    print("\nKey Takeaways:")
    print("  1. Use load_dataset() for simple workflows")
    print("  2. Use GhostLoadDataLoader for advanced customization")
    print("  3. Always enable validation in production")
    print("  4. Compute features for better ML performance")
    print("  5. Monitor data quality with quality reports")
    print("  6. Leverage geospatial and temporal patterns")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
