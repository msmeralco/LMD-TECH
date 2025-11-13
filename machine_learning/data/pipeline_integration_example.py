"""
End-to-End Integration Example: GhostLoad Mapper ML Pipeline
=============================================================

This script demonstrates the complete production workflow for detecting
electricity theft (non-technical losses) using the GhostLoad Mapper ML system.

Pipeline Flow:
    1. Data Loading (data_loader.py)
       - Load meter and transformer CSV files
       - Validate data quality
       - Extract consumption matrices
    
    2. Data Preprocessing (data_preprocessor.py)
       - Impute missing consumption values
       - Cap outliers at 3σ from transformer median
       - Normalize to [0,1] per transformer group
    
    3. Feature Engineering (feature_engineer.py)
       - Compute transformer baseline statistics
       - Calculate consumption trends (6-month slope)
       - Generate peer comparison ratios
    
    4. Anomaly Detection (scikit-learn Isolation Forest)
       - Train unsupervised anomaly detector
       - Score all meters for theft risk
       - Identify top suspicious meters

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import GhostLoadDataLoader, DataSchema
from data_preprocessor import DataPreprocessor, PreprocessorConfig, ImputationStrategy
from feature_engineer import FeatureEngineer, FeatureConfig, TrendMethod


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def create_synthetic_dataset():
    """
    Create synthetic meter dataset with known theft patterns.
    
    Returns:
        Tuple of (meters_df, transformers_df, ground_truth_labels)
    """
    print_section_header("CREATING SYNTHETIC DATASET")
    
    np.random.seed(42)
    n_meters = 500
    n_months = 12
    n_transformers = 25
    theft_rate = 0.15  # 15% theft prevalence (realistic for high-loss areas)
    
    print(f"Generating {n_meters} meters across {n_transformers} transformers")
    print(f"Theft rate: {theft_rate:.1%} (known ground truth for validation)")
    
    meters_data = []
    ground_truth = []
    
    for i in range(n_meters):
        transformer_id = f'TX_{(i % n_transformers):03d}'
        transformer_idx = i % n_transformers
        
        # Base consumption varies by transformer capacity
        base_consumption = 150 + transformer_idx * 20
        
        # Determine if meter is a thief
        is_theft = np.random.random() < theft_rate
        
        # Generate consumption time series
        consumption = []
        for month in range(n_months):
            if not is_theft:
                # Normal meter: stable with seasonal variation
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
                value = base_consumption * seasonal_factor + np.random.normal(0, 15)
            else:
                # Theft meter: declining trend after tampering
                if month < 3:
                    # Normal consumption before theft
                    value = base_consumption + np.random.normal(0, 15)
                else:
                    # Declining consumption after tampering (month 3)
                    decline_rate = np.random.uniform(5, 15)  # kWh per month
                    value = base_consumption - (month - 3) * decline_rate
                    value += np.random.normal(0, 10)
            
            consumption.append(max(5, value))  # Minimum 5 kWh (meter active)
        
        # Inject some missing values (10% randomly)
        consumption_with_nulls = consumption.copy()
        null_indices = np.random.choice(
            n_months, 
            size=int(n_months * 0.1), 
            replace=False
        )
        for idx in null_indices:
            consumption_with_nulls[idx] = np.nan
        
        meter_record = {
            'meter_id': f'MTR_{i:05d}',
            'transformer_id': transformer_id,
            'customer_class': 'residential',
            'latitude': 14.5995 + np.random.normal(0, 0.01),
            'longitude': 120.9842 + np.random.normal(0, 0.01),
        }
        
        # Add monthly consumption
        for month_idx, value in enumerate(consumption_with_nulls):
            meter_record[f'monthly_consumption_2024{month_idx+1:02d}'] = value
        
        meters_data.append(meter_record)
        ground_truth.append(1 if is_theft else 0)  # 1 = theft, 0 = normal
    
    meters_df = pd.DataFrame(meters_data)
    
    # Create transformers dataframe
    transformers_data = []
    for tx_id in range(n_transformers):
        transformers_data.append({
            'transformer_id': f'TX_{tx_id:03d}',
            'capacity_kva': np.random.choice([50, 75, 100, 150, 200]),
            'latitude': 14.5995 + np.random.normal(0, 0.02),
            'longitude': 120.9842 + np.random.normal(0, 0.02),
            'installation_year': np.random.randint(2010, 2020)
        })
    
    transformers_df = pd.DataFrame(transformers_data)
    
    print(f"✓ Dataset created: {len(meters_df)} meters, {len(transformers_df)} transformers")
    print(f"  Ground truth: {sum(ground_truth)} theft cases ({sum(ground_truth)/len(ground_truth):.1%})")
    
    return meters_df, transformers_df, np.array(ground_truth)


def main():
    """Execute complete ML pipeline."""
    
    print("\n" + "="*80)
    print("  GHOSTLOAD MAPPER - END-TO-END ML PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Step 1: Create synthetic dataset
    meters_df, transformers_df, ground_truth = create_synthetic_dataset()
    
    # Identify consumption columns
    consumption_cols = [col for col in meters_df.columns if col.startswith('monthly_consumption_')]
    
    # ==========================================================================
    # STAGE 1: DATA PREPROCESSING
    # ==========================================================================
    print_section_header("STAGE 1: DATA PREPROCESSING")
    
    preprocessor_config = PreprocessorConfig(
        outlier_threshold=3.0,
        imputation_strategy=ImputationStrategy.FORWARD_FILL,
        normalization_method='minmax',
        transformer_column='transformer_id',
        verbose=False
    )
    
    preprocessor = DataPreprocessor(preprocessor_config)
    preprocess_result = preprocessor.preprocess(meters_df, consumption_cols)
    
    print("\nPreprocessing Results:")
    print(f"  Missing values imputed: {preprocess_result.imputation_stats['imputed']:,}")
    print(f"  Outliers treated: {preprocess_result.outlier_stats['outliers_treated']:,}")
    print(f"  Normalization: {preprocess_result.normalization_stats['method']}")
    
    preprocessed_df = preprocess_result.data
    
    # ==========================================================================
    # STAGE 2: FEATURE ENGINEERING
    # ==========================================================================
    print_section_header("STAGE 2: FEATURE ENGINEERING")
    
    feature_config = FeatureConfig(
        trend_window=6,
        trend_method=TrendMethod.LINEAR_REGRESSION,
        baseline_statistic='median',
        variance_statistic='variance',
        transformer_column='transformer_id',
        robust_estimators=False,
        verbose=False
    )
    
    engineer = FeatureEngineer(feature_config)
    feature_result = engineer.engineer_features(preprocessed_df, consumption_cols)
    
    print("\nFeature Engineering Results:")
    print(f"  Features created: {len(feature_result.feature_names)}")
    for feature in feature_result.feature_names:
        coverage = feature_result.metadata['feature_coverage'][feature]['coverage']
        print(f"    - {feature}: {coverage:.1%} coverage")
    
    print(f"\n  Trend Statistics:")
    print(f"    Mean trend: {feature_result.trend_stats['mean_trend']:.2f} kWh/month")
    print(f"    Declining meters: {feature_result.trend_stats['pct_declining']:.1%}")
    
    print(f"\n  Consumption Ratio Statistics:")
    print(f"    Low consumers (< 0.5): {feature_result.ratio_stats['pct_low_consumers']:.1%}")
    print(f"    High consumers (> 2.0): {feature_result.ratio_stats['pct_high_consumers']:.1%}")
    
    features_df = feature_result.data
    
    # ==========================================================================
    # STAGE 3: ANOMALY DETECTION (ISOLATION FOREST)
    # ==========================================================================
    print_section_header("STAGE 3: ANOMALY DETECTION")
    
    # Prepare feature matrix
    feature_cols = [
        'transformer_baseline_median',
        'transformer_baseline_variance',
        'meter_consumption_trend',
        'consumption_ratio_to_transformer_median'
    ] + consumption_cols
    
    X = features_df[feature_cols].values
    
    # Handle any remaining NaN (shouldn't be any after preprocessing)
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize features (important for Isolation Forest)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"  Raw features: {len(consumption_cols)} consumption months")
    print(f"  Engineered features: {len(feature_cols) - len(consumption_cols)}")
    
    # Train Isolation Forest
    contamination = 0.15  # Expected theft rate
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\nTraining Isolation Forest (contamination={contamination:.1%})...")
    iso_forest.fit(X_scaled)
    
    # Predict anomalies
    predictions = iso_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_scores = iso_forest.score_samples(X_scaled)  # Lower = more anomalous
    
    # Convert predictions to binary (1 = theft, 0 = normal)
    predicted_theft = (predictions == -1).astype(int)
    
    # ==========================================================================
    # STAGE 4: EVALUATION
    # ==========================================================================
    print_section_header("STAGE 4: MODEL EVALUATION")
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(
        ground_truth, 
        predicted_theft,
        target_names=['Normal', 'Theft'],
        digits=3
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(ground_truth, predicted_theft)
    print("\nConfusion Matrix:")
    print(f"                  Predicted Normal  Predicted Theft")
    print(f"Actual Normal     {cm[0,0]:16d}  {cm[0,1]:15d}")
    print(f"Actual Theft      {cm[1,0]:16d}  {cm[1,1]:15d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nKey Metrics:")
    print(f"  Precision: {precision:.1%} (of flagged meters, % actually theft)")
    print(f"  Recall:    {recall:.1%} (of actual theft, % detected)")
    print(f"  F1-Score:  {f1:.1%}")
    
    # ==========================================================================
    # STAGE 5: INSIGHTS & RECOMMENDATIONS
    # ==========================================================================
    print_section_header("STAGE 5: ACTIONABLE INSIGHTS")
    
    # Add predictions to dataframe
    results_df = features_df.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['predicted_theft'] = predicted_theft
    results_df['actual_theft'] = ground_truth
    
    # Sort by anomaly score (most suspicious first)
    results_df = results_df.sort_values('anomaly_score')
    
    # Top 10 most suspicious meters
    print("\nTop 10 Most Suspicious Meters:")
    print("-" * 80)
    top_suspicious = results_df.head(10)
    
    display_cols = [
        'meter_id', 
        'transformer_id',
        'meter_consumption_trend',
        'consumption_ratio_to_transformer_median',
        'anomaly_score',
        'predicted_theft',
        'actual_theft'
    ]
    
    for idx, row in top_suspicious.iterrows():
        print(f"\n{row['meter_id']} (Transformer: {row['transformer_id']})")
        print(f"  Consumption trend:  {row['meter_consumption_trend']:7.2f} kWh/month")
        print(f"  Ratio to peer median: {row['consumption_ratio_to_transformer_median']:5.2f}")
        print(f"  Anomaly score:      {row['anomaly_score']:7.3f}")
        print(f"  Prediction: {'THEFT' if row['predicted_theft'] == 1 else 'NORMAL'} "
              f"(Actual: {'THEFT' if row['actual_theft'] == 1 else 'NORMAL'})")
    
    # Field inspection recommendations
    true_positives = results_df[
        (results_df['predicted_theft'] == 1) & (results_df['actual_theft'] == 1)
    ]
    false_positives = results_df[
        (results_df['predicted_theft'] == 1) & (results_df['actual_theft'] == 0)
    ]
    
    print("\n" + "-" * 80)
    print("\nField Inspection Recommendations:")
    print(f"  Total flagged meters: {predicted_theft.sum()}")
    print(f"  Confirmed theft cases: {len(true_positives)} ({len(true_positives)/predicted_theft.sum():.1%} precision)")
    print(f"  False alarms: {len(false_positives)}")
    
    # Prioritization by transformer
    print("\nHigh-Risk Transformers (>20% flagged meters):")
    transformer_risk = results_df.groupby('transformer_id').agg({
        'predicted_theft': ['sum', 'count']
    })
    transformer_risk.columns = ['flagged', 'total']
    transformer_risk['risk_rate'] = transformer_risk['flagged'] / transformer_risk['total']
    high_risk_tx = transformer_risk[transformer_risk['risk_rate'] > 0.20].sort_values('risk_rate', ascending=False)
    
    if len(high_risk_tx) > 0:
        for tx_id, row in high_risk_tx.iterrows():
            print(f"  {tx_id}: {row['flagged']}/{row['total']} meters flagged ({row['risk_rate']:.1%})")
    else:
        print("  None (all transformers have < 20% flagged meters)")
    
    # ==========================================================================
    # STAGE 6: FEATURE IMPORTANCE ANALYSIS
    # ==========================================================================
    print_section_header("STAGE 6: FEATURE IMPORTANCE")
    
    # Analyze which features distinguish theft best
    theft_cases = results_df[results_df['actual_theft'] == 1]
    normal_cases = results_df[results_df['actual_theft'] == 0]
    
    print("\nFeature Comparison (Theft vs Normal):")
    print("-" * 80)
    
    analysis_features = [
        ('meter_consumption_trend', 'kWh/month'),
        ('consumption_ratio_to_transformer_median', 'ratio'),
        ('transformer_baseline_variance', 'variance')
    ]
    
    for feature, unit in analysis_features:
        theft_mean = theft_cases[feature].mean()
        normal_mean = normal_cases[feature].mean()
        difference = theft_mean - normal_mean
        
        print(f"\n{feature}:")
        print(f"  Theft cases:  {theft_mean:8.2f} {unit}")
        print(f"  Normal cases: {normal_mean:8.2f} {unit}")
        print(f"  Difference:   {difference:8.2f} {unit} ({(difference/normal_mean)*100:+.1f}%)")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print_section_header("PIPELINE EXECUTION COMPLETE")
    
    print("\nSummary:")
    print(f"  Dataset: {len(meters_df)} meters, {len(transformers_df)} transformers")
    print(f"  Preprocessing: {preprocess_result.metadata['processing_time_seconds']:.2f}s")
    print(f"  Feature engineering: {feature_result.metadata['processing_time_seconds']:.2f}s")
    print(f"  Anomaly detection: Isolation Forest (100 trees)")
    print(f"\nPerformance:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1-Score:  {f1:.1%}")
    print(f"\nRecommendation:")
    print(f"  Inspect {predicted_theft.sum()} flagged meters for potential theft")
    print(f"  Expected true positives: ~{int(predicted_theft.sum() * precision)}")
    
    print("\n" + "="*80)
    print("  PIPELINE DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
    
    return results_df


if __name__ == "__main__":
    results = main()
    
    print("\nResults saved to 'results' DataFrame variable")
    print("You can further analyze with:")
    print("  - results[results['predicted_theft'] == 1]  # All flagged meters")
    print("  - results.groupby('transformer_id')['predicted_theft'].sum()  # Theft by transformer")
    print("  - results[['meter_consumption_trend', 'consumption_ratio_to_transformer_median']].corr()")
