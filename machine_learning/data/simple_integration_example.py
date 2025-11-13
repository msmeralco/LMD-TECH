"""
Simple Integration Example: Feature Engineering Pipeline
==========================================================

This script demonstrates how to use data_preprocessor.py and feature_engineer.py
together in a complete preprocessing and feature engineering workflow.

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessor import DataPreprocessor, PreprocessorConfig, ImputationStrategy
from feature_engineer import FeatureEngineer, FeatureConfig, TrendMethod


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def create_sample_dataset():
    """Create sample meter dataset."""
    print_section_header("CREATING SAMPLE DATASET")
    
    np.random.seed(42)
    n_meters = 200
    n_months = 12
    n_transformers = 15
    
    print(f"Generating {n_meters} meters across {n_transformers} transformers...")
    
    meters_data = []
    
    for i in range(n_meters):
        transformer_id = f'TX_{(i % n_transformers):03d}'
        transformer_idx = i % n_transformers
        
        # Base consumption varies by transformer
        base_consumption = 120 + transformer_idx * 15
        
        # 20% of meters have declining consumption (suspicious)
        is_suspicious = i < n_meters * 0.2
        
        # Generate consumption time series
        consumption = []
        for month in range(n_months):
            if not is_suspicious:
                # Normal meter: stable consumption
                value = base_consumption + np.random.normal(0, 12)
            else:
                # Suspicious meter: declining trend
                decline_rate = np.random.uniform(4, 10)
                value = base_consumption - month * decline_rate
                value += np.random.normal(0, 8)
            
            consumption.append(max(10, value))
        
        # Inject missing values (15% randomly)
        consumption_with_nulls = consumption.copy()
        null_indices = np.random.choice(n_months, size=2, replace=False)
        for idx in null_indices:
            consumption_with_nulls[idx] = np.nan
        
        meter_record = {
            'meter_id': f'MTR_{i:05d}',
            'transformer_id': transformer_id,
            'customer_class': 'residential',
        }
        
        # Add monthly consumption
        for month_idx, value in enumerate(consumption_with_nulls):
            meter_record[f'monthly_consumption_2024{month_idx+1:02d}'] = value
        
        meters_data.append(meter_record)
    
    meters_df = pd.DataFrame(meters_data)
    
    print(f"[OK] Dataset created: {len(meters_df)} meters across {n_transformers} transformers")
    print(f"     Missing values: {meters_df.filter(like='monthly_consumption').isnull().sum().sum()}")
    
    return meters_df


def main():
    """Execute preprocessing and feature engineering pipeline."""
    
    print("\n" + "="*80)
    print("  GHOSTLOAD MAPPER - PREPROCESSING & FEATURE ENGINEERING DEMO")
    print("="*80)
    
    # Create sample dataset
    meters_df = create_sample_dataset()
    consumption_cols = [col for col in meters_df.columns if col.startswith('monthly_consumption_')]
    
    # ==========================================================================
    # STAGE 1: DATA PREPROCESSING
    # ==========================================================================
    print_section_header("STAGE 1: DATA PREPROCESSING")
    
    print("\nConfiguring preprocessor...")
    preprocessor_config = PreprocessorConfig(
        outlier_threshold=3.0,
        imputation_strategy=ImputationStrategy.FORWARD_FILL,
        normalization_method='minmax',
        transformer_column='transformer_id',
        verbose=True
    )
    
    preprocessor = DataPreprocessor(preprocessor_config)
    print("\nExecuting preprocessing pipeline...")
    preprocess_result = preprocessor.preprocess(meters_df, consumption_cols)
    
    print("\n" + preprocessor.get_preprocessing_summary(preprocess_result))
    
    preprocessed_df = preprocess_result.data
    
    # ==========================================================================
    # STAGE 2: FEATURE ENGINEERING
    # ==========================================================================
    print_section_header("STAGE 2: FEATURE ENGINEERING")
    
    print("\nConfiguring feature engineer...")
    feature_config = FeatureConfig(
        trend_window=6,
        trend_method=TrendMethod.LINEAR_REGRESSION,
        baseline_statistic='median',
        variance_statistic='variance',
        transformer_column='transformer_id',
        robust_estimators=False,
        verbose=True
    )
    
    engineer = FeatureEngineer(feature_config)
    print("\nExecuting feature engineering pipeline...")
    feature_result = engineer.engineer_features(preprocessed_df, consumption_cols)
    
    print("\n" + engineer.get_feature_summary(feature_result))
    
    features_df = feature_result.data
    
    # ==========================================================================
    # STAGE 3: ANALYSIS & INSIGHTS
    # ==========================================================================
    print_section_header("STAGE 3: FEATURE ANALYSIS")
    
    # Analyze engineered features
    print("\nFeature Statistics:")
    print("-" * 80)
    
    feature_cols = [
        'transformer_baseline_median',
        'transformer_baseline_variance',
        'meter_consumption_trend',
        'consumption_ratio_to_transformer_median'
    ]
    
    stats_df = features_df[feature_cols].describe()
    print(stats_df.to_string())
    
    # Identify suspicious meters based on feature thresholds
    print("\n" + "-" * 80)
    print("Suspicious Meter Detection (Rule-Based):")
    print("-" * 80)
    
    # Define suspicion criteria
    low_ratio_threshold = 0.6  # < 60% of transformer median
    negative_trend_threshold = -5  # Declining > 5 kWh/month
    
    suspicious_meters = features_df[
        (features_df['consumption_ratio_to_transformer_median'] < low_ratio_threshold) &
        (features_df['meter_consumption_trend'] < negative_trend_threshold)
    ]
    
    print(f"\nCriteria:")
    print(f"  1. Consumption ratio < {low_ratio_threshold} (abnormally low vs peers)")
    print(f"  2. Trend slope < {negative_trend_threshold} kWh/month (declining consumption)")
    
    print(f"\nResults:")
    print(f"  Suspicious meters identified: {len(suspicious_meters)} / {len(features_df)}")
    print(f"  Suspicion rate: {len(suspicious_meters)/len(features_df):.1%}")
    
    if len(suspicious_meters) > 0:
        print(f"\nTop 10 Most Suspicious Meters:")
        print("-" * 80)
        
        # Sort by combined suspicion score (lower is worse)
        suspicious_meters['suspicion_score'] = (
            suspicious_meters['consumption_ratio_to_transformer_median'] +
            suspicious_meters['meter_consumption_trend'] / 100  # Normalize trend
        )
        
        top_suspicious = suspicious_meters.nsmallest(10, 'suspicion_score')
        
        display_cols = [
            'meter_id',
            'transformer_id',
            'consumption_ratio_to_transformer_median',
            'meter_consumption_trend',
            'transformer_baseline_median'
        ]
        
        print(top_suspicious[display_cols].to_string(index=False))
    
    # Transformer-level analysis
    print("\n" + "-" * 80)
    print("Transformer Risk Analysis:")
    print("-" * 80)
    
    features_df['is_suspicious'] = (
        (features_df['consumption_ratio_to_transformer_median'] < low_ratio_threshold) &
        (features_df['meter_consumption_trend'] < negative_trend_threshold)
    ).astype(int)
    
    transformer_analysis = features_df.groupby('transformer_id').agg({
        'is_suspicious': ['sum', 'count'],
        'consumption_ratio_to_transformer_median': 'mean',
        'meter_consumption_trend': 'mean'
    })
    
    transformer_analysis.columns = [
        'suspicious_count', 'total_meters', 'avg_ratio', 'avg_trend'
    ]
    transformer_analysis['suspicion_rate'] = (
        transformer_analysis['suspicious_count'] / transformer_analysis['total_meters']
    )
    
    # High-risk transformers (>20% suspicious meters)
    high_risk = transformer_analysis[transformer_analysis['suspicion_rate'] > 0.20]
    high_risk = high_risk.sort_values('suspicion_rate', ascending=False)
    
    if len(high_risk) > 0:
        print(f"\nHigh-Risk Transformers (>20% suspicious meters):")
        for tx_id, row in high_risk.head(10).iterrows():
            print(f"\n  {tx_id}:")
            print(f"    Suspicious: {row['suspicious_count']:.0f} / {row['total_meters']:.0f} "
                  f"({row['suspicion_rate']:.1%})")
            print(f"    Avg ratio to median: {row['avg_ratio']:.2f}")
            print(f"    Avg trend: {row['avg_trend']:.2f} kWh/month")
    else:
        print("\n  No high-risk transformers detected (all < 20% suspicion rate)")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print_section_header("PIPELINE EXECUTION SUMMARY")
    
    print(f"\nDataset: {len(meters_df)} meters, {len(consumption_cols)} months")
    print(f"\nPreprocessing:")
    print(f"  Missing values imputed: {preprocess_result.imputation_stats['imputed']}")
    print(f"  Outliers treated: {preprocess_result.outlier_stats['outliers_treated']}")
    print(f"  Processing time: {preprocess_result.metadata['processing_time_seconds']:.2f}s")
    
    print(f"\nFeature Engineering:")
    print(f"  Features created: {len(feature_result.feature_names)}")
    print(f"  Transformers analyzed: {feature_result.metadata['n_transformers']}")
    print(f"  Processing time: {feature_result.metadata['processing_time_seconds']:.2f}s")
    
    print(f"\nSuspicious Meter Detection:")
    print(f"  Total flagged: {len(suspicious_meters)} ({len(suspicious_meters)/len(features_df):.1%})")
    print(f"  High-risk transformers: {len(high_risk)}")
    
    print("\n" + "="*80)
    print("  DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("  1. Train Isolation Forest or other ML model on engineered features")
    print("  2. Deploy field inspections to flagged meters")
    print("  3. Update model with inspection results (supervised learning)")
    print("  4. Monitor consumption trends for early detection")
    
    return features_df


if __name__ == "__main__":
    results = main()
    
    print("\nResults saved to 'results' DataFrame variable")
    print("Explore the engineered features:")
    print("  - results[['meter_id', 'consumption_ratio_to_transformer_median', 'meter_consumption_trend']].head()")
    print("  - results.groupby('transformer_id')['meter_consumption_trend'].mean()")
