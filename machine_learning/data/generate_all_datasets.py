"""
Generate All Datasets for GhostLoad Mapper Development
=======================================================

This script generates multiple datasets for different development phases:
1. Demo dataset (fast iteration, testing)
2. Development dataset (local ML training)
3. Validation dataset (model testing)
4. Full dataset (production-scale simulation)

Run this once to generate all datasets needed for the hackathon.
"""

import logging
from pathlib import Path
from datetime import datetime
from synthetic_data_generator import GeneratorConfig, SyntheticDataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_demo_dataset():
    """
    Generate small demo dataset for rapid testing and presentation.
    
    - 10 transformers
    - 200 meters
    - 6 months of data
    - ~15 anomalies (7.5%)
    
    Use: Quick demos, UI testing, presentation backup
    Generation time: <1 second
    """
    logger.info("=" * 80)
    logger.info("Generating DEMO dataset...")
    logger.info("=" * 80)
    
    config = GeneratorConfig(
        random_seed=100,
        num_transformers=10,
        num_meters=200,
        num_months=6,
        anomaly_rate=0.075,
        output_dir=Path("../datasets/demo")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    logger.info(f"✅ Demo dataset generated: {len(outputs['meters_df'])} meters, "
                f"{len(outputs['anomaly_labels_df'])} anomalies")
    logger.info(f"   Location: {config.output_dir}")
    logger.info("")
    
    return outputs


def generate_development_dataset():
    """
    Generate medium-sized dataset for ML development.
    
    - 30 transformers
    - 1,000 meters
    - 12 months of data
    - ~75 anomalies (7.5%)
    
    Use: Local ML training, feature engineering, algorithm testing
    Generation time: ~2 seconds
    """
    logger.info("=" * 80)
    logger.info("Generating DEVELOPMENT dataset...")
    logger.info("=" * 80)
    
    config = GeneratorConfig(
        random_seed=200,
        num_transformers=30,
        num_meters=1000,
        num_months=12,
        anomaly_rate=0.075,
        output_dir=Path("../datasets/development")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    logger.info(f"✅ Development dataset generated: {len(outputs['meters_df'])} meters, "
                f"{len(outputs['anomaly_labels_df'])} anomalies")
    logger.info(f"   Location: {config.output_dir}")
    logger.info("")
    
    return outputs


def generate_validation_dataset():
    """
    Generate dataset for model validation with different anomaly rate.
    
    - 30 transformers
    - 1,000 meters
    - 12 months of data
    - ~100 anomalies (10%)
    
    Use: Model testing, threshold tuning, validation metrics
    Generation time: ~2 seconds
    """
    logger.info("=" * 80)
    logger.info("Generating VALIDATION dataset...")
    logger.info("=" * 80)
    
    config = GeneratorConfig(
        random_seed=300,
        num_transformers=30,
        num_meters=1000,
        num_months=12,
        anomaly_rate=0.10,  # Higher anomaly rate for validation
        output_dir=Path("../datasets/validation")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    logger.info(f"✅ Validation dataset generated: {len(outputs['meters_df'])} meters, "
                f"{len(outputs['anomaly_labels_df'])} anomalies")
    logger.info(f"   Location: {config.output_dir}")
    logger.info("")
    
    return outputs


def generate_full_dataset():
    """
    Generate full-scale dataset for production simulation.
    
    - 50 transformers
    - 2,000 meters
    - 12 months of data
    - ~150 anomalies (7.5%)
    
    Use: Production testing, performance benchmarking, final validation
    Generation time: ~3 seconds
    """
    logger.info("=" * 80)
    logger.info("Generating FULL dataset...")
    logger.info("=" * 80)
    
    config = GeneratorConfig(
        random_seed=42,  # Production seed
        num_transformers=50,
        num_meters=2000,
        num_months=12,
        anomaly_rate=0.075,
        output_dir=Path("../datasets/production")
    )
    
    pipeline = SyntheticDataPipeline(config)
    outputs = pipeline.generate_all()
    pipeline.save_outputs(outputs)
    
    logger.info(f"✅ Full dataset generated: {len(outputs['meters_df'])} meters, "
                f"{len(outputs['anomaly_labels_df'])} anomalies")
    logger.info(f"   Location: {config.output_dir}")
    logger.info("")
    
    return outputs


def generate_extreme_scenarios():
    """
    Generate edge case datasets for robustness testing.
    
    - Low anomaly scenario (5%)
    - High anomaly scenario (15%)
    - Large dataset (5,000 meters)
    """
    logger.info("=" * 80)
    logger.info("Generating EXTREME SCENARIO datasets...")
    logger.info("=" * 80)
    
    # Low anomaly scenario
    logger.info("  → Low anomaly scenario (5%)...")
    config_low = GeneratorConfig(
        random_seed=400,
        num_transformers=20,
        num_meters=500,
        num_months=12,
        anomaly_rate=0.05,
        output_dir=Path("../datasets/scenarios/low_anomaly")
    )
    pipeline = SyntheticDataPipeline(config_low)
    outputs_low = pipeline.generate_all()
    pipeline.save_outputs(outputs_low)
    logger.info(f"    ✅ Low anomaly: {len(outputs_low['anomaly_labels_df'])} anomalies")
    
    # High anomaly scenario
    logger.info("  → High anomaly scenario (15%)...")
    config_high = GeneratorConfig(
        random_seed=500,
        num_transformers=20,
        num_meters=500,
        num_months=12,
        anomaly_rate=0.15,
        output_dir=Path("../datasets/scenarios/high_anomaly")
    )
    pipeline = SyntheticDataPipeline(config_high)
    outputs_high = pipeline.generate_all()
    pipeline.save_outputs(outputs_high)
    logger.info(f"    ✅ High anomaly: {len(outputs_high['anomaly_labels_df'])} anomalies")
    
    # Large dataset
    logger.info("  → Large dataset (5,000 meters)...")
    config_large = GeneratorConfig(
        random_seed=600,
        num_transformers=100,
        num_meters=5000,
        num_months=12,
        anomaly_rate=0.075,
        output_dir=Path("../datasets/scenarios/large_scale")
    )
    pipeline = SyntheticDataPipeline(config_large)
    outputs_large = pipeline.generate_all()
    pipeline.save_outputs(outputs_large)
    logger.info(f"    ✅ Large scale: {len(outputs_large['meters_df'])} meters")
    
    logger.info("")


def generate_summary_report(start_time):
    """Generate summary of all datasets created."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 80)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 80)
    
    datasets_info = [
        ("Demo", "../datasets/demo", "200 meters, 6 months"),
        ("Development", "../datasets/development", "1,000 meters, 12 months"),
        ("Validation", "../datasets/validation", "1,000 meters, 12 months, 10% anomalies"),
        ("Production", "../datasets/production", "2,000 meters, 12 months"),
        ("Low Anomaly", "../datasets/scenarios/low_anomaly", "500 meters, 5% anomalies"),
        ("High Anomaly", "../datasets/scenarios/high_anomaly", "500 meters, 15% anomalies"),
        ("Large Scale", "../datasets/scenarios/large_scale", "5,000 meters"),
    ]
    
    logger.info("\nDatasets Generated:")
    logger.info("-" * 80)
    for name, path, description in datasets_info:
        logger.info(f"  {name:15s} → {path:40s} ({description})")
    
    logger.info(f"\nTotal Generation Time: {duration:.2f} seconds")
    logger.info(f"Generated at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL DATASETS READY FOR ML DEVELOPMENT!")
    logger.info("=" * 80)
    
    # Create a README in datasets folder
    datasets_readme = Path("../datasets/README.md")
    datasets_readme.parent.mkdir(parents=True, exist_ok=True)
    
    with open(datasets_readme, 'w') as f:
        f.write("# Generated Datasets for GhostLoad Mapper ML Development\n\n")
        f.write(f"**Generated**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Generation Time**: {duration:.2f} seconds\n\n")
        f.write("## Datasets\n\n")
        
        for name, path, description in datasets_info:
            f.write(f"### {name}\n")
            f.write(f"- **Location**: `{path}`\n")
            f.write(f"- **Description**: {description}\n")
            f.write(f"- **Files**: transformers.csv, meter_consumption.csv, anomaly_labels.csv, transformers.geojson\n\n")
        
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n\n")
        f.write("# Load demo dataset\n")
        f.write("meters = pd.read_csv('demo/meter_consumption.csv')\n")
        f.write("anomalies = pd.read_csv('demo/anomaly_labels.csv')\n")
        f.write("```\n")
    
    logger.info(f"\n📄 Summary report saved to: {datasets_readme}")


def main():
    """Generate all datasets for ML development."""
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("  GHOSTLOAD MAPPER - SYNTHETIC DATA GENERATION")
    print("  Generating all datasets for ML development phase")
    print("=" * 80 + "\n")
    
    try:
        # Generate core datasets
        generate_demo_dataset()
        generate_development_dataset()
        generate_validation_dataset()
        generate_full_dataset()
        
        # Generate edge case scenarios
        generate_extreme_scenarios()
        
        # Generate summary
        generate_summary_report(start_time)
        
        print("\n✨ SUCCESS! All datasets generated and ready to use.\n")
        print("📂 Datasets location: machine_learning/datasets/")
        print("\n💡 Next steps:")
        print("   1. Load datasets in your ML training script")
        print("   2. Train Isolation Forest model")
        print("   3. Validate on validation dataset")
        print("   4. Test end-to-end with production dataset\n")
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
