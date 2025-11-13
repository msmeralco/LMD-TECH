"""
GhostLoad Mapper - Model Training Script
=========================================

Simple wrapper script that uses the existing professional training pipeline.

The complete ML system already exists in machine_learning/:
├── data/: data_loader.py, data_preprocessor.py, feature_engineer.py
├── models/: isolation_forest_model.py, dbscan_model.py, model_registry.py
├── training/: model_trainer.py, hyperparameter_tuner.py
├── evaluation/: anomaly_scorer.py, risk_assessor.py, metrics_calculator.py
├── utils/: config_loader.py, logger.py, data_validator.py
└── pipeline/: training_pipeline.py ← This orchestrates everything!

Core ML Features (all implemented):
✓ Transformer Baseline Computation (feature_engineer.py)
✓ Isolation Forest (isolation_forest_model.py)
✓ DBSCAN (dbscan_model.py)
✓ Rule-Based Risk Scoring (risk_assessor.py)

Dataset Options (change in code below):
📊 RECOMMENDED: datasets/development (1,000 meters, 7.6% anomalies) - BEST FOR HACKATHON
   - Balanced, realistic, fast training (~3 min)
   - 76 theft cases to detect, diverse risk bands
   
🎯 Alternative: datasets/scenarios/high_anomaly (500 meters, 14.8% anomalies) - DRAMATIC DEMO
   - Shows impressive detection results
   - Perfect for judges who want to see action
   
⚡ Quick Test: datasets/demo (200 meters, 9.5% anomalies) - FASTEST
   - For testing only, small dataset

This script simply runs the existing training_pipeline.py with proper configuration.

Author: ML Team
Date: November 13, 2025
Status: Production Ready
"""

import sys
from pathlib import Path

# Add machine_learning to path for imports
machine_learning_dir = Path(__file__).parent
sys.path.insert(0, str(machine_learning_dir.parent))

print("\n" + "=" * 80)
print("GHOSTLOAD MAPPER - ML MODEL TRAINING".center(80))
print("=" * 80)
print("\nUsing existing professional pipeline with all ML features:")
print("  * Transformer Baseline Computation")
print("  * Isolation Forest Anomaly Detection")
print("  * DBSCAN Spatial Clustering (optional)")
print("  * Rule-Based Risk Scoring")
print()

try:
    # Import the professional training pipeline
    from machine_learning.pipeline.training_pipeline import TrainingPipeline
    
    # Configuration
    config_path = machine_learning_dir / "config" / "config.yaml"
    
    # DATASET SELECTION FOR HACKATHON:
    # - demo: 200 meters (quick test)
    # - development: 1,000 meters (RECOMMENDED for hackathon - balanced, realistic)
    # - production: 2,000 meters (realistic but slower)
    # - validation: 1,000 meters (testing)
    # - scenarios/high_anomaly: 500 meters (dramatic demo - 14.8% anomaly rate!)
    
    dataset_dir = machine_learning_dir / "datasets" / "development"  # BEST FOR HACKATHON
    output_dir = machine_learning_dir.parent / "output"
    
    print("Configuration:")
    print(f"   * Config: {config_path}")
    print(f"   * Dataset: development (1,000 meters, 76 anomalies, 7.6% theft rate)")
    print(f"   * Output: {output_dir}")
    print()
    
    # Initialize pipeline
    print("Initializing training pipeline...")
    pipeline = TrainingPipeline(
        config_path=str(config_path),
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir)
    )
    
    print("Starting training (this will take 2-3 minutes)...")
    print()
    
    # Run the complete pipeline
    results = pipeline.run()
    
    # Display results
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!".center(80))
    print("=" * 80)
    print()
    print("Results:")
    print(f"   * Total time: {results.execution_time:.1f}s")
    print(f"   * Model saved: {output_dir / 'latest' / 'trained_model.pkl'}")
    print()
    print("Next Steps:")
    print("   1. Test inference:")
    print("      python machine_learning/pipeline/inference_pipeline.py")
    print()
    print("   2. Integrate with backend:")
    print("      from machine_learning.pipeline.inference_pipeline import predict_meter_risk")
    print()
    print("   3. Start making predictions!")
    print()
    
except ImportError as e:
    print(f"Import error: {e}")
    print()
    print("The training pipeline modules are not available.")
    print("This usually means there are missing dependencies or path issues.")
    print()
    print("Please check:")
    print("  1. All required modules exist in machine_learning/")
    print("  2. Dependencies are installed: pip install -r requirements.txt")
    print()
    sys.exit(1)
    
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
