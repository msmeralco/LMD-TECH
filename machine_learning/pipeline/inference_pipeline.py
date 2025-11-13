"""
Inference Pipeline for GhostLoad Mapper
========================================

This module provides real-time inference for new meter data using trained models.
Perfect for FastAPI backend integration during hackathon.

Author: ML Team
Date: November 13, 2025
Status: Production Ready for Hackathon

Key Features:
- Load trained model from disk
- Process new meter readings
- Generate risk predictions
- Return JSON for API responses
- Simple one-line usage

Backend Integration:
-------------------
from machine_learning.pipeline.inference_pipeline import predict_meter_risk

# In your FastAPI endpoint
result = predict_meter_risk(meter_id="M12345", consumption_data=data)
# Returns: {"meter_id": "M12345", "risk_level": "HIGH", "anomaly_score": 0.85, ...}
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference pipeline - beginner friendly!"""
    
    # Model settings
    model_path: Union[str, Path] = "output/latest/trained_model.pkl"
    
    # Thresholds for risk classification (same as training)
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4
    
    # Feature engineering (must match training)
    enable_temporal_features: bool = True
    enable_statistical_features: bool = True
    
    # Output settings
    include_explanation: bool = True
    output_format: str = "json"  # or "dict"
    
    # Performance
    verbose: bool = True


# ==============================================================================
# RISK PREDICTION RESULT
# ==============================================================================

@dataclass
class RiskPrediction:
    """Result from inference pipeline - perfect for FastAPI responses!"""
    
    meter_id: str
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    anomaly_score: float
    confidence: float
    explanation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional details
    consumption_pattern: Optional[str] = None
    spatial_risk: Optional[float] = None
    temporal_risk: Optional[float] = None
    
    def to_json(self) -> str:
        """Convert to JSON string for API response"""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return asdict(self)


# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================

class InferencePipeline:
    """
    Production-ready inference pipeline for hackathon backend integration.
    
    This is the FINAL piece - use this in your FastAPI endpoints!
    
    Example Usage (Beginner Friendly):
    ----------------------------------
    # Step 1: Initialize once (when app starts)
    pipeline = InferencePipeline()
    
    # Step 2: Predict risk for new meter data
    result = pipeline.predict({
        'meter_id': 'M12345',
        'consumption': [100, 120, 115, ...],
        'transformer_id': 'T001'
    })
    
    # Step 3: Return to FastAPI
    return result.to_dict()  # Perfect for JSON response!
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize inference pipeline.
        
        Args:
            config: Configuration object (None = use defaults)
        """
        self.config = config or InferenceConfig()
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load trained model
        self._load_model()
        
        logger.info("✅ Inference Pipeline Ready for Production!")
    
    def _load_model(self):
        """Load trained model from disk"""
        try:
            model_path = Path(self.config.model_path)
            
            if not model_path.exists():
                # Try alternative locations
                alternative_paths = [
                    Path("machine_learning/output/latest/trained_model.pkl"),
                    Path("output/latest/trained_model.pkl"),
                    Path("trained_model.pkl")
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        break
                else:
                    raise FileNotFoundError(
                        f"❌ Model not found! Please train a model first using training_pipeline.py\n"
                        f"   Searched locations: {self.config.model_path}, {alternative_paths}"
                    )
            
            # Load model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract components
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
            else:
                self.model = model_data
            
            logger.info(f"✅ Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def predict(
        self,
        meter_data: Union[Dict, pd.DataFrame],
        return_json: bool = False
    ) -> Union[RiskPrediction, List[RiskPrediction], str]:
        """
        Predict risk for new meter data (MAIN METHOD FOR BACKEND).
        
        Args:
            meter_data: Either:
                - Dictionary with meter info (single prediction)
                - DataFrame with multiple meters (batch prediction)
            return_json: Return JSON string instead of object
        
        Returns:
            RiskPrediction object(s) or JSON string
        
        Example:
            # Single meter
            result = pipeline.predict({
                'meter_id': 'M12345',
                'consumption': [100, 120, 115],
                'transformer_id': 'T001'
            })
            
            # Multiple meters (batch)
            results = pipeline.predict(dataframe)
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(meter_data, dict):
                df = pd.DataFrame([meter_data])
                single_prediction = True
            else:
                df = meter_data.copy()
                single_prediction = False
            
            # Engineer features (same as training)
            features = self._engineer_features(df)
            
            # Get predictions
            predictions = self._predict_batch(features, df)
            
            # Format results
            if single_prediction:
                result = predictions[0]
                return result.to_json() if return_json else result
            else:
                return json.dumps([p.to_dict() for p in predictions]) if return_json else predictions
        
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features (must match training pipeline).
        
        This creates the same features used during training.
        """
        # Create new DataFrame for engineered features only
        features = pd.DataFrame()
        
        # 1. Temporal features (if consumption data available)
        if 'consumption' in df.columns or any('consumption' in col for col in df.columns):
            consumption_cols = [col for col in df.columns if 'consumption' in col.lower()]
            
            if consumption_cols:
                # Calculate statistics - exactly matching training
                features['consumption_mean'] = df[consumption_cols].mean(axis=1)
                features['consumption_std'] = df[consumption_cols].std(axis=1)
                features['consumption_max'] = df[consumption_cols].max(axis=1)
                features['consumption_min'] = df[consumption_cols].min(axis=1)
                features['consumption_range'] = features['consumption_max'] - features['consumption_min']
                
                # Trend
                features['consumption_trend'] = df[consumption_cols].apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    axis=1
                )
                
                # Variability (coefficient of variation)
                features['consumption_cv'] = features['consumption_std'] / (features['consumption_mean'] + 1e-6)
        
        # 2. Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _predict_batch(
        self,
        features: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> List[RiskPrediction]:
        """Generate predictions for batch of meters"""
        results = []
        
        # Get anomaly scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features)
            # Normalize to 0-1
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        else:
            scores = self.model.predict(features)
            scores_normalized = scores
        
        # Generate predictions for each meter
        for idx, score in enumerate(scores_normalized):
            meter_id = original_data.iloc[idx].get('meter_id', f'METER_{idx}')
            
            # Classify risk level
            if score >= self.config.high_risk_threshold:
                risk_level = "HIGH"
                explanation = "⚠️ High anomaly detected - Prioritize for field inspection"
            elif score >= self.config.medium_risk_threshold:
                risk_level = "MEDIUM"
                explanation = "⚡ Moderate anomaly - Monitor closely"
            else:
                risk_level = "LOW"
                explanation = "✅ Normal consumption pattern"
            
            # Calculate confidence
            if risk_level == "HIGH":
                confidence = score
            elif risk_level == "MEDIUM":
                confidence = 0.7
            else:
                confidence = 1.0 - score
            
            # Create prediction object
            prediction = RiskPrediction(
                meter_id=str(meter_id),
                risk_level=risk_level,
                anomaly_score=float(score),
                confidence=float(confidence),
                explanation=explanation,
                consumption_pattern="ANOMALOUS" if score > 0.5 else "NORMAL"
            )
            
            results.append(prediction)
        
        return results
    
    def predict_from_csv(
        self,
        csv_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> List[RiskPrediction]:
        """
        Predict risk from CSV file (useful for batch processing).
        
        Args:
            csv_path: Path to CSV with meter data
            output_path: Optional path to save results JSON
        
        Returns:
            List of predictions
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Processing {len(df)} meters from {csv_path}")
        
        # Get predictions
        predictions = self.predict(df)
        
        # Save to file if requested
        if output_path:
            output_data = [p.to_dict() for p in predictions]
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"💾 Results saved to: {output_path}")
        
        return predictions


# ==============================================================================
# CONVENIENCE FUNCTIONS (FOR BACKEND TEAM)
# ==============================================================================

def predict_meter_risk(
    meter_id: str,
    consumption_data: List[float],
    transformer_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    🎯 SIMPLEST WAY TO GET PREDICTIONS (Perfect for FastAPI!)
    
    This is what your backend team should use in FastAPI endpoints.
    
    Args:
        meter_id: Meter identifier
        consumption_data: List of consumption readings
        transformer_id: Optional transformer ID
        **kwargs: Additional meter attributes
    
    Returns:
        Dictionary ready for JSON response
    
    Example (In FastAPI):
    --------------------
    @app.post("/predict")
    async def predict_risk(meter_id: str, consumption: List[float]):
        result = predict_meter_risk(meter_id, consumption)
        return result  # Already a dict - perfect for FastAPI!
    
    Example Response:
    ----------------
    {
        "meter_id": "M12345",
        "risk_level": "HIGH",
        "anomaly_score": 0.85,
        "confidence": 0.85,
        "explanation": "⚠️ High anomaly detected - Prioritize for field inspection",
        "timestamp": "2025-11-13T15:30:00"
    }
    """
    # Initialize pipeline (cached after first call)
    if not hasattr(predict_meter_risk, '_pipeline'):
        predict_meter_risk._pipeline = InferencePipeline()
    
    # Prepare data
    meter_data = {
        'meter_id': meter_id,
        'transformer_id': transformer_id or 'UNKNOWN',
        **kwargs
    }
    
    # Add consumption data
    for i, value in enumerate(consumption_data):
        meter_data[f'consumption_{i}'] = value
    
    # Get prediction
    result = predict_meter_risk._pipeline.predict(meter_data)
    
    return result.to_dict()


def predict_batch_from_dataframe(
    df: pd.DataFrame,
    save_to_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Predict risk for multiple meters at once (batch processing).
    
    Args:
        df: DataFrame with meter data
        save_to_file: Optional path to save results
    
    Returns:
        List of prediction dictionaries
    """
    # Initialize pipeline
    if not hasattr(predict_batch_from_dataframe, '_pipeline'):
        predict_batch_from_dataframe._pipeline = InferencePipeline()
    
    # Get predictions
    predictions = predict_batch_from_dataframe._pipeline.predict(df)
    
    # Convert to dicts
    results = [p.to_dict() for p in predictions]
    
    # Save if requested
    if save_to_file:
        with open(save_to_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Batch results saved to: {save_to_file}")
    
    return results


# ==============================================================================
# SELF-TEST (FOR BEGINNERS)
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 INFERENCE PIPELINE - SELF TEST")
    print("="*70 + "\n")
    
    print("This test checks if inference pipeline is ready for your backend!\n")
    
    # Test 1: Check if model exists
    print("[1/4] Checking for trained model...")
    model_paths = [
        Path("output/latest/trained_model.pkl"),
        Path("machine_learning/output/latest/trained_model.pkl"),
        Path("trained_model.pkl")
    ]
    
    model_found = False
    for path in model_paths:
        if path.exists():
            print(f"    ✅ Found model at: {path}")
            model_found = True
            break
    
    if not model_found:
        print("    ❌ No trained model found!")
        print("    📝 ACTION REQUIRED: Train a model first:")
        print("       Run: python machine_learning/pipeline/training_pipeline.py")
        print("\n" + "="*70)
        sys.exit(1)
    
    # Test 2: Initialize pipeline
    print("\n[2/4] Initializing inference pipeline...")
    try:
        pipeline = InferencePipeline()
        print("    ✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"    ❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 3: Test single prediction
    print("\n[3/4] Testing single meter prediction...")
    try:
        test_data = {
            'meter_id': 'TEST_001',
            'consumption_0': 100,
            'consumption_1': 120,
            'consumption_2': 115,
            'consumption_3': 140,
            'consumption_4': 110,
            'transformer_id': 'T001'
        }
        
        result = pipeline.predict(test_data)
        print(f"    ✅ Prediction successful!")
        print(f"    📊 Result:")
        print(f"       - Meter ID: {result.meter_id}")
        print(f"       - Risk Level: {result.risk_level}")
        print(f"       - Anomaly Score: {result.anomaly_score:.3f}")
        print(f"       - Confidence: {result.confidence:.3f}")
        print(f"       - Explanation: {result.explanation}")
    except Exception as e:
        print(f"    ❌ Prediction failed: {e}")
        sys.exit(1)
    
    # Test 4: Test convenience function
    print("\n[4/4] Testing convenience function (for backend)...")
    try:
        result = predict_meter_risk(
            meter_id='TEST_002',
            consumption_data=[100, 120, 115, 140, 110],
            transformer_id='T001'
        )
        print("    ✅ Convenience function works!")
        print(f"    📊 Backend-ready JSON:")
        print(f"       {json.dumps(result, indent=8)}")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - INFERENCE PIPELINE READY FOR BACKEND!")
    print("="*70)
    print("\n📝 NEXT STEPS FOR BACKEND INTEGRATION:")
    print("\n1. Copy this to your FastAPI backend:")
    print("""
    from machine_learning.pipeline.inference_pipeline import predict_meter_risk
    
    @app.post("/api/predict")
    async def predict_risk(meter_id: str, consumption: List[float]):
        result = predict_meter_risk(meter_id, consumption)
        return result  # Returns JSON automatically!
    """)
    print("\n2. Test the endpoint:")
    print("""
    curl -X POST "http://localhost:8000/api/predict" \\
         -H "Content-Type: application/json" \\
         -d '{"meter_id": "M001", "consumption": [100, 120, 115]}'
    """)
    print("\n3. You're done! 🎉")
    print("="*70 + "\n")
