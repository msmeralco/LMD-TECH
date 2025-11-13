"""
Example: Creating a Custom Anomaly Detector
===========================================

This example demonstrates how to create a custom anomaly detector
by inheriting from BaseAnomalyDetector and implementing the required
abstract methods.

We'll create a simple Statistical Distance Detector that uses
Mahalanobis distance for anomaly scoring.

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import numpy as np
from numpy.typing import NDArray
from typing import Optional
import logging

from base_model import (
    BaseAnomalyDetector,
    ModelConfig,
    ModelType,
    PredictionResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM DETECTOR IMPLEMENTATION
# ============================================================================

class StatisticalDistanceDetector(BaseAnomalyDetector):
    """
    Statistical Distance Anomaly Detector using Mahalanobis Distance.
    
    This detector computes anomaly scores based on the Mahalanobis distance
    from the training data distribution. Samples far from the distribution
    center receive higher anomaly scores.
    
    Algorithm:
        1. Training: Compute mean and covariance matrix
        2. Prediction: Compute Mahalanobis distance for each sample
        3. Score: Distance normalized by quantile
    
    Mahalanobis Distance:
        D(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        
    Where:
        μ = mean vector
        Σ = covariance matrix
        x = sample vector
    
    Hyperparameters:
        regularization: Regularization term for covariance matrix
                       (prevents singular matrix)
    
    Example:
        >>> detector = StatisticalDistanceDetector()
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
        >>> print(f"Mean score: {scores.mean():.3f}")
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        regularization: float = 1e-6
    ):
        """
        Initialize Statistical Distance Detector.
        
        Args:
            config: Model configuration (uses defaults if None)
            regularization: Regularization term for covariance matrix
        """
        # Initialize base class
        super().__init__(config=config)
        
        # Store hyperparameters
        self.regularization = regularization
        
        # Internal state (set during training)
        self._mean: Optional[NDArray[np.float64]] = None
        self._cov: Optional[NDArray[np.float64]] = None
        self._cov_inv: Optional[NDArray[np.float64]] = None
    
    def _fit_implementation(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.int_]] = None
    ) -> None:
        """
        Train the detector by computing distribution statistics.
        
        Computes:
        - Mean vector (μ)
        - Covariance matrix (Σ)
        - Inverse covariance matrix (Σ⁻¹)
        
        Args:
            X: Training data (n_samples, n_features)
            y: Not used (unsupervised)
        """
        # Compute mean vector
        self._mean = X.mean(axis=0)
        
        # Compute covariance matrix
        self._cov = np.cov(X, rowvar=False)
        
        # Add regularization to prevent singular matrix
        # Σ_reg = Σ + λI
        self._cov += self.regularization * np.eye(self._cov.shape[0])
        
        # Compute inverse covariance matrix
        try:
            self._cov_inv = np.linalg.inv(self._cov)
        except np.linalg.LinAlgError:
            logger.warning(
                f"Covariance matrix is singular. "
                f"Increasing regularization to {self.regularization * 10}"
            )
            self._cov += (self.regularization * 9) * np.eye(self._cov.shape[0])
            self._cov_inv = np.linalg.inv(self._cov)
        
        # Update metadata with hyperparameters
        self.metadata.hyperparameters.update({
            'algorithm': 'Mahalanobis Distance',
            'regularization': self.regularization,
            'mean': self._mean.tolist(),
            'covariance_determinant': float(np.linalg.det(self._cov))
        })
        
        logger.info(
            f"Trained Statistical Distance Detector:\n"
            f"  Mean: {self._mean}\n"
            f"  Covariance det: {np.linalg.det(self._cov):.6f}"
        )
    
    def _predict_implementation(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute anomaly scores using Mahalanobis distance.
        
        Formula:
            score(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            anomaly_scores: 1D array of shape (n_samples,)
                          Higher scores = more anomalous
        """
        # Compute deviations from mean: (x - μ)
        deviations = X - self._mean
        
        # Compute Mahalanobis distance for each sample
        # D(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        scores = np.zeros(X.shape[0])
        
        for i, dev in enumerate(deviations):
            # (x - μ)ᵀ Σ⁻¹ (x - μ)
            mahal_sq = dev @ self._cov_inv @ dev
            scores[i] = np.sqrt(max(0, mahal_sq))  # Ensure non-negative
        
        return scores


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    """Demonstrate custom detector usage."""
    
    print("\n" + "="*80)
    print("STATISTICAL DISTANCE DETECTOR - DEMONSTRATION")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. GENERATE SYNTHETIC DATA
    # ========================================================================
    
    print("-" * 80)
    print("1. Generate Synthetic Dataset")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Normal samples: Gaussian distribution
    n_normal = 200
    n_features = 5
    X_normal = np.random.randn(n_normal, n_features)
    
    # Anomaly samples: Outliers (5x standard deviation)
    n_anomalies = 20
    X_anomalies = np.random.randn(n_anomalies, n_features) * 5
    
    # Combine datasets
    X_train = X_normal  # Train on normal data only
    X_test = np.vstack([X_normal[-50:], X_anomalies])  # Test on mixed data
    y_test = np.array([0]*50 + [1]*20)  # True labels (0=normal, 1=anomaly)
    
    print(f"Training set: {X_train.shape[0]} normal samples")
    print(f"Test set: {X_test.shape[0]} samples (50 normal + 20 anomalies)")
    print()
    
    # ========================================================================
    # 2. CREATE AND CONFIGURE DETECTOR
    # ========================================================================
    
    print("-" * 80)
    print("2. Create Statistical Distance Detector")
    print("-" * 80)
    
    config = ModelConfig(
        model_type=ModelType.CUSTOM,
        model_name="statistical_distance_v1",
        contamination=0.1,  # Expect ~10% anomalies
        normalize_scores=True,
        random_state=42,
        verbose=True
    )
    
    detector = StatisticalDistanceDetector(
        config=config,
        regularization=1e-6
    )
    
    print(f"Created detector:\n{detector}\n")
    
    # ========================================================================
    # 3. TRAIN DETECTOR
    # ========================================================================
    
    print("-" * 80)
    print("3. Train Detector")
    print("-" * 80)
    
    detector.fit(X_train)
    
    print(f"\nTraining Summary:")
    print(f"  Samples: {detector.metadata.training_samples}")
    print(f"  Features: {detector.metadata.training_features}")
    print(f"  Time: {detector.metadata.training_time_seconds:.3f}s")
    print(f"  Threshold: {detector.threshold:.4f}")
    print()
    
    # ========================================================================
    # 4. PREDICT ANOMALY SCORES
    # ========================================================================
    
    print("-" * 80)
    print("4. Predict Anomaly Scores")
    print("-" * 80)
    
    # Get comprehensive results
    result = detector.predict(X_test, return_probabilities=True)
    
    print(f"Prediction Result:")
    print(f"  Samples: {len(result)}")
    print(f"  Score range: [{result.anomaly_scores.min():.3f}, {result.anomaly_scores.max():.3f}]")
    print(f"  Mean score: {result.anomaly_scores.mean():.3f}")
    print(f"  Std score: {result.anomaly_scores.std():.3f}")
    print()
    
    # ========================================================================
    # 5. EVALUATE DETECTION PERFORMANCE
    # ========================================================================
    
    print("-" * 80)
    print("5. Evaluate Detection Performance")
    print("-" * 80)
    
    # Binary predictions
    y_pred = result.predictions
    
    # Compute metrics manually (no sklearn dependency)
    tp = np.sum((y_test == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_test == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_test == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_test == 1) & (y_pred == 0))  # False Negatives
    
    accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Classification Metrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print()
    
    print(f"Confusion Matrix:")
    print(f"  [[TN={tn:3d}  FP={fp:3d}]")
    print(f"   [FN={fn:3d}  TP={tp:3d}]]")
    print()
    
    # Analyze scores by class
    normal_scores = result.anomaly_scores[y_test == 0]
    anomaly_scores = result.anomaly_scores[y_test == 1]
    
    print(f"Score Distribution:")
    print(f"  Normal samples:  mean={normal_scores.mean():.3f}, std={normal_scores.std():.3f}")
    print(f"  Anomaly samples: mean={anomaly_scores.mean():.3f}, std={anomaly_scores.std():.3f}")
    print(f"  Separation:      {anomaly_scores.mean() - normal_scores.mean():.3f}")
    print()
    
    # ========================================================================
    # 6. SAVE AND LOAD MODEL
    # ========================================================================
    
    print("-" * 80)
    print("6. Model Persistence")
    print("-" * 80)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "statistical_detector.pkl"
        
        # Save model
        detector.save(save_path, include_metadata=True)
        print(f"Model saved to: {save_path}")
        print(f"Model ID: {detector.metadata.model_id}")
        print()
        
        # Load model
        loaded_detector = StatisticalDistanceDetector.load(save_path)
        print(f"Model loaded successfully")
        
        # Verify predictions match
        loaded_scores = loaded_detector.predict(X_test)
        
        if np.allclose(result.anomaly_scores, loaded_scores):
            print(f"[OK] Loaded model produces identical predictions")
        else:
            print(f"[FAIL] Loaded model predictions differ")
        print()
    
    # ========================================================================
    # 7. MODEL INTROSPECTION
    # ========================================================================
    
    print("-" * 80)
    print("7. Model Introspection")
    print("-" * 80)
    
    info = detector.get_model_info()
    
    print(f"Model Information:")
    print(f"  Class: {info['class_name']}")
    print(f"  Status: {info['is_fitted']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Threshold: {info['threshold']:.4f}")
    print(f"  Hyperparameters:")
    for key, value in info['metadata']['hyperparameters'].items():
        if key == 'mean':
            print(f"    {key}: {value[:3]}... (truncated)")
        elif isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    print()
    
    print("="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
    
    print("Summary:")
    print(f"  [OK] Custom detector successfully created")
    print(f"  [OK] Training completed in {detector.metadata.training_time_seconds:.3f}s")
    print(f"  [OK] Detection performance: F1={f1:.3f}, Recall={recall:.3f}")
    print(f"  [OK] Model persistence working correctly")
    print()


if __name__ == "__main__":
    main()
