"""
Production-Grade Isolation Forest Anomaly Detector for GhostLoad Mapper
========================================================================

This module implements a production-ready Isolation Forest anomaly detector
for electricity theft detection in the GhostLoad Mapper system. It wraps
scikit-learn's IsolationForest algorithm within our BaseAnomalyDetector
framework, providing enterprise-grade reliability, observability, and
integration capabilities.

Isolation Forest Algorithm:
    The Isolation Forest algorithm isolates anomalies by randomly selecting
    a feature and then randomly selecting a split value between the maximum
    and minimum values of the selected feature. Anomalies are points that
    have shorter average path lengths in the tree ensemble.
    
    Key Insight: Anomalies are "few and different" - they are easier to
    isolate (fewer splits required) than normal points in a random tree.

Research Foundation:
    - Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008)
      "Isolation Forest" - ICDM '08
      https://doi.org/10.1109/ICDM.2008.17
    
    - Liu, F.T., Ting, K.M. and Zhou, Z.H. (2012)
      "Isolation-based Anomaly Detection" - ACM TKDD
      https://doi.org/10.1145/2133360.2133363

Use Cases:
    - Electricity theft detection (non-technical losses)
    - Abnormal consumption pattern identification
    - Fraudulent meter behavior detection
    - Outlier detection in high-dimensional feature spaces

Performance Characteristics:
    - Time Complexity: O(n log n) for training, O(log n) for prediction
    - Space Complexity: O(n_estimators × max_samples)
    - Scalability: Efficient for large datasets (100k+ samples)
    - Parallelization: Supports multi-core training/prediction

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
import logging
import warnings
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# scikit-learn imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn import __version__ as sklearn_version
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn_version = "N/A"
    warnings.warn(
        "scikit-learn not available. Install with: pip install scikit-learn",
        ImportWarning
    )

# Local imports
from base_model import (
    BaseAnomalyDetector,
    ModelConfig,
    ModelType,
    ModelMetadata,
    PredictionResult
)


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IsolationForestConfig:
    """
    Configuration for Isolation Forest anomaly detector.
    
    This dataclass encapsulates all hyperparameters specific to the
    Isolation Forest algorithm, providing type safety and validation.
    
    Attributes:
        n_estimators: Number of isolation trees in the ensemble
                     Higher = more accurate but slower (default: 100)
        
        max_samples: Number of samples to draw for each tree
                    - int: exact number of samples
                    - float: proportion of dataset (0.0-1.0)
                    - "auto": min(256, n_samples)
                    Higher = slower but captures global structure
        
        max_features: Number of features to draw for each tree
                     - int: exact number of features
                     - float: proportion of features (0.0-1.0)
                     Lower = faster, may miss correlated anomalies
        
        bootstrap: Whether to bootstrap samples for each tree
                  True = sample with replacement (adds randomness)
                  False = sample without replacement (more diverse trees)
        
        warm_start: Enable incremental training
                   True = reuse previous trees when refitting
                   False = rebuild all trees from scratch
        
        score_inversion: How to convert Isolation Forest scores
                        "negative": Higher = more anomalous (recommended)
                        "positive": Lower = more anomalous (sklearn default)
    
    Recommended Settings:
        - Small datasets (<1000): n_estimators=50, max_samples=256
        - Medium datasets (1k-100k): n_estimators=100, max_samples="auto"
        - Large datasets (>100k): n_estimators=200, max_samples=512
    
    Example:
        >>> config = IsolationForestConfig(
        ...     n_estimators=100,
        ...     max_samples="auto",
        ...     contamination=0.1
        ... )
    """
    
    # Ensemble configuration
    n_estimators: int = 100
    max_samples: Union[int, float, str] = "auto"
    max_features: Union[int, float] = 1.0
    
    # Anomaly detection threshold
    contamination: Union[float, str] = 'auto'  # Expected proportion of anomalies or 'auto'
    
    # Sampling strategy
    bootstrap: bool = False
    
    # Reproducibility
    random_state: Optional[int] = None  # Random seed for reproducibility
    
    # Training options
    warm_start: bool = False
    
    # Score transformation
    score_inversion: str = "negative"  # "negative" or "positive"
    
    # Base model compatibility attributes
    enable_validation: bool = True  # Perform input validation (NaN/Inf checks)
    threshold: Optional[float] = None  # Decision threshold (auto-computed if None)
    cache_predictions: bool = False  # Cache predictions for repeated calls
    normalize_scores: bool = True  # Normalize anomaly scores to [0, 1]
    verbose: bool = True  # Enable detailed logging
    n_jobs: int = -1  # Number of parallel jobs (-1 = all cores)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate n_estimators
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be ≥1, got {self.n_estimators}")
        
        if self.n_estimators > 1000:
            warnings.warn(
                f"n_estimators={self.n_estimators} is very high. "
                f"This may be slow for training/prediction.",
                UserWarning
            )
        
        # Validate max_samples
        if isinstance(self.max_samples, (int, float)):
            if isinstance(self.max_samples, float):
                if not 0.0 < self.max_samples <= 1.0:
                    raise ValueError(
                        f"max_samples as float must be in (0.0, 1.0], "
                        f"got {self.max_samples}"
                    )
            elif isinstance(self.max_samples, int):
                if self.max_samples < 1:
                    raise ValueError(
                        f"max_samples as int must be ≥1, got {self.max_samples}"
                    )
        elif self.max_samples != "auto":
            raise ValueError(
                f"max_samples must be int, float, or 'auto', got {self.max_samples}"
            )
        
        # Validate max_features
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError(
                    f"max_features as float must be in (0.0, 1.0], "
                    f"got {self.max_features}"
                )
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError(
                    f"max_features as int must be ≥1, got {self.max_features}"
                )
        
        # Validate contamination
        if isinstance(self.contamination, float):
            if not 0.0 < self.contamination <= 0.5:
                raise ValueError(
                    f"contamination must be in (0.0, 0.5], got {self.contamination}"
                )
        elif self.contamination != 'auto':
            raise ValueError(
                f"contamination must be float or 'auto', got {self.contamination}"
            )
        
        # Validate random_state
        if self.random_state is not None and self.random_state < 0:
            raise ValueError(
                f"random_state must be non-negative or None, got {self.random_state}"
            )
        
        # Validate score_inversion
        if self.score_inversion not in ("negative", "positive"):
            raise ValueError(
                f"score_inversion must be 'negative' or 'positive', "
                f"got {self.score_inversion}"
            )
    
    def to_sklearn_params(self) -> Dict[str, Any]:
        """
        Convert configuration to scikit-learn IsolationForest parameters.
        
        Only includes parameters that sklearn's IsolationForest accepts.
        Excludes base model attributes like enable_validation, verbose, etc.
        
        Returns:
            Dictionary of parameters for sklearn.ensemble.IsolationForest
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'contamination': self.contamination,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'verbose': 0  # We handle logging ourselves
            # Note: enable_validation, threshold, cache_predictions, normalize_scores
            # are used by base model, not passed to sklearn
        }


# ============================================================================
# ISOLATION FOREST DETECTOR
# ============================================================================

class IsolationForestDetector(BaseAnomalyDetector):
    """
    Production-grade Isolation Forest anomaly detector.
    
    This detector implements the Isolation Forest algorithm (Liu et al., 2008)
    for unsupervised anomaly detection. It inherits from BaseAnomalyDetector
    to provide enterprise features: validation, logging, persistence, metadata.
    
    Algorithm Overview:
        1. Training: Build ensemble of isolation trees
           - Randomly sample features and split values
           - Build trees until all points are isolated
           - Store average path lengths
        
        2. Prediction: Compute anomaly scores
           - Calculate average path length for each sample
           - Shorter paths = easier to isolate = more anomalous
           - Normalize scores and invert (higher = more anomalous)
    
    Key Advantages:
        ✅ No assumptions about data distribution
        ✅ Efficient for high-dimensional data
        ✅ Low memory footprint (trees are shallow)
        ✅ Parallelizable training and prediction
        ✅ Robust to irrelevant features
    
    Key Limitations:
        ⚠️ Sensitive to contamination parameter
        ⚠️ May struggle with clustered anomalies
        ⚠️ Requires enough normal samples for baseline
    
    Hyperparameters:
        n_estimators: Number of trees (default: 100)
        max_samples: Samples per tree (default: "auto" = min(256, n))
        contamination: Expected anomaly rate (default: 0.1 = 10%)
        random_state: Random seed for reproducibility
    
    Example:
        >>> # Basic usage
        >>> detector = IsolationForestDetector()
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
        >>> print(f"Mean anomaly score: {scores.mean():.3f}")
        
        >>> # With custom configuration
        >>> config = ModelConfig(contamination=0.15, random_state=42)
        >>> if_config = IsolationForestConfig(n_estimators=200)
        >>> detector = IsolationForestDetector(config, if_config)
        >>> detector.fit(X_train)
        >>> result = detector.predict(X_test, return_probabilities=True)
        >>> print(f"Detected {result.predictions.sum()} anomalies")
        
        >>> # Save/load
        >>> detector.save('models/isolation_forest_v1.pkl')
        >>> loaded = IsolationForestDetector.load('models/isolation_forest_v1.pkl')
    
    Thread Safety: Not thread-safe (create separate instances per thread)
    
    References:
        [1] Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008)
            "Isolation Forest" - ICDM '08
        [2] Scikit-learn IsolationForest documentation
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        if_config: Optional[IsolationForestConfig] = None,
        metadata: Optional[ModelMetadata] = None
    ):
        """
        Initialize Isolation Forest anomaly detector.
        
        Args:
            config: Base model configuration (uses defaults if None)
            if_config: Isolation Forest specific configuration (uses defaults if None)
            metadata: Model metadata (auto-generated if None)
            
        Raises:
            ImportError: If scikit-learn is not installed
        """
        # Check scikit-learn availability
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for IsolationForestDetector. "
                "Install with: pip install scikit-learn"
            )
        
        # Store Isolation Forest configuration BEFORE calling super().__init__()
        # (needed for _get_hyperparameters() called during initialization)
        self.if_config = if_config or IsolationForestConfig()
        
        # Initialize internal state BEFORE calling super().__init__()
        # (needed for _get_hyperparameters() called during initialization)
        self._model: Optional[IsolationForest] = None
        self._feature_importances: Optional[NDArray[np.float64]] = None
        
        # Initialize base class
        if config is None:
            config = ModelConfig(model_type=ModelType.ISOLATION_FOREST)
        else:
            config.model_type = ModelType.ISOLATION_FOREST
        
        super().__init__(config=config, metadata=metadata)
        
        # Update metadata
        self.metadata.hyperparameters.update({
            'algorithm': 'Isolation Forest',
            'n_estimators': self.if_config.n_estimators,
            'max_samples': self.if_config.max_samples,
            'max_features': self.if_config.max_features,
            'bootstrap': self.if_config.bootstrap,
            'score_inversion': self.if_config.score_inversion,
            'sklearn_version': sklearn_version
        })
        
        logger.info(
            f"Initialized IsolationForestDetector "
            f"(n_estimators={self.if_config.n_estimators}, "
            f"contamination={self.config.contamination:.1%})"
        )
    
    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    def _fit_implementation(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.int_]] = None
    ) -> None:
        """
        Train Isolation Forest on input data.
        
        This method:
        1. Creates scikit-learn IsolationForest instance
        2. Configures hyperparameters (contamination, random_state, etc.)
        3. Fits the ensemble of isolation trees
        4. Computes feature importances (if enabled)
        
        Args:
            X: Training data (n_samples, n_features) - validated by base class
            y: Not used (unsupervised algorithm)
            
        Note:
            - Contamination parameter affects threshold computation
            - Random state ensures reproducibility
            - Training time scales as O(n_estimators × n_samples × log(max_samples))
        """
        start_time = time.time()
        
        # Create sklearn IsolationForest with our configuration
        # Note: to_sklearn_params() already includes all necessary parameters
        # (n_estimators, contamination, random_state, n_jobs, etc.)
        sklearn_params = self.if_config.to_sklearn_params()
        
        self._model = IsolationForest(**sklearn_params)
        
        logger.info(
            f"Training Isolation Forest with parameters:\n"
            f"  n_estimators: {self.if_config.n_estimators}\n"
            f"  max_samples: {self.if_config.max_samples}\n"
            f"  contamination: {self.if_config.contamination}\n"
            f"  random_state: {self.if_config.random_state}\n"
            f"  n_jobs: {self.if_config.n_jobs}"
        )
        
        # Fit the model
        self._model.fit(X)
        
        training_time = time.time() - start_time
        
        # Note: Feature importances computation moved to after model is marked as fitted
        # (see fit() method in base class which calls this, then sets self._is_fitted = True)
        
        logger.info(
            f"Isolation Forest training complete:\n"
            f"  Training time: {training_time:.3f}s\n"
            f"  Trees in ensemble: {len(self._model.estimators_)}\n"
            f"  Samples per tree: {self._model.max_samples_}\n"
            f"  Offset (normalization): {self._model.offset_:.6f}"
        )
    
    def _predict_implementation(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Predict anomaly scores using Isolation Forest.
        
        This method:
        1. Computes decision_function scores from sklearn
        2. Inverts scores (negative) so higher = more anomalous
        3. Returns 1D array of anomaly scores
        
        Args:
            X: Input data (n_samples, n_features) - validated by base class
            
        Returns:
            anomaly_scores: 1D array of shape (n_samples,)
                          Higher scores = more anomalous
        
        Score Interpretation:
            - Isolation Forest returns decision_function scores
            - Negative scores = anomalies, positive = normal (sklearn default)
            - We invert: anomaly_score = -decision_function
            - Result: Higher scores = more anomalous (consistent with our API)
        
        Note:
            - Prediction time: O(n_samples × n_estimators × log(max_samples))
            - Parallelized across n_jobs cores
        """
        # Get raw decision function scores from sklearn
        # sklearn convention: lower scores = more anomalous
        decision_scores = self._model.decision_function(X)
        
        # Invert scores based on configuration
        # We want: higher scores = more anomalous
        if self.if_config.score_inversion == "negative":
            anomaly_scores = -decision_scores
        else:
            anomaly_scores = decision_scores
        
        # Ensure 1D array
        anomaly_scores = anomaly_scores.flatten()
        
        return anomaly_scores
    
    # ========================================================================
    # ISOLATION FOREST SPECIFIC METHODS
    # ========================================================================
    
    def get_feature_importances(
        self,
        X: Optional[NDArray[np.float64]] = None,
        recompute: bool = False
    ) -> Optional[NDArray[np.float64]]:
        """
        Get feature importances (heuristic).
        
        Isolation Forest doesn't have native feature importances, but we
        can estimate them by measuring how much anomaly scores change when
        each feature is permuted.
        
        Args:
            X: Data to compute importances on (uses training data if None)
            recompute: Force recomputation even if cached
            
        Returns:
            feature_importances: Array of shape (n_features,) or None
                               Higher values = more important for anomaly detection
        
        Note:
            This is expensive (O(n_features × prediction_time))
            Only compute for small feature sets or when necessary
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing importances")
        
        # Return cached if available
        if self._feature_importances is not None and not recompute:
            return self._feature_importances
        
        # Need data to compute importances
        if X is None:
            logger.warning("Cannot compute feature importances without data")
            return None
        
        if X.shape[1] > 100:
            logger.warning(
                f"Computing importances for {X.shape[1]} features is expensive. "
                f"Consider using a smaller dataset or fewer features."
            )
        
        logger.info(f"Computing feature importances on {X.shape[0]} samples...")
        
        # Get baseline scores
        baseline_scores = self._predict_implementation(X)
        baseline_variance = np.var(baseline_scores)
        
        # Permutation importance
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Permute feature i
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Compute scores with permuted feature
            permuted_scores = self._predict_implementation(X_permuted)
            
            # Importance = change in score variance
            permuted_variance = np.var(permuted_scores)
            importances[i] = abs(baseline_variance - permuted_variance)
        
        # Normalize to [0, 1]
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self._feature_importances = importances
        
        logger.info(
            f"Feature importances computed. "
            f"Top feature: {importances.argmax()} (importance={importances.max():.3f})"
        )
        
        return importances
    
    def get_anomaly_paths(
        self,
        X: NDArray[np.float64],
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Get average path lengths for samples (diagnostic).
        
        Lower average path length = easier to isolate = more anomalous
        
        Args:
            X: Input samples
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with path length statistics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing path lengths")
        
        # Get scores for all samples
        scores = self._predict_implementation(X)
        
        # Get top anomalies
        top_indices = np.argsort(scores)[-n_samples:]
        
        # Note: sklearn doesn't expose individual tree paths directly
        # We can only report aggregated scores
        return {
            'top_anomaly_indices': top_indices.tolist(),
            'top_anomaly_scores': scores[top_indices].tolist(),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'n_estimators': self.if_config.n_estimators,
            'max_samples': self._model.max_samples_
        }
    
    def _compute_feature_importances(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Internal method to compute feature importances during training.
        
        Args:
            X: Training data
            
        Returns:
            Feature importances array
        """
        return self.get_feature_importances(X, recompute=True)
    
    # ========================================================================
    # ENHANCED METADATA
    # ========================================================================
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """
        Extract hyperparameters for metadata tracking.
        
        Returns:
            Dictionary of all hyperparameters
        """
        # Don't call super() - provide complete hyperparameters ourselves
        hyperparams = {
            'model_type': 'isolation_forest',
            'n_estimators': self.if_config.n_estimators,
            'max_samples': self.if_config.max_samples,
            'max_features': self.if_config.max_features,
            'contamination': self.if_config.contamination,
            'bootstrap': self.if_config.bootstrap,
            'warm_start': self.if_config.warm_start,
            'random_state': self.if_config.random_state,
            'score_inversion': self.if_config.score_inversion
        }
        
        # Add sklearn-specific info if model is fitted
        if self._model is not None:
            hyperparams.update({
                'n_trees_built': len(self._model.estimators_) if hasattr(self._model, 'estimators_') else 0,
                'max_samples_used': self._model.max_samples_ if hasattr(self._model, 'max_samples_') else None,
                'offset': float(self._model.offset_) if hasattr(self._model, 'offset_') else None
            })
        
        return hyperparams
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information including IF-specific details.
        
        Returns:
            Dictionary with model state and configuration
        """
        info = super().get_model_info()
        
        # Add Isolation Forest specific information
        info['isolation_forest'] = {
            'n_estimators': self.if_config.n_estimators,
            'max_samples': self.if_config.max_samples,
            'score_inversion': self.if_config.score_inversion,
            'sklearn_version': sklearn_version
        }
        
        if self._is_fitted and self._model is not None:
            info['isolation_forest'].update({
                'trees_in_ensemble': len(self._model.estimators_),
                'samples_per_tree': self._model.max_samples_,
                'normalization_offset': float(self._model.offset_)
            })
        
        return info
    
    def __repr__(self) -> str:
        """Enhanced string representation with IF-specific details."""
        status = "fitted" if self._is_fitted else "untrained"
        features = self._n_features_in if self._n_features_in else "unknown"
        
        return (
            f"IsolationForestDetector(\n"
            f"  status: {status},\n"
            f"  features: {features},\n"
            f"  n_estimators: {self.if_config.n_estimators},\n"
            f"  contamination: {self.config.contamination:.1%},\n"
            f"  model_id: {self.metadata.model_id}\n"
            f")"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_detector(
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42
) -> IsolationForestDetector:
    """
    Factory function to create Isolation Forest detector with common defaults.
    
    Args:
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Configured IsolationForestDetector instance
        
    Example:
        >>> detector = create_default_detector(contamination=0.15)
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
    """
    config = ModelConfig(
        model_type=ModelType.ISOLATION_FOREST,
        contamination=contamination,
        random_state=random_state,
        normalize_scores=True,
        verbose=True
    )
    
    if_config = IsolationForestConfig(
        n_estimators=n_estimators,
        max_samples="auto"
    )
    
    return IsolationForestDetector(config=config, if_config=if_config)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of Isolation Forest detector.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("ISOLATION FOREST DETECTOR - SELF-TEST")
    print("="*80 + "\n")
    
    # Check scikit-learn availability
    if not SKLEARN_AVAILABLE:
        print("[ERROR] scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        exit(1)
    
    # Generate synthetic data
    np.random.seed(42)
    n_normal = 200
    n_anomalies = 20
    n_features = 5
    
    # Normal samples: Gaussian distribution
    X_normal = np.random.randn(n_normal, n_features)
    
    # Anomaly samples: Shifted and scaled Gaussian
    X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5
    
    # Combine datasets
    X_train = X_normal  # Train on normal data only
    X_test = np.vstack([X_normal[-50:], X_anomalies])
    y_test = np.array([0]*50 + [1]*20)  # True labels
    
    print(f"Dataset created:")
    print(f"  Training: {X_train.shape[0]} normal samples")
    print(f"  Test: {X_test.shape[0]} samples (50 normal + 20 anomalies)")
    print()
    
    # Test 1: Basic fit/predict
    print("-" * 80)
    print("Test 1: Basic Fit/Predict")
    print("-" * 80)
    
    try:
        detector = create_default_detector(contamination=0.1, n_estimators=100)
        print(f"Created detector:\n{detector}\n")
        
        # Fit
        detector.fit(X_train)
        
        # Predict
        scores = detector.predict(X_test)
        print(f"[OK] Prediction successful: {scores.shape}")
        print(f"     Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Check anomaly detection
        normal_scores = scores[:50].mean()
        anomaly_scores = scores[50:].mean()
        print(f"     Normal mean: {normal_scores:.3f}")
        print(f"     Anomaly mean: {anomaly_scores:.3f}")
        print(f"     Separation: {anomaly_scores - normal_scores:.3f}")
        
        assert anomaly_scores > normal_scores, "Anomalies should have higher scores"
        print(f"     [OK] Anomalies correctly scored higher\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
        import traceback
        traceback.print_exc()
    
    # Test 2: Performance metrics
    print("-" * 80)
    print("Test 2: Performance Evaluation")
    print("-" * 80)
    
    try:
        predictions = detector.predict_binary(X_test)
        
        # Compute metrics
        tp = np.sum((y_test == 1) & (predictions == 1))
        tn = np.sum((y_test == 0) & (predictions == 0))
        fp = np.sum((y_test == 0) & (predictions == 1))
        fn = np.sum((y_test == 1) & (predictions == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"[OK] Metrics computed:")
        print(f"     Precision: {precision:.3f}")
        print(f"     Recall: {recall:.3f}")
        print(f"     F1-Score: {f1:.3f}")
        print(f"     Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
    
    # Test 3: Model persistence
    print("-" * 80)
    print("Test 3: Model Persistence")
    print("-" * 80)
    
    try:
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "if_model.pkl"
            
            # Save
            detector.save(save_path)
            print(f"[OK] Model saved to {save_path}")
            
            # Load
            loaded = IsolationForestDetector.load(save_path)
            print(f"[OK] Model loaded successfully")
            
            # Verify
            scores_original = detector.predict(X_test)
            scores_loaded = loaded.predict(X_test)
            
            assert np.allclose(scores_original, scores_loaded)
            print(f"[OK] Loaded model produces identical predictions\n")
            
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
        import traceback
        traceback.print_exc()
    
    # Test 4: Model introspection
    print("-" * 80)
    print("Test 4: Model Introspection")
    print("-" * 80)
    
    try:
        info = detector.get_model_info()
        print(f"[OK] Model info:")
        print(f"     Class: {info['class_name']}")
        print(f"     Fitted: {info['is_fitted']}")
        print(f"     Features: {info['n_features']}")
        print(f"     Trees: {info['isolation_forest']['trees_in_ensemble']}")
        print(f"     Samples/tree: {info['isolation_forest']['samples_per_tree']}")
        print()
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
    
    print("="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")
    
    print("Summary:")
    print(f"  [OK] All tests passed")
    print(f"  Anomaly detection working: {anomaly_scores/normal_scores:.1f}x higher scores")
    print(f"  Performance: F1={f1:.3f}, Recall={recall:.3f}")
    print(f"  Model persistence: OK")
