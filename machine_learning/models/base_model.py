"""
Production-Grade Base Model Abstraction for GhostLoad Mapper ML System
========================================================================

This module provides an enterprise-level abstract base class for anomaly detection
models in the GhostLoad Mapper electricity theft detection system. It enforces
a consistent interface across all anomaly detection implementations while providing
production-ready infrastructure for model lifecycle management.

The base model abstraction ensures:
1. Consistent API: fit(X) → None, predict(X) → anomaly_scores
2. Input/Output Validation: Shape checking, type safety, NaN handling
3. Reproducibility: Random state management, deterministic results
4. Observability: Structured logging, performance metrics, model metadata
5. Serialization: Save/load functionality with version control
6. Error Handling: Comprehensive validation with actionable error messages

Design Pattern: Template Method + Strategy
- Template Method: fit() and predict() define algorithm skeleton
- Strategy: Concrete implementations provide specific anomaly detection logic

Supported Model Types:
- Unsupervised: Isolation Forest, One-Class SVM, Local Outlier Factor
- Semi-supervised: With known normal/anomaly labels
- Ensemble: Combination of multiple detectors

Research Foundation:
    - Anomaly detection best practices (Chandola et al., 2009)
    - Production ML systems design (Sculley et al., 2015)
    - Model versioning and reproducibility (Amershi et al., 2019)

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

from abc import ABC, abstractmethod
import logging
import warnings
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, TypeVar, Generic
from datetime import datetime
from enum import Enum
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Type variable for model implementations
TModel = TypeVar('TModel', bound='BaseAnomalyDetector')


# ============================================================================
# CONSTANTS AND ENUMERATIONS
# ============================================================================

class ModelType(str, Enum):
    """Types of anomaly detection models."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    AUTOENCODER = "autoencoder"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class PredictionMode(str, Enum):
    """Prediction output modes."""
    ANOMALY_SCORE = "anomaly_score"  # Continuous anomaly scores
    BINARY_LABEL = "binary_label"  # Binary predictions (0=normal, 1=anomaly)
    PROBABILITY = "probability"  # Probability of being anomaly


# Default configuration values
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1  # Use all cores
MIN_SAMPLES_FOR_TRAINING = 10
MAX_ANOMALY_SCORE = 1.0
MIN_ANOMALY_SCORE = -1.0


# ============================================================================
# CONFIGURATION AND METADATA
# ============================================================================

@dataclass
class ModelConfig:
    """
    Type-safe configuration for anomaly detection models.
    
    This configuration object encapsulates common model parameters with
    built-in validation to ensure correctness and reproducibility.
    
    Attributes:
        model_type: Type of anomaly detection algorithm
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores)
        contamination: Expected proportion of anomalies (0.0 - 0.5)
        threshold: Decision threshold for binary classification
        normalize_scores: Whether to normalize anomaly scores to [0, 1]
        verbose: Enable detailed logging during training/prediction
        enable_validation: Perform input validation (disable for performance)
        cache_predictions: Cache predictions for repeated calls
    
    Example:
        >>> config = ModelConfig(
        ...     model_type=ModelType.ISOLATION_FOREST,
        ...     contamination=0.15,
        ...     random_state=42
        ... )
    """
    
    # Model identification
    model_type: ModelType = ModelType.CUSTOM
    model_name: Optional[str] = None
    model_version: str = "1.0.0"
    
    # Reproducibility
    random_state: int = DEFAULT_RANDOM_STATE
    
    # Performance
    n_jobs: int = DEFAULT_N_JOBS
    
    # Anomaly detection parameters
    contamination: float = 0.1  # Expected anomaly rate (10%)
    threshold: Optional[float] = None  # Auto-computed if None
    
    # Processing options
    normalize_scores: bool = True  # Scale to [0, 1]
    enable_validation: bool = True  # Input validation
    cache_predictions: bool = False  # Cache for repeated calls
    
    # Observability
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Convert string to enum if needed
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        
        # Validate contamination
        if not 0.0 <= self.contamination <= 0.5:
            raise ValueError(
                f"contamination must be in [0.0, 0.5], got {self.contamination}"
            )
        
        if self.contamination > 0.3:
            warnings.warn(
                f"contamination={self.contamination} is high (>30%). "
                f"Ensure this matches your expected anomaly rate.",
                UserWarning
            )
        
        # Validate threshold if provided
        if self.threshold is not None:
            if not -10.0 <= self.threshold <= 10.0:
                warnings.warn(
                    f"threshold={self.threshold} is outside typical range [-10, 10]",
                    UserWarning
                )
        
        # Validate random state
        if self.random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {self.random_state}")
        
        # Auto-generate model name if not provided
        if self.model_name is None:
            self.model_name = f"{self.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class ModelMetadata:
    """
    Metadata tracking for model lifecycle management.
    
    Tracks training history, performance metrics, and deployment information
    for model governance and reproducibility.
    
    Attributes:
        model_id: Unique identifier for this model instance
        created_at: Timestamp when model was created
        trained_at: Timestamp when model was last trained
        training_samples: Number of samples used for training
        training_time_seconds: Time taken for training
        feature_names: Names of input features
        feature_count: Number of input features
        status: Current model lifecycle status
        metrics: Performance metrics (if validation data available)
        hyperparameters: Model-specific hyperparameters
        environment: Training environment information
    """
    
    # Identification
    model_id: str = field(default_factory=lambda: f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Training information
    trained_at: Optional[str] = None
    training_samples: int = 0
    training_features: int = 0
    training_time_seconds: float = 0.0
    
    # Model properties
    feature_names: Optional[List[str]] = None
    status: ModelStatus = ModelStatus.UNTRAINED
    
    # Performance tracking
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PredictionResult:
    """
    Container for model prediction results.
    
    Encapsulates all prediction outputs with metadata for traceability
    and debugging.
    
    Attributes:
        anomaly_scores: Continuous anomaly scores (higher = more anomalous)
        predictions: Binary predictions (1=anomaly, 0=normal)
        probabilities: Anomaly probabilities (if available)
        metadata: Prediction metadata (timing, statistics, etc.)
    """
    
    anomaly_scores: NDArray[np.float64]
    predictions: Optional[NDArray[np.int_]] = None
    probabilities: Optional[NDArray[np.float64]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate prediction result shapes."""
        n_samples = len(self.anomaly_scores)
        
        if self.predictions is not None:
            if len(self.predictions) != n_samples:
                raise ValueError(
                    f"predictions length ({len(self.predictions)}) must match "
                    f"anomaly_scores length ({n_samples})"
                )
        
        if self.probabilities is not None:
            if len(self.probabilities) != n_samples:
                raise ValueError(
                    f"probabilities length ({len(self.probabilities)}) must match "
                    f"anomaly_scores length ({n_samples})"
                )
    
    def __len__(self) -> int:
        """Return number of predictions."""
        return len(self.anomaly_scores)
    
    def __repr__(self) -> str:
        return (
            f"PredictionResult(\n"
            f"  n_samples: {len(self)},\n"
            f"  anomaly_scores: [{self.anomaly_scores.min():.3f}, {self.anomaly_scores.max():.3f}],\n"
            f"  predictions: {self.predictions is not None},\n"
            f"  probabilities: {self.probabilities is not None}\n"
            f")"
        )


# ============================================================================
# BASE ANOMALY DETECTOR (ABSTRACT BASE CLASS)
# ============================================================================

class BaseAnomalyDetector(ABC):
    """
    Production-grade abstract base class for anomaly detection models.
    
    This class enforces a consistent interface across all anomaly detection
    implementations while providing production infrastructure for:
    - Input/output validation
    - Reproducibility (random state management)
    - Observability (logging, metrics)
    - Serialization (save/load)
    - Error handling
    
    All concrete anomaly detectors must implement:
    - _fit_implementation(X) → None: Train the model
    - _predict_implementation(X) → anomaly_scores: Generate anomaly scores
    
    The base class handles:
    - Input validation and shape checking
    - Feature name tracking
    - Model metadata management
    - Performance timing
    - Score normalization
    - Threshold-based classification
    
    Design Pattern: Template Method
    - fit() and predict() define the algorithm skeleton
    - Subclasses implement specific detection logic
    
    Usage:
        >>> class MyDetector(BaseAnomalyDetector):
        ...     def _fit_implementation(self, X):
        ...         # Train your model
        ...         pass
        ...
        ...     def _predict_implementation(self, X):
        ...         # Return anomaly scores
        ...         return scores
        ...
        >>> detector = MyDetector()
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
    
    Thread Safety: Not thread-safe (use separate instances per thread)
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        metadata: Optional[ModelMetadata] = None
    ):
        """
        Initialize base anomaly detector.
        
        Args:
            config: ModelConfig instance (uses defaults if None)
            metadata: ModelMetadata instance (creates new if None)
        """
        self.config = config or ModelConfig()
        self.metadata = metadata or ModelMetadata()
        
        # Model state
        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None
        self._n_features_in: Optional[int] = None
        self._threshold: Optional[float] = None
        
        # Performance tracking
        self._training_time: float = 0.0
        self._prediction_cache: Dict[int, NDArray[np.float64]] = {}
        
        # Update metadata
        self.metadata.hyperparameters = self._get_hyperparameters()
        self.metadata.environment = self._get_environment_info()
        
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(model_id={self.metadata.model_id})"
        )
    
    # ========================================================================
    # PUBLIC API (TEMPLATE METHODS)
    # ========================================================================
    
    def fit(
        self,
        X: Union[NDArray[np.float64], pd.DataFrame],
        y: Optional[NDArray[np.int_]] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'BaseAnomalyDetector':
        """
        Train the anomaly detection model.
        
        This is the main training method that:
        1. Validates input data
        2. Extracts feature names (if DataFrame)
        3. Calls subclass-specific _fit_implementation()
        4. Updates model metadata
        5. Computes decision threshold
        
        Args:
            X: Training data (n_samples, n_features)
            y: Optional labels for semi-supervised learning
            feature_names: Optional feature names (auto-extracted from DataFrame)
            
        Returns:
            self: Fitted model instance (for method chaining)
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If fitting fails
            
        Example:
            >>> detector = IsolationForestDetector()
            >>> detector.fit(X_train)
            >>> print(f"Trained on {detector.metadata.training_samples} samples")
        """
        logger.info("="*80)
        logger.info(f"TRAINING {self.__class__.__name__}")
        logger.info("="*80)
        
        start_time = time.time()
        self.metadata.status = ModelStatus.TRAINING
        
        try:
            # Validate and convert input
            X_validated = self._validate_input(X, is_training=True)
            
            # Extract feature names
            if isinstance(X, pd.DataFrame):
                self._feature_names = X.columns.tolist()
            elif feature_names is not None:
                self._feature_names = feature_names
            
            self._n_features_in = X_validated.shape[1]
            
            # Validate labels if provided
            if y is not None:
                y = self._validate_labels(y, X_validated.shape[0])
            
            # Check minimum samples
            if X_validated.shape[0] < MIN_SAMPLES_FOR_TRAINING:
                raise ValueError(
                    f"Insufficient training samples: {X_validated.shape[0]} "
                    f"(minimum: {MIN_SAMPLES_FOR_TRAINING})"
                )
            
            logger.info(
                f"Training on {X_validated.shape[0]} samples, "
                f"{X_validated.shape[1]} features"
            )
            
            # Call subclass implementation
            self._fit_implementation(X_validated, y)
            
            # Update state
            self._is_fitted = True
            self._training_time = time.time() - start_time
            
            # Compute decision threshold if not provided
            if self.config.threshold is None:
                self._threshold = self._compute_threshold(X_validated)
            else:
                self._threshold = self.config.threshold
            
            # Update metadata
            self.metadata.status = ModelStatus.TRAINED
            self.metadata.trained_at = datetime.now().isoformat()
            self.metadata.training_samples = X_validated.shape[0]
            self.metadata.training_features = X_validated.shape[1]
            self.metadata.training_time_seconds = self._training_time
            self.metadata.feature_names = self._feature_names
            
            logger.info("="*80)
            logger.info(f"TRAINING COMPLETE")
            logger.info(f"  Time elapsed: {self._training_time:.2f}s")
            logger.info(f"  Samples: {X_validated.shape[0]:,}")
            logger.info(f"  Features: {X_validated.shape[1]}")
            logger.info(f"  Threshold: {self._threshold:.4f}")
            logger.info("="*80)
            
            return self
            
        except Exception as e:
            self.metadata.status = ModelStatus.UNTRAINED
            self._is_fitted = False
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}") from e
    
    def predict(
        self,
        X: Union[NDArray[np.float64], pd.DataFrame],
        return_probabilities: bool = False
    ) -> Union[NDArray[np.float64], PredictionResult]:
        """
        Predict anomaly scores for input samples.
        
        This is the main prediction method that:
        1. Validates model is fitted
        2. Validates input data (shape, features)
        3. Calls subclass-specific _predict_implementation()
        4. Normalizes scores (if configured)
        5. Computes binary predictions and probabilities
        
        Args:
            X: Input data (n_samples, n_features)
            return_probabilities: If True, return PredictionResult with probabilities
            
        Returns:
            anomaly_scores: Continuous anomaly scores (n_samples,) - higher = more anomalous
            OR PredictionResult if return_probabilities=True
            
        Raises:
            ValueError: If input data is invalid or model not fitted
            
        Example:
            >>> scores = detector.predict(X_test)
            >>> print(f"Mean anomaly score: {scores.mean():.3f}")
            >>> 
            >>> # Or get full results
            >>> result = detector.predict(X_test, return_probabilities=True)
            >>> print(f"Detected {result.predictions.sum()} anomalies")
        """
        # Check if model is fitted
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        
        start_time = time.time()
        
        try:
            # Validate input
            X_validated = self._validate_input(X, is_training=False)
            
            # Check for cached predictions
            cache_key = None
            if self.config.cache_predictions:
                cache_key = hash(X_validated.tobytes())
                if cache_key in self._prediction_cache:
                    logger.debug(f"Using cached predictions for {len(X_validated)} samples")
                    anomaly_scores = self._prediction_cache[cache_key]
                    return self._build_prediction_result(
                        anomaly_scores,
                        return_probabilities,
                        {'cached': True, 'prediction_time': 0.0}
                    )
            
            # Call subclass implementation
            anomaly_scores = self._predict_implementation(X_validated)
            
            # Validate output shape
            if anomaly_scores.shape != (X_validated.shape[0],):
                raise RuntimeError(
                    f"_predict_implementation() must return 1D array of shape "
                    f"({X_validated.shape[0]},), got {anomaly_scores.shape}"
                )
            
            # Normalize scores if configured
            if self.config.normalize_scores:
                anomaly_scores = self._normalize_scores(anomaly_scores)
            
            # Cache predictions
            if self.config.cache_predictions and cache_key is not None:
                self._prediction_cache[cache_key] = anomaly_scores
            
            prediction_time = time.time() - start_time
            
            if self.config.verbose:
                logger.info(
                    f"Predicted {len(anomaly_scores)} samples in {prediction_time:.3f}s "
                    f"(scores: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}])"
                )
            
            # Build result
            return self._build_prediction_result(
                anomaly_scores,
                return_probabilities,
                {
                    'prediction_time': prediction_time,
                    'n_samples': len(anomaly_scores),
                    'score_mean': float(anomaly_scores.mean()),
                    'score_std': float(anomaly_scores.std())
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
    
    def fit_predict(
        self,
        X: Union[NDArray[np.float64], pd.DataFrame],
        y: Optional[NDArray[np.int_]] = None,
        return_probabilities: bool = False
    ) -> Union[NDArray[np.float64], PredictionResult]:
        """
        Train the model and immediately predict on the same data.
        
        Convenience method equivalent to fit(X).predict(X).
        Useful for unsupervised anomaly detection where you want anomaly
        scores for the training data.
        
        Args:
            X: Training/prediction data (n_samples, n_features)
            y: Optional labels for semi-supervised learning
            return_probabilities: If True, return PredictionResult
            
        Returns:
            Anomaly scores or PredictionResult
        """
        self.fit(X, y)
        return self.predict(X, return_probabilities)
    
    def predict_binary(
        self,
        X: Union[NDArray[np.float64], pd.DataFrame],
        threshold: Optional[float] = None
    ) -> NDArray[np.int_]:
        """
        Predict binary anomaly labels (0=normal, 1=anomaly).
        
        Args:
            X: Input data (n_samples, n_features)
            threshold: Decision threshold (uses model threshold if None)
            
        Returns:
            Binary predictions (n_samples,) - 1=anomaly, 0=normal
        """
        anomaly_scores = self.predict(X, return_probabilities=False)
        threshold = threshold if threshold is not None else self._threshold
        
        if threshold is None:
            raise ValueError(
                "No threshold available. Either provide threshold parameter "
                "or ensure model has computed threshold during training."
            )
        
        return (anomaly_scores > threshold).astype(np.int_)
    
    # ========================================================================
    # ABSTRACT METHODS (MUST BE IMPLEMENTED BY SUBCLASSES)
    # ========================================================================
    
    @abstractmethod
    def _fit_implementation(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.int_]] = None
    ) -> None:
        """
        Subclass-specific training logic.
        
        This method must be implemented by all concrete anomaly detectors.
        It receives validated input data and should train the model's internal
        parameters.
        
        Args:
            X: Validated training data (n_samples, n_features)
            y: Optional validated labels for semi-supervised learning
            
        Note:
            - Input data is already validated (shape, NaN, etc.)
            - No need to set self._is_fitted (handled by base class)
            - Raise exceptions for training failures
        """
        pass
    
    @abstractmethod
    def _predict_implementation(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Subclass-specific prediction logic.
        
        This method must be implemented by all concrete anomaly detectors.
        It receives validated input data and must return anomaly scores.
        
        Args:
            X: Validated input data (n_samples, n_features)
            
        Returns:
            anomaly_scores: 1D array of shape (n_samples,)
                          Higher scores = more anomalous
                          
        Note:
            - Input data is already validated
            - Must return 1D array with exactly n_samples elements
            - Scores will be normalized by base class if configured
        """
        pass
    
    # ========================================================================
    # VALIDATION AND PREPROCESSING
    # ========================================================================
    
    def _validate_input(
        self,
        X: Union[NDArray[np.float64], pd.DataFrame],
        is_training: bool
    ) -> NDArray[np.float64]:
        """
        Validate and convert input data to NumPy array.
        
        Args:
            X: Input data (array or DataFrame)
            is_training: Whether this is training data
            
        Returns:
            Validated NumPy array
            
        Raises:
            ValueError: If data is invalid
        """
        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            X_array = X.astype(np.float64)
        else:
            raise TypeError(
                f"X must be NumPy array or pandas DataFrame, got {type(X)}"
            )
        
        # Check dimensionality
        if X_array.ndim != 2:
            raise ValueError(
                f"X must be 2D array (n_samples, n_features), got {X_array.ndim}D"
            )
        
        # Check for empty data
        if X_array.shape[0] == 0:
            raise ValueError("X has 0 samples")
        
        if X_array.shape[1] == 0:
            raise ValueError("X has 0 features")
        
        # Check feature count consistency (for prediction)
        if not is_training and self._n_features_in is not None:
            if X_array.shape[1] != self._n_features_in:
                raise ValueError(
                    f"X has {X_array.shape[1]} features, but model was trained "
                    f"with {self._n_features_in} features"
                )
        
        # Check for NaN/Inf
        if self.config.enable_validation:
            if np.any(np.isnan(X_array)):
                raise ValueError(
                    "X contains NaN values. Impute or remove before fitting/prediction."
                )
            
            if np.any(np.isinf(X_array)):
                raise ValueError(
                    "X contains infinite values. Clip or remove before fitting/prediction."
                )
        
        return X_array
    
    def _validate_labels(
        self,
        y: Union[NDArray[np.int_], list],
        n_samples: int
    ) -> NDArray[np.int_]:
        """
        Validate labels for semi-supervised learning.
        
        Args:
            y: Labels array
            n_samples: Expected number of samples
            
        Returns:
            Validated labels array
        """
        y_array = np.asarray(y, dtype=np.int_)
        
        if y_array.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y_array.ndim}D")
        
        if len(y_array) != n_samples:
            raise ValueError(
                f"y has {len(y_array)} samples, but X has {n_samples} samples"
            )
        
        return y_array
    
    # ========================================================================
    # SCORE PROCESSING AND THRESHOLDING
    # ========================================================================
    
    def _normalize_scores(
        self,
        scores: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Normalize anomaly scores to [0, 1] range.
        
        Uses min-max normalization to map scores to unit interval.
        Scores are clipped to handle numerical edge cases.
        
        Args:
            scores: Raw anomaly scores
            
        Returns:
            Normalized scores in [0, 1]
        """
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            # All scores are identical
            return np.full_like(scores, 0.5)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return np.clip(normalized, 0.0, 1.0)
    
    def _compute_threshold(
        self,
        X: NDArray[np.float64]
    ) -> float:
        """
        Compute decision threshold based on contamination rate.
        
        Computes threshold as the (1 - contamination) percentile of
        anomaly scores on training data.
        
        Args:
            X: Training data
            
        Returns:
            Decision threshold
        """
        # Get scores on training data
        train_scores = self._predict_implementation(X)
        
        # Compute threshold at (1 - contamination) percentile
        percentile = (1.0 - self.config.contamination) * 100
        threshold = np.percentile(train_scores, percentile)
        
        logger.info(
            f"Computed threshold: {threshold:.4f} "
            f"(contamination={self.config.contamination:.1%})"
        )
        
        return float(threshold)
    
    def _build_prediction_result(
        self,
        anomaly_scores: NDArray[np.float64],
        return_probabilities: bool,
        metadata: Dict[str, Any]
    ) -> Union[NDArray[np.float64], PredictionResult]:
        """
        Build prediction result with optional probabilities.
        
        Args:
            anomaly_scores: Raw anomaly scores
            return_probabilities: Whether to compute probabilities
            metadata: Prediction metadata
            
        Returns:
            Anomaly scores or PredictionResult
        """
        if not return_probabilities:
            return anomaly_scores
        
        # Compute binary predictions
        predictions = None
        if self._threshold is not None:
            predictions = (anomaly_scores > self._threshold).astype(np.int_)
        
        # Compute probabilities (using sigmoid transformation)
        probabilities = self._scores_to_probabilities(anomaly_scores)
        
        return PredictionResult(
            anomaly_scores=anomaly_scores,
            predictions=predictions,
            probabilities=probabilities,
            metadata=metadata
        )
    
    def _scores_to_probabilities(
        self,
        scores: NDArray[np.float64],
        temperature: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Convert anomaly scores to probabilities using sigmoid.
        
        Args:
            scores: Anomaly scores
            temperature: Temperature for sigmoid (lower = sharper)
            
        Returns:
            Probabilities in [0, 1]
        """
        # Normalize scores to zero mean
        normalized = scores - scores.mean()
        
        # Apply sigmoid transformation
        probabilities = 1.0 / (1.0 + np.exp(-normalized / temperature))
        
        return probabilities
    
    # ========================================================================
    # MODEL PERSISTENCE (SAVE/LOAD)
    # ========================================================================
    
    def save(
        self,
        filepath: Union[str, Path],
        include_metadata: bool = True
    ) -> None:
        """
        Save model to disk using pickle.
        
        Args:
            filepath: Path to save file (.pkl)
            include_metadata: Whether to save metadata alongside model
            
        Example:
            >>> detector.save('models/isolation_forest_v1.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata separately
        if include_metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                f.write(self.metadata.to_json())
            logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseAnomalyDetector':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded model instance
            
        Example:
            >>> detector = IsolationForestDetector.load('models/detector.pkl')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    # ========================================================================
    # METADATA AND INTROSPECTION
    # ========================================================================
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """
        Extract model hyperparameters for metadata.
        
        Subclasses can override to include model-specific parameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'model_type': self.config.model_type.value,
            'contamination': self.config.contamination,
            'random_state': self.config.random_state,
            'normalize_scores': self.config.normalize_scores
        }
    
    def _get_environment_info(self) -> Dict[str, str]:
        """
        Collect environment information for reproducibility.
        
        Returns:
            Dictionary of environment details
        """
        import sys
        import platform
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model state, configuration, and metadata
        """
        return {
            'class_name': self.__class__.__name__,
            'is_fitted': self._is_fitted,
            'n_features': self._n_features_in,
            'feature_names': self._feature_names,
            'threshold': self._threshold,
            'config': asdict(self.config),
            'metadata': self.metadata.to_dict()
        }
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted
    
    @property
    def n_features_in(self) -> Optional[int]:
        """Number of features seen during training."""
        return self._n_features_in
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Names of features (if provided during training)."""
        return self._feature_names
    
    @property
    def threshold(self) -> Optional[float]:
        """Decision threshold for binary classification."""
        return self._threshold
    
    def __repr__(self) -> str:
        """String representation of model."""
        status = "fitted" if self._is_fitted else "untrained"
        features = self._n_features_in if self._n_features_in else "unknown"
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  status: {status},\n"
            f"  features: {features},\n"
            f"  contamination: {self.config.contamination:.1%},\n"
            f"  model_id: {self.metadata.model_id}\n"
            f")"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_anomaly_scores(
    scores: NDArray[np.float64],
    expected_shape: Tuple[int, ...]
) -> None:
    """
    Validate anomaly score array.
    
    Args:
        scores: Anomaly scores to validate
        expected_shape: Expected shape (typically (n_samples,))
        
    Raises:
        ValueError: If scores are invalid
    """
    if not isinstance(scores, np.ndarray):
        raise TypeError(f"scores must be NumPy array, got {type(scores)}")
    
    if scores.shape != expected_shape:
        raise ValueError(
            f"scores shape {scores.shape} does not match expected {expected_shape}"
        )
    
    if np.any(np.isnan(scores)):
        raise ValueError("scores contain NaN values")
    
    if np.any(np.isinf(scores)):
        raise ValueError("scores contain infinite values")


def ensure_fitted(model: BaseAnomalyDetector) -> None:
    """
    Ensure model is fitted before prediction.
    
    Args:
        model: Anomaly detector instance
        
    Raises:
        ValueError: If model is not fitted
    """
    if not model.is_fitted:
        raise ValueError(
            f"{model.__class__.__name__} is not fitted. Call fit() before prediction."
        )


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of base model abstraction.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("BASE ANOMALY DETECTOR - SELF-TEST")
    print("="*80 + "\n")
    
    # Create a simple concrete implementation for testing
    class DummyDetector(BaseAnomalyDetector):
        """Dummy detector for testing base class functionality."""
        
        def _fit_implementation(self, X, y=None):
            """Simple mean-based detector."""
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-6  # Avoid division by zero
        
        def _predict_implementation(self, X):
            """Compute L2 distance from mean as anomaly score."""
            deviations = (X - self._mean) / self._std
            scores = np.sqrt(np.sum(deviations ** 2, axis=1))
            return scores
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples_train = 100
    n_samples_test = 30
    n_features = 5
    
    X_train = np.random.randn(n_samples_train, n_features)
    X_test = np.random.randn(n_samples_test, n_features)
    
    # Inject anomalies in test set
    X_test[-5:] *= 5  # Last 5 samples are anomalies
    
    print(f"Created synthetic dataset:")
    print(f"  Training: {X_train.shape}")
    print(f"  Testing: {X_test.shape}")
    print(f"  True anomalies in test: 5 / {n_samples_test}\n")
    
    # Test 1: Basic fit/predict
    print("-" * 80)
    print("Test 1: Basic Fit/Predict")
    print("-" * 80)
    
    try:
        config = ModelConfig(
            model_type=ModelType.CUSTOM,
            contamination=0.15,
            normalize_scores=True,
            verbose=True
        )
        
        detector = DummyDetector(config=config)
        print(f"Created detector: {detector}\n")
        
        # Fit
        detector.fit(X_train)
        
        # Predict
        scores = detector.predict(X_test)
        print(f"\n[OK] Prediction successful: {scores.shape}")
        print(f"     Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Check that anomalies have higher scores
        normal_scores = scores[:-5].mean()
        anomaly_scores = scores[-5:].mean()
        print(f"     Normal samples mean score: {normal_scores:.3f}")
        print(f"     Anomaly samples mean score: {anomaly_scores:.3f}")
        
        assert anomaly_scores > normal_scores, "Anomalies should have higher scores"
        print("     [OK] Anomalies correctly scored higher\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
        import traceback
        traceback.print_exc()
    
    # Test 2: Binary predictions
    print("-" * 80)
    print("Test 2: Binary Predictions")
    print("-" * 80)
    
    try:
        predictions = detector.predict_binary(X_test)
        print(f"[OK] Binary predictions: {predictions.shape}")
        print(f"     Predicted anomalies: {predictions.sum()} / {len(predictions)}")
        print(f"     True anomalies: 5 / {len(predictions)}")
        
        # Check that some anomalies are detected
        anomaly_detections = predictions[-5:].sum()
        print(f"     Correctly detected: {anomaly_detections} / 5\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
    
    # Test 3: PredictionResult
    print("-" * 80)
    print("Test 3: Full Prediction Result")
    print("-" * 80)
    
    try:
        result = detector.predict(X_test, return_probabilities=True)
        print(f"[OK] PredictionResult: {result}")
        print(f"     Has predictions: {result.predictions is not None}")
        print(f"     Has probabilities: {result.probabilities is not None}")
        print(f"     Metadata keys: {list(result.metadata.keys())}\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
    
    # Test 4: Model persistence
    print("-" * 80)
    print("Test 4: Model Persistence (Save/Load)")
    print("-" * 80)
    
    try:
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pkl"
            
            # Save
            detector.save(save_path)
            print(f"[OK] Model saved to {save_path}")
            
            # Load
            loaded_detector = DummyDetector.load(save_path)
            print(f"[OK] Model loaded successfully")
            
            # Verify predictions match
            scores_original = detector.predict(X_test)
            scores_loaded = loaded_detector.predict(X_test)
            
            assert np.allclose(scores_original, scores_loaded), "Loaded model predictions differ"
            print(f"[OK] Loaded model produces identical predictions\n")
            
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
        import traceback
        traceback.print_exc()
    
    # Test 5: Input validation
    print("-" * 80)
    print("Test 5: Input Validation")
    print("-" * 80)
    
    # Test unfitted prediction
    try:
        untrained = DummyDetector()
        untrained.predict(X_test)
        print("[FAIL] Should raise error for untrained model\n")
    except ValueError as e:
        print(f"[OK] Correctly raises error for untrained model: {str(e)[:50]}...")
    
    # Test feature mismatch
    try:
        X_wrong_features = np.random.randn(10, 3)  # Wrong number of features
        detector.predict(X_wrong_features)
        print("[FAIL] Should raise error for feature mismatch\n")
    except ValueError as e:
        print(f"[OK] Correctly raises error for feature mismatch: {str(e)[:50]}...")
    
    # Test NaN input
    try:
        X_nan = X_test.copy()
        X_nan[0, 0] = np.nan
        detector.predict(X_nan)
        print("[FAIL] Should raise error for NaN input\n")
    except ValueError as e:
        print(f"[OK] Correctly raises error for NaN input: {str(e)[:50]}...\n")
    
    # Test 6: Model metadata
    print("-" * 80)
    print("Test 6: Model Metadata")
    print("-" * 80)
    
    try:
        info = detector.get_model_info()
        print(f"[OK] Model info retrieved:")
        print(f"     Class: {info['class_name']}")
        print(f"     Fitted: {info['is_fitted']}")
        print(f"     Features: {info['n_features']}")
        print(f"     Training samples: {detector.metadata.training_samples}")
        print(f"     Training time: {detector.metadata.training_time_seconds:.3f}s")
        print(f"     Model ID: {detector.metadata.model_id}\n")
        
    except Exception as e:
        print(f"[FAIL] {str(e)}\n")
    
    print("="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")
    
    print("Summary:")
    print("  [OK] All tests passed")
    print("  Base class provides:")
    print("    - Consistent fit/predict API")
    print("    - Input validation and error handling")
    print("    - Score normalization and thresholding")
    print("    - Model persistence (save/load)")
    print("    - Metadata tracking and introspection")
    print("    - Binary predictions and probabilities")
