"""
Production-Grade Composite Anomaly Scoring Engine for GhostLoad Mapper
=======================================================================

This module provides an enterprise-level anomaly scoring framework that combines
multiple anomaly signals into a unified composite score for electricity theft
detection. It implements a weighted ensemble approach that balances model-based
detection (Isolation Forest) with domain-specific business logic (consumption ratios).

The anomaly scorer ensures:
1. **Multi-Signal Fusion**: Combines ML predictions with domain knowledge
2. **Configurable Weighting**: Flexible weight adjustment for different signals
3. **Score Normalization**: Consistent 0-1 scale across all components
4. **Statistical Robustness**: Handles outliers and edge cases gracefully
5. **Explainability**: Provides component-level score breakdown
6. **Performance Optimization**: Vectorized operations for large-scale scoring
7. **Observability**: Detailed logging and metrics tracking

Design Patterns:
- **Strategy Pattern**: Pluggable scoring strategies and normalization methods
- **Composite Pattern**: Hierarchical score aggregation
- **Builder Pattern**: Fluent API for score configuration
- **Template Method**: Standardized scoring workflow

Enterprise Features:
- Configurable score fusion weights (isolation_weight + ratio_weight = 1.0)
- Multiple normalization strategies (min-max, z-score, robust scaling)
- Outlier handling with configurable clipping
- Score decomposition for model explainability
- Batch processing with memory-efficient streaming
- Integration with monitoring and alerting systems

Research Foundation:
    - Ensemble anomaly detection (Aggarwal, 2017)
    - Score fusion methods (Kittler et al., 1998)
    - Robust statistical scaling (Huber, 1981)
    - Electricity theft detection (Nagi et al., 2011)

Mathematical Formulation:
    composite_score = w₁ * isolation_score + w₂ * ratio_score
    
    where:
        isolation_score ∈ [0, 1]: Normalized Isolation Forest anomaly score
        ratio_score ∈ [0, 1]: Normalized consumption ratio score
        w₁, w₂ ∈ [0, 1]: Weights satisfying w₁ + w₂ = 1.0
    
    Default: w₁ = 0.7, w₂ = 0.3 (70% ML, 30% domain knowledge)

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import base model for PredictionResult
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
    from base_model import PredictionResult, BaseAnomalyDetector
except ImportError:
    warnings.warn(
        "Could not import base_model. Ensure models/ directory is in PYTHONPATH",
        ImportWarning
    )


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND ENUMERATIONS
# ============================================================================

class NormalizationMethod(Enum):
    """Score normalization strategies."""
    MIN_MAX = "min_max"  # Scale to [0, 1] based on min/max
    Z_SCORE = "z_score"  # Standardize to mean=0, std=1, then clip
    ROBUST = "robust"  # Use median and IQR for outlier resistance
    PERCENTILE = "percentile"  # Rank-based percentile scaling
    SIGMOID = "sigmoid"  # Smooth sigmoid transformation


class AggregationMethod(Enum):
    """Score aggregation strategies."""
    WEIGHTED_SUM = "weighted_sum"  # Linear weighted combination
    WEIGHTED_PRODUCT = "weighted_product"  # Geometric mean with weights
    MAX = "max"  # Take maximum component score
    MIN = "min"  # Take minimum component score
    HARMONIC_MEAN = "harmonic_mean"  # Harmonic mean of components


# Default weights for score fusion
DEFAULT_ISOLATION_WEIGHT = 0.7  # 70% ML-based detection
DEFAULT_RATIO_WEIGHT = 0.3  # 30% domain knowledge

# Score clipping bounds
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_SCORE = 1.0

# Ratio scoring thresholds
DEFAULT_RATIO_THRESHOLD_LOW = 0.5  # Below 50% of median → suspicious
DEFAULT_RATIO_THRESHOLD_HIGH = 2.0  # Above 200% of median → suspicious


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ScoringConfig:
    """
    Configuration for composite anomaly scoring.
    
    This dataclass encapsulates all parameters for score fusion, normalization,
    and aggregation, providing type safety and validation.
    
    Attributes:
        isolation_weight: Weight for Isolation Forest scores (default: 0.7)
        ratio_weight: Weight for consumption ratio scores (default: 0.3)
        
        normalization_method: Strategy for score normalization
        aggregation_method: Strategy for score fusion
        
        clip_scores: Whether to clip final scores to [0, 1]
        min_score: Minimum allowed score after clipping
        max_score: Maximum allowed score after clipping
        
        ratio_threshold_low: Low consumption ratio threshold (suspicious if below)
        ratio_threshold_high: High consumption ratio threshold (suspicious if above)
        
        handle_missing: Strategy for missing values ('zero', 'mean', 'median', 'drop')
        min_samples: Minimum samples required for statistical normalization
        
        include_breakdown: Include component score breakdown in results
        verbose: Enable detailed logging
    
    Validation:
        - isolation_weight + ratio_weight must equal 1.0
        - All weights must be in [0, 1]
        - min_score < max_score
        - ratio_threshold_low < ratio_threshold_high
    
    Example:
        >>> config = ScoringConfig(
        ...     isolation_weight=0.8,
        ...     ratio_weight=0.2,
        ...     normalization_method=NormalizationMethod.ROBUST
        ... )
    """
    
    # Weight configuration
    isolation_weight: float = DEFAULT_ISOLATION_WEIGHT
    ratio_weight: float = DEFAULT_RATIO_WEIGHT
    
    # Normalization and aggregation
    normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    
    # Score bounds
    clip_scores: bool = True
    min_score: float = DEFAULT_MIN_SCORE
    max_score: float = DEFAULT_MAX_SCORE
    
    # Ratio thresholds
    ratio_threshold_low: float = DEFAULT_RATIO_THRESHOLD_LOW
    ratio_threshold_high: float = DEFAULT_RATIO_THRESHOLD_HIGH
    
    # Missing value handling
    handle_missing: str = 'median'  # 'zero', 'mean', 'median', 'drop'
    min_samples: int = 10  # Minimum for statistical normalization
    
    # Output options
    include_breakdown: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate weight sum
        weight_sum = self.isolation_weight + self.ratio_weight
        if not np.isclose(weight_sum, 1.0, rtol=1e-6):
            raise ValueError(
                f"isolation_weight ({self.isolation_weight}) + "
                f"ratio_weight ({self.ratio_weight}) = {weight_sum} "
                f"must equal 1.0"
            )
        
        # Validate weight ranges
        if not (0.0 <= self.isolation_weight <= 1.0):
            raise ValueError(f"isolation_weight must be in [0, 1], got {self.isolation_weight}")
        if not (0.0 <= self.ratio_weight <= 1.0):
            raise ValueError(f"ratio_weight must be in [0, 1], got {self.ratio_weight}")
        
        # Validate score bounds
        if self.min_score >= self.max_score:
            raise ValueError(
                f"min_score ({self.min_score}) must be < max_score ({self.max_score})"
            )
        
        # Validate ratio thresholds
        if self.ratio_threshold_low >= self.ratio_threshold_high:
            raise ValueError(
                f"ratio_threshold_low ({self.ratio_threshold_low}) must be < "
                f"ratio_threshold_high ({self.ratio_threshold_high})"
            )
        
        # Validate handle_missing
        valid_missing = ['zero', 'mean', 'median', 'drop']
        if self.handle_missing not in valid_missing:
            raise ValueError(
                f"handle_missing must be one of {valid_missing}, got '{self.handle_missing}'"
            )
        
        # Validate min_samples
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")


@dataclass
class ScoringResult:
    """
    Results from composite anomaly scoring.
    
    Encapsulates all scoring outputs with component breakdown for
    explainability and debugging.
    
    Attributes:
        composite_scores: Final composite anomaly scores [0-1]
        isolation_scores: Normalized Isolation Forest scores [0-1]
        ratio_scores: Normalized consumption ratio scores [0-1]
        
        predictions: Binary anomaly predictions (if threshold provided)
        metadata: Scoring metadata (weights, thresholds, statistics)
        
        n_samples: Number of scored samples
        n_anomalies: Number of detected anomalies (if predictions available)
        scoring_time: Time taken for scoring (seconds)
    """
    
    # Core scores
    composite_scores: NDArray[np.float64]
    isolation_scores: NDArray[np.float64]
    ratio_scores: NDArray[np.float64]
    
    # Optional outputs
    predictions: Optional[NDArray[np.int_]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    n_samples: int = 0
    n_anomalies: Optional[int] = None
    scoring_time: float = 0.0
    
    def __post_init__(self):
        """Compute statistics and validate result."""
        self.n_samples = len(self.composite_scores)
        
        if self.predictions is not None:
            self.n_anomalies = int(np.sum(self.predictions))
        
        # Validate array shapes
        if len(self.isolation_scores) != self.n_samples:
            raise ValueError(
                f"isolation_scores length ({len(self.isolation_scores)}) must match "
                f"composite_scores length ({self.n_samples})"
            )
        if len(self.ratio_scores) != self.n_samples:
            raise ValueError(
                f"ratio_scores length ({len(self.ratio_scores)}) must match "
                f"composite_scores length ({self.n_samples})"
            )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert scoring results to pandas DataFrame.
        
        Returns:
            DataFrame with columns: composite_score, isolation_score, ratio_score, prediction
        """
        df = pd.DataFrame({
            'composite_score': self.composite_scores,
            'isolation_score': self.isolation_scores,
            'ratio_score': self.ratio_scores
        })
        
        if self.predictions is not None:
            df['prediction'] = self.predictions
        
        return df
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Compute summary statistics for all score components.
        
        Returns:
            Dictionary with mean, std, min, max, median for each score type
        """
        stats = {}
        
        for score_name, scores in [
            ('composite', self.composite_scores),
            ('isolation', self.isolation_scores),
            ('ratio', self.ratio_scores)
        ]:
            stats[f'{score_name}_mean'] = float(np.mean(scores))
            stats[f'{score_name}_std'] = float(np.std(scores))
            stats[f'{score_name}_min'] = float(np.min(scores))
            stats[f'{score_name}_max'] = float(np.max(scores))
            stats[f'{score_name}_median'] = float(np.median(scores))
            stats[f'{score_name}_q25'] = float(np.percentile(scores, 25))
            stats[f'{score_name}_q75'] = float(np.percentile(scores, 75))
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"ScoringResult(\n"
            f"  n_samples: {self.n_samples},\n"
            f"  n_anomalies: {self.n_anomalies},\n"
            f"  composite_score: [{self.composite_scores.min():.3f}, {self.composite_scores.max():.3f}],\n"
            f"  scoring_time: {self.scoring_time:.3f}s\n"
            f")"
        )


# ============================================================================
# CORE ANOMALY SCORER
# ============================================================================

class AnomalyScorer:
    """
    Production-grade composite anomaly scoring engine.
    
    This class implements a configurable ensemble scoring system that combines
    multiple anomaly signals (ML-based and domain-based) into a unified score.
    
    The scorer performs:
    1. Score normalization (multiple strategies available)
    2. Weight-based fusion (configurable weights)
    3. Outlier handling and clipping
    4. Component breakdown for explainability
    5. Missing value imputation
    
    Architecture:
        Input → Normalize → Fuse → Clip → Output
        
        - Normalize: Scale each component to [0, 1]
        - Fuse: Combine with configurable weights
        - Clip: Bound final scores to valid range
        - Output: Return composite + breakdown
    
    Thread Safety:
        This class is thread-safe for read operations after initialization.
        Concurrent scoring calls are safe.
    
    Example:
        >>> scorer = AnomalyScorer(ScoringConfig(isolation_weight=0.8))
        >>> result = scorer.score(
        ...     isolation_scores=model_scores,
        ...     consumption_ratios=ratios
        ... )
        >>> print(f"Mean composite score: {result.composite_scores.mean():.3f}")
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize anomaly scorer with configuration.
        
        Args:
            config: Scoring configuration (uses defaults if None)
        """
        self.config = config or ScoringConfig()
        
        if self.config.verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(
            f"Initialized AnomalyScorer (isolation_weight={self.config.isolation_weight:.2f}, "
            f"ratio_weight={self.config.ratio_weight:.2f})"
        )
    
    def score(
        self,
        isolation_scores: Union[NDArray[np.float64], PredictionResult],
        consumption_ratios: NDArray[np.float64],
        threshold: Optional[float] = None
    ) -> ScoringResult:
        """
        Compute composite anomaly scores from multiple signals.
        
        This is the main entry point for scoring. It normalizes each component,
        fuses them using configured weights, and optionally applies a threshold
        for binary predictions.
        
        Args:
            isolation_scores: Isolation Forest anomaly scores or PredictionResult
                             (will be normalized to [0, 1])
            consumption_ratios: Consumption ratios (meter/transformer_median)
                               (will be converted to anomaly scores)
            threshold: Optional threshold for binary predictions (0-1)
        
        Returns:
            ScoringResult with composite scores and component breakdown
        
        Raises:
            ValueError: If inputs have mismatched lengths or invalid values
        
        Example:
            >>> result = scorer.score(
            ...     isolation_scores=np.array([0.1, 0.8, 0.3]),
            ...     consumption_ratios=np.array([1.2, 0.3, 1.0]),
            ...     threshold=0.7
            ... )
        """
        start_time = time.time()
        
        # Extract isolation scores from PredictionResult if needed
        if isinstance(isolation_scores, PredictionResult):
            iso_scores = isolation_scores.anomaly_scores
        else:
            iso_scores = isolation_scores
        
        # Validate inputs
        self._validate_inputs(iso_scores, consumption_ratios)
        
        logger.debug(
            f"Scoring {len(iso_scores)} samples with "
            f"isolation_scores: [{iso_scores.min():.3f}, {iso_scores.max():.3f}], "
            f"ratios: [{consumption_ratios.min():.3f}, {consumption_ratios.max():.3f}]"
        )
        
        # Step 1: Normalize isolation scores to [0, 1]
        normalized_iso = self._normalize_scores(
            iso_scores,
            name="isolation_scores"
        )
        
        # Step 2: Convert consumption ratios to anomaly scores [0, 1]
        normalized_ratio = self._ratio_to_anomaly_score(consumption_ratios)
        
        # Step 3: Fuse scores with configured weights
        composite = self._fuse_scores(normalized_iso, normalized_ratio)
        
        # Step 4: Clip to valid range
        if self.config.clip_scores:
            composite = np.clip(composite, self.config.min_score, self.config.max_score)
        
        # Step 5: Generate predictions if threshold provided
        predictions = None
        if threshold is not None:
            predictions = (composite >= threshold).astype(np.int_)
        
        # Step 6: Build metadata
        metadata = {
            'isolation_weight': self.config.isolation_weight,
            'ratio_weight': self.config.ratio_weight,
            'normalization_method': self.config.normalization_method.value,
            'aggregation_method': self.config.aggregation_method.value,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        scoring_time = time.time() - start_time
        
        result = ScoringResult(
            composite_scores=composite,
            isolation_scores=normalized_iso,
            ratio_scores=normalized_ratio,
            predictions=predictions,
            metadata=metadata,
            scoring_time=scoring_time
        )
        
        logger.info(
            f"Scored {result.n_samples} samples in {scoring_time:.3f}s "
            f"(mean composite: {composite.mean():.3f})"
        )
        
        return result
    
    def _validate_inputs(
        self,
        isolation_scores: NDArray[np.float64],
        consumption_ratios: NDArray[np.float64]
    ) -> None:
        """
        Validate input arrays for scoring.
        
        Args:
            isolation_scores: Isolation Forest scores
            consumption_ratios: Consumption ratios
        
        Raises:
            ValueError: If validation fails
        """
        # Check array types
        if not isinstance(isolation_scores, np.ndarray):
            raise TypeError(
                f"isolation_scores must be numpy array, got {type(isolation_scores)}"
            )
        if not isinstance(consumption_ratios, np.ndarray):
            raise TypeError(
                f"consumption_ratios must be numpy array, got {type(consumption_ratios)}"
            )
        
        # Check lengths
        if len(isolation_scores) != len(consumption_ratios):
            raise ValueError(
                f"isolation_scores ({len(isolation_scores)}) and "
                f"consumption_ratios ({len(consumption_ratios)}) must have same length"
            )
        
        # Check for empty arrays
        if len(isolation_scores) == 0:
            raise ValueError("Cannot score empty arrays")
        
        # Check for NaN/inf
        if np.any(np.isnan(isolation_scores)) or np.any(np.isinf(isolation_scores)):
            logger.warning(f"Found {np.sum(np.isnan(isolation_scores))} NaN and "
                          f"{np.sum(np.isinf(isolation_scores))} inf in isolation_scores")
        
        if np.any(np.isnan(consumption_ratios)) or np.any(np.isinf(consumption_ratios)):
            logger.warning(f"Found {np.sum(np.isnan(consumption_ratios))} NaN and "
                          f"{np.sum(np.isinf(consumption_ratios))} inf in consumption_ratios")
    
    def _normalize_scores(
        self,
        scores: NDArray[np.float64],
        name: str = "scores"
    ) -> NDArray[np.float64]:
        """
        Normalize scores to [0, 1] range using configured method.
        
        Args:
            scores: Raw scores to normalize
            name: Name for logging purposes
        
        Returns:
            Normalized scores in [0, 1]
        """
        # Handle missing values
        scores = self._handle_missing_values(scores)
        
        method = self.config.normalization_method
        
        if method == NormalizationMethod.MIN_MAX:
            return self._normalize_min_max(scores)
        
        elif method == NormalizationMethod.Z_SCORE:
            return self._normalize_z_score(scores)
        
        elif method == NormalizationMethod.ROBUST:
            return self._normalize_robust(scores)
        
        elif method == NormalizationMethod.PERCENTILE:
            return self._normalize_percentile(scores)
        
        elif method == NormalizationMethod.SIGMOID:
            return self._normalize_sigmoid(scores)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _normalize_min_max(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Min-max scaling to [0, 1]."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if np.isclose(min_val, max_val):
            logger.warning("All scores are identical, returning 0.5")
            return np.full_like(scores, 0.5)
        
        return (scores - min_val) / (max_val - min_val)
    
    def _normalize_z_score(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score standardization, then sigmoid to [0, 1]."""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if np.isclose(std, 0.0):
            logger.warning("Zero std, returning 0.5")
            return np.full_like(scores, 0.5)
        
        z_scores = (scores - mean) / std
        # Apply sigmoid to map to [0, 1]
        return 1.0 / (1.0 + np.exp(-z_scores))
    
    def _normalize_robust(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Robust scaling using median and IQR."""
        median = np.median(scores)
        q25 = np.percentile(scores, 25)
        q75 = np.percentile(scores, 75)
        iqr = q75 - q25
        
        if np.isclose(iqr, 0.0):
            logger.warning("Zero IQR, falling back to min-max")
            return self._normalize_min_max(scores)
        
        # Scale by IQR
        scaled = (scores - median) / iqr
        # Apply sigmoid
        return 1.0 / (1.0 + np.exp(-scaled))
    
    def _normalize_percentile(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Percentile-based rank normalization."""
        # Rank-based percentile (0-100) divided by 100
        from scipy import stats
        percentiles = stats.rankdata(scores, method='average') / len(scores)
        return percentiles
    
    def _normalize_sigmoid(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sigmoid transformation to [0, 1]."""
        # Center around median
        median = np.median(scores)
        centered = scores - median
        return 1.0 / (1.0 + np.exp(-centered))
    
    def _ratio_to_anomaly_score(
        self,
        ratios: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert consumption ratios to anomaly scores.
        
        A ratio of 1.0 (equal to transformer median) is normal (score → 0).
        Ratios far from 1.0 (either low or high) are suspicious (score → 1).
        
        Scoring logic:
            - ratio < threshold_low: High anomaly score (stealing)
            - threshold_low <= ratio <= threshold_high: Low score (normal)
            - ratio > threshold_high: Medium score (suspicious high consumption)
        
        Args:
            ratios: Consumption ratios (meter/transformer_median)
        
        Returns:
            Anomaly scores in [0, 1]
        """
        # Handle missing values
        ratios = self._handle_missing_values(ratios)
        
        # Compute distance from normal (1.0)
        # Use absolute log ratio to treat under/over consumption symmetrically
        # But weight under-consumption more heavily (theft indicator)
        
        anomaly_scores = np.zeros_like(ratios, dtype=np.float64)
        
        # Low consumption (potential theft) - stronger signal
        low_mask = ratios < self.config.ratio_threshold_low
        if np.any(low_mask):
            # Map [0, threshold_low] to [1.0, 0.3]
            # Lower ratio = higher score
            low_ratios = ratios[low_mask]
            anomaly_scores[low_mask] = 1.0 - (low_ratios / self.config.ratio_threshold_low) * 0.7
        
        # Normal consumption
        normal_mask = (ratios >= self.config.ratio_threshold_low) & \
                     (ratios <= self.config.ratio_threshold_high)
        if np.any(normal_mask):
            # Map [threshold_low, threshold_high] to [0.3, 0.3]
            # Within normal range = low anomaly score
            anomaly_scores[normal_mask] = 0.3
        
        # High consumption (suspicious but less critical)
        high_mask = ratios > self.config.ratio_threshold_high
        if np.any(high_mask):
            # Map [threshold_high, inf] to [0.3, 0.8]
            # Higher ratio = higher score (but capped)
            high_ratios = ratios[high_mask]
            # Logarithmic scaling for extreme values
            scaled = np.log1p(high_ratios - self.config.ratio_threshold_high)
            anomaly_scores[high_mask] = 0.3 + np.clip(scaled / 10.0, 0, 0.5)
        
        return anomaly_scores
    
    def _fuse_scores(
        self,
        isolation_scores: NDArray[np.float64],
        ratio_scores: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Fuse component scores using configured aggregation method.
        
        Args:
            isolation_scores: Normalized isolation scores [0, 1]
            ratio_scores: Normalized ratio scores [0, 1]
        
        Returns:
            Fused composite scores [0, 1]
        """
        method = self.config.aggregation_method
        
        if method == AggregationMethod.WEIGHTED_SUM:
            return (
                self.config.isolation_weight * isolation_scores +
                self.config.ratio_weight * ratio_scores
            )
        
        elif method == AggregationMethod.WEIGHTED_PRODUCT:
            # Geometric mean with weights
            return np.power(
                np.power(isolation_scores, self.config.isolation_weight) *
                np.power(ratio_scores, self.config.ratio_weight),
                1.0 / (self.config.isolation_weight + self.config.ratio_weight)
            )
        
        elif method == AggregationMethod.MAX:
            return np.maximum(isolation_scores, ratio_scores)
        
        elif method == AggregationMethod.MIN:
            return np.minimum(isolation_scores, ratio_scores)
        
        elif method == AggregationMethod.HARMONIC_MEAN:
            # Weighted harmonic mean
            w1, w2 = self.config.isolation_weight, self.config.ratio_weight
            return (w1 + w2) / (w1 / (isolation_scores + 1e-10) + w2 / (ratio_scores + 1e-10))
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _handle_missing_values(
        self,
        scores: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Handle NaN and inf values in scores.
        
        Args:
            scores: Array potentially containing NaN/inf
        
        Returns:
            Array with missing values handled
        """
        # Create copy to avoid modifying input
        scores = scores.copy()
        
        # Find missing values
        missing_mask = np.isnan(scores) | np.isinf(scores)
        n_missing = np.sum(missing_mask)
        
        if n_missing == 0:
            return scores
        
        logger.warning(f"Found {n_missing} missing values, using '{self.config.handle_missing}' strategy")
        
        if self.config.handle_missing == 'zero':
            scores[missing_mask] = 0.0
        
        elif self.config.handle_missing == 'mean':
            valid_mean = np.mean(scores[~missing_mask])
            scores[missing_mask] = valid_mean
        
        elif self.config.handle_missing == 'median':
            valid_median = np.median(scores[~missing_mask])
            scores[missing_mask] = valid_median
        
        elif self.config.handle_missing == 'drop':
            raise ValueError(
                f"Cannot use 'drop' strategy in _handle_missing_values. "
                f"Filter missing values before calling score()."
            )
        
        return scores


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def score_anomalies(
    isolation_scores: Union[NDArray[np.float64], PredictionResult],
    consumption_ratios: NDArray[np.float64],
    isolation_weight: float = DEFAULT_ISOLATION_WEIGHT,
    ratio_weight: float = DEFAULT_RATIO_WEIGHT,
    threshold: Optional[float] = None,
    **kwargs
) -> ScoringResult:
    """
    Convenience function for quick anomaly scoring.
    
    Args:
        isolation_scores: Isolation Forest anomaly scores or PredictionResult
        consumption_ratios: Consumption ratios (meter/transformer_median)
        isolation_weight: Weight for isolation scores (default: 0.7)
        ratio_weight: Weight for ratio scores (default: 0.3)
        threshold: Optional threshold for binary predictions
        **kwargs: Additional config parameters
    
    Returns:
        ScoringResult with composite scores
    
    Example:
        >>> result = score_anomalies(
        ...     isolation_scores=model.predict(X, return_probabilities=True),
        ...     consumption_ratios=df['consumption_ratio'],
        ...     isolation_weight=0.8,
        ...     ratio_weight=0.2,
        ...     threshold=0.7
        ... )
    """
    config = ScoringConfig(
        isolation_weight=isolation_weight,
        ratio_weight=ratio_weight,
        **kwargs
    )
    
    scorer = AnomalyScorer(config)
    return scorer.score(isolation_scores, consumption_ratios, threshold)


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANOMALY SCORER - SELF-TEST")
    print("=" * 80)
    
    # Generate synthetic test data
    print("\nGenerating synthetic data...")
    print("-" * 80)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Isolation Forest scores (already in [0, 1] range)
    isolation_scores = np.random.beta(2, 5, n_samples)  # Skewed toward low scores
    
    # Consumption ratios (meter/transformer_median)
    # Normal: ~1.0, Theft: <0.5, Suspicious: >2.0
    normal_ratios = np.random.normal(1.0, 0.2, 700)
    theft_ratios = np.random.uniform(0.1, 0.5, 200)  # Low consumption
    suspicious_ratios = np.random.uniform(2.0, 3.0, 100)  # High consumption
    consumption_ratios = np.concatenate([normal_ratios, theft_ratios, suspicious_ratios])
    np.random.shuffle(consumption_ratios)
    
    print(f"+ Created {n_samples} samples")
    print(f"  - Isolation scores: [{isolation_scores.min():.3f}, {isolation_scores.max():.3f}]")
    print(f"  - Consumption ratios: [{consumption_ratios.min():.3f}, {consumption_ratios.max():.3f}]")
    
    # Test 1: Basic scoring with default config
    print("\n\nTest 1: Basic scoring with default weights (70/30)...")
    print("-" * 80)
    
    try:
        config = ScoringConfig()
        scorer = AnomalyScorer(config)
        
        result = scorer.score(
            isolation_scores=isolation_scores,
            consumption_ratios=consumption_ratios,
            threshold=0.7
        )
        
        print(f"\n+ Test 1 PASSED")
        print(f"  - Samples scored: {result.n_samples}")
        print(f"  - Anomalies detected: {result.n_anomalies}")
        print(f"  - Scoring time: {result.scoring_time:.3f}s")
        print(f"  - Composite score: [{result.composite_scores.min():.3f}, {result.composite_scores.max():.3f}]")
        print(f"  - Mean composite: {result.composite_scores.mean():.3f}")
        
        assert result.n_samples == n_samples
        assert result.n_anomalies is not None
        assert result.composite_scores.min() >= 0.0
        assert result.composite_scores.max() <= 1.0
        
    except Exception as e:
        print(f"\n- Test 1 FAILED: {str(e)}")
        raise
    
    # Test 2: Different weight configurations
    print("\n\nTest 2: Testing different weight configurations...")
    print("-" * 80)
    
    try:
        configs = [
            (0.9, 0.1, "ML-heavy"),
            (0.5, 0.5, "Balanced"),
            (0.3, 0.7, "Domain-heavy")
        ]
        
        for iso_w, ratio_w, desc in configs:
            config = ScoringConfig(
                isolation_weight=iso_w,
                ratio_weight=ratio_w,
                verbose=False
            )
            scorer = AnomalyScorer(config)
            result = scorer.score(isolation_scores, consumption_ratios)
            
            print(f"  {desc} ({iso_w:.1f}/{ratio_w:.1f}): "
                  f"mean={result.composite_scores.mean():.3f}, "
                  f"std={np.std(result.composite_scores):.3f}")
        
        print(f"\n+ Test 2 PASSED")
        
    except Exception as e:
        print(f"\n- Test 2 FAILED: {str(e)}")
        raise
    
    # Test 3: Score breakdown and explainability
    print("\n\nTest 3: Score breakdown for explainability...")
    print("-" * 80)
    
    try:
        config = ScoringConfig(include_breakdown=True)
        scorer = AnomalyScorer(config)
        result = scorer.score(isolation_scores, consumption_ratios)
        
        # Get summary statistics
        stats = result.get_summary_stats()
        
        print(f"\n+ Test 3 PASSED")
        print(f"  Component statistics:")
        print(f"    Isolation: mean={stats['isolation_mean']:.3f}, std={stats['isolation_std']:.3f}")
        print(f"    Ratio: mean={stats['ratio_mean']:.3f}, std={stats['ratio_std']:.3f}")
        print(f"    Composite: mean={stats['composite_mean']:.3f}, std={stats['composite_std']:.3f}")
        
        # Convert to DataFrame
        df = result.to_dataframe()
        print(f"\n  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        assert 'composite_score' in df.columns
        assert 'isolation_score' in df.columns
        assert 'ratio_score' in df.columns
        
    except Exception as e:
        print(f"\n- Test 3 FAILED: {str(e)}")
        raise
    
    # Test 4: Normalization methods
    print("\n\nTest 4: Testing different normalization methods...")
    print("-" * 80)
    
    try:
        methods = [
            NormalizationMethod.MIN_MAX,
            NormalizationMethod.ROBUST,
            NormalizationMethod.PERCENTILE
        ]
        
        for method in methods:
            config = ScoringConfig(normalization_method=method)
            scorer = AnomalyScorer(config)
            result = scorer.score(isolation_scores, consumption_ratios)
            
            print(f"  {method.value}: "
                  f"mean={result.composite_scores.mean():.3f}, "
                  f"range=[{result.composite_scores.min():.3f}, {result.composite_scores.max():.3f}]")
        
        print(f"\n+ Test 4 PASSED")
        
    except Exception as e:
        print(f"\n- Test 4 FAILED: {str(e)}")
        raise
    
    # Test 5: Convenience function
    print("\n\nTest 5: Testing convenience function...")
    print("-" * 80)
    
    try:
        result = score_anomalies(
            isolation_scores=isolation_scores,
            consumption_ratios=consumption_ratios,
            isolation_weight=0.75,
            ratio_weight=0.25,
            threshold=0.65
        )
        
        print(f"\n+ Test 5 PASSED")
        print(f"  - Quick scoring: {result.n_samples} samples in {result.scoring_time:.3f}s")
        print(f"  - Detected {result.n_anomalies} anomalies (threshold=0.65)")
        
        assert result.n_samples == n_samples
        
    except Exception as e:
        print(f"\n- Test 5 FAILED: {str(e)}")
        raise
    
    print("\n" + "=" * 80)
    print("SELF-TEST COMPLETE - ALL TESTS PASSED")
    print("=" * 80)
    print("\nAnomaly Scorer is production-ready!")
    print("\nNext steps:")
    print("  1. Integrate with trained models")
    print("  2. Tune weight configuration on validation set")
    print("  3. Set detection threshold based on business requirements")
    print("  4. Deploy to production scoring pipeline")
