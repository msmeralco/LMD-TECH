"""
Production-Grade Feature Engineering for GhostLoad Mapper ML System
=====================================================================

This module provides world-class feature engineering capabilities for electrical meter
consumption anomaly detection, implementing research-backed techniques for non-technical
loss (NTL) identification in power distribution systems.

The feature engineer computes transformer-level baselines and meter-level behavioral
indicators that expose patterns characteristic of electricity theft:

1. Transformer Baseline Statistics:
   - transformer_baseline_median: Robust central tendency of all connected meters
   - transformer_baseline_variance: Consumption variability across peer meters

2. Temporal Consumption Trends:
   - meter_consumption_trend: Linear regression slope over last 6 months
   - Captures declining consumption patterns (common in theft scenarios)

3. Relative Consumption Indicators:
   - consumption_ratio_to_transformer_median: Individual vs. peer comparison
   - Identifies meters with abnormally low consumption relative to peers

Architecture:
    - FeatureConfig: Type-safe configuration with validation
    - TransformerBaselineComputer: Computes group-level statistical baselines
    - ConsumptionTrendAnalyzer: Temporal trend detection via robust regression
    - RelativeConsumptionComputer: Peer-based comparative features
    - FeatureEngineer: Main facade orchestrating feature computation pipeline
    - FeatureResult: Type-safe container for engineered features + metadata

Design Principles:
    - Transformer-awareness: Group-based features prevent cross-contamination
    - Statistical robustness: Median-based estimators resistant to outliers
    - Temporal sensitivity: Capture time-evolving consumption patterns
    - Explainability: Features directly interpretable by domain experts
    - Deterministic: Reproducible results for ML experiment consistency
    - Defensive: Comprehensive input validation and error handling

Research Foundation:
    - Median-based baselines: Robust to outlier contamination (Huber, 1981)
    - Trend analysis: Detects gradual consumption changes (Nagi et al., 2011)
    - Peer comparison: Leverages collective behavior (Glauner et al., 2017)
    - References:
        * Nagi et al. (2011): "Non-Technical Loss Detection for Metered Customers"
        * Glauner et al. (2017): "Large-scale Detection of Non-Technical Losses"
        * Huber (1981): "Robust Statistics" - Wiley Series in Probability

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import curve_fit


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

class TrendMethod(str, Enum):
    """Methods for computing consumption trends."""
    LINEAR_REGRESSION = "linear_regression"  # OLS linear fit (default)
    THEIL_SEN = "theil_sen"  # Robust Theil-Sen estimator
    SIMPLE_DIFFERENCE = "simple_difference"  # End - Start difference
    PERCENT_CHANGE = "percent_change"  # Percentage change over period


class BaselineStatistic(str, Enum):
    """Statistical measures for transformer baselines."""
    MEDIAN = "median"  # Robust central tendency (default)
    MEAN = "mean"  # Arithmetic mean
    TRIMMED_MEAN = "trimmed_mean"  # Mean after removing extreme values


class VarianceStatistic(str, Enum):
    """Measures of consumption variability."""
    VARIANCE = "variance"  # Sample variance (default)
    STD = "std"  # Standard deviation
    MAD = "mad"  # Median Absolute Deviation (most robust)
    IQR = "iqr"  # Interquartile Range


# Default configuration values
DEFAULT_TREND_WINDOW = 6  # Months for trend computation
DEFAULT_MIN_METERS_PER_TRANSFORMER = 3  # Minimum meters for reliable statistics
DEFAULT_TREND_METHOD = TrendMethod.LINEAR_REGRESSION
DEFAULT_BASELINE_STATISTIC = BaselineStatistic.MEDIAN
DEFAULT_VARIANCE_STATISTIC = VarianceStatistic.VARIANCE


# Feature name constants (for consistent naming across codebase)
FEATURE_TRANSFORMER_BASELINE_MEDIAN = 'transformer_baseline_median'
FEATURE_TRANSFORMER_BASELINE_VARIANCE = 'transformer_baseline_variance'
FEATURE_METER_CONSUMPTION_TREND = 'meter_consumption_trend'
FEATURE_CONSUMPTION_RATIO_TO_MEDIAN = 'consumption_ratio_to_transformer_median'


# ============================================================================
# CONFIGURATION AND VALIDATION
# ============================================================================

@dataclass
class FeatureConfig:
    """
    Type-safe configuration for feature engineering pipeline.
    
    This configuration object encapsulates all feature computation parameters
    with built-in validation to ensure correctness and prevent silent failures.
    
    Attributes:
        trend_window: Number of most recent months for trend computation
        trend_method: Method for computing consumption trends
        baseline_statistic: Statistic for transformer baseline
        variance_statistic: Measure of transformer consumption variability
        min_meters_per_transformer: Minimum meters required for transformer stats
        consumption_column_prefix: Prefix for consumption time series columns
        transformer_column: Column name containing transformer IDs
        meter_id_column: Column name containing meter IDs
        min_valid_months_for_trend: Minimum non-null months for trend calculation
        handle_zero_consumption: How to treat zero values (True=keep, False=treat as null)
        robust_estimators: Use robust statistical estimators (Theil-Sen, MAD)
        verbose: Enable detailed logging of feature computation
    
    Example:
        >>> config = FeatureConfig(
        ...     trend_window=12,
        ...     trend_method=TrendMethod.THEIL_SEN,
        ...     robust_estimators=True
        ... )
    """
    
    # Trend computation
    trend_window: int = DEFAULT_TREND_WINDOW
    trend_method: TrendMethod = DEFAULT_TREND_METHOD
    min_valid_months_for_trend: int = 3
    
    # Transformer baseline
    baseline_statistic: BaselineStatistic = DEFAULT_BASELINE_STATISTIC
    variance_statistic: VarianceStatistic = DEFAULT_VARIANCE_STATISTIC
    min_meters_per_transformer: int = DEFAULT_MIN_METERS_PER_TRANSFORMER
    
    # Data schema
    consumption_column_prefix: str = 'monthly_consumption_'
    transformer_column: str = 'transformer_id'
    meter_id_column: str = 'meter_id'
    
    # Quality controls
    handle_zero_consumption: bool = False  # Treat zeros as missing (inactive meters)
    robust_estimators: bool = True  # Prefer robust statistical methods
    
    # Observability
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Convert string to enum if needed
        if isinstance(self.trend_method, str):
            self.trend_method = TrendMethod(self.trend_method)
        if isinstance(self.baseline_statistic, str):
            self.baseline_statistic = BaselineStatistic(self.baseline_statistic)
        if isinstance(self.variance_statistic, str):
            self.variance_statistic = VarianceStatistic(self.variance_statistic)
        
        # Validate trend window
        if self.trend_window < 2:
            raise ValueError(
                f"trend_window must be at least 2 for trend computation, got {self.trend_window}"
            )
        
        if self.trend_window > 120:
            warnings.warn(
                f"trend_window={self.trend_window} is very large (>10 years). "
                f"Consider using shorter windows for responsive anomaly detection.",
                UserWarning
            )
        
        # Validate minimum valid months
        if self.min_valid_months_for_trend < 2:
            raise ValueError(
                f"min_valid_months_for_trend must be at least 2, got {self.min_valid_months_for_trend}"
            )
        
        if self.min_valid_months_for_trend > self.trend_window:
            raise ValueError(
                f"min_valid_months_for_trend ({self.min_valid_months_for_trend}) cannot exceed "
                f"trend_window ({self.trend_window})"
            )
        
        # Validate minimum meters
        if self.min_meters_per_transformer < 1:
            raise ValueError(
                f"min_meters_per_transformer must be positive, got {self.min_meters_per_transformer}"
            )
        
        if self.min_meters_per_transformer < 3:
            warnings.warn(
                f"min_meters_per_transformer={self.min_meters_per_transformer} is low. "
                f"Statistical baselines may be unreliable. Consider >= 3 meters.",
                UserWarning
            )
        
        # Auto-enable robust estimators if requested
        if self.robust_estimators:
            if self.trend_method == TrendMethod.LINEAR_REGRESSION:
                logger.info("Switching to Theil-Sen trend estimation (robust_estimators=True)")
                self.trend_method = TrendMethod.THEIL_SEN
            
            if self.variance_statistic == VarianceStatistic.VARIANCE:
                logger.info("Switching to MAD variance estimation (robust_estimators=True)")
                self.variance_statistic = VarianceStatistic.MAD


# ============================================================================
# TRANSFORMER BASELINE COMPUTATION
# ============================================================================

class TransformerBaselineComputer:
    """
    Computes statistical baselines for transformer groups.
    
    This class aggregates consumption patterns across all meters connected to
    each transformer, computing robust central tendency (median) and variability
    (variance) measures. These baselines serve as peer-group norms for identifying
    outlier meters.
    
    Design Rationale:
        - Median over mean: Resistant to extreme outliers from legitimate high consumers
        - Variance tracking: High variance indicates heterogeneous customer mix
        - Per-transformer grouping: Prevents comparing 50kVA residential vs 500kVA industrial
    
    Research Note:
        Peer-based baselines are fundamental to NTL detection (Glauner et al., 2017).
        We use all historical months (not just recent) to establish stable baselines
        resistant to short-term fluctuations.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize baseline computer with configuration.
        
        Args:
            config: FeatureConfig instance with computation parameters
        """
        self.config = config
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def compute_transformer_baselines(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Compute transformer-level baseline statistics.
        
        For each transformer, computes:
        1. Baseline median: Median consumption across all meters and months
        2. Baseline variance: Variability of consumption patterns
        
        These baselines capture the "normal" consumption profile for each
        transformer's customer mix, enabling detection of anomalous meters.
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names
            
        Returns:
            DataFrame with added transformer baseline columns
            
        Raises:
            ValueError: If transformer_column not in DataFrame
            
        Example:
            >>> computer = TransformerBaselineComputer(config)
            >>> df_features = computer.compute_transformer_baselines(df, consumption_cols)
            >>> print(df_features['transformer_baseline_median'].head())
        """
        if self.config.transformer_column not in df.columns:
            raise ValueError(
                f"Transformer column '{self.config.transformer_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        
        df_features = df.copy()
        transformer_stats = {}
        
        logger.info(
            f"Computing transformer baselines using {self.config.baseline_statistic.value} "
            f"and {self.config.variance_statistic.value}"
        )
        
        # Compute baselines for each transformer
        for transformer_id in df[self.config.transformer_column].unique():
            mask = df[self.config.transformer_column] == transformer_id
            n_meters = mask.sum()
            
            if n_meters < self.config.min_meters_per_transformer:
                logger.warning(
                    f"Transformer {transformer_id} has only {n_meters} meters "
                    f"(minimum: {self.config.min_meters_per_transformer}). "
                    f"Baseline may be unreliable."
                )
            
            # Extract consumption data for this transformer
            transformer_consumption = df.loc[mask, consumption_cols].values.flatten()
            
            # Remove NaN and optionally zeros
            valid_consumption = transformer_consumption[~np.isnan(transformer_consumption)]
            if not self.config.handle_zero_consumption:
                valid_consumption = valid_consumption[valid_consumption > 0]
            
            if len(valid_consumption) == 0:
                logger.warning(
                    f"Transformer {transformer_id} has no valid consumption data. "
                    f"Setting baselines to NaN."
                )
                baseline_median = np.nan
                baseline_variance = np.nan
            else:
                # Compute baseline statistic
                baseline_median = self._compute_baseline(
                    valid_consumption,
                    self.config.baseline_statistic
                )
                
                # Compute variance statistic
                baseline_variance = self._compute_variance(
                    valid_consumption,
                    self.config.variance_statistic
                )
            
            # Assign to all meters on this transformer
            df_features.loc[mask, FEATURE_TRANSFORMER_BASELINE_MEDIAN] = baseline_median
            df_features.loc[mask, FEATURE_TRANSFORMER_BASELINE_VARIANCE] = baseline_variance
            
            # Store for inspection
            transformer_stats[transformer_id] = {
                'n_meters': n_meters,
                'n_valid_values': len(valid_consumption),
                'baseline_median': baseline_median,
                'baseline_variance': baseline_variance
            }
        
        self.baseline_stats = transformer_stats
        
        logger.info(
            f"Computed baselines for {len(transformer_stats)} transformers "
            f"({len(df_features)} meters total)"
        )
        
        return df_features
    
    def _compute_baseline(
        self,
        data: NDArray[np.float64],
        statistic: BaselineStatistic
    ) -> float:
        """
        Compute baseline statistic (central tendency).
        
        Args:
            data: 1D array of consumption values
            statistic: Type of baseline statistic
            
        Returns:
            Baseline value
        """
        if statistic == BaselineStatistic.MEDIAN:
            return float(np.median(data))
        
        elif statistic == BaselineStatistic.MEAN:
            return float(np.mean(data))
        
        elif statistic == BaselineStatistic.TRIMMED_MEAN:
            # Remove top and bottom 10% before computing mean
            return float(stats.trim_mean(data, proportiontocut=0.1))
        
        else:
            return float(np.median(data))  # Fallback to median
    
    def _compute_variance(
        self,
        data: NDArray[np.float64],
        statistic: VarianceStatistic
    ) -> float:
        """
        Compute variance statistic (dispersion measure).
        
        Args:
            data: 1D array of consumption values
            statistic: Type of variance statistic
            
        Returns:
            Variance value
        """
        if statistic == VarianceStatistic.VARIANCE:
            return float(np.var(data, ddof=1))  # Sample variance
        
        elif statistic == VarianceStatistic.STD:
            return float(np.std(data, ddof=1))  # Sample standard deviation
        
        elif statistic == VarianceStatistic.MAD:
            # Median Absolute Deviation (most robust)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return float(mad * 1.4826)  # Scale to match std for normal distribution
        
        elif statistic == VarianceStatistic.IQR:
            q75, q25 = np.percentile(data, [75, 25])
            return float(q75 - q25)
        
        else:
            return float(np.var(data, ddof=1))  # Fallback to variance


# ============================================================================
# CONSUMPTION TREND ANALYSIS
# ============================================================================

class ConsumptionTrendAnalyzer:
    """
    Computes temporal consumption trends for anomaly detection.
    
    This class analyzes time-series consumption patterns to detect gradual changes
    that may indicate electricity theft. A declining trend (negative slope) over
    recent months is a strong indicator of tampering or illegal bypass.
    
    Methods:
        - Linear regression: OLS fit to time series (fast, assumes linear trend)
        - Theil-Sen: Robust median-based regression (resistant to outliers)
        - Simple difference: End value - start value (interpretable)
        - Percent change: Relative change over period (normalized)
    
    Design Rationale:
        We focus on recent months (default: 6) rather than entire history to
        capture evolving behavior. Thieves typically show gradual decline after
        tampering installation, not immediate drop.
    
    Research Note:
        Trend features are highly discriminative for NTL (Nagi et al., 2011).
        We prefer Theil-Sen over OLS for robustness to measurement errors.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize trend analyzer with configuration.
        
        Args:
            config: FeatureConfig instance with trend parameters
        """
        self.config = config
        self.trend_stats: Dict[str, Any] = {}
        
    def compute_consumption_trends(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Compute consumption trends over recent months.
        
        For each meter, computes the slope of consumption over the last N months
        (configured via trend_window). Negative slopes indicate declining consumption,
        which may signal electricity theft.
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names (temporally ordered)
            
        Returns:
            DataFrame with added meter_consumption_trend column
            
        Example:
            >>> analyzer = ConsumptionTrendAnalyzer(config)
            >>> df_features = analyzer.compute_consumption_trends(df, consumption_cols)
            >>> declining_meters = df_features[df_features['meter_consumption_trend'] < -10]
        """
        df_features = df.copy()
        
        # Select last N months for trend computation
        if len(consumption_cols) < self.config.trend_window:
            logger.warning(
                f"Only {len(consumption_cols)} months available, but trend_window={self.config.trend_window}. "
                f"Using all available months."
            )
            trend_cols = consumption_cols
        else:
            trend_cols = consumption_cols[-self.config.trend_window:]
        
        logger.info(
            f"Computing consumption trends using {self.config.trend_method.value} "
            f"over {len(trend_cols)} months"
        )
        
        # Compute trend for each meter
        trends = []
        valid_trends = 0
        
        for idx, row in df_features.iterrows():
            consumption_series = row[trend_cols].values.astype(np.float64)
            
            # Check for sufficient valid data
            valid_mask = ~np.isnan(consumption_series)
            if not self.config.handle_zero_consumption:
                valid_mask &= (consumption_series > 0)
            
            n_valid = valid_mask.sum()
            
            if n_valid < self.config.min_valid_months_for_trend:
                trends.append(np.nan)
                continue
            
            # Compute trend
            trend = self._compute_trend(
                consumption_series[valid_mask],
                np.arange(len(consumption_series))[valid_mask],
                self.config.trend_method
            )
            
            trends.append(trend)
            if not np.isnan(trend):
                valid_trends += 1
        
        df_features[FEATURE_METER_CONSUMPTION_TREND] = trends
        
        logger.info(
            f"Computed trends for {valid_trends}/{len(df_features)} meters "
            f"({valid_trends/len(df_features):.1%} coverage)"
        )
        
        # Store statistics
        valid_trend_values = np.array([t for t in trends if not np.isnan(t)])
        if len(valid_trend_values) > 0:
            self.trend_stats = {
                'n_valid': valid_trends,
                'mean_trend': float(np.mean(valid_trend_values)),
                'median_trend': float(np.median(valid_trend_values)),
                'std_trend': float(np.std(valid_trend_values)),
                'pct_declining': float((valid_trend_values < 0).sum() / len(valid_trend_values))
            }
        
        return df_features
    
    def _compute_trend(
        self,
        y: NDArray[np.float64],
        x: NDArray[np.int_],
        method: TrendMethod
    ) -> float:
        """
        Compute trend slope using specified method.
        
        Args:
            y: Consumption values (dependent variable)
            x: Time indices (independent variable)
            method: Trend computation method
            
        Returns:
            Trend slope (units: consumption_units/month)
        """
        if len(y) < 2:
            return np.nan
        
        try:
            if method == TrendMethod.LINEAR_REGRESSION:
                # Ordinary Least Squares regression
                slope, _ = np.polyfit(x, y, deg=1)
                return float(slope)
            
            elif method == TrendMethod.THEIL_SEN:
                # Robust Theil-Sen estimator (median of all pairwise slopes)
                result = stats.theilslopes(y, x)
                return float(result.slope)
            
            elif method == TrendMethod.SIMPLE_DIFFERENCE:
                # Simple end - start difference
                return float(y[-1] - y[0])
            
            elif method == TrendMethod.PERCENT_CHANGE:
                # Percentage change (avoid division by zero)
                if y[0] == 0:
                    return np.nan
                return float((y[-1] - y[0]) / y[0] * 100)
            
            else:
                # Fallback to linear regression
                slope, _ = np.polyfit(x, y, deg=1)
                return float(slope)
                
        except Exception as e:
            logger.debug(f"Trend computation failed: {str(e)}")
            return np.nan


# ============================================================================
# RELATIVE CONSUMPTION FEATURES
# ============================================================================

class RelativeConsumptionComputer:
    """
    Computes consumption features relative to transformer baselines.
    
    This class creates comparative features that measure how each meter's consumption
    deviates from its peer group (other meters on the same transformer). Meters with
    consistently low ratios may be bypassing the meter or tampering.
    
    Features:
        - consumption_ratio_to_transformer_median: Individual / Group Median
        - Values << 1.0 indicate abnormally low consumption
        - Values >> 1.0 indicate abnormally high consumption
    
    Design Rationale:
        Absolute consumption alone is insufficient (industrial vs residential).
        Relative comparison to peers on same infrastructure provides normalized
        anomaly signal independent of customer class.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize relative consumption computer.
        
        Args:
            config: FeatureConfig instance
        """
        self.config = config
        self.ratio_stats: Dict[str, float] = {}
        
    def compute_consumption_ratios(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Compute consumption ratio to transformer median baseline.
        
        For each meter, computes the ratio of its mean consumption to the
        transformer baseline median. This ratio is normalized such that:
        - ratio ≈ 1.0: Meter consumes near transformer median (normal)
        - ratio << 1.0: Meter consumes much less than peers (suspicious)
        - ratio >> 1.0: Meter consumes much more than peers (high consumer)
        
        Args:
            df: DataFrame with meter consumption data (must have transformer baselines)
            consumption_cols: List of consumption column names
            
        Returns:
            DataFrame with added consumption_ratio_to_transformer_median column
            
        Raises:
            ValueError: If transformer baseline columns missing
            
        Example:
            >>> computer = RelativeConsumptionComputer(config)
            >>> df_features = computer.compute_consumption_ratios(df, consumption_cols)
            >>> suspicious = df_features[df_features['consumption_ratio_to_transformer_median'] < 0.3]
        """
        # Validate baseline columns exist
        if FEATURE_TRANSFORMER_BASELINE_MEDIAN not in df.columns:
            raise ValueError(
                f"Transformer baseline not computed. Run compute_transformer_baselines() first."
            )
        
        df_features = df.copy()
        
        logger.info("Computing consumption ratios relative to transformer baselines")
        
        # Compute mean consumption for each meter
        meter_mean_consumption = df_features[consumption_cols].mean(axis=1, skipna=True)
        
        # Compute ratio to transformer baseline
        baseline_median = df_features[FEATURE_TRANSFORMER_BASELINE_MEDIAN]
        
        # Avoid division by zero
        consumption_ratio = np.where(
            baseline_median > 0,
            meter_mean_consumption / baseline_median,
            np.nan
        )
        
        df_features[FEATURE_CONSUMPTION_RATIO_TO_MEDIAN] = consumption_ratio
        
        # Compute statistics
        valid_ratios = consumption_ratio[~np.isnan(consumption_ratio)]
        if len(valid_ratios) > 0:
            self.ratio_stats = {
                'n_valid': len(valid_ratios),
                'mean_ratio': float(np.mean(valid_ratios)),
                'median_ratio': float(np.median(valid_ratios)),
                'std_ratio': float(np.std(valid_ratios)),
                'pct_low_consumers': float((valid_ratios < 0.5).sum() / len(valid_ratios)),
                'pct_high_consumers': float((valid_ratios > 2.0).sum() / len(valid_ratios))
            }
            
            logger.info(
                f"Computed consumption ratios: "
                f"{self.ratio_stats['pct_low_consumers']:.1%} low consumers, "
                f"{self.ratio_stats['pct_high_consumers']:.1%} high consumers"
            )
        
        return df_features


# ============================================================================
# MAIN FEATURE ENGINEER FACADE
# ============================================================================

@dataclass
class FeatureResult:
    """
    Container for feature engineering pipeline results.
    
    Attributes:
        data: DataFrame with original data + engineered features
        feature_names: List of newly created feature column names
        transformer_baseline_stats: Statistics from baseline computation
        trend_stats: Statistics from trend analysis
        ratio_stats: Statistics from relative consumption computation
        metadata: Additional metadata (timing, coverage, warnings)
    """
    data: pd.DataFrame
    feature_names: List[str]
    transformer_baseline_stats: Dict[str, Any]
    trend_stats: Dict[str, Any]
    ratio_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"FeatureResult(\n"
            f"  rows: {len(self.data)},\n"
            f"  features: {self.feature_names},\n"
            f"  transformers: {len(self.transformer_baseline_stats)},\n"
            f"  trend_coverage: {self.trend_stats.get('n_valid', 0)}/{len(self.data)}\n"
            f")"
        )


class FeatureEngineer:
    """
    Production-grade feature engineering pipeline for GhostLoad Mapper.
    
    This is the main facade class that orchestrates the complete feature engineering
    workflow for electricity theft detection:
    
    1. Transformer Baseline Computation:
       - transformer_baseline_median: Robust central tendency per transformer
       - transformer_baseline_variance: Consumption variability measure
    
    2. Temporal Trend Analysis:
       - meter_consumption_trend: Slope of consumption over recent months
       - Detects gradual decline patterns characteristic of theft
    
    3. Relative Consumption Features:
       - consumption_ratio_to_transformer_median: Individual vs peer comparison
       - Normalizes for different customer classes and transformer capacities
    
    The pipeline is designed for anomaly detection in electricity distribution,
    with specific attention to:
    - Domain expertise: Features interpretable by utility engineers
    - Statistical robustness: Median-based estimators resistant to outliers
    - Transformer-awareness: Group-based features prevent cross-contamination
    - Temporal sensitivity: Capture evolving consumption patterns
    - Explainability: Direct mapping to theft indicators
    
    Usage:
        >>> config = FeatureConfig(trend_window=12, robust_estimators=True)
        >>> engineer = FeatureEngineer(config)
        >>> result = engineer.engineer_features(meters_df, consumption_cols)
        >>> features_df = result.data
        >>> print(result.trend_stats)
    
    Design Pattern: Facade + Pipeline + Strategy
    Thread Safety: Not thread-safe (use separate instances per thread)
    
    Research Foundation:
        Features are based on published NTL detection research:
        - Peer comparison: Glauner et al. (2017)
        - Trend analysis: Nagi et al. (2011)
        - Robust statistics: Huber (1981)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: FeatureConfig instance (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        
        # Initialize pipeline components
        self.baseline_computer = TransformerBaselineComputer(self.config)
        self.trend_analyzer = ConsumptionTrendAnalyzer(self.config)
        self.ratio_computer = RelativeConsumptionComputer(self.config)
        
        logger.info(f"Initialized FeatureEngineer with config: {self.config}")
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        consumption_cols: Optional[List[str]] = None
    ) -> FeatureResult:
        """
        Execute complete feature engineering pipeline.
        
        This method runs all feature computation steps in sequence:
        1. Validate input data
        2. Compute transformer baseline statistics
        3. Analyze consumption trends
        4. Calculate relative consumption ratios
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names (auto-detected if None)
            
        Returns:
            FeatureResult containing data with new features and statistics
            
        Raises:
            ValueError: If required columns missing or data invalid
            
        Example:
            >>> engineer = FeatureEngineer()
            >>> result = engineer.engineer_features(meters_df)
            >>> print(result.feature_names)
            >>> suspicious = result.data[result.data['consumption_ratio_to_transformer_median'] < 0.3]
        """
        logger.info("="*80)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Auto-detect consumption columns if not provided
        if consumption_cols is None:
            consumption_cols = [
                col for col in df.columns 
                if col.startswith(self.config.consumption_column_prefix)
            ]
            if not consumption_cols:
                raise ValueError(
                    f"No consumption columns found with prefix "
                    f"'{self.config.consumption_column_prefix}'"
                )
            consumption_cols = sorted(consumption_cols)  # Ensure temporal order
        
        logger.info(
            f"Processing {len(df)} meters with {len(consumption_cols)} consumption months"
        )
        
        # Validate required columns
        if self.config.transformer_column not in df.columns:
            raise ValueError(
                f"Required column '{self.config.transformer_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        
        df_features = df.copy()
        
        # Stage 1: Transformer Baseline Computation
        logger.info("\n--- Stage 1: Transformer Baseline Computation ---")
        df_features = self.baseline_computer.compute_transformer_baselines(
            df_features, consumption_cols
        )
        
        # Stage 2: Consumption Trend Analysis
        logger.info("\n--- Stage 2: Consumption Trend Analysis ---")
        df_features = self.trend_analyzer.compute_consumption_trends(
            df_features, consumption_cols
        )
        
        # Stage 3: Relative Consumption Features
        logger.info("\n--- Stage 3: Relative Consumption Computation ---")
        df_features = self.ratio_computer.compute_consumption_ratios(
            df_features, consumption_cols
        )
        
        # Collect feature names
        feature_names = [
            FEATURE_TRANSFORMER_BASELINE_MEDIAN,
            FEATURE_TRANSFORMER_BASELINE_VARIANCE,
            FEATURE_METER_CONSUMPTION_TREND,
            FEATURE_CONSUMPTION_RATIO_TO_MEDIAN
        ]
        
        # Validate feature completeness
        feature_coverage = {}
        for feature in feature_names:
            if feature in df_features.columns:
                valid_count = df_features[feature].notna().sum()
                coverage = valid_count / len(df_features)
                feature_coverage[feature] = {
                    'valid_count': int(valid_count),
                    'coverage': float(coverage)
                }
                
                if coverage < 0.8:
                    logger.warning(
                        f"Feature '{feature}' has low coverage: {coverage:.1%} "
                        f"({valid_count}/{len(df_features)} meters)"
                    )
        
        # Compute metadata
        elapsed_time = time.time() - start_time
        metadata = {
            'processing_time_seconds': elapsed_time,
            'n_meters': len(df),
            'n_months': len(consumption_cols),
            'consumption_columns': consumption_cols,
            'feature_coverage': feature_coverage,
            'n_transformers': df[self.config.transformer_column].nunique(),
            'config': {
                'trend_window': self.config.trend_window,
                'trend_method': self.config.trend_method.value,
                'baseline_statistic': self.config.baseline_statistic.value,
                'variance_statistic': self.config.variance_statistic.value,
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
        logger.info(f"  Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"  Meters processed: {len(df):,}")
        logger.info(f"  Features created: {len(feature_names)}")
        logger.info(f"  Transformers analyzed: {metadata['n_transformers']}")
        logger.info("="*80 + "\n")
        
        return FeatureResult(
            data=df_features,
            feature_names=feature_names,
            transformer_baseline_stats=self.baseline_computer.baseline_stats,
            trend_stats=self.trend_analyzer.trend_stats,
            ratio_stats=self.ratio_computer.ratio_stats,
            metadata=metadata
        )
    
    def get_feature_summary(self, result: FeatureResult) -> str:
        """
        Generate human-readable summary of feature engineering results.
        
        Args:
            result: FeatureResult from engineer_features() method
            
        Returns:
            Formatted string summary
        """
        lines = [
            "="*80,
            "FEATURE ENGINEERING SUMMARY",
            "="*80,
            "",
            f"Dataset: {result.metadata['n_meters']:,} meters × {result.metadata['n_months']} months",
            f"Processing time: {result.metadata['processing_time_seconds']:.2f}s",
            f"Transformers analyzed: {result.metadata['n_transformers']}",
            "",
            "--- Features Created ---"
        ]
        
        for feature in result.feature_names:
            coverage_info = result.metadata['feature_coverage'].get(feature, {})
            valid = coverage_info.get('valid_count', 0)
            pct = coverage_info.get('coverage', 0)
            lines.append(f"  {feature}: {valid:,} values ({pct:.1%} coverage)")
        
        lines.extend([
            "",
            "--- Transformer Baseline Statistics ---",
            f"  Transformers processed: {len(result.transformer_baseline_stats)}",
            f"  Baseline method: {self.config.baseline_statistic.value}",
            f"  Variance method: {self.config.variance_statistic.value}",
        ])
        
        if result.trend_stats:
            lines.extend([
                "",
                "--- Consumption Trend Statistics ---",
                f"  Valid trends: {result.trend_stats.get('n_valid', 0):,}",
                f"  Mean trend: {result.trend_stats.get('mean_trend', 0):.2f} units/month",
                f"  Declining meters: {result.trend_stats.get('pct_declining', 0):.1%}",
            ])
        
        if result.ratio_stats:
            lines.extend([
                "",
                "--- Relative Consumption Statistics ---",
                f"  Valid ratios: {result.ratio_stats.get('n_valid', 0):,}",
                f"  Mean ratio: {result.ratio_stats.get('mean_ratio', 0):.2f}",
                f"  Low consumers (< 0.5): {result.ratio_stats.get('pct_low_consumers', 0):.1%}",
                f"  High consumers (> 2.0): {result.ratio_stats.get('pct_high_consumers', 0):.1%}",
            ])
        
        lines.append("="*80)
        
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def engineer_consumption_features(
    df: pd.DataFrame,
    consumption_cols: Optional[List[str]] = None,
    trend_window: int = 6,
    trend_method: str = "linear_regression",
    transformer_column: str = "transformer_id",
    robust_estimators: bool = True
) -> FeatureResult:
    """
    Convenience function for quick feature engineering with default settings.
    
    Args:
        df: DataFrame with meter consumption data
        consumption_cols: List of consumption column names (auto-detected if None)
        trend_window: Number of recent months for trend computation
        trend_method: Method for computing trends
        transformer_column: Column name for transformer grouping
        robust_estimators: Use robust statistical methods
        
    Returns:
        FeatureResult with engineered features and statistics
        
    Example:
        >>> from feature_engineer import engineer_consumption_features
        >>> result = engineer_consumption_features(
        ...     meters_df,
        ...     trend_window=12,
        ...     robust_estimators=True
        ... )
        >>> features_df = result.data
    """
    config = FeatureConfig(
        trend_window=trend_window,
        trend_method=TrendMethod(trend_method),
        transformer_column=transformer_column,
        robust_estimators=robust_estimators
    )
    
    engineer = FeatureEngineer(config)
    return engineer.engineer_features(df, consumption_cols)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of feature engineering capabilities.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEER - SELF-TEST")
    print("="*80 + "\n")
    
    # Create synthetic test data
    np.random.seed(42)
    n_meters = 150
    n_months = 12
    n_transformers = 10
    
    # Generate sample data with realistic patterns
    test_data = {
        'meter_id': [f'MTR_{i:04d}' for i in range(n_meters)],
        'transformer_id': [f'TX_{i%n_transformers:02d}' for i in range(n_meters)],
    }
    
    # Generate consumption with various patterns
    for month in range(n_months):
        consumption = []
        
        for i in range(n_meters):
            # Base consumption varies by transformer (simulate different capacity)
            transformer_idx = i % n_transformers
            base = 100 + transformer_idx * 50
            
            # Normal meters: stable consumption with noise
            if i < n_meters * 0.7:
                value = base + np.random.normal(0, 10)
            
            # Declining meters (potential theft): decreasing trend
            elif i < n_meters * 0.85:
                decline_rate = 5  # kWh per month
                value = base - (month * decline_rate) + np.random.normal(0, 5)
            
            # High consumers: legitimate high usage
            else:
                value = base * 2 + np.random.normal(0, 20)
            
            consumption.append(max(0, value))  # Ensure non-negative
        
        test_data[f'monthly_consumption_2024{month+1:02d}'] = consumption
    
    test_df = pd.DataFrame(test_data)
    
    print(f"Created test dataset: {len(test_df)} meters × {n_months} months")
    print(f"Transformers: {test_df['transformer_id'].nunique()}")
    print()
    
    # Test feature engineering
    print("Testing feature engineering pipeline...")
    print("-" * 80)
    
    try:
        engineer = FeatureEngineer(
            FeatureConfig(
                trend_window=6,
                trend_method=TrendMethod.LINEAR_REGRESSION,
                baseline_statistic=BaselineStatistic.MEDIAN,
                variance_statistic=VarianceStatistic.VARIANCE,
                robust_estimators=False,
                verbose=True
            )
        )
        
        consumption_cols = [col for col in test_df.columns if col.startswith('monthly_consumption_')]
        result = engineer.engineer_features(test_df, consumption_cols)
        
        print("\n✓ Feature engineering successful!")
        print("\nSummary:")
        print(engineer.get_feature_summary(result))
        
        # Validate results
        assert len(result.feature_names) == 4, f"Expected 4 features, got {len(result.feature_names)}"
        
        for feature in result.feature_names:
            assert feature in result.data.columns, f"Feature {feature} not in output"
        
        # Check transformer baselines computed
        baseline_median = result.data[FEATURE_TRANSFORMER_BASELINE_MEDIAN]
        assert baseline_median.notna().all(), "Transformer baseline has NaN values"
        
        # Check trends computed
        trends = result.data[FEATURE_METER_CONSUMPTION_TREND]
        assert trends.notna().sum() > 0, "No valid trends computed"
        
        # Check consumption ratios
        ratios = result.data[FEATURE_CONSUMPTION_RATIO_TO_MEDIAN]
        assert ratios.notna().sum() > 0, "No valid consumption ratios"
        assert (ratios[ratios.notna()] > 0).all(), "Negative consumption ratios detected"
        
        # Identify suspicious meters
        suspicious_threshold = 0.5
        suspicious_meters = result.data[
            (result.data[FEATURE_CONSUMPTION_RATIO_TO_MEDIAN] < suspicious_threshold) &
            (result.data[FEATURE_METER_CONSUMPTION_TREND] < -5)
        ]
        
        print(f"\n✓ All validation checks passed!")
        print(f"\nSuspicious meters detected: {len(suspicious_meters)} "
              f"({len(suspicious_meters)/len(test_df):.1%})")
        print(f"  Criteria: ratio < {suspicious_threshold} AND trend < -5 kWh/month")
        
        if len(suspicious_meters) > 0:
            print("\nTop 5 most suspicious meters:")
            display_cols = ['meter_id', 'transformer_id', 
                          FEATURE_CONSUMPTION_RATIO_TO_MEDIAN,
                          FEATURE_METER_CONSUMPTION_TREND]
            print(suspicious_meters[display_cols].head().to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Error during feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")
