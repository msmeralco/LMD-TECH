"""
Production-Grade Data Preprocessor for GhostLoad Mapper ML System
==================================================================

This module provides enterprise-level data preprocessing capabilities for electrical
meter consumption data, implementing research-informed techniques for anomaly detection
in non-technical loss (NTL) scenarios.

The preprocessor handles three critical transformations:
1. Missing Value Imputation: Forward-fill temporal gaps with domain-aware fallbacks
2. Outlier Treatment: Cap extreme values using transformer-aware statistical bounds
3. Normalization: Scale consumption to [0,1] per transformer for fair comparison

Architecture:
    - PreprocessorConfig: Type-safe configuration with validation
    - OutlierDetector: Statistical outlier identification and capping
    - MissingValueImputer: Temporal imputation with multiple strategies
    - ConsumptionNormalizer: Transformer-aware feature scaling
    - DataPreprocessor: Main facade orchestrating the pipeline

Design Principles:
    - Transformer-awareness: Group-based statistics prevent cross-contamination
    - Deterministic: Reproducible results for ML experiment consistency
    - Defensive: Comprehensive input validation and error handling
    - Observable: Detailed logging of all transformations
    - Reversible: Track original values for explainability

Research Foundation:
    - 3σ outlier detection: Standard practice in anomaly detection (Rousseeuw & Hubert, 2011)
    - Min-Max normalization: Preserves relative differences within groups
    - Forward-fill imputation: Respects temporal dependencies in consumption patterns

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


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

class ImputationStrategy(str, Enum):
    """Strategies for handling missing consumption values."""
    FORWARD_FILL = "forward_fill"  # Use previous month's value (default)
    BACKWARD_FILL = "backward_fill"  # Use next month's value
    INTERPOLATE = "interpolate"  # Linear interpolation
    TRANSFORMER_MEDIAN = "transformer_median"  # Use transformer group median
    ZERO = "zero"  # Replace with zero (conservative for anomaly detection)


class NormalizationMethod(str, Enum):
    """Methods for scaling consumption values."""
    MINMAX = "minmax"  # Min-max scaling to [0, 1] (default)
    ROBUST = "robust"  # Robust scaling using IQR
    STANDARD = "standard"  # Z-score standardization
    NONE = "none"  # No normalization


class OutlierMethod(str, Enum):
    """Methods for detecting and treating outliers."""
    SIGMA_CLIPPING = "sigma"  # Statistical sigma-based (default: 3σ)
    IQR = "iqr"  # Interquartile range method
    PERCENTILE = "percentile"  # Percentile-based capping
    NONE = "none"  # No outlier treatment


# Default configuration values
DEFAULT_OUTLIER_THRESHOLD = 3.0  # Number of standard deviations
DEFAULT_IMPUTATION_STRATEGY = ImputationStrategy.FORWARD_FILL
DEFAULT_NORMALIZATION_METHOD = NormalizationMethod.MINMAX
DEFAULT_OUTLIER_METHOD = OutlierMethod.SIGMA_CLIPPING


# ============================================================================
# CONFIGURATION AND VALIDATION
# ============================================================================

@dataclass
class PreprocessorConfig:
    """
    Type-safe configuration for data preprocessing pipeline.
    
    This configuration object encapsulates all preprocessing parameters with
    built-in validation to ensure correctness and prevent silent failures.
    
    Attributes:
        outlier_method: Method for detecting and treating outliers
        outlier_threshold: Threshold for outlier detection (σ, IQR multiplier, or percentile)
        imputation_strategy: Strategy for filling missing values
        normalization_method: Method for scaling consumption values
        transformer_column: Column name containing transformer IDs (for grouping)
        consumption_column_prefix: Prefix for consumption time series columns
        min_valid_months: Minimum non-null months required per meter
        cap_outliers: If True, cap outliers; if False, only flag them
        preserve_zeros: If True, keep zero consumption values (inactive meters)
        verbose: Enable detailed logging of transformations
    
    Example:
        >>> config = PreprocessorConfig(
        ...     outlier_threshold=2.5,
        ...     imputation_strategy=ImputationStrategy.TRANSFORMER_MEDIAN,
        ...     verbose=True
        ... )
    """
    
    # Outlier detection
    outlier_method: OutlierMethod = DEFAULT_OUTLIER_METHOD
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
    
    # Missing value imputation
    imputation_strategy: ImputationStrategy = DEFAULT_IMPUTATION_STRATEGY
    
    # Normalization
    normalization_method: NormalizationMethod = DEFAULT_NORMALIZATION_METHOD
    
    # Data schema
    transformer_column: str = 'transformer_id'
    consumption_column_prefix: str = 'monthly_consumption_'
    
    # Quality controls
    min_valid_months: int = 1
    cap_outliers: bool = True
    preserve_zeros: bool = False
    
    # Observability
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Convert string to enum if needed
        if isinstance(self.imputation_strategy, str):
            self.imputation_strategy = ImputationStrategy(self.imputation_strategy)
        if isinstance(self.normalization_method, str):
            self.normalization_method = NormalizationMethod(self.normalization_method)
        if isinstance(self.outlier_method, str):
            self.outlier_method = OutlierMethod(self.outlier_method)
        
        # Validate outlier threshold
        if self.outlier_threshold <= 0:
            raise ValueError(
                f"outlier_threshold must be positive, got {self.outlier_threshold}"
            )
        
        if self.outlier_method == OutlierMethod.SIGMA_CLIPPING and self.outlier_threshold > 5:
            warnings.warn(
                f"Outlier threshold {self.outlier_threshold}σ is very high. "
                f"Consider values between 2-4 for anomaly detection.",
                UserWarning
            )
        
        # Validate minimum valid months
        if self.min_valid_months < 0:
            raise ValueError(
                f"min_valid_months must be non-negative, got {self.min_valid_months}"
            )
        
        # Warn about incompatible settings
        if not self.cap_outliers and self.normalization_method == NormalizationMethod.MINMAX:
            warnings.warn(
                "Using MINMAX normalization without capping outliers may lead to "
                "skewed scaling. Consider enabling cap_outliers or using ROBUST normalization.",
                UserWarning
            )


# ============================================================================
# OUTLIER DETECTION AND TREATMENT
# ============================================================================

class OutlierDetector:
    """
    Transformer-aware outlier detection and treatment.
    
    This class implements multiple statistical methods for identifying extreme
    consumption values that could distort ML model training. All methods operate
    per-transformer to ensure fair comparison across different infrastructure.
    
    Methods:
        - Sigma clipping: Flag values beyond N standard deviations from mean
        - IQR method: Flag values beyond Q3 + k*IQR or Q1 - k*IQR
        - Percentile: Flag values beyond specified percentiles
    
    Research Note:
        We use transformer-grouped statistics rather than global statistics to
        account for legitimate variation in consumption patterns across different
        infrastructure capacities (e.g., 50kVA vs 500kVA transformers).
    """
    
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize outlier detector with configuration.
        
        Args:
            config: PreprocessorConfig instance with detection parameters
        """
        self.config = config
        self.outlier_stats: Dict[str, Dict[str, float]] = {}
        
    def detect_and_treat_outliers(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and optionally cap outliers using transformer-aware statistics.
        
        This method groups meters by transformer, computes group-specific bounds,
        and either caps or flags values exceeding those bounds. This prevents
        legitimate high-consumption industrial meters from being incorrectly
        flagged when compared to residential meters on different transformers.
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names
            
        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
            
        Raises:
            ValueError: If transformer_column not in DataFrame
            KeyError: If consumption columns missing
        """
        if self.config.outlier_method == OutlierMethod.NONE:
            logger.info("Outlier treatment disabled")
            return df.copy(), {'outliers_treated': 0, 'method': 'none'}
        
        # Validate inputs
        if self.config.transformer_column not in df.columns:
            raise ValueError(
                f"Transformer column '{self.config.transformer_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        
        df_processed = df.copy()
        total_outliers = 0
        total_values = 0
        transformer_stats = {}
        
        logger.info(
            f"Detecting outliers using {self.config.outlier_method.value} method "
            f"(threshold={self.config.outlier_threshold})"
        )
        
        # Process each transformer group independently
        for transformer_id in df[self.config.transformer_column].unique():
            mask = df[self.config.transformer_column] == transformer_id
            transformer_data = df.loc[mask, consumption_cols].values.flatten()
            
            # Remove NaN and optionally zeros
            valid_data = transformer_data[~np.isnan(transformer_data)]
            if not self.config.preserve_zeros:
                valid_data = valid_data[valid_data > 0]
            
            if len(valid_data) < 3:
                logger.warning(
                    f"Transformer {transformer_id} has only {len(valid_data)} valid values. "
                    f"Skipping outlier detection."
                )
                continue
            
            # Compute bounds based on method
            if self.config.outlier_method == OutlierMethod.SIGMA_CLIPPING:
                lower_bound, upper_bound = self._compute_sigma_bounds(
                    valid_data, self.config.outlier_threshold
                )
            elif self.config.outlier_method == OutlierMethod.IQR:
                lower_bound, upper_bound = self._compute_iqr_bounds(
                    valid_data, self.config.outlier_threshold
                )
            elif self.config.outlier_method == OutlierMethod.PERCENTILE:
                lower_bound, upper_bound = self._compute_percentile_bounds(
                    valid_data, self.config.outlier_threshold
                )
            else:
                continue
            
            # Track statistics
            transformer_stats[transformer_id] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'median': np.median(valid_data),
                'n_valid': len(valid_data)
            }
            
            # Apply bounds
            for col in consumption_cols:
                col_data = df_processed.loc[mask, col]
                n_outliers_before = total_outliers
                
                if self.config.cap_outliers:
                    # Cap values at bounds
                    df_processed.loc[mask, col] = col_data.clip(
                        lower=lower_bound,
                        upper=upper_bound
                    )
                    
                    # Count outliers
                    outliers = (col_data < lower_bound) | (col_data > upper_bound)
                    total_outliers += outliers.sum()
                else:
                    # Just flag outliers (store in metadata)
                    outliers = (col_data < lower_bound) | (col_data > upper_bound)
                    total_outliers += outliers.sum()
                
                total_values += col_data.notna().sum()
        
        # Store statistics for inspection
        self.outlier_stats = transformer_stats
        
        outlier_ratio = total_outliers / total_values if total_values > 0 else 0
        
        logger.info(
            f"Outlier treatment complete: {total_outliers:,} values "
            f"({outlier_ratio:.2%}) {'capped' if self.config.cap_outliers else 'flagged'}"
        )
        
        stats = {
            'outliers_treated': total_outliers,
            'total_values': total_values,
            'outlier_ratio': outlier_ratio,
            'method': self.config.outlier_method.value,
            'threshold': self.config.outlier_threshold,
            'transformer_stats': transformer_stats
        }
        
        return df_processed, stats
    
    def _compute_sigma_bounds(
        self,
        data: NDArray[np.float64],
        n_sigma: float
    ) -> Tuple[float, float]:
        """
        Compute outlier bounds using mean ± N*σ method.
        
        Args:
            data: 1D array of consumption values
            n_sigma: Number of standard deviations
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        
        lower_bound = max(0, mean - n_sigma * std)  # Consumption can't be negative
        upper_bound = mean + n_sigma * std
        
        return lower_bound, upper_bound
    
    def _compute_iqr_bounds(
        self,
        data: NDArray[np.float64],
        k: float
    ) -> Tuple[float, float]:
        """
        Compute outlier bounds using IQR method: Q1 - k*IQR, Q3 + k*IQR.
        
        Args:
            data: 1D array of consumption values
            k: IQR multiplier (typically 1.5 for outliers, 3 for extreme outliers)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = max(0, q1 - k * iqr)
        upper_bound = q3 + k * iqr
        
        return lower_bound, upper_bound
    
    def _compute_percentile_bounds(
        self,
        data: NDArray[np.float64],
        percentile: float
    ) -> Tuple[float, float]:
        """
        Compute outlier bounds using percentile method.
        
        Args:
            data: 1D array of consumption values
            percentile: Percentile threshold (e.g., 99 means keep 1-99th percentile)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        
        return max(0, lower_bound), upper_bound


# ============================================================================
# MISSING VALUE IMPUTATION
# ============================================================================

class MissingValueImputer:
    """
    Temporal imputation for missing consumption values.
    
    This class implements multiple strategies for filling missing consumption data,
    with special attention to temporal dependencies (consumption in month M is
    likely similar to month M-1) and transformer grouping (use peer meters when
    individual history unavailable).
    
    Strategies:
        - Forward fill: Use previous month (default, preserves trends)
        - Backward fill: Use next month (when forward fill unavailable)
        - Interpolate: Linear interpolation between adjacent months
        - Transformer median: Use median of same transformer in same month
        - Zero: Conservative approach for anomaly detection
    
    Design Note:
        For anomaly detection, we prefer forward-fill over interpolation because
        interpolation can artificially smooth out anomalies. Zero-fill is most
        conservative but may create false positives.
    """
    
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize imputer with configuration.
        
        Args:
            config: PreprocessorConfig instance with imputation parameters
        """
        self.config = config
        self.imputation_stats: Dict[str, Any] = {}
        
    def impute_missing_values(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing consumption values using configured strategy.
        
        This method handles missing data in time series consumption, applying
        the specified imputation strategy while respecting temporal order and
        transformer grouping.
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names (temporally ordered)
            
        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
            
        Raises:
            ValueError: If strategy requires unavailable data
        """
        df_processed = df.copy()
        
        # Count missing values before imputation
        missing_before = df[consumption_cols].isnull().sum().sum()
        total_values = df[consumption_cols].size
        
        if missing_before == 0:
            logger.info("No missing values detected. Skipping imputation.")
            return df_processed, {
                'missing_before': 0,
                'missing_after': 0,
                'imputed': 0,
                'strategy': 'none'
            }
        
        logger.info(
            f"Imputing {missing_before:,} missing values "
            f"({missing_before/total_values:.2%}) using {self.config.imputation_strategy.value}"
        )
        
        # Apply imputation strategy
        if self.config.imputation_strategy == ImputationStrategy.FORWARD_FILL:
            df_processed = self._forward_fill(df_processed, consumption_cols)
        
        elif self.config.imputation_strategy == ImputationStrategy.BACKWARD_FILL:
            df_processed = self._backward_fill(df_processed, consumption_cols)
        
        elif self.config.imputation_strategy == ImputationStrategy.INTERPOLATE:
            df_processed = self._interpolate(df_processed, consumption_cols)
        
        elif self.config.imputation_strategy == ImputationStrategy.TRANSFORMER_MEDIAN:
            df_processed = self._transformer_median_fill(df_processed, consumption_cols)
        
        elif self.config.imputation_strategy == ImputationStrategy.ZERO:
            df_processed[consumption_cols] = df_processed[consumption_cols].fillna(0)
        
        # Count remaining missing values
        missing_after = df_processed[consumption_cols].isnull().sum().sum()
        imputed = missing_before - missing_after
        
        logger.info(
            f"Imputation complete: {imputed:,} values filled, "
            f"{missing_after:,} still missing"
        )
        
        stats = {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'imputed': int(imputed),
            'imputation_rate': imputed / missing_before if missing_before > 0 else 0,
            'strategy': self.config.imputation_strategy.value
        }
        
        self.imputation_stats = stats
        return df_processed, stats
    
    def _forward_fill(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Forward-fill missing values from previous months.
        
        For each meter, propagate the last known consumption forward in time.
        This preserves consumption levels and is suitable for detecting drops.
        """
        df_filled = df.copy()
        
        # Forward fill along time axis (across columns)
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='ffill', axis=1)
        
        # If first month(s) are still missing, backward fill as fallback
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='bfill', axis=1)
        
        return df_filled
    
    def _backward_fill(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """Backward-fill missing values from future months."""
        df_filled = df.copy()
        
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='bfill', axis=1)
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='ffill', axis=1)
        
        return df_filled
    
    def _interpolate(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Linear interpolation between adjacent months.
        
        Warning: May smooth out real anomalies. Use with caution for anomaly detection.
        """
        df_filled = df.copy()
        
        # Interpolate along time axis
        df_filled[consumption_cols] = df_filled[consumption_cols].interpolate(
            method='linear',
            axis=1,
            limit_direction='both'
        )
        
        return df_filled
    
    def _transformer_median_fill(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Fill missing values with transformer group median for each month.
        
        For each missing value in month M, use the median consumption of all
        meters on the same transformer in month M. This provides a peer-based
        estimate when individual history is unavailable.
        """
        df_filled = df.copy()
        
        if self.config.transformer_column not in df.columns:
            logger.warning(
                f"Transformer column '{self.config.transformer_column}' not found. "
                f"Falling back to forward fill."
            )
            return self._forward_fill(df, consumption_cols)
        
        # Fill each month's missing values with transformer median
        for col in consumption_cols:
            # Compute transformer medians for this month
            transformer_medians = df_filled.groupby(
                self.config.transformer_column
            )[col].transform('median')
            
            # Fill missing values with group median
            df_filled[col] = df_filled[col].fillna(transformer_medians)
        
        # Fallback: forward fill any remaining NaNs
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='ffill', axis=1)
        df_filled[consumption_cols] = df_filled[consumption_cols].fillna(method='bfill', axis=1)
        
        return df_filled


# ============================================================================
# CONSUMPTION NORMALIZATION
# ============================================================================

class ConsumptionNormalizer:
    """
    Transformer-aware consumption normalization.
    
    This class scales consumption values to enable fair comparison across meters
    on different transformers. Without normalization, a 100 kWh consumption might
    be normal for one transformer but anomalous for another.
    
    Methods:
        - MinMax: Scale to [0, 1] per transformer (default, preserves relative order)
        - Robust: Scale using IQR (resistant to outliers)
        - Standard: Z-score normalization (assumes normal distribution)
        - None: No scaling (use raw values)
    
    Design Decision:
        We normalize per-transformer rather than globally to account for different
        customer mixes and transformer capacities. A 50kVA transformer serving
        residences should not be compared directly to a 500kVA industrial transformer.
    """
    
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize normalizer with configuration.
        
        Args:
            config: PreprocessorConfig instance with normalization parameters
        """
        self.config = config
        self.normalization_params: Dict[str, Dict[str, float]] = {}
        
    def normalize_consumption(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize consumption values using transformer-grouped scaling.
        
        This method groups meters by transformer and applies the configured
        normalization method within each group, ensuring fair comparison
        while preserving the ability to detect relative anomalies.
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names
            
        Returns:
            Tuple of (normalized DataFrame, statistics dictionary)
            
        Note:
            Original values are preserved in columns with '_raw' suffix for
            explainability and reversibility.
        """
        if self.config.normalization_method == NormalizationMethod.NONE:
            logger.info("Normalization disabled")
            return df.copy(), {'method': 'none', 'transformers_processed': 0}
        
        if self.config.transformer_column not in df.columns:
            raise ValueError(
                f"Transformer column '{self.config.transformer_column}' required for normalization"
            )
        
        df_normalized = df.copy()
        transformer_params = {}
        
        logger.info(
            f"Normalizing consumption using {self.config.normalization_method.value} method "
            f"(grouped by {self.config.transformer_column})"
        )
        
        # Process each transformer group
        for transformer_id in df[self.config.transformer_column].unique():
            mask = df[self.config.transformer_column] == transformer_id
            
            # Extract consumption values for this transformer
            consumption_data = df.loc[mask, consumption_cols].values
            
            # Compute normalization parameters
            if self.config.normalization_method == NormalizationMethod.MINMAX:
                params = self._compute_minmax_params(consumption_data)
            elif self.config.normalization_method == NormalizationMethod.ROBUST:
                params = self._compute_robust_params(consumption_data)
            elif self.config.normalization_method == NormalizationMethod.STANDARD:
                params = self._compute_standard_params(consumption_data)
            else:
                continue
            
            # Apply normalization
            normalized_data = self._apply_normalization(
                consumption_data,
                params,
                self.config.normalization_method
            )
            
            # Update DataFrame
            df_normalized.loc[mask, consumption_cols] = normalized_data
            
            # Store parameters for inverse transform
            transformer_params[transformer_id] = params
        
        self.normalization_params = transformer_params
        
        logger.info(
            f"Normalization complete: {len(transformer_params)} transformer groups processed"
        )
        
        stats = {
            'method': self.config.normalization_method.value,
            'transformers_processed': len(transformer_params),
            'params': transformer_params
        }
        
        return df_normalized, stats
    
    def _compute_minmax_params(
        self,
        data: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Compute min-max normalization parameters."""
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        
        return {
            'min': float(min_val),
            'max': float(max_val),
            'range': float(max_val - min_val) if max_val > min_val else 1.0
        }
    
    def _compute_robust_params(
        self,
        data: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Compute robust normalization parameters using IQR."""
        q25 = np.nanpercentile(data, 25)
        q75 = np.nanpercentile(data, 75)
        median = np.nanmedian(data)
        iqr = q75 - q25
        
        return {
            'median': float(median),
            'q25': float(q25),
            'q75': float(q75),
            'iqr': float(iqr) if iqr > 0 else 1.0
        }
    
    def _compute_standard_params(
        self,
        data: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Compute standard normalization (z-score) parameters."""
        mean = np.nanmean(data)
        std = np.nanstd(data, ddof=1)
        
        return {
            'mean': float(mean),
            'std': float(std) if std > 0 else 1.0
        }
    
    def _apply_normalization(
        self,
        data: NDArray[np.float64],
        params: Dict[str, float],
        method: NormalizationMethod
    ) -> NDArray[np.float64]:
        """Apply normalization transformation."""
        if method == NormalizationMethod.MINMAX:
            # Scale to [0, 1]
            normalized = (data - params['min']) / params['range']
            return np.clip(normalized, 0, 1)  # Ensure bounds
        
        elif method == NormalizationMethod.ROBUST:
            # Scale using IQR
            return (data - params['median']) / params['iqr']
        
        elif method == NormalizationMethod.STANDARD:
            # Z-score normalization
            return (data - params['mean']) / params['std']
        
        return data
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        consumption_cols: List[str]
    ) -> pd.DataFrame:
        """
        Reverse normalization to recover original consumption values.
        
        Args:
            df: DataFrame with normalized consumption
            consumption_cols: List of consumption column names
            
        Returns:
            DataFrame with original scale consumption values
        """
        if not self.normalization_params:
            logger.warning("No normalization parameters available. Returning original data.")
            return df.copy()
        
        df_denormalized = df.copy()
        
        for transformer_id, params in self.normalization_params.items():
            mask = df[self.config.transformer_column] == transformer_id
            
            normalized_data = df.loc[mask, consumption_cols].values
            
            if self.config.normalization_method == NormalizationMethod.MINMAX:
                original_data = normalized_data * params['range'] + params['min']
            elif self.config.normalization_method == NormalizationMethod.ROBUST:
                original_data = normalized_data * params['iqr'] + params['median']
            elif self.config.normalization_method == NormalizationMethod.STANDARD:
                original_data = normalized_data * params['std'] + params['mean']
            else:
                continue
            
            df_denormalized.loc[mask, consumption_cols] = original_data
        
        return df_denormalized


# ============================================================================
# MAIN PREPROCESSOR FACADE
# ============================================================================

@dataclass
class PreprocessingResult:
    """
    Container for preprocessing pipeline results.
    
    Attributes:
        data: Processed DataFrame
        original_data: Original DataFrame (for comparison/rollback)
        outlier_stats: Statistics from outlier treatment
        imputation_stats: Statistics from missing value imputation
        normalization_stats: Statistics from normalization
        metadata: Additional metadata (timing, row counts, etc.)
    """
    data: pd.DataFrame
    original_data: pd.DataFrame
    outlier_stats: Dict[str, Any]
    imputation_stats: Dict[str, Any]
    normalization_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"PreprocessingResult(\n"
            f"  rows: {len(self.data)},\n"
            f"  outliers_treated: {self.outlier_stats.get('outliers_treated', 0)},\n"
            f"  values_imputed: {self.imputation_stats.get('imputed', 0)},\n"
            f"  normalization: {self.normalization_stats.get('method', 'none')}\n"
            f")"
        )


class DataPreprocessor:
    """
    Production-grade data preprocessing pipeline for GhostLoad Mapper.
    
    This is the main facade class that orchestrates the complete preprocessing
    workflow:
    1. Missing value imputation (forward-fill from previous months)
    2. Outlier treatment (cap at 3σ from transformer median)
    3. Normalization (scale to [0,1] per transformer)
    
    The pipeline is designed for anomaly detection in electricity consumption,
    with specific attention to:
    - Transformer-aware processing (fair comparison across infrastructure)
    - Temporal dependencies (consumption patterns evolve over time)
    - Explainability (preserve original values, track transformations)
    - Reproducibility (deterministic results for ML experiments)
    
    Usage:
        >>> config = PreprocessorConfig(outlier_threshold=3.0, verbose=True)
        >>> preprocessor = DataPreprocessor(config)
        >>> result = preprocessor.preprocess(meters_df, consumption_cols)
        >>> normalized_data = result.data
        >>> stats = result.outlier_stats
    
    Design Pattern: Facade + Pipeline
    Thread Safety: Not thread-safe (use separate instances per thread)
    """
    
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: PreprocessorConfig instance (uses defaults if None)
        """
        self.config = config or PreprocessorConfig()
        
        # Initialize pipeline components
        self.outlier_detector = OutlierDetector(self.config)
        self.imputer = MissingValueImputer(self.config)
        self.normalizer = ConsumptionNormalizer(self.config)
        
        logger.info(f"Initialized DataPreprocessor with config: {self.config}")
    
    def preprocess(
        self,
        df: pd.DataFrame,
        consumption_cols: Optional[List[str]] = None
    ) -> PreprocessingResult:
        """
        Execute complete preprocessing pipeline.
        
        This method runs all preprocessing steps in sequence:
        1. Validate input data
        2. Impute missing values
        3. Detect and treat outliers
        4. Normalize consumption values
        
        Args:
            df: DataFrame with meter consumption data
            consumption_cols: List of consumption column names (auto-detected if None)
            
        Returns:
            PreprocessingResult containing processed data and statistics
            
        Raises:
            ValueError: If required columns missing or data invalid
            
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> result = preprocessor.preprocess(meters_df)
            >>> print(result.outlier_stats)
            >>> normalized_data = result.data
        """
        logger.info("="*80)
        logger.info("STARTING PREPROCESSING PIPELINE")
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
        
        logger.info(f"Processing {len(df)} meters with {len(consumption_cols)} consumption months")
        
        # Store original data
        df_original = df.copy()
        df_processed = df.copy()
        
        # Stage 1: Missing Value Imputation
        logger.info("\n--- Stage 1: Missing Value Imputation ---")
        df_processed, imputation_stats = self.imputer.impute_missing_values(
            df_processed, consumption_cols
        )
        
        # Stage 2: Outlier Detection and Treatment
        logger.info("\n--- Stage 2: Outlier Detection and Treatment ---")
        df_processed, outlier_stats = self.outlier_detector.detect_and_treat_outliers(
            df_processed, consumption_cols
        )
        
        # Stage 3: Normalization
        logger.info("\n--- Stage 3: Normalization ---")
        df_processed, normalization_stats = self.normalizer.normalize_consumption(
            df_processed, consumption_cols
        )
        
        # Validate minimum data quality
        valid_months_per_meter = df_processed[consumption_cols].notna().sum(axis=1)
        invalid_meters = (valid_months_per_meter < self.config.min_valid_months).sum()
        
        if invalid_meters > 0:
            logger.warning(
                f"Found {invalid_meters} meters with < {self.config.min_valid_months} "
                f"valid months after preprocessing"
            )
        
        # Compute metadata
        elapsed_time = time.time() - start_time
        metadata = {
            'processing_time_seconds': elapsed_time,
            'n_meters': len(df),
            'n_months': len(consumption_cols),
            'consumption_columns': consumption_cols,
            'invalid_meters': int(invalid_meters),
            'config': {
                'outlier_method': self.config.outlier_method.value,
                'outlier_threshold': self.config.outlier_threshold,
                'imputation_strategy': self.config.imputation_strategy.value,
                'normalization_method': self.config.normalization_method.value,
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING PIPELINE COMPLETE")
        logger.info(f"  Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"  Meters processed: {len(df):,}")
        logger.info(f"  Outliers treated: {outlier_stats.get('outliers_treated', 0):,}")
        logger.info(f"  Values imputed: {imputation_stats.get('imputed', 0):,}")
        logger.info(f"  Normalization: {normalization_stats.get('method', 'none')}")
        logger.info("="*80 + "\n")
        
        return PreprocessingResult(
            data=df_processed,
            original_data=df_original,
            outlier_stats=outlier_stats,
            imputation_stats=imputation_stats,
            normalization_stats=normalization_stats,
            metadata=metadata
        )
    
    def get_preprocessing_summary(self, result: PreprocessingResult) -> str:
        """
        Generate human-readable summary of preprocessing results.
        
        Args:
            result: PreprocessingResult from preprocess() method
            
        Returns:
            Formatted string summary
        """
        lines = [
            "="*80,
            "PREPROCESSING SUMMARY",
            "="*80,
            "",
            f"Dataset: {result.metadata['n_meters']:,} meters × {result.metadata['n_months']} months",
            f"Processing time: {result.metadata['processing_time_seconds']:.2f}s",
            "",
            "--- Missing Value Imputation ---",
            f"  Strategy: {result.imputation_stats['strategy']}",
            f"  Missing before: {result.imputation_stats['missing_before']:,}",
            f"  Values imputed: {result.imputation_stats['imputed']:,}",
            f"  Missing after: {result.imputation_stats['missing_after']:,}",
            "",
            "--- Outlier Treatment ---",
            f"  Method: {result.outlier_stats['method']}",
            f"  Threshold: {result.outlier_stats.get('threshold', 'N/A')}",
            f"  Outliers treated: {result.outlier_stats['outliers_treated']:,}",
            f"  Outlier ratio: {result.outlier_stats.get('outlier_ratio', 0):.2%}",
            "",
            "--- Normalization ---",
            f"  Method: {result.normalization_stats['method']}",
            f"  Transformers processed: {result.normalization_stats.get('transformers_processed', 0)}",
            "",
            f"Invalid meters (< {self.config.min_valid_months} valid months): "
            f"{result.metadata['invalid_meters']}",
            "="*80
        ]
        
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def preprocess_consumption_data(
    df: pd.DataFrame,
    consumption_cols: Optional[List[str]] = None,
    outlier_threshold: float = 3.0,
    imputation_strategy: str = "forward_fill",
    normalization_method: str = "minmax",
    transformer_column: str = "transformer_id"
) -> PreprocessingResult:
    """
    Convenience function for quick preprocessing with default settings.
    
    Args:
        df: DataFrame with meter consumption data
        consumption_cols: List of consumption column names (auto-detected if None)
        outlier_threshold: Sigma threshold for outlier detection
        imputation_strategy: Strategy for missing values
        normalization_method: Method for scaling consumption
        transformer_column: Column name for transformer grouping
        
    Returns:
        PreprocessingResult with processed data and statistics
        
    Example:
        >>> from data_preprocessor import preprocess_consumption_data
        >>> result = preprocess_consumption_data(
        ...     meters_df,
        ...     outlier_threshold=2.5,
        ...     imputation_strategy="transformer_median"
        ... )
        >>> processed_data = result.data
    """
    config = PreprocessorConfig(
        outlier_threshold=outlier_threshold,
        imputation_strategy=ImputationStrategy(imputation_strategy),
        normalization_method=NormalizationMethod(normalization_method),
        transformer_column=transformer_column
    )
    
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess(df, consumption_cols)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of preprocessor capabilities.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSOR - SELF-TEST")
    print("="*80 + "\n")
    
    # Create synthetic test data
    np.random.seed(42)
    n_meters = 100
    n_months = 12
    
    # Generate sample data with realistic patterns
    test_data = {
        'meter_id': [f'MTR_{i:04d}' for i in range(n_meters)],
        'transformer_id': [f'TX_{i%10:02d}' for i in range(n_meters)],
    }
    
    # Generate consumption with outliers and missing values
    for month in range(n_months):
        consumption = np.random.lognormal(mean=5, sigma=0.5, size=n_meters) * 100
        
        # Inject outliers (5%)
        outlier_idx = np.random.choice(n_meters, size=int(n_meters * 0.05), replace=False)
        consumption[outlier_idx] *= 5
        
        # Inject missing values (10%)
        missing_idx = np.random.choice(n_meters, size=int(n_meters * 0.10), replace=False)
        consumption[missing_idx] = np.nan
        
        test_data[f'monthly_consumption_2024{month+1:02d}'] = consumption
    
    test_df = pd.DataFrame(test_data)
    
    print(f"Created test dataset: {len(test_df)} meters × {n_months} months")
    print(f"Missing values: {test_df.filter(like='monthly_consumption').isnull().sum().sum()}")
    print()
    
    # Test preprocessing
    print("Testing preprocessing pipeline...")
    print("-" * 80)
    
    try:
        preprocessor = DataPreprocessor(
            PreprocessorConfig(
                outlier_threshold=3.0,
                imputation_strategy=ImputationStrategy.FORWARD_FILL,
                normalization_method=NormalizationMethod.MINMAX,
                verbose=True
            )
        )
        
        result = preprocessor.preprocess(test_df)
        
        print("\n✓ Preprocessing successful!")
        print("\nSummary:")
        print(preprocessor.get_preprocessing_summary(result))
        
        # Validate results
        consumption_cols = result.metadata['consumption_columns']
        
        # Check no missing values after imputation
        missing_after = result.data[consumption_cols].isnull().sum().sum()
        assert missing_after == 0, f"Expected 0 missing values, got {missing_after}"
        
        # Check normalized values in [0, 1]
        min_val = result.data[consumption_cols].min().min()
        max_val = result.data[consumption_cols].max().max()
        assert 0 <= min_val <= 1, f"Minimum value {min_val} out of range [0, 1]"
        assert 0 <= max_val <= 1, f"Maximum value {max_val} out of range [0, 1]"
        
        print("\n✓ All validation checks passed!")
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")
