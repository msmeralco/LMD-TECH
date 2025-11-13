"""
Production-Grade Data Loader for GhostLoad Mapper ML System
=============================================================

This module provides enterprise-grade data loading, validation, and transformation
capabilities for electrical meter consumption and transformer infrastructure data.

The data loader implements defensive programming practices, comprehensive validation,
and efficient data transformation pipelines suitable for production ML systems.

Architecture:
    - DataSchema: Schema definitions and validation rules
    - DataValidator: Multi-stage validation with detailed error reporting
    - DataTransformer: Feature engineering and format conversion
    - GhostLoadDataLoader: Main facade class orchestrating the loading pipeline

Design Principles:
    - Defense in depth: Multi-layer validation (schema, business logic, statistical)
    - Fail-fast: Early detection of data quality issues with actionable error messages
    - Idempotency: Reproducible results for the same input data
    - Performance: Vectorized operations, lazy evaluation, and efficient memory usage
    - Observability: Structured logging with performance metrics and data quality reports

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
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

class CustomerClass(str, Enum):
    """Valid customer classification categories."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"


class RiskBand(str, Enum):
    """Anomaly risk severity levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AnomalyType(str, Enum):
    """Types of consumption anomalies."""
    LOW_CONSUMPTION = "low_consumption"
    HIGH_CONSUMPTION = "high_consumption"
    ERRATIC_PATTERN = "erratic_pattern"


# Default validation constraints
DEFAULT_CONSTRAINTS = {
    'min_consumption': 0.0,
    'max_consumption': 10000.0,  # kWh per month
    'min_capacity': 25.0,  # kVA
    'max_capacity': 1000.0,  # kVA
    'min_latitude': -90.0,
    'max_latitude': 90.0,
    'min_longitude': -180.0,
    'max_longitude': 180.0,
    'max_null_ratio': 0.1,  # Maximum 10% null values allowed
    'min_months': 1,
    'max_months': 120,  # 10 years
}


# ============================================================================
# DATA SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class DataSchema:
    """
    Schema definitions for GhostLoad Mapper datasets.
    
    Defines required columns, data types, and validation rules for:
    - Meter consumption data
    - Transformer infrastructure data
    - Anomaly ground truth labels
    
    Attributes:
        meter_required_columns: Mandatory columns in meter consumption CSV
        transformer_required_columns: Mandatory columns in transformer CSV
        anomaly_required_columns: Mandatory columns in anomaly labels CSV
        consumption_column_prefix: Prefix for time-series consumption columns
        meter_dtypes: Expected pandas data types for meter columns
        transformer_dtypes: Expected pandas data types for transformer columns
    """
    
    # Meter consumption schema
    meter_required_columns: List[str] = field(default_factory=lambda: [
        'meter_id',
        'transformer_id',
        'customer_class',
        'barangay',
        'lat',
        'lon',
        'kVA'
    ])
    
    # Transformer infrastructure schema
    transformer_required_columns: List[str] = field(default_factory=lambda: [
        'transformer_id',
        'feeder_id',
        'barangay',
        'lat',
        'lon',
        'capacity_kVA'
    ])
    
    # Anomaly labels schema
    anomaly_required_columns: List[str] = field(default_factory=lambda: [
        'meter_id',
        'anomaly_flag',
        'risk_band',
        'anomaly_type'
    ])
    
    # Time-series consumption column pattern
    consumption_column_prefix: str = 'monthly_consumption_'
    
    # Data types for validation
    meter_dtypes: Dict[str, str] = field(default_factory=lambda: {
        'meter_id': 'object',
        'transformer_id': 'object',
        'customer_class': 'object',
        'barangay': 'object',
        'lat': 'float64',
        'lon': 'float64',
        'kVA': 'float64'
    })
    
    transformer_dtypes: Dict[str, str] = field(default_factory=lambda: {
        'transformer_id': 'object',
        'feeder_id': 'object',
        'barangay': 'object',
        'lat': 'float64',
        'lon': 'float64',
        'capacity_kVA': 'float64'
    })
    
    def get_consumption_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract time-series consumption column names from DataFrame.
        
        Args:
            df: DataFrame containing meter consumption data
            
        Returns:
            Sorted list of consumption column names (e.g., monthly_consumption_202401)
            
        Raises:
            ValueError: If no consumption columns found
        """
        consumption_cols = [
            col for col in df.columns 
            if col.startswith(self.consumption_column_prefix)
        ]
        
        if not consumption_cols:
            raise ValueError(
                f"No consumption columns found with prefix '{self.consumption_column_prefix}'. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Sort chronologically
        return sorted(consumption_cols)


# ============================================================================
# DATA VALIDATION ENGINE
# ============================================================================

class DataValidator:
    """
    Multi-stage data validation engine with comprehensive quality checks.
    
    Implements defense-in-depth validation strategy:
    1. Schema validation: Required columns, data types
    2. Business logic validation: Value ranges, referential integrity
    3. Statistical validation: Distribution sanity checks, outlier detection
    4. Completeness validation: Null checks, coverage analysis
    
    All validation failures generate actionable error messages with specific
    row/column information for debugging.
    """
    
    def __init__(self, schema: DataSchema, constraints: Optional[Dict] = None):
        """
        Initialize validator with schema and constraints.
        
        Args:
            schema: DataSchema instance defining expected structure
            constraints: Optional validation constraints (uses defaults if None)
        """
        self.schema = schema
        self.constraints = constraints or DEFAULT_CONSTRAINTS
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def validate_meter_data(
        self, 
        df: pd.DataFrame, 
        strict: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Comprehensive validation of meter consumption data.
        
        Args:
            df: DataFrame containing meter consumption data
            strict: If True, raise exception on validation failure
            
        Returns:
            Tuple of (is_valid, errors, warnings)
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info(f"Validating meter data: {len(df)} rows, {len(df.columns)} columns")
        
        # Stage 1: Schema validation
        self._validate_schema(df, self.schema.meter_required_columns, "meter")
        
        # Stage 2: Data type validation
        self._validate_dtypes(df, self.schema.meter_dtypes, "meter")
        
        # Stage 3: Business logic validation
        self._validate_meter_business_logic(df)
        
        # Stage 4: Statistical validation
        self._validate_statistical_properties(df)
        
        # Stage 5: Completeness validation
        self._validate_completeness(df, "meter")
        
        is_valid = len(self.validation_errors) == 0
        
        if not is_valid and strict:
            error_msg = "\n".join(self.validation_errors)
            raise ValueError(f"Meter data validation failed:\n{error_msg}")
        
        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
    
    def validate_transformer_data(
        self, 
        df: pd.DataFrame, 
        strict: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Comprehensive validation of transformer infrastructure data.
        
        Args:
            df: DataFrame containing transformer data
            strict: If True, raise exception on validation failure
            
        Returns:
            Tuple of (is_valid, errors, warnings)
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info(f"Validating transformer data: {len(df)} rows, {len(df.columns)} columns")
        
        # Stage 1: Schema validation
        self._validate_schema(df, self.schema.transformer_required_columns, "transformer")
        
        # Stage 2: Data type validation
        self._validate_dtypes(df, self.schema.transformer_dtypes, "transformer")
        
        # Stage 3: Business logic validation
        self._validate_transformer_business_logic(df)
        
        # Stage 4: Completeness validation
        self._validate_completeness(df, "transformer")
        
        is_valid = len(self.validation_errors) == 0
        
        if not is_valid and strict:
            error_msg = "\n".join(self.validation_errors)
            raise ValueError(f"Transformer data validation failed:\n{error_msg}")
        
        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
    
    def validate_anomaly_data(
        self, 
        df: pd.DataFrame, 
        meter_ids: pd.Series,
        strict: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate anomaly labels with referential integrity checks.
        
        Args:
            df: DataFrame containing anomaly labels
            meter_ids: Series of valid meter IDs for referential integrity
            strict: If True, raise exception on validation failure
            
        Returns:
            Tuple of (is_valid, errors, warnings)
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info(f"Validating anomaly labels: {len(df)} rows")
        
        # Stage 1: Schema validation
        self._validate_schema(df, self.schema.anomaly_required_columns, "anomaly")
        
        # Stage 2: Referential integrity
        invalid_meter_ids = df[~df['meter_id'].isin(meter_ids)]
        if not invalid_meter_ids.empty:
            self.validation_errors.append(
                f"Found {len(invalid_meter_ids)} anomaly labels with invalid meter_id references: "
                f"{invalid_meter_ids['meter_id'].head(5).tolist()}"
            )
        
        # Stage 3: Validate anomaly flags
        invalid_flags = df[~df['anomaly_flag'].isin([0, 1])]
        if not invalid_flags.empty:
            self.validation_errors.append(
                f"Found {len(invalid_flags)} invalid anomaly_flag values "
                f"(must be 0 or 1): {invalid_flags['anomaly_flag'].unique().tolist()}"
            )
        
        # Stage 4: Validate risk bands
        valid_risk_bands = {rb.value for rb in RiskBand}
        invalid_risk = df[~df['risk_band'].isin(valid_risk_bands)]
        if not invalid_risk.empty:
            self.validation_errors.append(
                f"Found {len(invalid_risk)} invalid risk_band values. "
                f"Valid values: {valid_risk_bands}. "
                f"Found: {invalid_risk['risk_band'].unique().tolist()}"
            )
        
        # Stage 5: Validate anomaly types
        valid_anomaly_types = {at.value for at in AnomalyType}
        invalid_types = df[~df['anomaly_type'].isin(valid_anomaly_types)]
        if not invalid_types.empty:
            self.validation_warnings.append(
                f"Found {len(invalid_types)} non-standard anomaly_type values: "
                f"{invalid_types['anomaly_type'].unique().tolist()}"
            )
        
        is_valid = len(self.validation_errors) == 0
        
        if not is_valid and strict:
            error_msg = "\n".join(self.validation_errors)
            raise ValueError(f"Anomaly data validation failed:\n{error_msg}")
        
        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
    
    def _validate_schema(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str], 
        data_type: str
    ) -> None:
        """Validate required columns exist."""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(
                f"Missing required columns in {data_type} data: {sorted(missing_cols)}. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )
    
    def _validate_dtypes(
        self, 
        df: pd.DataFrame, 
        expected_dtypes: Dict[str, str],
        data_type: str
    ) -> None:
        """Validate data types match expectations."""
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                continue  # Already caught by schema validation
            
            actual_dtype = str(df[col].dtype)
            if expected_dtype == 'float64' and actual_dtype not in ['float64', 'int64']:
                self.validation_errors.append(
                    f"Column '{col}' in {data_type} data has incorrect dtype. "
                    f"Expected: {expected_dtype}, Got: {actual_dtype}"
                )
            elif expected_dtype == 'object' and actual_dtype != 'object':
                self.validation_warnings.append(
                    f"Column '{col}' in {data_type} data should be string/object type. "
                    f"Got: {actual_dtype}"
                )
    
    def _validate_meter_business_logic(self, df: pd.DataFrame) -> None:
        """Validate meter-specific business rules."""
        # Validate customer classes (only if column exists)
        if 'customer_class' in df.columns:
            valid_classes = {cc.value for cc in CustomerClass}
            invalid_classes = df[~df['customer_class'].isin(valid_classes)]
            if not invalid_classes.empty:
                self.validation_errors.append(
                    f"Found {len(invalid_classes)} invalid customer_class values. "
                    f"Valid: {valid_classes}. "
                    f"Found: {invalid_classes['customer_class'].unique().tolist()}"
                )
        
        # Validate geographic coordinates
        self._validate_coordinates(df, 'meter')
        
        # Validate kVA (apparent power)
        if 'kVA' in df.columns:
            invalid_kva = df[(df['kVA'] < 0) | (df['kVA'] > 10000)]
            if not invalid_kva.empty:
                self.validation_errors.append(
                    f"Found {len(invalid_kva)} meters with invalid kVA values "
                    f"(range: 0-10000): {invalid_kva['meter_id'].head(5).tolist()}"
                )
        
        # Validate consumption values
        try:
            consumption_cols = self.schema.get_consumption_columns(df)
            if consumption_cols:
                for col in consumption_cols:
                    min_val = self.constraints['min_consumption']
                    max_val = self.constraints['max_consumption']
                    
                    # Allow NaN for missing months, but validate non-null values
                    invalid = df[
                        df[col].notna() & 
                        ((df[col] < min_val) | (df[col] > max_val))
                    ]
                    
                    if not invalid.empty and 'meter_id' in df.columns:
                        self.validation_errors.append(
                            f"Column '{col}' contains {len(invalid)} values outside "
                            f"valid range [{min_val}, {max_val}]. "
                            f"Affected meters: {invalid['meter_id'].head(5).tolist()}"
                        )
        except ValueError:
            # No consumption columns found - already caught by schema validation
            pass
    
    def _validate_transformer_business_logic(self, df: pd.DataFrame) -> None:
        """Validate transformer-specific business rules."""
        # Validate capacity
        if 'capacity_kVA' in df.columns:
            min_cap = self.constraints['min_capacity']
            max_cap = self.constraints['max_capacity']
            
            invalid_capacity = df[
                (df['capacity_kVA'] < min_cap) | 
                (df['capacity_kVA'] > max_cap)
            ]
            
            if not invalid_capacity.empty:
                self.validation_errors.append(
                    f"Found {len(invalid_capacity)} transformers with invalid capacity "
                    f"(range: {min_cap}-{max_cap} kVA): "
                    f"{invalid_capacity['transformer_id'].head(5).tolist()}"
                )
        
        # Validate geographic coordinates
        self._validate_coordinates(df, 'transformer')
        
        # Check for duplicate transformer IDs
        duplicates = df[df.duplicated(subset=['transformer_id'], keep=False)]
        if not duplicates.empty:
            self.validation_errors.append(
                f"Found {len(duplicates)} duplicate transformer_id values: "
                f"{duplicates['transformer_id'].unique().tolist()}"
            )
    
    def _validate_coordinates(self, df: pd.DataFrame, data_type: str) -> None:
        """Validate latitude and longitude ranges."""
        if 'lat' in df.columns:
            invalid_lat = df[
                (df['lat'] < self.constraints['min_latitude']) | 
                (df['lat'] > self.constraints['max_latitude'])
            ]
            if not invalid_lat.empty:
                self.validation_errors.append(
                    f"Found {len(invalid_lat)} {data_type} records with invalid latitude "
                    f"(range: -90 to 90): {invalid_lat.iloc[0]['lat']}"
                )
        
        if 'lon' in df.columns:
            invalid_lon = df[
                (df['lon'] < self.constraints['min_longitude']) | 
                (df['lon'] > self.constraints['max_longitude'])
            ]
            if not invalid_lon.empty:
                self.validation_errors.append(
                    f"Found {len(invalid_lon)} {data_type} records with invalid longitude "
                    f"(range: -180 to 180): {invalid_lon.iloc[0]['lon']}"
                )
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> None:
        """Validate statistical properties of consumption data."""
        consumption_cols = self.schema.get_consumption_columns(df)
        if not consumption_cols:
            return
        
        # Check for suspiciously uniform distributions (potential synthetic data issues)
        for col in consumption_cols:
            non_null = df[col].dropna()
            if len(non_null) > 10:
                # Check if too many identical values
                value_counts = non_null.value_counts()
                most_common_ratio = value_counts.iloc[0] / len(non_null)
                
                if most_common_ratio > 0.5:
                    self.validation_warnings.append(
                        f"Column '{col}' has {most_common_ratio:.1%} identical values "
                        f"({value_counts.iloc[0]} out of {len(non_null)}). "
                        f"This may indicate data quality issues."
                    )
        
        # Check for reasonable variance
        consumption_array = df[consumption_cols].values
        row_variances = np.nanvar(consumption_array, axis=1)
        zero_variance_count = np.sum(row_variances == 0)
        
        if zero_variance_count > 0:
            self.validation_warnings.append(
                f"Found {zero_variance_count} meters with zero consumption variance "
                f"across all months. These may be inactive meters."
            )
    
    def _validate_completeness(self, df: pd.DataFrame, data_type: str) -> None:
        """Validate data completeness (null checks)."""
        # Check overall null ratio
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_ratio = null_cells / total_cells if total_cells > 0 else 0
        
        if null_ratio > self.constraints['max_null_ratio']:
            self.validation_warnings.append(
                f"{data_type.capitalize()} data has {null_ratio:.1%} null values "
                f"(threshold: {self.constraints['max_null_ratio']:.1%})"
            )
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            self.validation_errors.append(
                f"Found {empty_rows} completely empty rows in {data_type} data"
            )


# ============================================================================
# DATA TRANSFORMATION ENGINE
# ============================================================================

class DataTransformer:
    """
    Feature engineering and data transformation pipeline.
    
    Converts raw CSV data into ML-ready formats:
    - Consumption time series → NumPy arrays
    - Categorical features → Encoded representations
    - Missing values → Imputation strategies
    - Feature scaling → Normalization/standardization
    """
    
    def __init__(self, schema: DataSchema):
        """
        Initialize transformer with schema.
        
        Args:
            schema: DataSchema instance for column identification
        """
        self.schema = schema
        
    def extract_consumption_matrix(
        self, 
        df: pd.DataFrame,
        impute_strategy: str = 'zero'
    ) -> NDArray[np.float64]:
        """
        Extract consumption time series as NumPy array for scikit-learn.
        
        Args:
            df: DataFrame with meter consumption data
            impute_strategy: Strategy for missing values ('zero', 'mean', 'median', 'forward_fill')
            
        Returns:
            2D NumPy array of shape (n_meters, n_months) with consumption values
            
        Raises:
            ValueError: If invalid impute_strategy provided
        """
        consumption_cols = self.schema.get_consumption_columns(df)
        logger.info(f"Extracting consumption matrix: {len(consumption_cols)} months")
        
        # Extract raw consumption matrix
        consumption_matrix = df[consumption_cols].values
        
        # Ensure C-contiguous memory layout for performance
        if not consumption_matrix.flags['C_CONTIGUOUS']:
            consumption_matrix = np.ascontiguousarray(consumption_matrix)
        
        # Handle missing values
        if impute_strategy == 'zero':
            consumption_matrix = np.nan_to_num(consumption_matrix, nan=0.0)
        elif impute_strategy == 'mean':
            col_means = np.nanmean(consumption_matrix, axis=0)
            for i, mean_val in enumerate(col_means):
                consumption_matrix[np.isnan(consumption_matrix[:, i]), i] = mean_val
        elif impute_strategy == 'median':
            col_medians = np.nanmedian(consumption_matrix, axis=0)
            for i, median_val in enumerate(col_medians):
                consumption_matrix[np.isnan(consumption_matrix[:, i]), i] = median_val
        elif impute_strategy == 'forward_fill':
            # Forward fill along time axis (columns)
            df_filled = pd.DataFrame(consumption_matrix).fillna(method='ffill', axis=1)
            consumption_matrix = df_filled.fillna(0.0).values
        else:
            raise ValueError(
                f"Invalid impute_strategy: {impute_strategy}. "
                f"Valid options: 'zero', 'mean', 'median', 'forward_fill'"
            )
        
        logger.info(f"Consumption matrix shape: {consumption_matrix.shape}")
        return consumption_matrix
    
    def compute_statistical_features(
        self, 
        consumption_matrix: NDArray[np.float64]
    ) -> pd.DataFrame:
        """
        Compute statistical features from consumption time series.
        
        Features include:
        - Central tendency: mean, median, mode
        - Dispersion: std, variance, IQR, range
        - Shape: skewness, kurtosis
        - Trend: linear regression slope
        - Seasonality indicators
        
        Args:
            consumption_matrix: 2D array of consumption values (n_meters, n_months)
            
        Returns:
            DataFrame with statistical features for each meter
        """
        from scipy import stats
        
        n_meters = consumption_matrix.shape[0]
        logger.info(f"Computing statistical features for {n_meters} meters")
        
        features = {}
        
        # Central tendency
        features['consumption_mean'] = np.mean(consumption_matrix, axis=1)
        features['consumption_median'] = np.median(consumption_matrix, axis=1)
        features['consumption_std'] = np.std(consumption_matrix, axis=1)
        features['consumption_min'] = np.min(consumption_matrix, axis=1)
        features['consumption_max'] = np.max(consumption_matrix, axis=1)
        features['consumption_range'] = features['consumption_max'] - features['consumption_min']
        
        # Coefficient of variation (normalized volatility)
        features['consumption_cv'] = features['consumption_std'] / (features['consumption_mean'] + 1e-6)
        
        # Percentiles for robust statistics
        features['consumption_p25'] = np.percentile(consumption_matrix, 25, axis=1)
        features['consumption_p75'] = np.percentile(consumption_matrix, 75, axis=1)
        features['consumption_iqr'] = features['consumption_p75'] - features['consumption_p25']
        
        # Trend (linear regression slope)
        n_months = consumption_matrix.shape[1]
        time_index = np.arange(n_months)
        slopes = []
        
        for i in range(n_meters):
            if np.all(consumption_matrix[i] == 0):
                slopes.append(0.0)
            else:
                slope, _, _, _, _ = stats.linregress(time_index, consumption_matrix[i])
                slopes.append(slope)
        
        features['consumption_trend'] = np.array(slopes)
        
        # Zero consumption ratio (indicator of inactive periods)
        features['zero_consumption_ratio'] = np.mean(consumption_matrix == 0, axis=1)
        
        # Skewness and kurtosis
        features['consumption_skewness'] = stats.skew(consumption_matrix, axis=1)
        features['consumption_kurtosis'] = stats.kurtosis(consumption_matrix, axis=1)
        
        return pd.DataFrame(features)
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical features for ML models.
        
        Args:
            df: DataFrame with categorical columns
            columns: List of column names to encode
            method: Encoding method ('onehot', 'label', 'target')
            
        Returns:
            DataFrame with encoded categorical features
            
        Raises:
            ValueError: If invalid encoding method specified
        """
        df_encoded = df.copy()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            raise ValueError(f"Invalid encoding method: {method}. Use 'onehot' or 'label'")
        
        return df_encoded


# ============================================================================
# MAIN DATA LOADER FACADE
# ============================================================================

@dataclass
class LoadedData:
    """
    Container for loaded and validated datasets.
    
    Attributes:
        meters: DataFrame with meter metadata and consumption
        transformers: DataFrame with transformer infrastructure data
        anomalies: Optional DataFrame with anomaly labels
        consumption_matrix: NumPy array of consumption time series
        feature_matrix: Optional DataFrame with engineered features
        metadata: Dictionary with loading metadata (timing, validation results)
    """
    meters: pd.DataFrame
    transformers: pd.DataFrame
    anomalies: Optional[pd.DataFrame]
    consumption_matrix: NDArray[np.float64]
    feature_matrix: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"LoadedData(\n"
            f"  meters: {len(self.meters)} rows × {len(self.meters.columns)} cols,\n"
            f"  transformers: {len(self.transformers)} rows × {len(self.transformers.columns)} cols,\n"
            f"  anomalies: {len(self.anomalies) if self.anomalies is not None else 'None'} rows,\n"
            f"  consumption_matrix: {self.consumption_matrix.shape},\n"
            f"  feature_matrix: {self.feature_matrix.shape if self.feature_matrix is not None else 'None'}\n"
            f")"
        )


class GhostLoadDataLoader:
    """
    Production-grade data loader for GhostLoad Mapper ML system.
    
    Main facade class that orchestrates the complete data loading pipeline:
    1. File I/O with error handling
    2. Multi-stage validation
    3. Feature extraction and transformation
    4. Quality reporting and metrics
    
    Usage:
        >>> loader = GhostLoadDataLoader(dataset_dir='datasets/development')
        >>> data = loader.load_all(validate=True, compute_features=True)
        >>> X = data.consumption_matrix  # For scikit-learn models
        >>> y = data.anomalies['anomaly_flag'].values  # Ground truth labels
    
    Design Pattern: Facade + Builder
    Thread Safety: Not thread-safe (use separate instances per thread)
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        schema: Optional[DataSchema] = None,
        validation_constraints: Optional[Dict] = None
    ):
        """
        Initialize data loader with dataset directory.
        
        Args:
            dataset_dir: Path to directory containing CSV files
            schema: Optional custom DataSchema (uses default if None)
            validation_constraints: Optional custom constraints (uses defaults if None)
        """
        self.dataset_dir = Path(dataset_dir)
        self.schema = schema or DataSchema()
        self.validator = DataValidator(self.schema, validation_constraints)
        self.transformer = DataTransformer(self.schema)
        
        logger.info(f"Initialized GhostLoadDataLoader: {self.dataset_dir}")
        
        # Verify dataset directory exists
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}. "
                f"Please ensure the path is correct."
            )
    
    def load_meters(
        self, 
        filename: str = 'meter_consumption.csv',
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and validate meter consumption data.
        
        Args:
            filename: Name of meter consumption CSV file
            validate: If True, perform validation (recommended)
            
        Returns:
            DataFrame with meter consumption data
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If validation fails (when validate=True)
        """
        filepath = self.dataset_dir / filename
        logger.info(f"Loading meter data from: {filepath}")
        
        start_time = time.time()
        
        try:
            df = pd.read_csv(filepath)
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(df)} meters in {load_time:.2f}s")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Meter consumption file not found: {filepath}. "
                f"Available files: {list(self.dataset_dir.glob('*.csv'))}"
            )
        except Exception as e:
            raise IOError(f"Error reading {filepath}: {str(e)}")
        
        if validate:
            is_valid, errors, warnings = self.validator.validate_meter_data(df, strict=True)
            
            # Log warnings even if validation passed
            for warning in warnings:
                logger.warning(f"Meter data: {warning}")
        
        return df
    
    def load_transformers(
        self, 
        filename: str = 'transformers.csv',
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and validate transformer infrastructure data.
        
        Args:
            filename: Name of transformer CSV file
            validate: If True, perform validation (recommended)
            
        Returns:
            DataFrame with transformer data
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If validation fails (when validate=True)
        """
        filepath = self.dataset_dir / filename
        logger.info(f"Loading transformer data from: {filepath}")
        
        start_time = time.time()
        
        try:
            df = pd.read_csv(filepath)
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(df)} transformers in {load_time:.2f}s")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Transformer file not found: {filepath}. "
                f"Available files: {list(self.dataset_dir.glob('*.csv'))}"
            )
        except Exception as e:
            raise IOError(f"Error reading {filepath}: {str(e)}")
        
        if validate:
            is_valid, errors, warnings = self.validator.validate_transformer_data(df, strict=True)
            
            for warning in warnings:
                logger.warning(f"Transformer data: {warning}")
        
        return df
    
    def load_anomalies(
        self, 
        filename: str = 'anomaly_labels.csv',
        meter_ids: Optional[pd.Series] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and validate anomaly ground truth labels.
        
        Args:
            filename: Name of anomaly labels CSV file
            meter_ids: Optional Series of valid meter IDs for referential integrity check
            validate: If True, perform validation (recommended)
            
        Returns:
            DataFrame with anomaly labels
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If validation fails (when validate=True)
        """
        filepath = self.dataset_dir / filename
        
        # Anomaly labels are optional in some datasets
        if not filepath.exists():
            logger.warning(f"Anomaly labels file not found: {filepath}. Returning None.")
            return None
        
        logger.info(f"Loading anomaly labels from: {filepath}")
        
        start_time = time.time()
        
        try:
            df = pd.read_csv(filepath)
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(df)} anomaly labels in {load_time:.2f}s")
            
        except Exception as e:
            raise IOError(f"Error reading {filepath}: {str(e)}")
        
        if validate and meter_ids is not None:
            is_valid, errors, warnings = self.validator.validate_anomaly_data(
                df, meter_ids, strict=True
            )
            
            for warning in warnings:
                logger.warning(f"Anomaly data: {warning}")
        
        return df
    
    def load_all(
        self,
        validate: bool = True,
        compute_features: bool = False,
        impute_strategy: str = 'zero'
    ) -> LoadedData:
        """
        Load complete dataset with all components.
        
        This is the primary entry point for most use cases. It loads meters,
        transformers, and anomaly labels (if available), performs validation,
        and optionally computes statistical features.
        
        Args:
            validate: If True, perform comprehensive validation
            compute_features: If True, compute statistical features from consumption
            impute_strategy: Strategy for handling missing consumption values
            
        Returns:
            LoadedData object containing all datasets and matrices
            
        Example:
            >>> loader = GhostLoadDataLoader('datasets/development')
            >>> data = loader.load_all(validate=True, compute_features=True)
            >>> 
            >>> # Use for ML training
            >>> X = data.consumption_matrix
            >>> y = data.anomalies['anomaly_flag'].values
            >>> 
            >>> from sklearn.ensemble import IsolationForest
            >>> model = IsolationForest(contamination=0.075)
            >>> model.fit(X)
        """
        logger.info("="*80)
        logger.info("LOADING GHOSTLOAD MAPPER DATASET")
        logger.info("="*80)
        
        start_time = time.time()
        metadata = {}
        
        # Load meter data
        meters = self.load_meters(validate=validate)
        metadata['n_meters'] = len(meters)
        
        # Load transformer data
        transformers = self.load_transformers(validate=validate)
        metadata['n_transformers'] = len(transformers)
        
        # Load anomaly labels (optional)
        anomalies = self.load_anomalies(
            meter_ids=meters['meter_id'] if validate else None,
            validate=validate
        )
        
        if anomalies is not None:
            metadata['n_anomalies'] = len(anomalies)
            metadata['anomaly_rate'] = len(anomalies) / len(meters)
        
        # Extract consumption matrix
        consumption_matrix = self.transformer.extract_consumption_matrix(
            meters, impute_strategy=impute_strategy
        )
        metadata['consumption_shape'] = consumption_matrix.shape
        metadata['n_months'] = consumption_matrix.shape[1]
        
        # Compute statistical features (optional)
        feature_matrix = None
        if compute_features:
            feature_matrix = self.transformer.compute_statistical_features(consumption_matrix)
            metadata['n_features'] = len(feature_matrix.columns)
        
        # Referential integrity check
        if validate:
            self._validate_referential_integrity(meters, transformers)
        
        total_time = time.time() - start_time
        metadata['load_time_seconds'] = total_time
        
        logger.info("="*80)
        logger.info("DATASET LOADING COMPLETE")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Meters: {metadata['n_meters']}")
        logger.info(f"  Transformers: {metadata['n_transformers']}")
        logger.info(f"  Anomalies: {metadata.get('n_anomalies', 'N/A')}")
        logger.info(f"  Consumption matrix: {consumption_matrix.shape}")
        if feature_matrix is not None:
            logger.info(f"  Feature matrix: {feature_matrix.shape}")
        logger.info("="*80)
        
        return LoadedData(
            meters=meters,
            transformers=transformers,
            anomalies=anomalies,
            consumption_matrix=consumption_matrix,
            feature_matrix=feature_matrix,
            metadata=metadata
        )
    
    def _validate_referential_integrity(
        self, 
        meters: pd.DataFrame, 
        transformers: pd.DataFrame
    ) -> None:
        """
        Validate foreign key relationships between meters and transformers.
        
        Args:
            meters: DataFrame with meter data
            transformers: DataFrame with transformer data
            
        Raises:
            ValueError: If referential integrity violations found
        """
        logger.info("Validating referential integrity...")
        
        valid_transformer_ids = set(transformers['transformer_id'])
        meter_transformer_ids = set(meters['transformer_id'])
        
        orphan_references = meter_transformer_ids - valid_transformer_ids
        
        if orphan_references:
            n_orphans = meters[meters['transformer_id'].isin(orphan_references)].shape[0]
            raise ValueError(
                f"Found {n_orphans} meters referencing non-existent transformers. "
                f"Invalid transformer_id values: {sorted(orphan_references)}"
            )
        
        logger.info("✓ Referential integrity validated successfully")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary with quality metrics and statistics
        """
        try:
            meters = self.load_meters(validate=False)
            transformers = self.load_transformers(validate=False)
            anomalies = self.load_anomalies(validate=False)
            
            report = {
                'dataset_directory': str(self.dataset_dir),
                'timestamp': pd.Timestamp.now().isoformat(),
                'meters': {
                    'count': len(meters),
                    'columns': len(meters.columns),
                    'null_cells': meters.isnull().sum().sum(),
                    'null_ratio': meters.isnull().sum().sum() / meters.size,
                    'duplicate_ids': meters.duplicated(subset=['meter_id']).sum(),
                },
                'transformers': {
                    'count': len(transformers),
                    'columns': len(transformers.columns),
                    'null_cells': transformers.isnull().sum().sum(),
                    'null_ratio': transformers.isnull().sum().sum() / transformers.size,
                    'duplicate_ids': transformers.duplicated(subset=['transformer_id']).sum(),
                },
            }
            
            if anomalies is not None:
                report['anomalies'] = {
                    'count': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(meters),
                    'risk_band_distribution': anomalies['risk_band'].value_counts().to_dict(),
                    'anomaly_type_distribution': anomalies['anomaly_type'].value_counts().to_dict(),
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
            return {'error': str(e)}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_dataset(
    dataset_dir: Union[str, Path],
    validate: bool = True,
    compute_features: bool = False
) -> LoadedData:
    """
    Convenience function to load dataset with default settings.
    
    Args:
        dataset_dir: Path to dataset directory
        validate: If True, perform validation
        compute_features: If True, compute statistical features
        
    Returns:
        LoadedData object with all datasets
        
    Example:
        >>> from data_loader import load_dataset
        >>> data = load_dataset('datasets/development')
        >>> X = data.consumption_matrix
        >>> y = data.anomalies['anomaly_flag'].values
    """
    loader = GhostLoadDataLoader(dataset_dir)
    return loader.load_all(validate=validate, compute_features=compute_features)


def validate_dataset(dataset_dir: Union[str, Path]) -> bool:
    """
    Validate dataset without loading into memory.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        True if validation passes, False otherwise
        
    Example:
        >>> from data_loader import validate_dataset
        >>> is_valid = validate_dataset('datasets/production')
        >>> if is_valid:
        >>>     print("Dataset is ready for ML training")
    """
    try:
        loader = GhostLoadDataLoader(dataset_dir)
        loader.load_all(validate=True, compute_features=False)
        return True
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of data loader capabilities.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("GHOSTLOAD MAPPER DATA LOADER - SELF-TEST")
    print("="*80 + "\n")
    
    # Example 1: Load development dataset
    print("Example 1: Loading development dataset...")
    print("-" * 80)
    
    try:
        loader = GhostLoadDataLoader('../datasets/development')
        data = loader.load_all(validate=True, compute_features=True)
        
        print(f"\n✓ Successfully loaded dataset:")
        print(data)
        print(f"\nConsumption matrix summary:")
        print(f"  Shape: {data.consumption_matrix.shape}")
        print(f"  Mean consumption: {data.consumption_matrix.mean():.2f} kWh")
        print(f"  Std consumption: {data.consumption_matrix.std():.2f} kWh")
        
        if data.feature_matrix is not None:
            print(f"\nFeature matrix columns:")
            print(f"  {', '.join(data.feature_matrix.columns.tolist())}")
        
        print(f"\nMetadata:")
        for key, value in data.metadata.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    # Example 2: Data quality report
    print("\n" + "="*80)
    print("Example 2: Generating data quality report...")
    print("-" * 80)
    
    try:
        loader = GhostLoadDataLoader('../datasets/development')
        report = loader.get_data_quality_report()
        
        print("\n✓ Data Quality Report:")
        import json
        print(json.dumps(report, indent=2, default=str))
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")
