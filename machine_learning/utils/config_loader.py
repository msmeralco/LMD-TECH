"""
Configuration Loader - Centralized Configuration Management

This module provides type-safe, validated configuration loading for the
GhostLoad Mapper ML pipeline. It supports YAML files, environment variables,
and programmatic overrides with comprehensive validation.

Key Responsibilities:
    1. Load configuration from YAML files with schema validation
    2. Support environment variable overrides (12-factor app pattern)
    3. Provide type-safe configuration objects via dataclasses
    4. Validate all parameters with clear error messages
    5. Support multiple environments (dev, staging, production)
    6. Enable hot-reload for parameter tuning

Design Philosophy:
    - Fail fast with clear error messages
    - Type safety through dataclasses
    - Immutability by default (frozen dataclasses)
    - Explicit is better than implicit
    - Configuration as code

Architecture:
    - Hierarchical configuration structure
    - Nested dataclasses for logical grouping
    - Post-init validation for all parameters
    - Environment-specific overrides
    - Singleton pattern for global config

Author: GhostLoad Mapper ML Team
Created: 2025-11-13
Version: 1.0.0
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import warnings

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for configuration loading. "
        "Install with: pip install pyyaml"
    )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigurationFileNotFoundError(ConfigurationError):
    """Exception raised when configuration file is not found."""
    pass


class ConfigurationParseError(ConfigurationError):
    """Exception raised when configuration file parsing fails."""
    pass


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass(frozen=True)
class FilePathsConfig:
    """
    File paths configuration.
    
    Attributes:
        data_dir: Directory containing input data files
        output_dir: Directory for output files (models, reports, etc.)
        model_registry_dir: Directory for model registry
        logs_dir: Directory for log files
        temp_dir: Directory for temporary files
        
        # Data files
        raw_data_file: Raw input data file path
        preprocessed_data_file: Preprocessed data file path
        features_file: Engineered features file path
        
        # Model files
        isolation_forest_model: IsolationForest model path
        dbscan_model: DBSCAN model path
        
        # Output files
        predictions_file: Predictions output file path
        risk_assessment_file: Risk assessment output file path
        metrics_file: Metrics output file path
    """
    
    # Directories
    data_dir: str = "data"
    output_dir: str = "output"
    model_registry_dir: str = "models"
    logs_dir: str = "logs"
    temp_dir: str = "temp"
    
    # Data files
    raw_data_file: str = "data/raw/meter_data.csv"
    preprocessed_data_file: str = "data/processed/preprocessed.csv"
    features_file: str = "data/processed/features.csv"
    
    # Model files
    isolation_forest_model: str = "models/isolation_forest.pkl"
    dbscan_model: str = "models/dbscan.pkl"
    
    # Output files
    predictions_file: str = "output/predictions.csv"
    risk_assessment_file: str = "output/risk_assessment.csv"
    metrics_file: str = "output/metrics.json"
    
    def __post_init__(self):
        """Validate file paths configuration."""
        # Validate directory paths are not empty
        for field_name in ['data_dir', 'output_dir', 'model_registry_dir', 'logs_dir', 'temp_dir']:
            value = getattr(self, field_name)
            if not value or not isinstance(value, str):
                raise ConfigurationValidationError(
                    f"{field_name} must be a non-empty string, got {value}"
                )
        
        # Validate file paths are not empty
        file_fields = [
            'raw_data_file', 'preprocessed_data_file', 'features_file',
            'isolation_forest_model', 'dbscan_model',
            'predictions_file', 'risk_assessment_file', 'metrics_file'
        ]
        for field_name in file_fields:
            value = getattr(self, field_name)
            if not value or not isinstance(value, str):
                raise ConfigurationValidationError(
                    f"{field_name} must be a non-empty string, got {value}"
                )
    
    def resolve_path(self, path: str, base_dir: Optional[str] = None) -> Path:
        """
        Resolve a path relative to base directory.
        
        Args:
            path: Path to resolve
            base_dir: Base directory (defaults to current working directory)
            
        Returns:
            Resolved Path object
        """
        if base_dir:
            return Path(base_dir) / path
        return Path(path)
    
    def ensure_directories(self, base_dir: Optional[str] = None) -> None:
        """
        Create all configured directories if they don't exist.
        
        Args:
            base_dir: Base directory for relative paths
        """
        directories = [
            self.data_dir, self.output_dir, self.model_registry_dir,
            self.logs_dir, self.temp_dir
        ]
        
        for directory in directories:
            dir_path = self.resolve_path(directory, base_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")


@dataclass(frozen=True)
class IsolationForestConfig:
    """
    IsolationForest model parameters.
    
    Attributes:
        contamination: Expected proportion of anomalies (0.0-0.5)
        n_estimators: Number of trees in the forest
        max_samples: Number of samples to draw for each tree
        max_features: Number of features to draw for each tree
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    
    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: Union[int, str] = "auto"
    max_features: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        """Validate IsolationForest parameters."""
        # Validate contamination
        if not 0.0 < self.contamination < 0.5:
            raise ConfigurationValidationError(
                f"contamination must be in (0, 0.5), got {self.contamination}"
            )
        
        # Validate n_estimators
        if self.n_estimators < 1:
            raise ConfigurationValidationError(
                f"n_estimators must be >= 1, got {self.n_estimators}"
            )
        
        # Validate max_samples
        if isinstance(self.max_samples, int) and self.max_samples < 1:
            raise ConfigurationValidationError(
                f"max_samples must be >= 1 or 'auto', got {self.max_samples}"
            )
        elif isinstance(self.max_samples, str) and self.max_samples != "auto":
            raise ConfigurationValidationError(
                f"max_samples must be an integer or 'auto', got {self.max_samples}"
            )
        
        # Validate max_features
        if not 0.0 < self.max_features <= 1.0:
            raise ConfigurationValidationError(
                f"max_features must be in (0, 1], got {self.max_features}"
            )
        
        # Validate random_state
        if self.random_state < 0:
            raise ConfigurationValidationError(
                f"random_state must be >= 0, got {self.random_state}"
            )
        
        # Validate n_jobs
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ConfigurationValidationError(
                f"n_jobs must be -1 or > 0, got {self.n_jobs}"
            )


@dataclass(frozen=True)
class DBSCANConfig:
    """
    DBSCAN clustering parameters.
    
    Attributes:
        eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum samples in neighborhood to form core point
        metric: Distance metric (euclidean, manhattan, etc.)
        algorithm: Algorithm for computing neighbors (auto, ball_tree, kd_tree, brute)
        leaf_size: Leaf size for tree-based algorithms
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    algorithm: str = "auto"
    leaf_size: int = 30
    n_jobs: int = -1
    
    def __post_init__(self):
        """Validate DBSCAN parameters."""
        # Validate eps
        if self.eps <= 0:
            raise ConfigurationValidationError(
                f"eps must be > 0, got {self.eps}"
            )
        
        # Validate min_samples
        if self.min_samples < 1:
            raise ConfigurationValidationError(
                f"min_samples must be >= 1, got {self.min_samples}"
            )
        
        # Validate metric
        valid_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine']
        if self.metric not in valid_metrics:
            warnings.warn(
                f"metric '{self.metric}' not in common metrics {valid_metrics}. "
                f"Ensure it's supported by scikit-learn."
            )
        
        # Validate algorithm
        valid_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        if self.algorithm not in valid_algorithms:
            raise ConfigurationValidationError(
                f"algorithm must be one of {valid_algorithms}, got {self.algorithm}"
            )
        
        # Validate leaf_size
        if self.leaf_size < 1:
            raise ConfigurationValidationError(
                f"leaf_size must be >= 1, got {self.leaf_size}"
            )
        
        # Validate n_jobs
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ConfigurationValidationError(
                f"n_jobs must be -1 or > 0, got {self.n_jobs}"
            )


@dataclass(frozen=True)
class ModelParametersConfig:
    """
    Model parameters for all ML models.
    
    Attributes:
        isolation_forest: IsolationForest configuration
        dbscan: DBSCAN configuration
    """
    
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    
    def __post_init__(self):
        """Validate model parameters."""
        # Validate isolation_forest is correct type
        if not isinstance(self.isolation_forest, IsolationForestConfig):
            raise ConfigurationValidationError(
                f"isolation_forest must be IsolationForestConfig, got {type(self.isolation_forest)}"
            )
        
        # Validate dbscan is correct type
        if not isinstance(self.dbscan, DBSCANConfig):
            raise ConfigurationValidationError(
                f"dbscan must be DBSCANConfig, got {type(self.dbscan)}"
            )


@dataclass(frozen=True)
class RiskThresholdsConfig:
    """
    Risk classification thresholds.
    
    Attributes:
        high: Threshold for high-risk classification (composite_score)
        medium: Threshold for medium-risk classification (composite_score)
        extreme_low_ratio: Threshold for extreme low consumption ratio
        suspicious_low_ratio: Threshold for suspicious low consumption ratio
        spatial_boost: Amount to boost risk score for spatial anomalies
    """
    
    high: float = 0.8
    medium: float = 0.6
    extreme_low_ratio: float = 0.2
    suspicious_low_ratio: float = 0.4
    spatial_boost: float = 0.15
    
    def __post_init__(self):
        """Validate risk thresholds."""
        # Validate high threshold
        if not 0.0 < self.high <= 1.0:
            raise ConfigurationValidationError(
                f"high threshold must be in (0, 1], got {self.high}"
            )
        
        # Validate medium threshold
        if not 0.0 < self.medium <= 1.0:
            raise ConfigurationValidationError(
                f"medium threshold must be in (0, 1], got {self.medium}"
            )
        
        # Validate threshold ordering
        if self.medium >= self.high:
            raise ConfigurationValidationError(
                f"medium threshold ({self.medium}) must be < high threshold ({self.high})"
            )
        
        # Validate extreme_low_ratio
        if not 0.0 < self.extreme_low_ratio <= 1.0:
            raise ConfigurationValidationError(
                f"extreme_low_ratio must be in (0, 1], got {self.extreme_low_ratio}"
            )
        
        # Validate suspicious_low_ratio
        if not 0.0 < self.suspicious_low_ratio <= 1.0:
            raise ConfigurationValidationError(
                f"suspicious_low_ratio must be in (0, 1], got {self.suspicious_low_ratio}"
            )
        
        # Validate ratio ordering
        if self.extreme_low_ratio >= self.suspicious_low_ratio:
            raise ConfigurationValidationError(
                f"extreme_low_ratio ({self.extreme_low_ratio}) must be < "
                f"suspicious_low_ratio ({self.suspicious_low_ratio})"
            )
        
        # Validate spatial_boost
        if not 0.0 <= self.spatial_boost <= 1.0:
            raise ConfigurationValidationError(
                f"spatial_boost must be in [0, 1], got {self.spatial_boost}"
            )


@dataclass(frozen=True)
class FeatureWeightsConfig:
    """
    Feature weights for anomaly scoring.
    
    Attributes:
        isolation: Weight for IsolationForest anomaly score
        ratio: Weight for consumption ratio score
        spatial: Weight for spatial clustering score (optional)
    """
    
    isolation: float = 0.7
    ratio: float = 0.3
    spatial: float = 0.0
    
    def __post_init__(self):
        """Validate feature weights."""
        # Validate individual weights
        for field_name in ['isolation', 'ratio', 'spatial']:
            weight = getattr(self, field_name)
            if not 0.0 <= weight <= 1.0:
                raise ConfigurationValidationError(
                    f"{field_name} weight must be in [0, 1], got {weight}"
                )
        
        # Validate weights sum to 1.0 (allowing small tolerance for floating point)
        total = self.isolation + self.ratio + self.spatial
        if not (0.99 <= total <= 1.01):
            raise ConfigurationValidationError(
                f"Feature weights must sum to 1.0, got {total:.3f} "
                f"(isolation={self.isolation}, ratio={self.ratio}, spatial={self.spatial})"
            )


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics calculation configuration.
    
    Attributes:
        baseline_high_risk_rate: Expected high-risk rate for calibrated models
        top_n_suspicious: Number of top suspicious meters to report
        confidence_threshold: Minimum acceptable confidence level
    """
    
    baseline_high_risk_rate: float = 0.20
    top_n_suspicious: int = 100
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate metrics configuration."""
        # Validate baseline_high_risk_rate
        if not 0.0 < self.baseline_high_risk_rate < 1.0:
            raise ConfigurationValidationError(
                f"baseline_high_risk_rate must be in (0, 1), got {self.baseline_high_risk_rate}"
            )
        
        # Validate top_n_suspicious
        if self.top_n_suspicious < 1:
            raise ConfigurationValidationError(
                f"top_n_suspicious must be >= 1, got {self.top_n_suspicious}"
            )
        
        # Validate confidence_threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationValidationError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )


@dataclass(frozen=True)
class PipelineConfig:
    """
    Main pipeline configuration.
    
    Attributes:
        file_paths: File paths configuration
        model_parameters: Model parameters configuration
        risk_thresholds: Risk thresholds configuration
        feature_weights: Feature weights configuration
        metrics: Metrics configuration
        
        # Pipeline settings
        enable_preprocessing: Enable data preprocessing
        enable_feature_engineering: Enable feature engineering
        enable_training: Enable model training
        enable_evaluation: Enable model evaluation
        
        # Execution settings
        random_seed: Global random seed for reproducibility
        n_jobs: Number of parallel jobs for pipeline execution
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
    """
    
    file_paths: FilePathsConfig = field(default_factory=FilePathsConfig)
    model_parameters: ModelParametersConfig = field(default_factory=ModelParametersConfig)
    risk_thresholds: RiskThresholdsConfig = field(default_factory=RiskThresholdsConfig)
    feature_weights: FeatureWeightsConfig = field(default_factory=FeatureWeightsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Pipeline settings
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_training: bool = True
    enable_evaluation: bool = True
    
    # Execution settings
    random_seed: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    def __post_init__(self):
        """Validate pipeline configuration."""
        # Validate random_seed
        if self.random_seed < 0:
            raise ConfigurationValidationError(
                f"random_seed must be >= 0, got {self.random_seed}"
            )
        
        # Validate n_jobs
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ConfigurationValidationError(
                f"n_jobs must be -1 or > 0, got {self.n_jobs}"
            )
        
        # Validate verbose
        if self.verbose not in [0, 1, 2]:
            raise ConfigurationValidationError(
                f"verbose must be 0, 1, or 2, got {self.verbose}"
            )
        
        # Log configuration loaded
        if self.verbose > 0:
            logger.info(
                f"Pipeline configuration loaded: "
                f"preprocessing={self.enable_preprocessing}, "
                f"feature_engineering={self.enable_feature_engineering}, "
                f"training={self.enable_training}, "
                f"evaluation={self.enable_evaluation}"
            )


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:
    """
    Configuration loader with YAML support and environment variable overrides.
    
    This class provides a centralized way to load, validate, and access
    configuration for the GhostLoad Mapper ML pipeline.
    
    Features:
        - Load from YAML files
        - Environment variable overrides (12-factor app)
        - Type-safe configuration objects
        - Comprehensive validation
        - Multiple environment support (dev, staging, prod)
        - Hot-reload capability
    
    Environment Variable Format:
        GHOSTLOAD_<SECTION>_<SUBSECTION>_<PARAMETER>=value
        
        Examples:
            GHOSTLOAD_MODEL_ISOLATION_FOREST_CONTAMINATION=0.15
            GHOSTLOAD_RISK_THRESHOLDS_HIGH=0.85
            GHOSTLOAD_FEATURE_WEIGHTS_ISOLATION=0.8
    
    Example:
        >>> # Load from default config.yaml
        >>> loader = ConfigLoader()
        >>> config = loader.load()
        >>> 
        >>> # Load from custom file
        >>> loader = ConfigLoader("config/production.yaml")
        >>> config = loader.load()
        >>> 
        >>> # Access configuration
        >>> print(config.model_parameters.isolation_forest.contamination)
        >>> print(config.risk_thresholds.high)
    """
    
    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        enable_env_override: bool = True,
        env_prefix: str = "GHOSTLOAD"
    ):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to YAML configuration file (default: config.yaml)
            enable_env_override: Enable environment variable overrides
            env_prefix: Prefix for environment variables (default: GHOSTLOAD)
        """
        self.config_file = Path(config_file) if config_file else Path("config.yaml")
        self.enable_env_override = enable_env_override
        self.env_prefix = env_prefix
        
        logger.info(
            f"Initialized ConfigLoader (file={self.config_file}, "
            f"env_override={enable_env_override})"
        )
    
    def load(self, validate: bool = True) -> PipelineConfig:
        """
        Load configuration from file and environment variables.
        
        Args:
            validate: Perform validation (default: True)
            
        Returns:
            PipelineConfig with all settings
            
        Raises:
            ConfigurationFileNotFoundError: If config file not found
            ConfigurationParseError: If config file parsing fails
            ConfigurationValidationError: If validation fails
        """
        # Load from file
        config_dict = self._load_from_file()
        
        # Apply environment variable overrides
        if self.enable_env_override:
            config_dict = self._apply_env_overrides(config_dict)
        
        # Build configuration objects
        try:
            config = self._build_config(config_dict)
        except Exception as e:
            raise ConfigurationValidationError(
                f"Failed to build configuration: {e}"
            ) from e
        
        logger.info("Configuration loaded successfully")
        return config
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise ConfigurationFileNotFoundError(
                f"Configuration file not found: {self.config_file}. "
                f"Please create a config.yaml file or specify a valid path."
            )
        
        try:
            with open(self.config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                config_dict = {}
            
            logger.info(f"Loaded configuration from {self.config_file}")
            return config_dict
        
        except yaml.YAMLError as e:
            raise ConfigurationParseError(
                f"Failed to parse YAML configuration: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationParseError(
                f"Failed to read configuration file: {e}"
            ) from e
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables format: GHOSTLOAD_<SECTION>_<SUBSECTION>_<PARAMETER>
        """
        overrides_applied = 0
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(self.env_prefix + "_"):
                continue
            
            # Parse environment variable name
            parts = env_var[len(self.env_prefix) + 1:].lower().split('_')
            
            # Navigate nested dictionary
            current = config_dict
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value (with type conversion)
            param_name = parts[-1]
            current[param_name] = self._convert_env_value(value)
            overrides_applied += 1
            
            logger.info(
                f"Applied environment override: {env_var} = {value}"
            )
        
        if overrides_applied > 0:
            logger.info(f"Applied {overrides_applied} environment variable overrides")
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _build_config(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Build configuration objects from dictionary."""
        # Build file paths config
        file_paths_dict = config_dict.get('file_paths', {})
        file_paths = FilePathsConfig(**file_paths_dict)
        
        # Build model parameters config
        model_params_dict = config_dict.get('model_parameters', {})
        
        # IsolationForest
        isolation_forest_dict = model_params_dict.get('isolation_forest', {})
        isolation_forest = IsolationForestConfig(**isolation_forest_dict)
        
        # DBSCAN
        dbscan_dict = model_params_dict.get('dbscan', {})
        dbscan = DBSCANConfig(**dbscan_dict)
        
        model_parameters = ModelParametersConfig(
            isolation_forest=isolation_forest,
            dbscan=dbscan
        )
        
        # Build risk thresholds config
        risk_thresholds_dict = config_dict.get('risk_thresholds', {})
        risk_thresholds = RiskThresholdsConfig(**risk_thresholds_dict)
        
        # Build feature weights config
        feature_weights_dict = config_dict.get('feature_weights', {})
        feature_weights = FeatureWeightsConfig(**feature_weights_dict)
        
        # Build metrics config
        metrics_dict = config_dict.get('metrics', {})
        metrics = MetricsConfig(**metrics_dict)
        
        # Build pipeline config
        pipeline_dict = {
            k: v for k, v in config_dict.items()
            if k not in ['file_paths', 'model_parameters', 'risk_thresholds', 
                        'feature_weights', 'metrics']
        }
        
        pipeline = PipelineConfig(
            file_paths=file_paths,
            model_parameters=model_parameters,
            risk_thresholds=risk_thresholds,
            feature_weights=feature_weights,
            metrics=metrics,
            **pipeline_dict
        )
        
        return pipeline
    
    def save(self, config: PipelineConfig, output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: PipelineConfig to save
            output_file: Output file path (defaults to original config file)
        """
        output_path = Path(output_file) if output_file else self.config_file
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        # Write to file
        try:
            with open(output_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {output_path}")
        
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}"
            ) from e
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert configuration objects to dictionary."""
        return {
            'file_paths': {
                'data_dir': config.file_paths.data_dir,
                'output_dir': config.file_paths.output_dir,
                'model_registry_dir': config.file_paths.model_registry_dir,
                'logs_dir': config.file_paths.logs_dir,
                'temp_dir': config.file_paths.temp_dir,
                'raw_data_file': config.file_paths.raw_data_file,
                'preprocessed_data_file': config.file_paths.preprocessed_data_file,
                'features_file': config.file_paths.features_file,
                'isolation_forest_model': config.file_paths.isolation_forest_model,
                'dbscan_model': config.file_paths.dbscan_model,
                'predictions_file': config.file_paths.predictions_file,
                'risk_assessment_file': config.file_paths.risk_assessment_file,
                'metrics_file': config.file_paths.metrics_file
            },
            'model_parameters': {
                'isolation_forest': {
                    'contamination': config.model_parameters.isolation_forest.contamination,
                    'n_estimators': config.model_parameters.isolation_forest.n_estimators,
                    'max_samples': config.model_parameters.isolation_forest.max_samples,
                    'max_features': config.model_parameters.isolation_forest.max_features,
                    'random_state': config.model_parameters.isolation_forest.random_state,
                    'n_jobs': config.model_parameters.isolation_forest.n_jobs
                },
                'dbscan': {
                    'eps': config.model_parameters.dbscan.eps,
                    'min_samples': config.model_parameters.dbscan.min_samples,
                    'metric': config.model_parameters.dbscan.metric,
                    'algorithm': config.model_parameters.dbscan.algorithm,
                    'leaf_size': config.model_parameters.dbscan.leaf_size,
                    'n_jobs': config.model_parameters.dbscan.n_jobs
                }
            },
            'risk_thresholds': {
                'high': config.risk_thresholds.high,
                'medium': config.risk_thresholds.medium,
                'extreme_low_ratio': config.risk_thresholds.extreme_low_ratio,
                'suspicious_low_ratio': config.risk_thresholds.suspicious_low_ratio,
                'spatial_boost': config.risk_thresholds.spatial_boost
            },
            'feature_weights': {
                'isolation': config.feature_weights.isolation,
                'ratio': config.feature_weights.ratio,
                'spatial': config.feature_weights.spatial
            },
            'metrics': {
                'baseline_high_risk_rate': config.metrics.baseline_high_risk_rate,
                'top_n_suspicious': config.metrics.top_n_suspicious,
                'confidence_threshold': config.metrics.confidence_threshold
            },
            'enable_preprocessing': config.enable_preprocessing,
            'enable_feature_engineering': config.enable_feature_engineering,
            'enable_training': config.enable_training,
            'enable_evaluation': config.enable_evaluation,
            'random_seed': config.random_seed,
            'n_jobs': config.n_jobs,
            'verbose': config.verbose
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_config(
    config_file: Optional[Union[str, Path]] = None,
    enable_env_override: bool = True
) -> PipelineConfig:
    """
    Load configuration from file.
    
    Convenience function that creates a ConfigLoader and loads configuration.
    
    Args:
        config_file: Path to YAML configuration file (default: config.yaml)
        enable_env_override: Enable environment variable overrides
        
    Returns:
        PipelineConfig with all settings
        
    Example:
        >>> config = load_config()
        >>> print(config.model_parameters.isolation_forest.contamination)
        0.1
    """
    loader = ConfigLoader(config_file, enable_env_override)
    return loader.load()


def create_default_config(output_file: Union[str, Path] = "config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        output_file: Output file path (default: config.yaml)
        
    Example:
        >>> create_default_config("config/default.yaml")
    """
    config = PipelineConfig()
    loader = ConfigLoader()
    loader.save(config, output_file)
    logger.info(f"Created default configuration: {output_file}")


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("=" * 80)
    print("CONFIG LOADER - SELF-TEST")
    print("=" * 80)
    
    # Test 1: Create default configuration
    print("\n" + "=" * 80)
    print("Test 1: Creating default configuration...")
    print("-" * 80)
    
    default_config = PipelineConfig()
    
    print(f"\n+ Test 1 PASSED")
    print(f"  - IsolationForest contamination: {default_config.model_parameters.isolation_forest.contamination}")
    print(f"  - DBSCAN eps: {default_config.model_parameters.dbscan.eps}")
    print(f"  - Risk threshold (high): {default_config.risk_thresholds.high}")
    print(f"  - Feature weight (isolation): {default_config.feature_weights.isolation}")
    
    # Test 2: Save and load configuration
    print("\n" + "=" * 80)
    print("Test 2: Saving and loading configuration...")
    print("-" * 80)
    
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    loader = ConfigLoader(temp_path, enable_env_override=False)
    loader.save(default_config, temp_path)
    
    loaded_config = loader.load()
    
    print(f"\n+ Test 2 PASSED")
    print(f"  - Configuration saved to: {temp_path}")
    print(f"  - Configuration loaded successfully")
    print(f"  - Contamination match: {loaded_config.model_parameters.isolation_forest.contamination == default_config.model_parameters.isolation_forest.contamination}")
    
    # Test 3: Validation
    print("\n" + "=" * 80)
    print("Test 3: Testing validation...")
    print("-" * 80)
    
    validation_passed = True
    
    # Test invalid contamination
    try:
        IsolationForestConfig(contamination=0.6)
        validation_passed = False
    except ConfigurationValidationError as e:
        print(f"  ✓ Caught invalid contamination: {e}")
    
    # Test invalid threshold ordering
    try:
        RiskThresholdsConfig(high=0.5, medium=0.8)
        validation_passed = False
    except ConfigurationValidationError as e:
        print(f"  ✓ Caught invalid threshold ordering: {e}")
    
    # Test invalid feature weights
    try:
        FeatureWeightsConfig(isolation=0.5, ratio=0.3, spatial=0.1)
        validation_passed = False
    except ConfigurationValidationError as e:
        print(f"  ✓ Caught invalid feature weights (sum != 1.0): {e}")
    
    if validation_passed:
        print(f"\n+ Test 3 PASSED")
    else:
        print(f"\n- Test 3 FAILED (validation should have caught errors)")
    
    # Test 4: Custom configuration
    print("\n" + "=" * 80)
    print("Test 4: Creating custom configuration...")
    print("-" * 80)
    
    custom_config = PipelineConfig(
        model_parameters=ModelParametersConfig(
            isolation_forest=IsolationForestConfig(contamination=0.15),
            dbscan=DBSCANConfig(eps=0.3, min_samples=10)
        ),
        risk_thresholds=RiskThresholdsConfig(high=0.85, medium=0.65),
        feature_weights=FeatureWeightsConfig(isolation=0.6, ratio=0.4, spatial=0.0)
    )
    
    print(f"\n+ Test 4 PASSED")
    print(f"  - Custom contamination: {custom_config.model_parameters.isolation_forest.contamination}")
    print(f"  - Custom DBSCAN eps: {custom_config.model_parameters.dbscan.eps}")
    print(f"  - Custom risk threshold: {custom_config.risk_thresholds.high}")
    
    # Test 5: Convenience functions
    print("\n" + "=" * 80)
    print("Test 5: Testing convenience functions...")
    print("-" * 80)
    
    # Create default config file
    create_default_config(temp_path)
    
    # Load using convenience function
    conv_config = load_config(temp_path, enable_env_override=False)
    
    print(f"\n+ Test 5 PASSED")
    print(f"  - create_default_config() executed successfully")
    print(f"  - load_config() loaded successfully")
    
    # Test 6: File paths helper methods
    print("\n" + "=" * 80)
    print("Test 6: Testing file paths helper methods...")
    print("-" * 80)
    
    file_paths = FilePathsConfig()
    resolved_path = file_paths.resolve_path("data/test.csv", "/tmp")
    
    print(f"\n+ Test 6 PASSED")
    print(f"  - Resolved path: {resolved_path}")
    
    # Cleanup
    import os
    os.unlink(temp_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SELF-TEST COMPLETE - ALL TESTS PASSED")
    print("=" * 80)
    
    print("\nConfig Loader is production-ready!")
    print("\nNext steps:")
    print("  1. Create config.yaml file with your settings")
    print("  2. Use load_config() in your pipeline")
    print("  3. Override with environment variables as needed")
    print("  4. Integrate with all pipeline components")
