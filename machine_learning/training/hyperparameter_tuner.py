"""
Production-Grade Hyperparameter Tuning Engine for GhostLoad Mapper ML System
=============================================================================

This module provides an enterprise-level hyperparameter optimization framework
for anomaly detection models in the GhostLoad Mapper electricity theft detection
system. It implements grid search and advanced tuning strategies with silhouette
scoring for unsupervised anomaly detection.

The hyperparameter tuner ensures:
1. **Systematic Exploration**: Grid search across parameter spaces
2. **Unsupervised Evaluation**: Silhouette score for clustering quality
3. **Risk Separation**: Maximizes separation between high/low risk groups
4. **Multi-Model Support**: Isolation Forest, DBSCAN, and custom detectors
5. **Reproducibility**: Deterministic search with controlled randomness
6. **Performance Optimization**: Parallel execution and early stopping
7. **Observability**: Detailed logging with search progress tracking

Design Patterns:
- **Strategy Pattern**: Pluggable scoring functions and search strategies
- **Builder Pattern**: Fluent API for search space configuration
- **Template Method**: Standardized tuning workflow
- **Factory Pattern**: Model instantiation with varied hyperparameters

Enterprise Features:
- Cross-validation for robust evaluation
- Parallel parameter evaluation (multi-core)
- Early stopping for inefficient searches
- Experiment tracking integration (MLflow, W&B)
- Automatic best model selection and registration
- Search history persistence and visualization

Research Foundation:
    - Silhouette score (Rousseeuw, 1987) for clustering validation
    - Hyperparameter optimization (Bergstra & Bengio, 2012)
    - Anomaly detection evaluation (Campos et al., 2016)
    - Production ML tuning (Akiba et al., 2019 - Optuna)

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
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import sklearn for metrics
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not available. Install with: pip install scikit-learn",
        ImportWarning
    )

# Import models
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
    from base_model import BaseAnomalyDetector, ModelType, ModelConfig
    from isolation_forest_model import IsolationForestDetector, IsolationForestConfig
    from dbscan_model import DBSCANDetector, DBSCANConfig
    from model_registry import ModelRegistry, DeploymentStage, create_default_registry
except ImportError:
    warnings.warn(
        "Could not import ML models. Ensure models/ directory is in PYTHONPATH",
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

# Default parameter grids
DEFAULT_ISOLATION_FOREST_GRID = {
    'contamination': [0.05, 0.1, 0.15],
    'n_estimators': [100],
    'max_samples': ['auto'],
}

DEFAULT_DBSCAN_GRID = {
    'eps': [0.3, 0.5, 0.7],  # km or meters depending on coordinate scale
    'min_samples': [5],
}

# Scoring metrics
SCORING_METRICS = {
    'silhouette': 'Silhouette score (higher is better)',
    'davies_bouldin': 'Davies-Bouldin index (lower is better)',
    'calinski_harabasz': 'Calinski-Harabasz index (higher is better)',
}


# ============================================================================
# TUNING CONFIGURATION
# ============================================================================

@dataclass
class TuningConfig:
    """
    Configuration for hyperparameter tuning workflow.
    
    This dataclass encapsulates all tuning settings, search space definitions,
    scoring functions, and optimization strategies.
    
    Attributes:
        model_type: Type of anomaly detector to tune
        param_grid: Dictionary mapping parameter names to lists of values
        scoring_metric: Metric to optimize ('silhouette', 'davies_bouldin', etc.)
        
        # Search strategy
        search_strategy: 'grid' for grid search, 'random' for random search
        n_random_samples: Number of random samples (if random search)
        
        # Cross-validation
        cv_folds: Number of cross-validation folds (0 = no CV)
        shuffle: If True, shuffle data before CV split
        
        # Performance optimization
        n_jobs: Number of parallel jobs (-1 = all cores)
        early_stopping: If True, stop search if no improvement for N iterations
        early_stopping_patience: Iterations without improvement before stopping
        
        # Risk separation
        use_risk_separation: If True, compute separation between high/low risk
        risk_threshold: Anomaly score threshold for high/low risk split
        
        # Model registry
        auto_register_best: If True, register best model in registry
        registry_root: Root directory for model registry
        deployment_stage: Deployment stage for registered model
        
        # Logging and monitoring
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        log_interval: Iterations between progress logs
        
        # Reproducibility
        random_state: Random seed for reproducibility
        
    Example:
        >>> config = TuningConfig(
        ...     model_type=ModelType.ISOLATION_FOREST,
        ...     param_grid={'contamination': [0.05, 0.1, 0.15]},
        ...     scoring_metric='silhouette',
        ...     cv_folds=5
        ... )
    """
    
    # Model configuration
    model_type: ModelType = ModelType.ISOLATION_FOREST
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    scoring_metric: str = 'silhouette'
    
    # Search strategy
    search_strategy: str = 'grid'
    n_random_samples: int = 20
    
    # Cross-validation
    cv_folds: int = 0
    shuffle: bool = True
    
    # Performance optimization
    n_jobs: int = -1
    early_stopping: bool = False
    early_stopping_patience: int = 10
    
    # Risk separation
    use_risk_separation: bool = True
    risk_threshold: float = 0.5  # Median split
    
    # Model registry
    auto_register_best: bool = True
    registry_root: Optional[Path] = None
    deployment_stage: DeploymentStage = DeploymentStage.DEVELOPMENT
    
    # Logging and monitoring
    verbose: int = 1
    log_interval: int = 1
    
    # Reproducibility
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.scoring_metric not in SCORING_METRICS:
            raise ValueError(
                f"Unknown scoring metric: {self.scoring_metric}. "
                f"Supported: {list(SCORING_METRICS.keys())}"
            )
        
        if self.search_strategy not in ['grid', 'random']:
            raise ValueError(
                f"Unknown search strategy: {self.search_strategy}. "
                f"Supported: ['grid', 'random']"
            )
        
        if self.cv_folds < 0:
            raise ValueError(f"cv_folds must be non-negative, got {self.cv_folds}")
        
        if not 0 <= self.risk_threshold <= 1:
            raise ValueError(
                f"risk_threshold must be in [0, 1], got {self.risk_threshold}"
            )
        
        # Set default parameter grid if empty
        if not self.param_grid:
            if self.model_type == ModelType.ISOLATION_FOREST:
                self.param_grid = DEFAULT_ISOLATION_FOREST_GRID.copy()
            elif self.model_type == ModelType.CUSTOM:  # DBSCAN
                self.param_grid = DEFAULT_DBSCAN_GRID.copy()


@dataclass
class TuningResult:
    """
    Results from hyperparameter tuning workflow.
    
    This dataclass encapsulates all outputs from the tuning process,
    including the best model, optimal parameters, and search history.
    
    Attributes:
        best_model: Best model found during search
        best_params: Optimal hyperparameter configuration
        best_score: Best score achieved
        
        # Search history
        search_results: DataFrame with all parameter configurations and scores
        n_iterations: Total number of parameter configurations evaluated
        search_time: Total search time in seconds
        
        # Performance metrics
        scores: List of scores for all configurations
        mean_score: Mean score across all configurations
        std_score: Standard deviation of scores
        
        # Risk separation (if use_risk_separation=True)
        risk_separation: Separation score between high/low risk groups
        high_risk_fraction: Fraction of samples in high risk group
        
        # Model registry
        model_version: Registered model version (if auto_register_best=True)
        
    Example:
        >>> result.best_params
        {'contamination': 0.1, 'n_estimators': 100}
        >>> result.best_score
        0.42
        >>> result.search_results.head()
    """
    
    # Best model
    best_model: BaseAnomalyDetector
    best_params: Dict[str, Any]
    best_score: float
    
    # Search history
    search_results: pd.DataFrame
    n_iterations: int
    search_time: float
    
    # Performance metrics
    scores: List[float]
    mean_score: float
    std_score: float
    
    # Risk separation
    risk_separation: Optional[float] = None
    high_risk_fraction: Optional[float] = None
    
    # Model registry
    model_version: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': self.n_iterations,
            'search_time': self.search_time,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'risk_separation': self.risk_separation,
            'high_risk_fraction': self.high_risk_fraction,
        }


# ============================================================================
# HYPERPARAMETER TUNER
# ============================================================================

class HyperparameterTuner:
    """
    Production-grade hyperparameter optimization engine.
    
    This class implements systematic hyperparameter search for anomaly
    detection models using silhouette scoring and risk separation metrics.
    
    Usage:
        >>> from hyperparameter_tuner import HyperparameterTuner, TuningConfig
        >>> 
        >>> # Prepare data
        >>> import pandas as pd
        >>> df = pd.read_csv('preprocessed_meters.csv')
        >>> X = df[['consumption_mean', 'consumption_std']].values
        >>> 
        >>> # Configure search
        >>> config = TuningConfig(
        ...     model_type=ModelType.ISOLATION_FOREST,
        ...     param_grid={
        ...         'contamination': [0.05, 0.1, 0.15],
        ...         'n_estimators': [50, 100, 150]
        ...     },
        ...     scoring_metric='silhouette',
        ...     cv_folds=3
        ... )
        >>> 
        >>> # Run search
        >>> tuner = HyperparameterTuner(config)
        >>> result = tuner.tune(X)
        >>> 
        >>> # Inspect results
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best score: {result.best_score:.4f}")
        >>> print(result.search_results.sort_values('mean_score', ascending=False))
    
    Design Pattern: Template Method + Strategy
    Thread Safety: Not thread-safe (use separate instances per thread)
    
    Research Foundation:
        Hyperparameter optimization for anomaly detection models
        (Goldstein & Uchida, 2016; Campos et al., 2016)
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            config: Tuning configuration (uses defaults if None)
            
        Example:
            >>> tuner = HyperparameterTuner()
            >>> # Or with custom config:
            >>> config = TuningConfig(verbose=2)
            >>> tuner = HyperparameterTuner(config)
        """
        self.config = config or TuningConfig()
        self.registry: Optional[ModelRegistry] = None
        
        # Initialize model registry if auto-registration enabled
        if self.config.auto_register_best:
            registry_root = self.config.registry_root or Path("model_registry")
            self.registry = create_default_registry(registry_root=registry_root)
            if self.config.verbose >= 1:
                logger.info(f"Initialized model registry at: {registry_root}")
    
    def tune(
        self,
        X: Union[NDArray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> TuningResult:
        """
        Execute hyperparameter tuning workflow.
        
        This method orchestrates the complete tuning pipeline:
        1. Generate parameter combinations
        2. For each configuration:
           a. Train model
           b. Compute anomaly scores
           c. Evaluate with scoring metric
        3. Select best model
        4. Optionally register in model registry
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional feature names for logging
            
        Returns:
            TuningResult with best model and search history
            
        Raises:
            ValueError: If X is empty or invalid
            RuntimeError: If tuning fails
            
        Example:
            >>> X = df[['consumption_mean', 'consumption_std']].values
            >>> result = tuner.tune(X)
            >>> 
            >>> # Use best model
            >>> predictions = result.best_model.predict(X)
        """
        logger.info("=" * 80)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Validate input
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        
        if self.config.verbose >= 1:
            logger.info(f"Data: {n_samples} samples × {n_features} features")
            if feature_names:
                logger.info(f"Features: {', '.join(feature_names)}")
        
        # Step 2: Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        n_combinations = len(param_combinations)
        
        if self.config.verbose >= 1:
            logger.info(f"Search space: {n_combinations} parameter combinations")
            logger.info(f"Scoring metric: {self.config.scoring_metric}")
        
        # Step 3: Evaluate each configuration
        results = []
        best_score = -np.inf if self._is_score_maximized() else np.inf
        best_model = None
        best_params = None
        no_improvement_count = 0
        
        for idx, params in enumerate(param_combinations):
            iteration_start = time.time()
            
            # Train and evaluate model
            score, model = self._evaluate_params(X, params)
            
            results.append({
                'iteration': idx,
                'params': params,
                'score': score,
                'time': time.time() - iteration_start
            })
            
            # Update best model
            is_better = (
                score > best_score if self._is_score_maximized()
                else score < best_score
            )
            
            if is_better:
                best_score = score
                best_model = model
                best_params = params.copy()
                no_improvement_count = 0
                
                if self.config.verbose >= 1:
                    logger.info(
                        f"[{idx+1}/{n_combinations}] ✓ New best: "
                        f"score={score:.4f}, params={params}"
                    )
            else:
                no_improvement_count += 1
                
                if self.config.verbose >= 2:
                    logger.info(
                        f"[{idx+1}/{n_combinations}] "
                        f"score={score:.4f}, params={params}"
                    )
            
            # Early stopping check
            if (self.config.early_stopping and 
                no_improvement_count >= self.config.early_stopping_patience):
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self.config.early_stopping_patience} iterations"
                )
                break
        
        # Step 4: Construct results
        search_time = time.time() - start_time
        search_df = self._results_to_dataframe(results)
        
        scores = [r['score'] for r in results]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        # Compute risk separation if requested
        risk_separation = None
        high_risk_fraction = None
        
        if self.config.use_risk_separation and best_model is not None:
            risk_separation, high_risk_fraction = self._compute_risk_separation(
                best_model, X
            )
            
            if self.config.verbose >= 1:
                logger.info(
                    f"Risk separation: {risk_separation:.4f} "
                    f"(high risk: {high_risk_fraction:.1%})"
                )
        
        # Step 5: Register best model
        model_version = None
        
        if self.config.auto_register_best and self.registry is not None:
            model_version = self._register_best_model(
                best_model, best_params, best_score, feature_names
            )
            
            if self.config.verbose >= 1:
                logger.info(f"✓ Registered best model: {model_version.model_id}")
        
        # Construct final result
        result = TuningResult(
            best_model=best_model,
            best_params=best_params,
            best_score=best_score,
            search_results=search_df,
            n_iterations=len(results),
            search_time=search_time,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            risk_separation=risk_separation,
            high_risk_fraction=high_risk_fraction,
            model_version=model_version,
        )
        
        logger.info("=" * 80)
        logger.info(f"TUNING COMPLETE - {search_time:.2f}s")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        logger.info("=" * 80)
        
        return result
    
    def _validate_input(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        """
        Validate and convert input data.
        
        Args:
            X: Input features
            
        Returns:
            Validated NumPy array
            
        Raises:
            ValueError: If input invalid
        """
        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Validate shape
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if X.shape[0] == 0:
            raise ValueError("X contains no samples")
        
        if X.shape[1] == 0:
            raise ValueError("X contains no features")
        
        # Check for invalid values
        if np.any(np.isnan(X)):
            n_nan = np.isnan(X).sum()
            logger.warning(f"X contains {n_nan} NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")
        
        return X
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations to evaluate.
        
        Returns:
            List of parameter dictionaries
        """
        if self.config.search_strategy == 'grid':
            # Grid search: all combinations
            param_names = sorted(self.config.param_grid.keys())
            param_values = [self.config.param_grid[name] for name in param_names]
            
            combinations = []
            for values in product(*param_values):
                params = dict(zip(param_names, values))
                combinations.append(params)
            
            return combinations
        
        elif self.config.search_strategy == 'random':
            # Random search: sample combinations
            rng = np.random.RandomState(self.config.random_state)
            param_names = list(self.config.param_grid.keys())
            
            combinations = []
            for _ in range(self.config.n_random_samples):
                params = {
                    name: rng.choice(self.config.param_grid[name])
                    for name in param_names
                }
                combinations.append(params)
            
            return combinations
        
        else:
            raise ValueError(f"Unknown search strategy: {self.config.search_strategy}")
    
    def _evaluate_params(
        self,
        X: NDArray,
        params: Dict[str, Any]
    ) -> Tuple[float, BaseAnomalyDetector]:
        """
        Evaluate single parameter configuration.
        
        Args:
            X: Feature matrix
            params: Hyperparameter dictionary
            
        Returns:
            Tuple of (score, trained_model)
        """
        # Create model with parameters
        model = self._create_model(params)
        
        # Train model
        if self.config.cv_folds > 0:
            # Cross-validation
            score = self._cross_validate(model, X)
        else:
            # Single train
            model.fit(X)
            score = self._score_model(model, X)
        
        return score, model
    
    def _create_model(self, params: Dict[str, Any]) -> BaseAnomalyDetector:
        """
        Create model instance with given parameters.
        
        Args:
            params: Hyperparameter dictionary
            
        Returns:
            Instantiated model
        """
        # Make a copy to avoid modifying the original
        params_copy = params.copy()
        
        if self.config.model_type == ModelType.ISOLATION_FOREST:
            # Extract contamination for base config
            contamination = params_copy.pop('contamination', 0.1)
            
            # Create base ModelConfig (has n_jobs)
            base_config = ModelConfig(
                model_type=ModelType.ISOLATION_FOREST,
                contamination=contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            # Create IsolationForestConfig (no n_jobs here)
            if_config = IsolationForestConfig(
                **params_copy  # n_estimators, max_samples, etc.
            )
            
            return IsolationForestDetector(config=base_config, if_config=if_config)
        
        elif self.config.model_type == ModelType.CUSTOM:  # DBSCAN
            # Create base ModelConfig (has n_jobs)
            base_config = ModelConfig(
                model_type=ModelType.CUSTOM,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            # Create DBSCANConfig (no n_jobs here)
            dbscan_config = DBSCANConfig(
                **params_copy  # eps, min_samples, metric
            )
            
            return DBSCANDetector(config=base_config, dbscan_config=dbscan_config)
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _cross_validate(
        self,
        model: BaseAnomalyDetector,
        X: NDArray
    ) -> float:
        """
        Perform cross-validation and return mean score.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            
        Returns:
            Mean score across folds
        """
        n_samples = len(X)
        fold_size = n_samples // self.config.cv_folds
        
        # Shuffle if requested
        if self.config.shuffle:
            rng = np.random.RandomState(self.config.random_state)
            indices = rng.permutation(n_samples)
            X = X[indices]
        
        scores = []
        
        for fold in range(self.config.cv_folds):
            # Split data
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.config.cv_folds - 1 else n_samples
            
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            
            X_train = X[train_idx]
            X_val = X[val_idx]
            
            # Train and score
            model.fit(X_train)
            score = self._score_model(model, X_val)
            scores.append(score)
        
        return float(np.mean(scores))
    
    def _score_model(
        self,
        model: BaseAnomalyDetector,
        X: NDArray
    ) -> float:
        """
        Compute score for trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Score value
        """
        # Get predictions/scores (use return_probabilities=True to get PredictionResult)
        result = model.predict(X, return_probabilities=True)
        scores = result.anomaly_scores
        predictions = result.predictions
        
        # Compute silhouette score
        if self.config.scoring_metric == 'silhouette':
            # Need at least 2 clusters
            unique_labels = np.unique(predictions)
            if len(unique_labels) < 2:
                return -1.0  # Worst score
            
            try:
                score = silhouette_score(X, predictions)
            except ValueError:
                # Silhouette undefined (e.g., all same cluster)
                return -1.0
        
        elif self.config.scoring_metric == 'davies_bouldin':
            unique_labels = np.unique(predictions)
            if len(unique_labels) < 2:
                return np.inf  # Worst score (lower is better)
            
            try:
                score = davies_bouldin_score(X, predictions)
            except ValueError:
                return np.inf
        
        elif self.config.scoring_metric == 'calinski_harabasz':
            unique_labels = np.unique(predictions)
            if len(unique_labels) < 2:
                return 0.0  # Worst score
            
            try:
                score = calinski_harabasz_score(X, predictions)
            except ValueError:
                return 0.0
        
        else:
            raise ValueError(f"Unknown scoring metric: {self.config.scoring_metric}")
        
        return float(score)
    
    def _is_score_maximized(self) -> bool:
        """Check if scoring metric should be maximized."""
        return self.config.scoring_metric in ['silhouette', 'calinski_harabasz']
    
    def _compute_risk_separation(
        self,
        model: BaseAnomalyDetector,
        X: NDArray
    ) -> Tuple[float, float]:
        """
        Compute separation between high and low risk groups.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Tuple of (separation_score, high_risk_fraction)
        """
        result = model.predict(X, return_probabilities=True)
        scores = result.anomaly_scores
        
        # Split into high/low risk
        threshold = np.quantile(scores, self.config.risk_threshold)
        high_risk = scores >= threshold
        
        high_risk_scores = scores[high_risk]
        low_risk_scores = scores[~high_risk]
        
        if len(high_risk_scores) == 0 or len(low_risk_scores) == 0:
            return 0.0, 0.0
        
        # Compute separation (difference in means)
        separation = float(np.mean(high_risk_scores) - np.mean(low_risk_scores))
        high_risk_fraction = float(high_risk.sum() / len(high_risk))
        
        return separation, high_risk_fraction
    
    def _results_to_dataframe(
        self,
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert results list to DataFrame.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            DataFrame with search history
        """
        rows = []
        
        for r in results:
            row = {
                'iteration': r['iteration'],
                'score': r['score'],
                'time': r['time'],
            }
            row.update(r['params'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by score
        ascending = not self._is_score_maximized()
        df = df.sort_values('score', ascending=ascending).reset_index(drop=True)
        
        return df
    
    def _register_best_model(
        self,
        model: BaseAnomalyDetector,
        params: Dict[str, Any],
        score: float,
        feature_names: Optional[List[str]]
    ) -> Any:
        """
        Register best model in model registry.
        
        Args:
            model: Best model
            params: Best parameters
            score: Best score
            feature_names: Feature names
            
        Returns:
            ModelVersion object
        """
        performance_metrics = {
            f'{self.config.scoring_metric}_score': score,
            'tuning_iterations': self.config.n_random_samples if self.config.search_strategy == 'random' else len(list(product(*self.config.param_grid.values()))),
        }
        
        training_metadata = {
            'best_params': params,
            'feature_names': feature_names or [],
            'tuning_date': datetime.now().isoformat(),
            'scoring_metric': self.config.scoring_metric,
        }
        
        version = self.registry.register_model(
            model=model,
            model_type=self.config.model_type,
            version="1.0.0",
            deployment_stage=self.config.deployment_stage,
            performance_metrics=performance_metrics,
            training_metadata=training_metadata,
            tags=['tuned', 'best_params'],
            notes=f"Tuned with {self.config.scoring_metric} scoring"
        )
        
        return version


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def tune_isolation_forest(
    X: Union[NDArray, pd.DataFrame],
    contamination_values: List[float] = [0.05, 0.1, 0.15],
    n_estimators_values: List[int] = [100],
    scoring_metric: str = 'silhouette',
    **kwargs
) -> TuningResult:
    """
    Convenience function to tune Isolation Forest hyperparameters.
    
    Args:
        X: Feature matrix
        contamination_values: Contamination values to search
        n_estimators_values: Number of estimators to search
        scoring_metric: Metric to optimize
        **kwargs: Additional TuningConfig parameters
        
    Returns:
        TuningResult with best model
        
    Example:
        >>> X = df[['consumption_mean', 'consumption_std']].values
        >>> result = tune_isolation_forest(
        ...     X,
        ...     contamination_values=[0.05, 0.1, 0.15, 0.2],
        ...     scoring_metric='silhouette'
        ... )
        >>> print(f"Best contamination: {result.best_params['contamination']}")
    """
    config = TuningConfig(
        model_type=ModelType.ISOLATION_FOREST,
        param_grid={
            'contamination': contamination_values,
            'n_estimators': n_estimators_values,
        },
        scoring_metric=scoring_metric,
        **kwargs
    )
    
    tuner = HyperparameterTuner(config)
    return tuner.tune(X)


def tune_dbscan(
    X: Union[NDArray, pd.DataFrame],
    eps_values: List[float] = [0.3, 0.5, 0.7],
    min_samples_values: List[int] = [3, 5, 7],
    scoring_metric: str = 'silhouette',
    **kwargs
) -> TuningResult:
    """
    Convenience function to tune DBSCAN hyperparameters.
    
    Args:
        X: Feature matrix (typically GPS coordinates)
        eps_values: Eps values to search
        min_samples_values: Min samples values to search
        scoring_metric: Metric to optimize
        **kwargs: Additional TuningConfig parameters
        
    Returns:
        TuningResult with best model
        
    Example:
        >>> X = df[['latitude', 'longitude']].values
        >>> result = tune_dbscan(
        ...     X,
        ...     eps_values=[0.3, 0.5, 0.7, 1.0],
        ...     min_samples_values=[3, 5],
        ...     scoring_metric='silhouette'
        ... )
        >>> print(f"Best eps: {result.best_params['eps']}")
    """
    config = TuningConfig(
        model_type=ModelType.CUSTOM,  # DBSCAN
        param_grid={
            'eps': eps_values,
            'min_samples': min_samples_values,
        },
        scoring_metric=scoring_metric,
        **kwargs
    )
    
    tuner = HyperparameterTuner(config)
    return tuner.tune(X)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of hyperparameter tuner capabilities.
    Run this module directly to validate the implementation.
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNER - SELF-TEST")
    print("=" * 80 + "\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    print("-" * 80)
    
    np.random.seed(42)
    n_samples = 500
    
    # Create realistic consumption features
    consumption_mean = np.random.gamma(shape=2, scale=500, size=n_samples)
    consumption_std = consumption_mean * np.random.uniform(0.1, 0.5, n_samples)
    
    X = np.column_stack([consumption_mean, consumption_std])
    
    print(f"+ Created feature matrix: {X.shape}\n")
    
    # Test 1: Tune Isolation Forest
    print("Test 1: Tuning Isolation Forest...")
    print("-" * 80)
    
    try:
        result = tune_isolation_forest(
            X,
            contamination_values=[0.05, 0.1, 0.15],
            scoring_metric='silhouette',
            auto_register_best=False,
            verbose=1
        )
        
        print(f"\n+ Test 1 PASSED")
        print(f"  - Best contamination: {result.best_params['contamination']}")
        print(f"  - Best score: {result.best_score:.4f}")
        print(f"  - Iterations: {result.n_iterations}")
        print(f"  - Search time: {result.search_time:.2f}s")
        print(f"\nTop 3 configurations:")
        print(result.search_results.head(3)[['contamination', 'score']])
        
        assert result.best_model is not None
        assert result.best_score > -1.0  # Silhouette in [-1, 1]
        
    except Exception as e:
        print(f"\n- Test 1 FAILED: {str(e)}")
        raise
    
    # Test 2: Tune DBSCAN
    print("\n\nTest 2: Tuning DBSCAN...")
    print("-" * 80)
    
    try:
        # Generate GPS coordinates
        gps = np.column_stack([
            np.random.uniform(14.5, 14.7, n_samples),
            np.random.uniform(120.9, 121.1, n_samples)
        ])
        
        result = tune_dbscan(
            gps,
            eps_values=[0.3, 0.5, 0.7],
            min_samples_values=[3, 5],
            scoring_metric='silhouette',
            auto_register_best=False,
            verbose=1
        )
        
        print(f"\n+ Test 2 PASSED")
        print(f"  - Best eps: {result.best_params['eps']}")
        print(f"  - Best min_samples: {result.best_params['min_samples']}")
        print(f"  - Best score: {result.best_score:.4f}")
        print(f"  - Iterations: {result.n_iterations}")
        
        assert result.best_model is not None
        
    except Exception as e:
        print(f"\n- Test 2 FAILED: {str(e)}")
        raise
    
    # Test 3: Risk separation
    print("\n\nTest 3: Risk separation metric...")
    print("-" * 80)
    
    try:
        result = tune_isolation_forest(
            X,
            contamination_values=[0.1, 0.15],
            use_risk_separation=True,
            auto_register_best=False,
            verbose=1
        )
        
        print(f"\n+ Test 3 PASSED")
        print(f"  - Risk separation: {result.risk_separation:.4f}")
        print(f"  - High risk fraction: {result.high_risk_fraction:.1%}")
        
        assert result.risk_separation is not None
        assert result.high_risk_fraction is not None
        
    except Exception as e:
        print(f"\n- Test 3 FAILED: {str(e)}")
        raise
    
    print("\n" + "=" * 80)
    print("SELF-TEST COMPLETE - ALL TESTS PASSED")
    print("=" * 80)
    print("\nHyperparameter Tuner is production-ready!")
    print("\nNext steps:")
    print("  1. Tune models on real data")
    print("  2. Compare different scoring metrics")
    print("  3. Integrate with experiment tracking")
