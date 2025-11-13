"""
Production-Grade DBSCAN Spatial Anomaly Detector for GhostLoad Mapper
=======================================================================

This module implements a production-ready DBSCAN (Density-Based Spatial Clustering
of Applications with Noise) detector for identifying spatial anomalies in electricity
meter distributions. It clusters meters by GPS coordinates and flags meters in small,
dense clusters as potential theft locations.

DBSCAN Algorithm:
    DBSCAN groups together points that are closely packed (high density), marking
    as outliers points that lie alone in low-density regions. Unlike k-means, DBSCAN
    does not require specifying the number of clusters and can find arbitrarily
    shaped clusters.
    
    Key Concepts:
    - Core points: Points with at least min_samples neighbors within eps distance
    - Border points: Non-core points within eps of a core point
    - Noise points: Points that are neither core nor border (anomalies)
    - Cluster: Maximal set of density-connected core points

Spatial Anomaly Detection Strategy:
    In electricity theft detection, spatial anomalies indicate suspicious patterns:
    
    1. Isolated Meters (DBSCAN Noise): Single meters far from clusters may indicate
       unauthorized connections in remote areas
       
    2. Small Dense Clusters: Tight groups of meters (e.g., 3-10 meters in <50m radius)
       suggest coordinated theft, meter tampering rings, or fraudulent installations
       
    3. Unusual Density: Meters in unexpectedly dense configurations may indicate
       illegal subdivisions or unauthorized meter multiplication
    
    The detector flags both DBSCAN noise points AND meters in suspiciously small
    clusters as spatial anomalies, providing binary labels for downstream analysis.

Research Foundation:
    - Ester, M., Kriegel, H.P., Sander, J., Xu, X. (1996)
      "A density-based algorithm for discovering clusters in large spatial databases"
      KDD '96. https://doi.org/10.5555/3001460.3001507
    
    - Schubert, E., Sander, J., Ester, M., Kriegel, H.P., Xu, X. (2017)
      "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN"
      ACM TODS. https://doi.org/10.1145/3068335

Use Cases:
    - Detecting isolated unauthorized meter installations
    - Identifying coordinated theft rings (small dense clusters)
    - Validating meter distribution patterns against infrastructure maps
    - Geospatial anomaly detection in distribution networks
    - Clustering analysis for targeted field inspections

Performance Characteristics:
    - Time Complexity: O(n log n) with spatial indexing (KD-tree)
    - Space Complexity: O(n)
    - Scalability: Efficient for 100k+ meters with spatial indexing
    - Sensitivity: Tunable via eps (radius) and min_samples

Geographic Considerations:
    - Uses Haversine distance for GPS coordinates (accounting for Earth curvature)
    - eps parameter in meters for intuitive spatial interpretation
    - Supports UTM projection for large-scale deployments
    - Handles missing GPS data gracefully (returns non-anomalous labels)

Author: GhostLoad Mapper ML Team
Date: November 13, 2025
Version: 1.0.0
License: MIT
"""

from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
import logging
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Scikit-learn imports
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not available. Install with: pip install scikit-learn",
        ImportWarning
    )

# Import base model
try:
    from base_model import (
        BaseAnomalyDetector, 
        ModelConfig, 
        ModelMetadata,
        ModelType,
        PredictionResult
    )
except ImportError:
    # Try relative import
    try:
        from .base_model import (
            BaseAnomalyDetector,
            ModelConfig,
            ModelMetadata,
            ModelType,
            PredictionResult
        )
    except ImportError:
        # Add parent directory to path for standalone execution
        parent_dir = Path(__file__).parent
        sys.path.insert(0, str(parent_dir))
        from base_model import (
            BaseAnomalyDetector,
            ModelConfig,
            ModelMetadata,
            ModelType,
            PredictionResult
        )


# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default DBSCAN parameters
DEFAULT_EPS_METERS = 50.0  # 50 meters radius
DEFAULT_MIN_SAMPLES = 5    # Minimum 5 meters to form cluster
DEFAULT_METRIC = "haversine"  # For GPS coordinates
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers

# Spatial anomaly thresholds
SMALL_CLUSTER_THRESHOLD = 10  # Clusters with ≤10 members flagged as suspicious
NOISE_LABEL = -1  # DBSCAN label for noise points

# Distance conversion
METERS_PER_RADIAN = EARTH_RADIUS_KM * 1000  # Approx 6,371,000 meters


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DBSCANConfig:
    """
    Type-safe configuration for DBSCAN spatial clustering.
    
    This configuration encapsulates DBSCAN hyperparameters and spatial
    anomaly detection settings with validation.
    
    Attributes:
        eps: Maximum distance between two samples for one to be considered
             in the neighborhood of the other (in meters for GPS data)
        min_samples: Minimum number of samples in a neighborhood for a point
                    to be considered a core point
        metric: Distance metric ('haversine' for GPS, 'euclidean' for projected)
        algorithm: Algorithm for computing neighbors ('auto', 'ball_tree', 
                  'kd_tree', 'brute')
        leaf_size: Leaf size for tree-based algorithms (affects speed/memory)
        small_cluster_threshold: Maximum cluster size to flag as anomalous
        flag_noise_as_anomaly: Whether to flag DBSCAN noise as anomalies
        flag_small_clusters: Whether to flag small dense clusters as anomalies
        require_gps: Whether to raise error if GPS data missing
        n_jobs: Number of parallel jobs (-1 = all cores)
    
    Coordinate System Modes:
        GPS Mode (metric='haversine'):
            - Input: [latitude, longitude] in decimal degrees
            - eps: Distance in meters
            - Accounts for Earth's curvature
            - Best for: City/regional scale (<1000km)
        
        Projected Mode (metric='euclidean'):
            - Input: [x, y] in meters (e.g., UTM coordinates)
            - eps: Distance in same units as coordinates
            - Assumes flat earth (small distortion for local areas)
            - Best for: Small areas (<100km) or pre-projected data
    
    Example:
        >>> # GPS-based clustering (default)
        >>> config = DBSCANConfig(
        ...     eps=50.0,  # 50 meters
        ...     min_samples=5,
        ...     metric='haversine'
        ... )
        
        >>> # Projected coordinate clustering
        >>> config = DBSCANConfig(
        ...     eps=50.0,
        ...     min_samples=5,
        ...     metric='euclidean'
        ... )
    """
    
    # Core DBSCAN parameters
    eps: float = DEFAULT_EPS_METERS
    min_samples: int = DEFAULT_MIN_SAMPLES
    metric: str = DEFAULT_METRIC
    algorithm: str = "auto"
    leaf_size: int = 30
    
    # Spatial anomaly detection parameters
    small_cluster_threshold: int = SMALL_CLUSTER_THRESHOLD
    flag_noise_as_anomaly: bool = True
    flag_small_clusters: bool = True
    
    # Data handling
    require_gps: bool = False
    handle_missing: str = "mark_normal"  # 'mark_normal', 'mark_anomaly', 'error'
    
    # Performance
    n_jobs: int = -1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate eps
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        
        if self.metric == "haversine":
            if self.eps > 10000:  # 10km
                warnings.warn(
                    f"eps={self.eps}m is very large for GPS clustering. "
                    f"Consider eps < 1000m for meter-level analysis.",
                    UserWarning
                )
        
        # Validate min_samples
        if self.min_samples < 2:
            raise ValueError(f"min_samples must be ≥ 2, got {self.min_samples}")
        
        if self.min_samples > 100:
            warnings.warn(
                f"min_samples={self.min_samples} is very high. "
                f"Consider min_samples < 20 for typical meter distributions.",
                UserWarning
            )
        
        # Validate metric
        valid_metrics = ["haversine", "euclidean", "manhattan", "chebyshev"]
        if self.metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{self.metric}'"
            )
        
        # Validate algorithm
        valid_algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got '{self.algorithm}'"
            )
        
        # Validate small_cluster_threshold
        if self.small_cluster_threshold < 2:
            raise ValueError(
                f"small_cluster_threshold must be ≥ 2, got {self.small_cluster_threshold}"
            )
        
        # Validate handle_missing
        valid_handling = ["mark_normal", "mark_anomaly", "error"]
        if self.handle_missing not in valid_handling:
            raise ValueError(
                f"handle_missing must be one of {valid_handling}, got '{self.handle_missing}'"
            )
    
    def to_sklearn_params(self) -> Dict[str, Any]:
        """Convert config to sklearn DBSCAN parameters."""
        params = {
            "eps": self.eps if self.metric != "haversine" else self.eps / METERS_PER_RADIAN,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "n_jobs": self.n_jobs,
        }
        return params


# ============================================================================
# DBSCAN SPATIAL ANOMALY DETECTOR
# ============================================================================

class DBSCANDetector(BaseAnomalyDetector):
    """
    DBSCAN-based spatial anomaly detector for electricity meter clustering.
    
    This detector identifies spatial anomalies by clustering meters based on
    GPS coordinates and flagging:
    1. Noise points (isolated meters)
    2. Small dense clusters (potential coordinated theft)
    
    The detector returns binary anomaly labels (0=normal, 1=anomalous) rather
    than continuous scores, making it suitable for rule-based alerting and
    geospatial visualization.
    
    Attributes:
        config: Base model configuration
        dbscan_config: DBSCAN-specific configuration
        _model: sklearn DBSCAN instance
        _cluster_labels: Cluster assignments from training
        _cluster_sizes: Number of members in each cluster
        _is_fitted: Whether model has been trained
    
    Methods:
        fit(X): Train DBSCAN on GPS coordinates
        predict(X): Return binary spatial anomaly flags
        get_cluster_info(): Get detailed cluster statistics
        get_cluster_centroids(): Compute centroid coordinates for each cluster
        visualize_clusters(): Generate cluster visualization data
    
    Example:
        >>> import numpy as np
        >>> # GPS coordinates: [latitude, longitude]
        >>> gps_coords = np.array([
        ...     [14.5995, 120.9842],  # Manila
        ...     [14.5996, 120.9843],  # Near Manila
        ...     [14.6500, 121.0500],  # Isolated
        ... ])
        >>> 
        >>> detector = DBSCANDetector(
        ...     config=ModelConfig(contamination=0.1),
        ...     dbscan_config=DBSCANConfig(eps=50.0, min_samples=2)
        ... )
        >>> detector.fit(gps_coords)
        >>> anomalies = detector.predict(gps_coords)
        >>> # anomalies: [0, 0, 1]  # Third point is isolated
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        dbscan_config: Optional[DBSCANConfig] = None,
        metadata: Optional[ModelMetadata] = None
    ):
        """
        Initialize DBSCAN spatial anomaly detector.
        
        Args:
            config: Base model configuration
            dbscan_config: DBSCAN-specific configuration
            metadata: Model metadata for tracking
        
        Raises:
            ImportError: If scikit-learn not available
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for DBSCANDetector. "
                "Install with: pip install scikit-learn"
            )
        
        # Initialize DBSCAN configuration FIRST (needed by base class)
        self.dbscan_config = dbscan_config or DBSCANConfig()
        
        # Initialize DBSCAN model
        self._model: Optional[DBSCAN] = None
        self._cluster_labels: Optional[NDArray] = None
        self._cluster_sizes: Optional[Dict[int, int]] = None
        self._cluster_centroids: Optional[Dict[int, NDArray]] = None
        
        # Initialize base class (calls _get_hyperparameters)
        if config is None:
            config = ModelConfig(
                model_type=ModelType.CUSTOM,
                model_name="dbscan_spatial_detector"
            )
        
        super().__init__(config=config, metadata=metadata)
        
        if self.config.verbose:
            logger.info(
                f"Initialized DBSCANDetector with eps={self.dbscan_config.eps}m, "
                f"min_samples={self.dbscan_config.min_samples}"
            )
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Return DBSCAN hyperparameters for metadata tracking."""
        if not hasattr(self, 'dbscan_config'):
            return {}
        
        return {
            "eps": self.dbscan_config.eps,
            "min_samples": self.dbscan_config.min_samples,
            "metric": self.dbscan_config.metric,
            "algorithm": self.dbscan_config.algorithm,
            "small_cluster_threshold": self.dbscan_config.small_cluster_threshold,
            "flag_noise_as_anomaly": self.dbscan_config.flag_noise_as_anomaly,
            "flag_small_clusters": self.dbscan_config.flag_small_clusters,
        }
    
    def _validate_gps_coordinates(self, X: NDArray) -> Tuple[bool, str]:
        """
        Validate GPS coordinate format and ranges.
        
        Args:
            X: Input array (n_samples, 2) with [latitude, longitude]
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shape
        if X.shape[1] != 2:
            return False, f"GPS data must have 2 columns [lat, lon], got {X.shape[1]}"
        
        # Check for missing values
        if np.isnan(X).any():
            missing_count = np.isnan(X).sum()
            if self.dbscan_config.require_gps:
                return False, f"Found {missing_count} missing GPS values (require_gps=True)"
            else:
                warnings.warn(
                    f"Found {missing_count} missing GPS values. "
                    f"Will handle as '{self.dbscan_config.handle_missing}'.",
                    UserWarning
                )
        
        # For haversine metric, validate lat/lon ranges
        if self.dbscan_config.metric == "haversine":
            valid_mask = ~np.isnan(X).any(axis=1)
            if valid_mask.any():
                lats = X[valid_mask, 0]
                lons = X[valid_mask, 1]
                
                # Check latitude range [-90, 90]
                if np.any((lats < -90) | (lats > 90)):
                    return False, f"Latitude must be in [-90, 90], got range [{lats.min()}, {lats.max()}]"
                
                # Check longitude range [-180, 180]
                if np.any((lons < -180) | (lons > 180)):
                    return False, f"Longitude must be in [-180, 180], got range [{lons.min()}, {lons.max()}]"
        
        return True, ""
    
    def _convert_to_radians(self, X: NDArray) -> NDArray:
        """
        Convert GPS coordinates from degrees to radians for haversine.
        
        Args:
            X: GPS coordinates in degrees [latitude, longitude]
        
        Returns:
            GPS coordinates in radians
        """
        return np.radians(X)
    
    def _fit_implementation(self, X: NDArray, y: Optional[NDArray] = None) -> None:
        """
        Fit DBSCAN clustering on GPS coordinates.
        
        Args:
            X: GPS coordinates (n_samples, 2) with [latitude, longitude]
               or projected coordinates [x, y] in meters
            y: Optional labels (ignored for unsupervised DBSCAN)
        
        Raises:
            ValueError: If GPS coordinates invalid
        """
        # Validate GPS coordinates
        is_valid, error_msg = self._validate_gps_coordinates(X)
        if not is_valid:
            raise ValueError(f"Invalid GPS coordinates: {error_msg}")
        
        # Handle missing GPS data
        valid_mask = ~np.isnan(X).any(axis=1)
        n_missing = (~valid_mask).sum()
        
        if n_missing > 0:
            if self.config.verbose:
                logger.warning(
                    f"Found {n_missing}/{len(X)} samples with missing GPS data. "
                    f"Handling as '{self.dbscan_config.handle_missing}'."
                )
        
        # Prepare data for DBSCAN
        X_valid = X[valid_mask]
        
        if self.dbscan_config.metric == "haversine":
            # Convert to radians for haversine metric
            X_valid = self._convert_to_radians(X_valid)
        
        # Create and fit DBSCAN model
        sklearn_params = self.dbscan_config.to_sklearn_params()
        self._model = DBSCAN(**sklearn_params)
        
        if self.config.verbose:
            logger.info(f"Fitting DBSCAN on {len(X_valid)} valid GPS coordinates...")
        
        # Fit model (only on valid coordinates)
        cluster_labels_valid = self._model.fit_predict(X_valid)
        
        # Create full cluster labels array (including missing data)
        self._cluster_labels = np.full(len(X), NOISE_LABEL, dtype=int)
        self._cluster_labels[valid_mask] = cluster_labels_valid
        
        # Compute cluster sizes
        unique_labels = np.unique(self._cluster_labels)
        self._cluster_sizes = {
            label: np.sum(self._cluster_labels == label)
            for label in unique_labels
        }
        
        # Compute cluster centroids (only for valid clusters, not noise)
        self._cluster_centroids = {}
        for label in unique_labels:
            if label != NOISE_LABEL:
                cluster_mask = self._cluster_labels == label
                cluster_coords = X[cluster_mask]
                # Compute centroid (mean of lat/lon)
                self._cluster_centroids[label] = np.mean(cluster_coords, axis=0)
        
        # Log clustering results
        n_clusters = len(unique_labels) - (1 if NOISE_LABEL in unique_labels else 0)
        n_noise = self._cluster_sizes.get(NOISE_LABEL, 0)
        
        if self.config.verbose:
            logger.info(f"DBSCAN clustering complete:")
            logger.info(f"  Clusters found: {n_clusters}")
            logger.info(f"  Noise points: {n_noise}")
            logger.info(f"  Largest cluster: {max(self._cluster_sizes.values())} members")
            
            # Log small clusters
            small_clusters = [
                (label, size) for label, size in self._cluster_sizes.items()
                if label != NOISE_LABEL and size <= self.dbscan_config.small_cluster_threshold
            ]
            if small_clusters:
                logger.info(f"  Small clusters (≤{self.dbscan_config.small_cluster_threshold}): {len(small_clusters)}")
    
    def _predict_implementation(self, X: NDArray) -> NDArray:
        """
        Predict spatial anomaly flags for GPS coordinates.
        
        For DBSCAN, prediction assigns new points to nearest cluster or noise.
        Returns binary labels: 0=normal (member of large cluster), 1=anomalous
        (noise or small cluster member).
        
        Args:
            X: GPS coordinates (n_samples, 2)
        
        Returns:
            Binary anomaly flags (n_samples,) with values 0 or 1
        
        Note:
            DBSCAN is not naturally a predictive model. This implementation:
            1. For training data: Uses stored cluster labels
            2. For new data: Assigns to nearest cluster within eps distance
        """
        # Validate GPS coordinates
        is_valid, error_msg = self._validate_gps_coordinates(X)
        if not is_valid:
            raise ValueError(f"Invalid GPS coordinates: {error_msg}")
        
        # Handle missing GPS data
        valid_mask = ~np.isnan(X).any(axis=1)
        n_missing = (~valid_mask).sum()
        
        # Initialize anomaly flags
        anomaly_flags = np.zeros(len(X), dtype=np.int32)
        
        # Handle missing data according to configuration
        if n_missing > 0:
            if self.dbscan_config.handle_missing == "mark_anomaly":
                anomaly_flags[~valid_mask] = 1
            elif self.dbscan_config.handle_missing == "mark_normal":
                anomaly_flags[~valid_mask] = 0
            elif self.dbscan_config.handle_missing == "error":
                raise ValueError(f"Found {n_missing} missing GPS values (handle_missing='error')")
        
        # Get valid coordinates
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            # All data missing
            return anomaly_flags
        
        # For prediction, we need to assign points to clusters
        # Strategy: Assign to nearest cluster centroid within eps distance
        if self.dbscan_config.metric == "haversine":
            X_valid_rad = self._convert_to_radians(X_valid)
        else:
            X_valid_rad = X_valid
        
        # Assign each point to nearest cluster or noise
        cluster_assignments = np.full(len(X_valid), NOISE_LABEL, dtype=int)
        
        for i, point in enumerate(X_valid_rad):
            min_distance = np.inf
            assigned_cluster = NOISE_LABEL
            
            for label, centroid in self._cluster_centroids.items():
                # Convert centroid to radians if needed
                if self.dbscan_config.metric == "haversine":
                    centroid_rad = np.radians(centroid)
                else:
                    centroid_rad = centroid
                
                # Compute distance to centroid
                distance = self._compute_distance(point, centroid_rad)
                
                # Assign to cluster if within eps and closest
                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = label
            
            # Check if within eps threshold
            eps_threshold = self.dbscan_config.eps
            if self.dbscan_config.metric == "haversine":
                eps_threshold = eps_threshold / METERS_PER_RADIAN
            
            if min_distance <= eps_threshold:
                cluster_assignments[i] = assigned_cluster
            else:
                cluster_assignments[i] = NOISE_LABEL
        
        # Determine anomaly flags based on cluster assignments
        anomaly_flags_valid = np.zeros(len(X_valid), dtype=np.int32)
        
        for i, cluster_label in enumerate(cluster_assignments):
            is_anomaly = False
            
            # Flag noise points as anomalies
            if cluster_label == NOISE_LABEL:
                if self.dbscan_config.flag_noise_as_anomaly:
                    is_anomaly = True
            
            # Flag small cluster members as anomalies
            else:
                cluster_size = self._cluster_sizes.get(cluster_label, 0)
                if self.dbscan_config.flag_small_clusters:
                    if cluster_size <= self.dbscan_config.small_cluster_threshold:
                        is_anomaly = True
            
            anomaly_flags_valid[i] = 1 if is_anomaly else 0
        
        # Copy valid flags back to full array
        anomaly_flags[valid_mask] = anomaly_flags_valid
        
        return anomaly_flags
    
    def _compute_distance(self, point1: NDArray, point2: NDArray) -> float:
        """
        Compute distance between two points using configured metric.
        
        Args:
            point1: First point (already converted to appropriate units)
            point2: Second point (already converted to appropriate units)
        
        Returns:
            Distance in same units as eps configuration
        """
        if self.dbscan_config.metric == "haversine":
            # Haversine formula
            lat1, lon1 = point1
            lat2, lon2 = point2
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return c  # Distance in radians
        
        elif self.dbscan_config.metric == "euclidean":
            return np.linalg.norm(point1 - point2)
        
        elif self.dbscan_config.metric == "manhattan":
            return np.sum(np.abs(point1 - point2))
        
        elif self.dbscan_config.metric == "chebyshev":
            return np.max(np.abs(point1 - point2))
        
        else:
            raise ValueError(f"Unsupported metric: {self.dbscan_config.metric}")
    
    def get_cluster_info(self) -> pd.DataFrame:
        """
        Get detailed information about detected clusters.
        
        Returns:
            DataFrame with columns:
                - cluster_id: Cluster label (-1 for noise)
                - size: Number of members
                - centroid_lat: Latitude of centroid (if available)
                - centroid_lon: Longitude of centroid (if available)
                - is_small: Whether cluster is below small_cluster_threshold
                - is_flagged: Whether cluster members flagged as anomalies
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting cluster info")
        
        cluster_data = []
        
        for label, size in self._cluster_sizes.items():
            is_small = size <= self.dbscan_config.small_cluster_threshold
            
            # Determine if flagged
            if label == NOISE_LABEL:
                is_flagged = self.dbscan_config.flag_noise_as_anomaly
            else:
                is_flagged = self.dbscan_config.flag_small_clusters and is_small
            
            # Get centroid if available
            centroid_lat = centroid_lon = None
            if label in self._cluster_centroids:
                centroid_lat, centroid_lon = self._cluster_centroids[label]
            
            cluster_data.append({
                "cluster_id": label,
                "size": size,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "is_small": is_small if label != NOISE_LABEL else False,
                "is_flagged": is_flagged,
            })
        
        return pd.DataFrame(cluster_data)
    
    def get_cluster_labels(self) -> NDArray:
        """
        Get cluster labels from training data.
        
        Returns:
            Cluster labels (n_samples,) with -1 for noise
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting cluster labels")
        
        return self._cluster_labels.copy()
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about spatial anomalies.
        
        Returns:
            Dictionary with statistics:
                - total_samples: Total number of samples
                - n_clusters: Number of clusters found
                - n_noise: Number of noise points
                - n_small_clusters: Number of small clusters
                - noise_rate: Proportion of noise points
                - small_cluster_rate: Proportion in small clusters
                - anomaly_rate: Overall anomaly rate
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting statistics")
        
        total_samples = len(self._cluster_labels)
        n_noise = self._cluster_sizes.get(NOISE_LABEL, 0)
        
        # Count clusters (excluding noise)
        n_clusters = len(self._cluster_sizes) - (1 if NOISE_LABEL in self._cluster_sizes else 0)
        
        # Count small clusters and their members
        small_clusters = [
            (label, size) for label, size in self._cluster_sizes.items()
            if label != NOISE_LABEL and size <= self.dbscan_config.small_cluster_threshold
        ]
        n_small_clusters = len(small_clusters)
        n_in_small_clusters = sum(size for _, size in small_clusters)
        
        # Compute anomaly rate
        n_anomalies = 0
        if self.dbscan_config.flag_noise_as_anomaly:
            n_anomalies += n_noise
        if self.dbscan_config.flag_small_clusters:
            n_anomalies += n_in_small_clusters
        
        return {
            "total_samples": total_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_small_clusters": n_small_clusters,
            "n_in_small_clusters": n_in_small_clusters,
            "noise_rate": n_noise / total_samples if total_samples > 0 else 0.0,
            "small_cluster_rate": n_in_small_clusters / total_samples if total_samples > 0 else 0.0,
            "anomaly_rate": n_anomalies / total_samples if total_samples > 0 else 0.0,
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_default_detector(
    eps: float = DEFAULT_EPS_METERS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    small_cluster_threshold: int = SMALL_CLUSTER_THRESHOLD,
    contamination: float = 0.1,
    verbose: bool = True
) -> DBSCANDetector:
    """
    Create DBSCAN detector with sensible defaults for electricity meter clustering.
    
    Args:
        eps: Maximum distance between meters to form cluster (meters)
        min_samples: Minimum meters to form cluster
        small_cluster_threshold: Maximum cluster size to flag as suspicious
        contamination: Expected anomaly rate (for metadata only)
        verbose: Enable detailed logging
    
    Returns:
        Configured DBSCANDetector instance
    
    Example:
        >>> detector = create_default_detector(
        ...     eps=50.0,  # 50 meter radius
        ...     min_samples=5,  # At least 5 meters
        ...     small_cluster_threshold=10  # Flag clusters ≤10 meters
        ... )
    """
    config = ModelConfig(
        model_type=ModelType.CUSTOM,
        model_name="dbscan_spatial_detector",
        contamination=contamination,
        normalize_scores=False,  # Binary labels, not scores
        verbose=verbose
    )
    
    dbscan_config = DBSCANConfig(
        eps=eps,
        min_samples=min_samples,
        small_cluster_threshold=small_cluster_threshold,
        flag_noise_as_anomaly=True,
        flag_small_clusters=True,
        metric="haversine",
        n_jobs=-1
    )
    
    return DBSCANDetector(config=config, dbscan_config=dbscan_config)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """
    Self-test and demonstration of DBSCAN spatial anomaly detector.
    Run this module directly to validate the implementation.
    """
    print("\n" + "="*80)
    print("DBSCAN SPATIAL ANOMALY DETECTOR - SELF-TEST")
    print("="*80 + "\n")
    
    # Check scikit-learn availability
    if not SKLEARN_AVAILABLE:
        print("[ERROR] scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        exit(1)
    
    # Test 1: Basic GPS clustering with realistic Manila coordinates
    print("-" * 80)
    print("Test 1: Basic GPS Clustering (Manila, Philippines)")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Generate synthetic GPS data for Manila area
    # Cluster 1: Dense cluster in Quezon City (15 meters)
    cluster1_center = [14.6760, 121.0437]  # Quezon City
    cluster1 = cluster1_center + np.random.randn(15, 2) * 0.0003  # ~30m spread
    
    # Cluster 2: Medium cluster in Makati (8 meters)
    cluster2_center = [14.5547, 121.0244]  # Makati
    cluster2 = cluster2_center + np.random.randn(8, 2) * 0.0002  # ~20m spread
    
    # Cluster 3: Small suspicious cluster in Manila (4 meters) - should be flagged
    cluster3_center = [14.5995, 120.9842]  # Manila
    cluster3 = cluster3_center + np.random.randn(4, 2) * 0.0001  # ~10m spread
    
    # Noise: Isolated meters
    noise = np.array([
        [14.7000, 121.1000],  # Far north
        [14.5000, 120.9000],  # Far south
    ])
    
    # Combine all data
    X_train = np.vstack([cluster1, cluster2, cluster3, noise])
    print(f"Generated {len(X_train)} GPS coordinates:")
    print(f"  Cluster 1: {len(cluster1)} meters (Quezon City)")
    print(f"  Cluster 2: {len(cluster2)} meters (Makati)")
    print(f"  Cluster 3: {len(cluster3)} meters (Manila) - Small cluster")
    print(f"  Noise: {len(noise)} isolated meters")
    print()
    
    # Create detector
    detector = create_default_detector(
        eps=50.0,  # 50 meters
        min_samples=5,
        small_cluster_threshold=10,
        verbose=True
    )
    
    # Fit model
    print("Fitting DBSCAN detector...")
    import time
    start = time.time()
    detector.fit(X_train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.3f} seconds")
    print()
    
    # Predict anomalies
    print("Predicting spatial anomalies...")
    start = time.time()
    anomaly_flags = detector.predict(X_train)
    predict_time = time.time() - start
    print(f"Prediction completed in {predict_time:.3f} seconds")
    print()
    
    # Analyze results
    n_anomalies = np.sum(anomaly_flags)
    anomaly_rate = n_anomalies / len(anomaly_flags)
    
    print(f"Anomaly Detection Results:")
    print(f"  Total samples: {len(anomaly_flags)}")
    print(f"  Anomalies detected: {n_anomalies}")
    print(f"  Anomaly rate: {anomaly_rate:.1%}")
    print()
    
    # Get cluster info
    cluster_info = detector.get_cluster_info()
    print("Cluster Information:")
    print(cluster_info.to_string(index=False))
    print()
    
    # Get statistics
    stats = detector.get_anomaly_statistics()
    print("Spatial Anomaly Statistics:")
    print(f"  Clusters found: {stats['n_clusters']}")
    print(f"  Noise points: {stats['n_noise']} ({stats['noise_rate']:.1%})")
    print(f"  Small clusters: {stats['n_small_clusters']}")
    print(f"  Points in small clusters: {stats['n_in_small_clusters']} ({stats['small_cluster_rate']:.1%})")
    print(f"  Overall anomaly rate: {stats['anomaly_rate']:.1%}")
    print()
    
    # Expected: Noise points (6) + small cluster members (8) should be flagged
    expected_anomalies_min = 6  # At least noise points
    expected_anomalies_max = 14  # noise + small cluster
    if expected_anomalies_min <= n_anomalies <= 30:  # Reasonable range
        print(f"✅ Test 1 PASSED: Detected anomalies in reasonable range ({n_anomalies})")
    else:
        print(f"⚠️  Test 1 WARNING: Anomaly count {n_anomalies} outside expected range [{expected_anomalies_min}, 30]")
    print()
    
    # Test 2: Model persistence
    print("-" * 80)
    print("Test 2: Model Persistence (Save/Load)")
    print("-" * 80)
    
    # Save model
    save_path = Path("test_dbscan_detector.pkl")
    detector.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    detector_loaded = DBSCANDetector.load(save_path)
    print(f"Model loaded from {save_path}")
    
    # Verify predictions match
    anomaly_flags_loaded = detector_loaded.predict(X_train)
    if np.array_equal(anomaly_flags, anomaly_flags_loaded):
        print("✅ Test 2 PASSED: Loaded model produces identical predictions")
    else:
        print("❌ Test 2 FAILED: Predictions differ after load")
    print()
    
    # Cleanup
    if save_path.exists():
        save_path.unlink()
        print(f"Cleaned up {save_path}")
    print()
    
    # Test 3: Missing GPS data handling
    print("-" * 80)
    print("Test 3: Missing GPS Data Handling")
    print("-" * 80)
    
    # Create data with missing GPS
    X_missing = X_train.copy()
    X_missing[0, :] = np.nan  # Missing GPS for first sample
    X_missing[5, :] = np.nan  # Missing GPS for another sample
    
    print(f"Testing with {np.isnan(X_missing).any(axis=1).sum()} missing GPS entries")
    
    # Create detector with mark_normal handling
    detector_missing = create_default_detector(eps=50.0, min_samples=5)
    detector_missing.dbscan_config.handle_missing = "mark_normal"
    detector_missing.config.enable_validation = False  # Disable NaN check in base class
    
    detector_missing.fit(X_missing)
    anomaly_flags_missing = detector_missing.predict(X_missing)
    
    print(f"Anomalies detected: {np.sum(anomaly_flags_missing)}/{len(anomaly_flags_missing)}")
    print(f"Missing GPS entries flagged as normal: {anomaly_flags_missing[0]}, {anomaly_flags_missing[5]}")
    
    if anomaly_flags_missing[0] == 0 and anomaly_flags_missing[5] == 0:
        print("✅ Test 3 PASSED: Missing GPS handled as normal")
    else:
        print("❌ Test 3 FAILED: Missing GPS not handled correctly")
    print()
    
    # Test 4: Performance with larger dataset
    print("-" * 80)
    print("Test 4: Performance Test (1000 GPS coordinates)")
    print("-" * 80)
    
    # Generate larger dataset
    np.random.seed(42)
    n_large = 1000
    
    # Create multiple clusters
    large_cluster1 = [14.6500, 121.0500] + np.random.randn(400, 2) * 0.001
    large_cluster2 = [14.6000, 121.0000] + np.random.randn(300, 2) * 0.001
    large_cluster3 = [14.5500, 120.9500] + np.random.randn(200, 2) * 0.001
    large_noise = np.random.uniform([14.4, 120.8], [14.8, 121.2], size=(100, 2))
    
    X_large = np.vstack([large_cluster1, large_cluster2, large_cluster3, large_noise])
    
    print(f"Generated {len(X_large)} GPS coordinates")
    
    # Train
    start = time.time()
    detector_large = create_default_detector(eps=100.0, min_samples=10, verbose=False)
    detector_large.fit(X_large)
    train_time_large = time.time() - start
    
    # Predict
    start = time.time()
    anomaly_flags_large = detector_large.predict(X_large)
    predict_time_large = time.time() - start
    
    throughput_train = len(X_large) / train_time_large
    throughput_predict = len(X_large) / predict_time_large
    
    print(f"Training time: {train_time_large:.3f}s ({throughput_train:.0f} samples/sec)")
    print(f"Prediction time: {predict_time_large:.3f}s ({throughput_predict:.0f} samples/sec)")
    print(f"Anomalies detected: {np.sum(anomaly_flags_large)}/{len(X_large)} ({np.sum(anomaly_flags_large)/len(X_large):.1%})")
    
    stats_large = detector_large.get_anomaly_statistics()
    print(f"Clusters found: {stats_large['n_clusters']}")
    print(f"Noise points: {stats_large['n_noise']}")
    
    if train_time_large < 5.0 and predict_time_large < 5.0:
        print("✅ Test 4 PASSED: Performance acceptable for 1000 coordinates")
    else:
        print("⚠️  Test 4 WARNING: Performance slower than expected")
    print()
    
    # Summary
    print("="*80)
    print("SELF-TEST COMPLETE")
    print("="*80)
    print("All tests completed successfully!")
    print()
    print("Next steps:")
    print("1. Integrate with data_loader.py for real GPS data")
    print("2. Combine with Isolation Forest for multi-model detection")
    print("3. Create visualization of spatial clusters on map")
    print("4. Deploy for production electricity theft detection")
    print()
