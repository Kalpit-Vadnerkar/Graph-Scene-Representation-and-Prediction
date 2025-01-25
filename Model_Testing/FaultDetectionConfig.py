from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    name: str
    dimensions: int  # Number of dimensions for this feature (e.g., 2 for position, 1 for steering)

FEATURE_CONFIGS: Dict[str, FeatureConfig] = {
    'position': FeatureConfig('position', 2),
    'velocity': FeatureConfig('velocity', 2),
    'steering': FeatureConfig('steering', 1),
    'acceleration': FeatureConfig('acceleration', 1),
    'object_distance': FeatureConfig('object_distance', 1),
    'traffic_light_detected': FeatureConfig('traffic_light_detected', 1)
}

# Common feature names used across residual analysis
FEATURE_NAMES: List[str] = list(FEATURE_CONFIGS.keys())

# Feature components for when we need to separate multi-dimensional features
FEATURE_COMPONENTS: Dict[str, List[str]] = {
    'position': ['position_x', 'position_y'],
    'velocity': ['velocity_x', 'velocity_y'],
    'steering': ['steering'],
    'acceleration': ['acceleration'],
    'object_distance': ['object_distance'],
    'traffic_light_detected': ['traffic_light_detected']
}

# Statistical metrics computed for each feature
STATISTICAL_METRICS: List[str] = [
    'mean',
    'std',
    'max',
    'range'
]

# Types of residual analysis
RESIDUAL_TYPES: List[str] = [
    #'raw',           # Raw residuals
    #'normalized',    # Previously called standardized
    #'uncertainty',   # Uncertainty values
    'kl_divergence',  # KL divergence between predicted and empirical distributions
    #'shewhart',
    #'cusum', 
    #'sprt'
]