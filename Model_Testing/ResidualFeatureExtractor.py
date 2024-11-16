from Model_Testing.ResidualGenerator import ResidualFeatures
from Model_Testing.FaultDetectionConfig import (
    FEATURE_COMPONENTS,
    STATISTICAL_METRICS,
    RESIDUAL_TYPES
)

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

class ResidualFeatureExtractor:
    @staticmethod
    def compute_statistical_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(values) == 0:
            return {}
            
        flat_values = values.reshape(-1)
        
        features = {}
        try:
            for metric in STATISTICAL_METRICS:
                if metric == 'mean':
                    features[f'{prefix}_mean'] = float(np.mean(flat_values))
                elif metric == 'std':
                    features[f'{prefix}_std'] = float(np.std(flat_values))
                elif metric == 'max':
                    features[f'{prefix}_max'] = float(np.max(np.abs(flat_values)))
                elif metric == 'range':
                    features[f'{prefix}_range'] = float(np.ptp(flat_values))
                
        except Exception as e:
            print(f"Error computing features for {prefix}: {str(e)}")
            
        return features
    
    def extract_features(self, residuals: ResidualFeatures) -> Dict[str, Any]:
        features = {}
        
        # Process each type of residual
        for residual_type in RESIDUAL_TYPES:
            if residual_type == 'raw':
                residual_data = residuals.raw_residuals
            elif residual_type == 'normalized':
                residual_data = residuals.normalized_residuals
            else:  # uncertainty
                residual_data = residuals.uncertainties
            
            # Process each feature and its components
            for feature, components in FEATURE_COMPONENTS.items():
                data = residual_data[feature]
                
                if len(components) == 1:
                    # Single component feature (e.g., steering)
                    prefix = f"{components[0]}_{residual_type}"
                    features.update(self.compute_statistical_features(data.squeeze(), prefix))
                else:
                    # Multi-component feature (e.g., position with x,y)
                    for idx, component in enumerate(components):
                        prefix = f"{component}_{residual_type}"
                        features.update(self.compute_statistical_features(data[:, idx], prefix))
                        
        return features