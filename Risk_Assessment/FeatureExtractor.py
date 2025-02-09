from Risk_Assessment.ResidualGenerator import ResidualFeatures
from Risk_Assessment.FaultDetectionConfig import FEATURE_COMPONENTS

from typing import Dict, Any
import numpy as np

class FeatureExtractor:
    @staticmethod
    def compute_statistical_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(values) == 0:
            return {}
            
        flat_values = values.reshape(-1)
        features = {}
        
        try:
            metrics = ['mean', 'std', 'max', 'range']
            for metric in metrics:
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
        
        for feature, components in FEATURE_COMPONENTS.items():
            feature_residuals = residuals.residuals[feature]
            
            if len(components) == 1:
                for residual_type in feature_residuals:
                    prefix = f"{components[0]}_{residual_type}"
                    values = feature_residuals[residual_type].squeeze()
                    features.update(self.compute_statistical_features(values, prefix))
            else:
                for idx, component in enumerate(components):
                    for residual_type in feature_residuals:
                        prefix = f"{component}_{residual_type}"
                        values = feature_residuals[residual_type][:, idx]
                        features.update(self.compute_statistical_features(values, prefix))
                        
        return features