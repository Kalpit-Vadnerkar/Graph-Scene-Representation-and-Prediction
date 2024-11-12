from Model_Testing.ResidualGenerator import ResidualFeatures

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

class ResidualFeatureExtractor:
    @staticmethod
    def compute_statistical_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute statistical features with proper error handling"""
        if len(values) == 0:
            return {}
            
        flat_values = values.reshape(-1)
        time_steps = np.arange(len(flat_values))
        
        features = {}
        try:
            features.update({
                f'{prefix}_mean': float(np.mean(flat_values)),
                f'{prefix}_std': float(np.std(flat_values)),
                f'{prefix}_max': float(np.max(np.abs(flat_values))),
                f'{prefix}_range': float(np.ptp(flat_values))
            })
            
            if len(flat_values) > 1:
                features[f'{prefix}_trend'] = float(np.polyfit(time_steps, flat_values, 1)[0])
            
            if len(flat_values) > 2:
                features[f'{prefix}_skew'] = float(stats.skew(flat_values))
                
            if len(flat_values) > 3:
                features[f'{prefix}_kurtosis'] = float(stats.kurtosis(flat_values))
                
        except Exception as e:
            print(f"Error computing features for {prefix}: {str(e)}")
            
        return features
    
    def extract_features(self, residuals: ResidualFeatures) -> Dict[str, Any]:
        """Extract features with sequence tracking"""
        features = {
            'sequence_id': residuals.sequence_id,
            'timestamp': residuals.time
        }
        
        # Process each component
        components = [
            ('position_x', residuals.position_residuals[:, 0]),
            ('position_y', residuals.position_residuals[:, 1]),
            ('velocity_x', residuals.velocity_residuals[:, 0]),
            ('velocity_y', residuals.velocity_residuals[:, 1]),
            ('steering', residuals.steering_residuals.squeeze()),
            ('acceleration', residuals.acceleration_residuals.squeeze())
        ]
        
        for name, data in components:
            features.update(self.compute_statistical_features(data, name))
            
        # Process uncertainties
        uncertainty_components = [
            ('position_x_uncertainty', residuals.position_uncertainties[:, 0]),
            ('position_y_uncertainty', residuals.position_uncertainties[:, 1]),
            ('velocity_x_uncertainty', residuals.velocity_uncertainties[:, 0]),
            ('velocity_y_uncertainty', residuals.velocity_uncertainties[:, 1]),
            ('steering_uncertainty', residuals.steering_uncertainties.squeeze()),
            ('acceleration_uncertainty', residuals.acceleration_uncertainties.squeeze())
        ]
        
        for name, data in uncertainty_components:
            features.update(self.compute_statistical_features(data, name))
            
        # Remove condition from features as it should be a label
        return features