from dataclasses import dataclass
from typing import Dict, NamedTuple
import numpy as np
import torch

from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES, FEATURE_CONFIGS, RESIDUAL_TYPES, DELTA_VALUES
from Risk_Assessment.Residuals import RawResidual, KLDivergenceResidual, CUSUMResidual


class ResidualOutput(NamedTuple):
    residuals: Dict[str, Dict[str, np.ndarray]]  # Feature -> residual_type -> values
    timestamp: int

@dataclass
class ResidualFeatures:
    time: int
    condition: str
    residuals: Dict[str, Dict[str, np.ndarray]]

class ResidualGenerator:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.residual_types = ['raw', 'kl_divergence', 'cusum']
        self.residual_calculators = {
            'raw': RawResidual(),
            'kl_divergence': KLDivergenceResidual(),
            'cusum': CUSUMResidual()
        }

    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, 'torch.Tensor'],
                         timestamp: int) -> ResidualOutput:
        residuals = {feature: {} for feature in FEATURE_NAMES}
        
        for feature in FEATURE_NAMES:
            truth_np = ground_truth[feature].detach().cpu().numpy()
            
            # Handle special cases for object_distance and traffic_light_detected
            if feature in ['object_distance', 'traffic_light_detected']:
                pred_mean = predictions[feature]
                pred_var = np.ones_like(pred_mean)  # Use unit variance for these features
            else:
                pred_mean = predictions[f'{feature}_mean']
                pred_var = predictions[f'{feature}_var']
            
            truth_np = self._ensure_consistent_shape(truth_np, FEATURE_CONFIGS[feature].dimensions)
            pred_mean = self._ensure_consistent_shape(pred_mean, FEATURE_CONFIGS[feature].dimensions)
            pred_var = self._ensure_consistent_shape(pred_var, FEATURE_CONFIGS[feature].dimensions)
            
            for residual_type, calculator in self.residual_calculators.items():
                if residual_type == 'cusum':
                    calculator.set_delta(DELTA_VALUES[feature])
                residuals[feature][residual_type] = calculator.calculate(
                    truth_np, pred_mean, pred_var
                )
                
        return ResidualOutput(residuals, timestamp)
    
    def _ensure_consistent_shape(self, array: np.ndarray, feature_dim: int) -> np.ndarray:
        array = array.squeeze()
        if feature_dim == 1 and len(array.shape) == 1:
            array = array.reshape(-1, 1)
        return array