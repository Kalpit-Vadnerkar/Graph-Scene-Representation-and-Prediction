from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
import torch
from scipy.stats import norm

from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES, FEATURE_CONFIGS, RESIDUAL_TYPES
from Risk_Assessment.Residuals import *


class ResidualOutput(NamedTuple):
    residuals: Dict[str, Dict[str, np.ndarray]]  # Feature -> residual_type -> values
    timestamp: int

@dataclass
class ResidualFeatures:
    time: int
    condition: str
    residuals: Dict[str, Dict[str, np.ndarray]]  # Feature -> residual_type -> values
    
    @classmethod
    def create_from_data(cls, 
                        time: int, 
                        condition: str,
                        residuals: Dict[str, Dict[str, np.ndarray]]) -> 'ResidualFeatures':
        return cls(
            time=time,
            condition=condition,
            residuals=residuals
        )

class ResidualGenerator:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.residual_types = RESIDUAL_TYPES
        
        # Map residual types to their calculator classes
        residual_class_map = {
            'raw': RawResidual,
            'normalized': NormalizedResidual,
            'uncertainty': UncertaintyResidual,
            'kl_divergence': KLDivergenceResidual,
            'shewhart': ShewartResidual,
            'cusum': CUSUMResidual,
            'sprt': SPRTResidual
        }

        # Only initialize calculators for residual types that exist in RESIDUAL_TYPES
        self.residual_calculators = {
            residual_type: residual_class_map[residual_type]()
            for residual_type in RESIDUAL_TYPES
            if residual_type in residual_class_map
        }
    
    def ensure_consistent_shape(self, array: np.ndarray, feature_dim: int) -> np.ndarray:
        array = array.squeeze()
        if feature_dim == 1 and len(array.shape) == 1:
            array = array.reshape(-1, 1)
        return array
    
    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, 'torch.Tensor'],
                         timestamp: int) -> ResidualOutput:
        residuals = {feature: {} for feature in FEATURE_NAMES}
        
        for feature in FEATURE_NAMES:
            mean_key = f'{feature}_mean'
            var_key = f'{feature}_var'
            feature_dim = FEATURE_CONFIGS[feature].dimensions
            
            truth_np = ground_truth[feature].detach().cpu().numpy()
            
            if mean_key in predictions and feature in ground_truth:
                pred_mean = predictions[mean_key]
                pred_var = predictions[var_key]
            else:
                pred_mean = predictions[feature]
                pred_var = predictions[feature] / predictions[feature]
                
            truth_np = self.ensure_consistent_shape(truth_np, feature_dim)
            pred_mean = self.ensure_consistent_shape(pred_mean, feature_dim)
            pred_var = self.ensure_consistent_shape(pred_var, feature_dim)
            
            for residual_type, calculator in self.residual_calculators.items():
                residuals[feature][residual_type] = calculator.calculate(
                    truth_np, pred_mean, pred_var
                )
                
        return ResidualOutput(residuals, timestamp)