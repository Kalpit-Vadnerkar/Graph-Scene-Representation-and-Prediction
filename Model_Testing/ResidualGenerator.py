from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from Model_Testing.FaultDetectionConfig import FEATURE_NAMES, FEATURE_CONFIGS, RESIDUAL_TYPES
from scipy.stats import norm

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
        #self.residual_types = ['raw', 'normalized', 'uncertainty', 'kl_divergence']
        
    def compute_kl_divergence(self, 
                            true_values: np.ndarray, 
                            pred_mean: np.ndarray, 
                            pred_var: np.ndarray) -> np.ndarray:
        """
        Compute KL divergence between predicted distribution and empirical distribution
        Using Gaussian assumption for predicted distribution
        """
        # Assume empirical distribution is a point mass at true values
        # KL(P||Q) where P is empirical and Q is predicted Gaussian
        # For point mass P at x, KL(P||Q) = -log(Q(x))
        kl_div = 0.5 * (
            np.log(2 * np.pi * pred_var) + 
            (true_values - pred_mean)**2 / pred_var
        )
        return kl_div
    
    def ensure_consistent_shape(self, 
                              array: np.ndarray, 
                              feature_dim: int) -> np.ndarray:
        """Ensure array has consistent shape based on feature dimensions"""
        array = array.squeeze()
        if feature_dim == 1 and len(array.shape) == 1:
            array = array.reshape(-1, 1)
        return array
    
    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, torch.Tensor],
                         timestamp: int) -> ResidualOutput:
        """
        Compute all types of residuals for each feature and group them in a nested dictionary
        structure: {feature -> {residual_type -> values}}
        """
        residuals = {feature: {} for feature in FEATURE_NAMES}
        
        for feature in FEATURE_NAMES:
            mean_key = f'{feature}_mean'
            var_key = f'{feature}_var'
            feature_dim = FEATURE_CONFIGS[feature].dimensions
            
            # Convert ground truth to numpy and get predictions
            truth_np = ground_truth[feature].detach().cpu().numpy()
            
            if mean_key in predictions and feature in ground_truth:
                pred_mean = predictions[mean_key]
                pred_var = predictions[var_key]
            else:
                pred_mean = predictions[feature]
                pred_var = predictions[feature] / predictions[feature]  # Creates array of ones
                
            # Ensure consistent shapes
            truth_np = self.ensure_consistent_shape(truth_np, feature_dim)
            pred_mean = self.ensure_consistent_shape(pred_mean, feature_dim)
            pred_var = self.ensure_consistent_shape(pred_var, feature_dim)
            
            # Compute each type of residual
            residuals[feature] = {
                'raw': pred_mean - truth_np,
                'normalized': (pred_mean - truth_np) / np.sqrt(pred_var),
                'uncertainty': pred_var,
                'kl_divergence': self.compute_kl_divergence(truth_np, pred_mean, pred_var)
            }
                
        return ResidualOutput(residuals, timestamp)