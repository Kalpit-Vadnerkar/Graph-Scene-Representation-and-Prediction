from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from Model_Testing.FaultDetectionConfig import FEATURE_NAMES, FEATURE_CONFIGS

class ResidualOutput(NamedTuple):
    residuals: Dict[str, np.ndarray]
    normalized_residuals: Dict[str, np.ndarray]  
    uncertainties: Dict[str, np.ndarray]
    timestamp: int

@dataclass
class ResidualFeatures:
    time: int
    condition: str
    
    # Initialize dictionaries to store different types of residuals for each feature
    raw_residuals: Dict[str, np.ndarray]
    normalized_residuals: Dict[str, np.ndarray]  
    uncertainties: Dict[str, np.ndarray]
    
    @classmethod
    def create_from_data(cls, 
                        time: int, 
                        condition: str,
                        residuals: Dict[str, np.ndarray],
                        normalized_residuals: Dict[str, np.ndarray],
                        uncertainties: Dict[str, np.ndarray]) -> 'ResidualFeatures':
        return cls(
            time=time,
            condition=condition,
            raw_residuals=residuals,
            normalized_residuals=normalized_residuals,
            uncertainties=uncertainties
        )

class ResidualGenerator:
    def __init__(self, horizon: int):
        self.horizon = horizon
        
    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, torch.Tensor],
                         timestamp: int) -> ResidualOutput:
    
        residuals = {}
        normalized_residuals = {}
        uncertainties = {}
        
        for feature in FEATURE_NAMES:
            mean_key = f'{feature}_mean'
            var_key = f'{feature}_var'
            
            if mean_key in predictions and feature in ground_truth:
                truth_np = ground_truth[feature].detach().cpu().numpy()
                pred_mean = predictions[mean_key]
                pred_var = predictions[var_key]
                
                # Remove singleton dimensions and ensure consistent shapes
                truth_np = truth_np.squeeze()
                if FEATURE_CONFIGS[feature].dimensions == 1:
                    if len(pred_mean.shape) == 1:
                        pred_mean = pred_mean.reshape(-1, 1)
                    if len(pred_var.shape) == 1:
                        pred_var = pred_var.reshape(-1, 1)
                    if len(truth_np.shape) == 1:
                        truth_np = truth_np.reshape(-1, 1)
                
                residuals[feature] = pred_mean - truth_np
                normalized_residuals[feature] = (pred_mean - truth_np) / np.sqrt(pred_var)
                uncertainties[feature] = pred_var
            else:
                truth_np = ground_truth[feature].detach().cpu().numpy()
                pred_mean = predictions[feature]
                pred_var = predictions[feature] / predictions[feature] 

                truth_np = truth_np.squeeze()
                if FEATURE_CONFIGS[feature].dimensions == 1:
                    if len(pred_mean.shape) == 1:
                        pred_mean = pred_mean.reshape(-1, 1)
                    if len(pred_var.shape) == 1:
                        pred_var = pred_var.reshape(-1, 1)
                    if len(truth_np.shape) == 1:
                        truth_np = truth_np.reshape(-1, 1)
                
                residuals[feature] = pred_mean - truth_np
                normalized_residuals[feature] = (pred_mean - truth_np) / np.sqrt(pred_var)
                uncertainties[feature] = pred_var
                
        return ResidualOutput(residuals, normalized_residuals, uncertainties, timestamp)