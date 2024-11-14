import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

class ResidualOutput(NamedTuple):
    residuals: Dict[str, np.ndarray]
    uncertainties: Dict[str, np.ndarray]
    sequence_id: int
    timestamp: int

@dataclass
class ResidualFeatures:
    sequence_id: int
    time: int
    position_residuals: np.ndarray      # Shape: [window_size, 2]
    velocity_residuals: np.ndarray      # Shape: [window_size, 2]
    steering_residuals: np.ndarray      # Shape: [window_size, 1]
    acceleration_residuals: np.ndarray  # Shape: [window_size, 1]
    position_uncertainties: np.ndarray  # Shape: [window_size, 2]
    velocity_uncertainties: np.ndarray  # Shape: [window_size, 2]
    steering_uncertainties: np.ndarray  # Shape: [window_size, 1]
    acceleration_uncertainties: np.ndarray  # Shape: [window_size, 1]
    condition: str

class ResidualGenerator:
    def __init__(self, horizon: int):
        self.horizon = horizon
        
    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, torch.Tensor],
                         sequence_id: int,
                         timestamp: int) -> ResidualOutput:
    
        residuals = {}
        uncertainties = {}
        
        for feature in ['position', 'velocity', 'steering', 'acceleration']:
            mean_key = f'{feature}_mean'
            var_key = f'{feature}_var'
            
            if mean_key in predictions and feature in ground_truth:
                truth_np = ground_truth[feature].detach().cpu().numpy()
                pred_mean = predictions[mean_key]
                pred_var = predictions[var_key]
                
                # Remove singleton dimensions and ensure consistent shapes
                truth_np = truth_np.squeeze()
                if feature in ['steering', 'acceleration']:
                    if len(pred_mean.shape) == 1:
                        pred_mean = pred_mean.reshape(-1, 1)
                    if len(pred_var.shape) == 1:
                        pred_var = pred_var.reshape(-1, 1)
                    if len(truth_np.shape) == 1:
                        truth_np = truth_np.reshape(-1, 1)
                
                residuals[feature] = pred_mean - truth_np
                uncertainties[feature] = pred_var
                
        return ResidualOutput(residuals, uncertainties, sequence_id, timestamp)