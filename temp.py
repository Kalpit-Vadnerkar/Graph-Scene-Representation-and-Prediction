import numpy as np
import torch
from typing import Dict, List, Any
from dataclasses import dataclass
from Risk_Assessment.DataLoader import LoadedData
from Risk_Assessment.ResidualGenerator import ResidualOutput, ResidualFeatures
from Risk_Assessment.Residuals import NormalizedResidual
from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES, FEATURE_CONFIGS
from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import make_predictions, load_model

from model_config import CONFIG
import os

@dataclass
class ResidualStats:
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[str, float]

class ResidualAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.residual_calculator = NormalizedResidual()
        
    def load_data(self, model, condition: str) -> LoadedData:
        """Load dataset and predictions for a condition"""
        data_folder = os.path.join("Test_Dataset/Sequence_Dataset", condition)
        dataset = TrajectoryDataset(
            data_folder,
            position_scaling_factor=10,
            velocity_scaling_factor=10,
            steering_scaling_factor=10,
            acceleration_scaling_factor=10
        )
        predictions = make_predictions(model, dataset, self.config)
        return LoadedData(dataset=dataset, predictions=predictions)
    
    def _ensure_consistent_shape(self, array: np.ndarray, feature_dim: int) -> np.ndarray:
        """Ensure consistent array shape"""
        array = array.squeeze()
        if feature_dim == 1 and len(array.shape) == 1:
            array = array.reshape(-1, 1)
        return array
    
    def compute_residuals(self, 
                         predictions: Dict[str, np.ndarray],
                         ground_truth: Dict[str, 'torch.Tensor']) -> Dict[str, np.ndarray]:
        """Compute normalized residuals for all features"""
        residuals = {}
        
        for feature in FEATURE_NAMES:
            truth_np = ground_truth[feature].detach().cpu().numpy()
            
            # Handle special cases for object_distance and traffic_light_detected
            if feature in ['object_distance', 'traffic_light_detected']:
                pred_mean = predictions[feature]
                pred_var = np.ones_like(pred_mean)
            else:
                pred_mean = predictions[f'{feature}_mean']
                pred_var = predictions[f'{feature}_var']
            
            # Ensure consistent shapes
            truth_np = self._ensure_consistent_shape(truth_np, FEATURE_CONFIGS[feature].dimensions)
            pred_mean = self._ensure_consistent_shape(pred_mean, FEATURE_CONFIGS[feature].dimensions)
            pred_var = self._ensure_consistent_shape(pred_var, FEATURE_CONFIGS[feature].dimensions)
            
            # Calculate normalized residuals
            residuals[feature] = self.residual_calculator.calculate(truth_np, pred_mean, pred_var)
            
        return residuals
    
    def calculate_residual_statistics(self, residuals_list: List[Dict[str, np.ndarray]]) -> Dict[str, ResidualStats]:
        """Calculate statistics for each feature's residuals"""
        # Combine residuals across time steps
        combined_residuals = {}
        for feature in FEATURE_NAMES:
            feature_residuals = [res[feature].flatten() for res in residuals_list]
            combined_residuals[feature] = np.concatenate(feature_residuals)
        
        # Calculate statistics for each feature
        stats = {}
        for feature, residuals in combined_residuals.items():
            stats[feature] = ResidualStats(
                mean=np.mean(residuals),
                std=np.std(residuals),
                min=np.min(residuals),
                max=np.max(residuals),
                percentiles={
                    '25': np.percentile(residuals, 25),
                    '50': np.percentile(residuals, 50),
                    '75': np.percentile(residuals, 75),
                    '95': np.percentile(residuals, 95),
                    '99': np.percentile(residuals, 99)
                }
            )
        
        return stats

def main():
    # Initialize
    config = CONFIG  # Your config dictionary
    analyzer = ResidualAnalyzer(config)
    model = load_model(config)
    
    # Load nominal data
    loaded_data = analyzer.load_data(model, 'Nominal')
    
    # Collect residuals
    all_residuals = []
    for t in range(len(loaded_data.dataset)):
        if t >= len(loaded_data.predictions):
            break
            
        past, future, _, _ = loaded_data.dataset[t]
        residuals = analyzer.compute_residuals(loaded_data.predictions[t], future)
        all_residuals.append(residuals)
    
    # Calculate statistics
    stats = analyzer.calculate_residual_statistics(all_residuals)
    
    # Print results
    print("\nResidual Statistics for Nominal Condition:")
    print("=========================================")
    for feature, stat in stats.items():
        print(f"\n{feature}:")
        print(f"  Mean: {stat.mean:.4f}")
        print(f"  Std Dev: {stat.std:.4f}")
        print(f"  Range: [{stat.min:.4f}, {stat.max:.4f}]")
        print("  Percentiles:")
        for p, v in stat.percentiles.items():
            print(f"    {p}th: {v:.4f}")
        
        # CUSUM recommendation
        recommended_delta = stat.std / 2
        print(f"  Recommended CUSUM δ (std/2): {recommended_delta:.4f}")

if __name__ == "__main__":
    main()