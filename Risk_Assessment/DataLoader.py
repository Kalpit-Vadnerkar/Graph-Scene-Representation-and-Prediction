from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import make_predictions

from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import os

@dataclass
class LoadedData:
    dataset: TrajectoryDataset
    predictions: List[Dict[str, np.ndarray]]

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_data_and_predictions(self, model, condition: str) -> LoadedData:
        """Load dataset and generate predictions for a specific condition"""
        data_folder = os.path.join(self.config['test_data_folder'], condition)
        dataset = TrajectoryDataset(
            data_folder,
            position_scaling_factor=self.config['position_scaling_factor'],
            velocity_scaling_factor=self.config['velocity_scaling_factor'],
            steering_scaling_factor=self.config['steering_scaling_factor'],
            acceleration_scaling_factor=self.config['acceleration_scaling_factor']
        )
        
        predictions = make_predictions(model, dataset, self.config)
        return LoadedData(dataset=dataset, predictions=predictions)