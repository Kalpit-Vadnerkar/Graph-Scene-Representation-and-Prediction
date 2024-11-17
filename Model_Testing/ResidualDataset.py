from Model_Testing.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Model_Testing.ResidualFeatureExtractor import ResidualFeatureExtractor
from Model_Testing.FaultDetectionConfig import FEATURE_NAMES

from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class ResidualDataset:
    def __init__(self, horizon: int):
        """
        Initialize ResidualDataset with specified horizon.
        
        Args:
            horizon (int): Number of timesteps to look back for feature extraction
        """
        self.horizon = horizon
        self.residual_generator = ResidualGenerator(horizon)
        self.feature_extractor = ResidualFeatureExtractor()
        self.features: List[Dict[str, float]] = []
        self.labels: List[str] = []
        
    def process_sequence(self,
                        dataset: Any,
                        predictions: List[Dict[str, np.ndarray]],
                        condition: str) -> None:
        """
        Process a sequence of data to generate residual features.
        
        Args:
            dataset: Dataset containing ground truth values
            predictions: List of model predictions
            condition: Condition label for the sequence
        """
        start_idx = 0
        end_idx = len(dataset)
        
        for t in range(start_idx, end_idx - self.horizon):
            window_residuals = []
            
            try:
                # Look at past horizon steps for feature extraction
                for h in range(-self.horizon + 1, 1):
                    if t + h < start_idx or t + h >= end_idx:
                        continue
                        
                    past, future, _, _ = dataset[t + h]
                    pred_idx = t + h - start_idx
                    
                    if 0 <= pred_idx < len(predictions):
                        pred = predictions[pred_idx]
                        # Only use past data for residual computation
                        truth = past
                        
                        residual_output = self.residual_generator.compute_residuals(
                            pred, truth, t+h
                        )
                        window_residuals.append(residual_output.residuals)
                
                if len(window_residuals) == self.horizon:
                    # Create a nested dictionary to store all residuals for the window
                    # Structure: {feature -> residual_type -> timestep -> values}
                    residuals_dict = {}
                    
                    for feature in FEATURE_NAMES:
                        residuals_dict[feature] = {}
                        
                        # Initialize dictionary for each residual type
                        for residual_type in self.residual_generator.residual_types:
                            # Collect values across timesteps for this feature and residual type
                            values = [r[feature][residual_type] for r in window_residuals]
                            residuals_dict[feature][residual_type] = np.array(values)
                    
                    # Create ResidualFeatures object with the collected data
                    features = ResidualFeatures.create_from_data(
                        time=t,
                        condition=condition,
                        residuals=residuals_dict
                    )
                    
                    # Extract features using the feature extractor
                    feature_dict = self.feature_extractor.extract_features(features)
                    
                    # Store the features and label
                    self.features.append(feature_dict)
                    self.labels.append(condition)
                    
            except Exception as e:
                print(f"Error processing sequence at time {t}: {str(e)}")
                continue
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the collected dataset.
        
        Returns:
            Dict containing dataset statistics
        """
        if not self.features or not self.labels:
            return {"error": "No data collected"}
        
        label_counts = pd.Series(self.labels).value_counts()
        feature_names = list(self.features[0].keys()) if self.features else []
        
        return {
            "total_samples": len(self.labels),
            "label_distribution": label_counts.to_dict(),
            "num_features": len(feature_names),
            "feature_names": feature_names
        }
    
    def clear_dataset(self) -> None:
        """Clear all collected features and labels"""
        self.features = []
        self.labels = []