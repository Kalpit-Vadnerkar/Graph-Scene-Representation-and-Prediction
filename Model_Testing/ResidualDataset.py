from Model_Testing.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Model_Testing.ResidualFeatureExtractor import ResidualFeatureExtractor

from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class ResidualDataset:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.residual_generator = ResidualGenerator(horizon)
        self.feature_extractor = ResidualFeatureExtractor()
        self.features: List[Dict[str, float]] = []
        self.labels: List[str] = []
        
    def process_sequence(self,
                        dataset: Any,
                        predictions: List[Dict[str, np.ndarray]],
                        condition: str) -> None:
        
        start_idx = 0
        end_idx = len(dataset)
        for t in range(start_idx, end_idx - self.horizon):
            window_residuals = []
            window_std_residuals = []
            window_uncertainties = []
            
            try:
                # Only look at past horizon steps for feature extraction
                # This prevents data leakage from future steps
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
                        window_std_residuals.append(residual_output.standardized_residuals)
                        window_uncertainties.append(residual_output.uncertainties)
                
                if len(window_residuals) == self.horizon:
                    features = ResidualFeatures(
                        time=t,
                        
                        position_residuals=np.array([r['position'] for r in window_residuals]),
                        velocity_residuals=np.array([r['velocity'] for r in window_residuals]),
                        steering_residuals=np.array([r['steering'] for r in window_residuals]),
                        acceleration_residuals=np.array([r['acceleration'] for r in window_residuals]),
                        object_distance_residuals=np.array([r['object_distance'] for r in window_residuals]),
                        traffic_light_detected_residuals=np.array([r['traffic_light_detected'] for r in window_residuals]),

                        position_std_residuals=np.array([sr['position'] for sr in window_std_residuals]),
                        velocity_std_residuals=np.array([sr['velocity'] for sr in window_std_residuals]),
                        steering_std_residuals=np.array([sr['steering'] for sr in window_std_residuals]),
                        acceleration_std_residuals=np.array([sr['acceleration'] for sr in window_std_residuals]),
                        object_distance_std_residuals=np.array([sr['object_distance'] for sr in window_std_residuals]),
                        traffic_light_detected_std_residuals=np.array([sr['traffic_light_detected'] for sr in window_std_residuals]),
                        
                        position_uncertainties=np.array([u['position'] for u in window_uncertainties]),
                        velocity_uncertainties=np.array([u['velocity'] for u in window_uncertainties]),
                        steering_uncertainties=np.array([u['steering'] for u in window_uncertainties]),
                        acceleration_uncertainties=np.array([u['acceleration'] for u in window_uncertainties]),
                        object_distance_uncertainties=np.array([u['object_distance'] for u in window_uncertainties]),
                        traffic_light_detected_uncertainties=np.array([u['traffic_light_detected'] for u in window_uncertainties]),
                        
                        condition=condition
                    )
                    
                    feature_dict = self.feature_extractor.extract_features(features)
                    self.features.append(feature_dict)
                    self.labels.append(condition)
                    
            except Exception as e:
                print(f"Error processing sequence at time {t}: {str(e)}")
                continue