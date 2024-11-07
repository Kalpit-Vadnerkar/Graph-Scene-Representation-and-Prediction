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
        self.sequence_ids: List[int] = []
        
    def process_sequence(self,
                        dataset: Any,
                        predictions: List[Dict[str, np.ndarray]],
                        condition: str,
                        sequence_id: int,
                        start_idx: int,
                        end_idx: int) -> None:
        """Process a sequence with proper tracking of sequence IDs"""
        
        for t in range(start_idx, end_idx - self.horizon):
            window_residuals = []
            window_uncertainties = []
            
            try:
                for h in range(-self.horizon + 1, self.horizon + 1):
                    if t + h < start_idx or t + h >= end_idx:
                        continue
                        
                    past, future, _, _ = dataset[t + h]
                    pred_idx = t + h - start_idx
                    
                    if 0 <= pred_idx < len(predictions):
                        pred = predictions[pred_idx]
                        truth = future if h > 0 else past
                        
                        residual_output = self.residual_generator.compute_residuals(
                            pred, truth, sequence_id, t+h
                        )
                        window_residuals.append(residual_output.residuals)
                        window_uncertainties.append(residual_output.uncertainties)
                
                if len(window_residuals) == 2 * self.horizon:
                    features = ResidualFeatures(
                        sequence_id=sequence_id,
                        time=t,
                        position_residuals=np.array([r['position'] for r in window_residuals]),
                        velocity_residuals=np.array([r['velocity'] for r in window_residuals]),
                        steering_residuals=np.array([r['steering'] for r in window_residuals]),
                        acceleration_residuals=np.array([r['acceleration'] for r in window_residuals]),
                        position_uncertainties=np.array([u['position'] for u in window_uncertainties]),
                        velocity_uncertainties=np.array([u['velocity'] for u in window_uncertainties]),
                        steering_uncertainties=np.array([u['steering'] for u in window_uncertainties]),
                        acceleration_uncertainties=np.array([u['acceleration'] for u in window_uncertainties]),
                        condition=condition
                    )
                    
                    feature_dict = self.feature_extractor.extract_features(features)
                    self.features.append(feature_dict)
                    self.labels.append(condition)
                    self.sequence_ids.append(sequence_id)
                    
            except Exception as e:
                print(f"Error processing sequence {sequence_id} at time {t}: {str(e)}")
                continue