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
            window_normalized_residuals = [] 
            window_uncertainties = []
            
            try:
                # Only look at past horizon steps for feature extraction
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
                        window_normalized_residuals.append(residual_output.normalized_residuals)
                        window_uncertainties.append(residual_output.uncertainties)
                
                if len(window_residuals) == self.horizon:
                    # Create dictionaries to store all residual types for each feature
                    residuals_dict = {
                        feature: np.array([r[feature] for r in window_residuals])
                        for feature in FEATURE_NAMES
                    }
                    normalized_dict = {
                        feature: np.array([r[feature] for r in window_normalized_residuals])
                        for feature in FEATURE_NAMES
                    }
                    uncertainties_dict = {
                        feature: np.array([r[feature] for r in window_uncertainties])
                        for feature in FEATURE_NAMES
                    }
                    
                    features = ResidualFeatures.create_from_data(
                        time=t,
                        condition=condition,
                        residuals=residuals_dict,
                        normalized_residuals=normalized_dict,
                        uncertainties=uncertainties_dict
                    )
                    
                    feature_dict = self.feature_extractor.extract_features(features)
                    self.features.append(feature_dict)
                    self.labels.append(condition)
                    
            except Exception as e:
                print(f"Error processing sequence at time {t}: {str(e)}")
                continue