from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.DLModels import GraphAttentionLSTM
from Prediction_Model.model_utils import make_predictions

import torch
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import os
import time

@dataclass
class LoadedData:
    dataset: TrajectoryDataset
    predictions: List[Dict[str, np.ndarray]]

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_ensemble = []
        
    def load_ensemble(self):        
        for model_path in os.listdir(self.config['ensemble_model_path']):
            model = GraphAttentionLSTM(self.config)
            model.load_state_dict(torch.load(os.path.join(self.config['ensemble_model_path'], model_path), map_location=self.config['device'], weights_only=True))
            model.to(self.config['device'])
            model.eval()
            self.model_ensemble.append(model)

    def _calculate_epistemic_uncertainty(self, all_model_predictions: List[List[Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:
        """
        Calculate epistemic uncertainty across ensemble predictions.
        
        The predictive mean and variance are computed as:
        μ̂m,t = (1/N) ∑i=1,N μi,m,t
        σ̂²m,t = (1/N) ∑i=1,N [σ²i,m,t + μ²i,m,t] - μ̂²m,t
        
        Args:
            all_model_predictions: List of prediction lists from different models
            
        Returns:
            List of combined predictions with epistemic uncertainty
        """
        num_models = len(all_model_predictions)
        if num_models == 0:
            return []
        
        # Get number of samples
        num_samples = len(all_model_predictions[0])
        
        # Initialize combined predictions
        combined_predictions = []
        
        # Process each sample
        for sample_idx in range(num_samples):
            # Initialize combined prediction dictionary for this sample
            combined_pred = {}
            
            # Extract predictions for this sample from all models
            sample_predictions = [model_preds[sample_idx] for model_preds in all_model_predictions]
            
            # Get the set of all keys (features) from the first model's prediction
            feature_keys = sample_predictions[0].keys()
            
            for key in feature_keys:
                # Check if this is a mean or variance key
                if key.endswith('_mean'):
                    base_feature = key.replace('_mean', '')
                    var_key = f"{base_feature}_var"
                    
                    # Stack predictions from all models for this feature
                    mean_values = np.stack([pred[key] for pred in sample_predictions])
                    
                    # Calculate predictive mean (μ̂m,t)
                    predictive_mean = np.mean(mean_values, axis=0)
                    combined_pred[key] = predictive_mean
                    
                    # If corresponding variance exists, calculate epistemic uncertainty
                    if var_key in feature_keys:
                        var_values = np.stack([pred[var_key] for pred in sample_predictions])
                        
                        # Calculate part of the formula: (1/N) ∑i=1,N [σ²i,m,t + μ²i,m,t]
                        total_variance = np.mean(var_values + mean_values**2, axis=0)
                        
                        # Complete the formula: σ̂²m,t = (1/N) ∑i=1,N [σ²i,m,t + μ²i,m,t] - μ̂²m,t
                        predictive_variance = total_variance - predictive_mean**2
                        
                        # Ensure variance is positive
                        predictive_variance = np.maximum(predictive_variance, 1e-6)
                        
                        combined_pred[var_key] = predictive_variance
                elif not key.endswith('_var'):
                    # For other features (not ending with _mean or _var),
                    # just average the predictions
                    values = np.stack([pred[key] for pred in sample_predictions])
                    combined_pred[key] = np.mean(values, axis=0)
            
            combined_predictions.append(combined_pred)
        
        return combined_predictions

    def load_data_and_predictions(self, model, condition: str) -> LoadedData:
        """Load dataset, model ensemble, and generate predictions for a specific condition"""
        data_folder = os.path.join(self.config['test_data_folder'], condition)
        dataset = TrajectoryDataset(
            data_folder,
            position_scaling_factor=self.config['position_scaling_factor'],
            velocity_scaling_factor=self.config['velocity_scaling_factor'],
            steering_scaling_factor=self.config['steering_scaling_factor'],
            acceleration_scaling_factor=self.config['acceleration_scaling_factor']
        )
        
        total_inference_time = 0.0
        num_predictions = 0
        
        # If a single model is provided, use it
        if model is not None:
            start_time = time.time()
            predictions = make_predictions(model, dataset, self.config)
            inference_time = time.time() - start_time
            total_inference_time = inference_time
            num_predictions = len(dataset)
            combined_predictions = predictions
        else:
            # Otherwise, use the ensemble of models
            self.load_ensemble()
            
            if not self.model_ensemble:
                raise ValueError(f"No models found in ensemble directory: {self.config['ensemble_model_path']}")
            
            # Initialize list to store predictions from each model
            all_model_predictions = []
            
            # Generate predictions with each model in the ensemble
            for ensemble_model in self.model_ensemble:
                start_time = time.time()
                predictions = make_predictions(ensemble_model, dataset, self.config)
                model_inference_time = time.time() - start_time
                total_inference_time += model_inference_time
                num_predictions += len(dataset)
                all_model_predictions.append(predictions)
            
            # Epistemic Uncertainty Quantification
            combined_predictions = self._calculate_epistemic_uncertainty(all_model_predictions)
        
        # Calculate and print average inference time (once, at the end)
        avg_inference_time = total_inference_time / num_predictions if num_predictions > 0 else 0
        print(f"Average ST-GAT inference time per sample: {avg_inference_time*1000:.2f} ms")
        
        return LoadedData(dataset=dataset, predictions=combined_predictions)