from Risk_Assessment.DataLoader import DataLoader
from Risk_Assessment.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Risk_Assessment.FeatureExtractor import FeatureExtractor
from Risk_Assessment.FaultDetector import FaultDetector
from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class RiskAssessmentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader(config)
        self.horizon = config['output_seq_len']
        self.residual_generator = ResidualGenerator(horizon=self.horizon)
        self.feature_extractor = FeatureExtractor()
        self.fault_detector = FaultDetector()
        
        self.features_by_condition = {}
        self.labels_by_condition = {}
    
    def process_condition(self, model, condition: str):
        """Process a single condition"""
        # Load data
        loaded_data = self.data_loader.load_data_and_predictions(model, condition)
        
        # Generate residuals and extract features
        features = []
        labels = []
        
        for t in range(len(loaded_data.dataset)):
            past, future, _, _ = loaded_data.dataset[t]
            if t < len(loaded_data.predictions):
                residual_output = self.residual_generator.compute_residuals(
                    loaded_data.predictions[t],
                    future,
                    t
                )
                
                residual_features = ResidualFeatures(
                    time=t,
                    condition=condition,
                    residuals=residual_output.residuals
                )
                
                feature_dict = self.feature_extractor.extract_features(residual_features)
                features.append(feature_dict)
                labels.append(condition)
        
        self.features_by_condition[condition] = features
        self.labels_by_condition[condition] = labels
    
    def plot_confusion_matrix(self, results: Dict[str, Any], save_path: str = 'Results/confusion_matrix.png'):
        """
        Plot and save confusion matrix visualization.
        
        Args:
            results: Dictionary containing classification results
            save_path: Path where to save the confusion matrix plot
        """
        import os
        
        # Create Results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get confusion matrix and labels
        conf_matrix = results['confusion_matrix']
        report_lines = results['classification_report'].split('\n')
        class_labels = [line.split()[0] for line in report_lines[1:-5] if line.strip()]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Confusion matrix plot saved to {save_path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract and format feature importance from the random forest classifier.
        
        Returns:
            pd.DataFrame: DataFrame containing feature names and their importance scores
        """
        # Get feature names and importance scores
        feature_names = list(self.features_by_condition[self.config['conditions'][0]][0].keys())
        importance_scores = self.fault_detector.pipeline.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df

    def print_feature_importance(self):
        """Print feature importance in a formatted table."""
        importance_df = self.get_feature_importance()
        
        # Format the table
        table_data = []
        for _, row in importance_df.iterrows():
            table_data.append([
                row['Feature'],
                f"{row['Importance']:.4f}"
            ])
        
        # Print using tabulate
        print("\nFeature Importance:")
        print(tabulate(table_data, 
                      headers=['Feature', 'Importance Score'],
                      tablefmt='grid'))

    def run_fault_detection(self, model):
        """Run complete fault detection pipeline"""
        # Process all conditions
        for condition in self.config['conditions']:
            #self.process_condition(model, condition)
            self.process_condition_aligned(model, condition)
        
        # Combine all features and labels
        all_features = []
        all_labels = []
        for condition in self.config['conditions']:
            all_features.extend(self.features_by_condition[condition])
            all_labels.extend(self.labels_by_condition[condition])
        
        # Run fault detection
        results = self.fault_detector.train_and_evaluate(all_features, all_labels)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results)
        
        # Print feature importance
        self.print_feature_importance()
        
        return results
    
    
    def process_condition_aligned(self, model, condition: str):
        """
        Process a single condition with aligned observations and predictions
        This approach aligns predictions made at different times but targeting the same future timestamp
        """
        # Load data
        loaded_data = self.data_loader.load_data_and_predictions(model, condition)
        
        # Generate residuals and extract features
        features = []
        labels = []
        
        # Calculate valid range based on horizon
        horizon = self.config['output_seq_len']
        start_idx = 0
        end_idx = len(loaded_data.dataset)
        
        # For each target timestamp
        for target_t in range(start_idx + horizon, end_idx):
            # Dictionaries to collect all predictions and observations for this target timestamp
            predictions_for_t = {
                'position': [], 'position_mean': [], 'position_var': [],
                'velocity': [], 'velocity_mean': [], 'velocity_var': [],
                'steering': [], 'steering_mean': [], 'steering_var': [],
                'acceleration': [], 'acceleration_mean': [], 'acceleration_var': [],
                'object_distance': [], 'traffic_light_detected': []
            }
            
            observations_for_t = {
                'position': [],
                'velocity': [],
                'steering': [],
                'acceleration': [],
                'object_distance': [],
                'traffic_light_detected': []
            }
            
            valid_predictions = 0
            
            # Collect predictions made at different times for the same target time
            for origin_t in range(target_t - horizon, target_t):
                if origin_t < 0 or origin_t >= len(loaded_data.predictions):
                    continue
                    
                # Calculate which prediction step corresponds to target_t
                pred_offset = target_t - origin_t - 1  # -1 because prediction starts from next step
                
                # Only process if the offset is within the horizon
                if pred_offset < 0 or pred_offset >= horizon:
                    continue
                
                # Get actual observation at target_t
                past, future, _, _ = loaded_data.dataset[target_t]
                
                # Get prediction made at origin_t for target_t
                prediction = loaded_data.predictions[origin_t]
                
                # Check if we have valid prediction data
                if prediction is None or len(prediction) == 0:
                    continue
                    
                valid_predictions += 1
                
                # Add observation at target_t
                for feature in observations_for_t.keys():
                    # Ensure the feature exists in the future data
                    if feature in future:
                        # Get the actual observation - we need all the values, not just at an offset
                        # because future contains the ground truth for all future steps
                        observations_for_t[feature].append(future[feature])
                
                # Add prediction for target_t made at origin_t
                for feature in ['position', 'velocity', 'steering', 'acceleration']:
                    # Regular features have mean and variance
                    mean_key = f'{feature}_mean'
                    var_key = f'{feature}_var'
                    
                    if mean_key in prediction and var_key in prediction:
                        # Extract the specific prediction step (pred_offset)
                        # Based on DLModels.py, prediction[mean_key] shape is [batch, seq_len, dims]
                        # or [batch, seq_len] depending on the feature
                        try:
                            # For position and velocity (2D features)
                            if feature in ['position', 'velocity'] and prediction[mean_key].ndim > 2:
                                if pred_offset < prediction[mean_key].shape[0]:
                                    predictions_for_t[mean_key].append(prediction[mean_key][pred_offset])
                                    predictions_for_t[var_key].append(prediction[var_key][pred_offset])
                            # For steering and acceleration (1D features)
                            elif pred_offset < prediction[mean_key].shape[0]:
                                predictions_for_t[mean_key].append(prediction[mean_key][pred_offset])
                                predictions_for_t[var_key].append(prediction[var_key][pred_offset])
                        except Exception as e:
                            print(f"Error extracting {feature} at offset {pred_offset}: {str(e)}")
                            print(f"Shape: {prediction[mean_key].shape}")
                
                # Special case for object_distance and traffic_light_detected (no variance)
                for feature in ['object_distance', 'traffic_light_detected']:
                    if feature in prediction:
                        try:
                            if pred_offset < prediction[feature].shape[0]:
                                predictions_for_t[feature].append(prediction[feature][pred_offset])
                        except Exception as e:
                            print(f"Error extracting {feature} at offset {pred_offset}: {str(e)}")
                            print(f"Shape: {prediction[feature].shape}")
            
            # Only process if we have collected valid predictions
            if valid_predictions > 0:
                # Convert lists to tensors/arrays for compute_residuals
                processed_predictions = {}
                processed_observations = {}
                
                # Process predictions
                for key, values in predictions_for_t.items():
                    if values:
                        # Convert list of tensors to single array/tensor
                        if key.endswith('_mean') or key.endswith('_var'):
                            feature = key.split('_')[0]
                            # Handle the main feature keys to match what compute_residuals expects
                            if feature in ['position', 'velocity', 'steering', 'acceleration']:
                                # For these we need both mean and var
                                if len(values) > 0:
                                    try:
                                        import torch
                                        import numpy as np
                                        # Stack tensors if they're torch tensors
                                        if isinstance(values[0], torch.Tensor):
                                            processed_predictions[key] = torch.stack(values).detach().cpu().numpy()
                                        else:
                                            # Convert to numpy arrays if not already
                                            processed_predictions[key] = np.array(values)
                                    except Exception as e:
                                        print(f"Error processing {key}: {str(e)}")
                        else:
                            # Direct features (object_distance, traffic_light_detected)
                            if len(values) > 0:
                                try:
                                    import torch
                                    import numpy as np
                                    if isinstance(values[0], torch.Tensor):
                                        processed_predictions[key] = torch.stack(values).detach().cpu().numpy()
                                    else:
                                        processed_predictions[key] = np.array(values)
                                except Exception as e:
                                    print(f"Error processing {key}: {str(e)}")
                
                # Process observations
                for key, values in observations_for_t.items():
                    if values:
                        try:
                            import torch
                            # Stack tensors
                            processed_observations[key] = torch.stack(values)
                        except Exception as e:
                            print(f"Error processing observation {key}: {str(e)}")
                
                # Generate residuals if we have both predictions and observations
                if processed_predictions and processed_observations:
                    residual_output = self.residual_generator.compute_residuals(
                        processed_predictions,
                        processed_observations,
                        target_t
                    )
                    
                    residual_features = ResidualFeatures(
                        time=target_t,
                        condition=condition,
                        residuals=residual_output.residuals
                    )
                    
                    feature_dict = self.feature_extractor.extract_features(residual_features)
                    features.append(feature_dict)
                    labels.append(condition)
        
        self.features_by_condition[condition] = features
        self.labels_by_condition[condition] = labels
        
        return features, labels