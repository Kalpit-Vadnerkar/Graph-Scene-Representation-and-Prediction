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
        self.residual_generator = ResidualGenerator(horizon=config['output_seq_len'])
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
            self.process_condition(model, condition)
        
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
    
    
    def process_run(self,
                        dataset: Any,
                        predictions: List[Dict[str, np.ndarray]],
                        condition: str) -> None:
        """
        Process a run to generate time-step-based residual features.
        This approach looks at all the H' predictions made for timestep t.
        The implementation is unoptimized and may require debugging.
        
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