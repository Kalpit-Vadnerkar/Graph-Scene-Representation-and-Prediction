from Risk_Assessment.DataLoader import DataLoader
from Risk_Assessment.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Risk_Assessment.FeatureExtractor import DimensionReductionFeatureExtractor
from Risk_Assessment.FaultDetector import FaultDetector
from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import torch

class RiskAssessmentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader(config)
        self.horizon = config['output_seq_len']
        self.residual_generator = ResidualGenerator(horizon=self.horizon)
        self.feature_extractor = None # Will be initialized with specific n_components
        self.fault_detector = FaultDetector()
        
        self.features_by_condition = {}
        self.labels_by_condition = {}
        self.all_residuals = []  # Store all residuals for fitting PCA
    
    def generate_all_residuals(self, loaded_data_dict):
        """Generate residuals for all conditions using pre-loaded data."""
        self.all_residuals = []  # Reset residuals list
        
        # Process all conditions using pre-loaded data
        for condition in self.config['conditions']:
            loaded_data = loaded_data_dict[condition]
            #self.process_condition(loaded_data, condition)
            self.process_condition_aligned(loaded_data, condition)

    def process_condition(self, loaded_data, condition: str):
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
                self.all_residuals.append(residual_features)
    
    def plot_confusion_matrix(self, results: Dict[str, Any], save_path: str = 'Results/confusion_matrix.png'):
        """Plot and save confusion matrix visualization."""
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

    def plot_explained_variance(self, save_path: str = 'Results/explained_variance.png'):
        """Plot cumulative explained variance for each feature-residual type."""
        if not self.feature_extractor or not self.feature_extractor.is_fitted:
            raise ValueError("Feature extractor not fitted yet.")
        
        cumulative_variance = self.feature_extractor.get_cumulative_explained_variance()
        
        plt.figure(figsize=(15, 10))
        for key, variance in cumulative_variance.items():
            plt.plot(range(1, len(variance) + 1), variance, marker='o', label=key)
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_feature_importance(self, results: Dict[str, Any], save_path: str = 'Results/feature_importance.png'):
        """Plot feature importance by PCA component."""
        if not self.feature_extractor:
            raise ValueError("Feature extractor not initialized.")
        
        importance_by_component = self.feature_extractor.get_feature_importance_by_component(
            self.fault_detector.pipeline.named_steps['classifier']
        )
        
        # Prepare data for plotting
        features = []
        components = []
        importance_scores = []
        
        for feature, component_scores in importance_by_component.items():
            for component, score in component_scores.items():
                features.append(feature)
                components.append(f"PC{component}")
                importance_scores.append(score)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': features,
            'Component': components,
            'Importance': importance_scores
        })
        
        # Plot
        plt.figure(figsize=(15, 10))
        sns.barplot(data=df, x='Feature', y='Importance', hue='Component')
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance by PCA Component')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def print_feature_importance_by_component(self):
        """Print detailed feature importance breakdown by PCA component."""
        if not self.feature_extractor:
            raise ValueError("Feature extractor not initialized.")
            
        importance_by_component = self.feature_extractor.get_feature_importance_by_component(
            self.fault_detector.pipeline.named_steps['classifier']
        )
        
        # Prepare table data
        table_data = []
        for feature, component_scores in importance_by_component.items():
            for component, score in component_scores.items():
                table_data.append([
                    feature,
                    f"PC{component}",
                    f"{score:.4f}"
                ])
        
        # Sort by importance score
        table_data.sort(key=lambda x: float(x[2]), reverse=True)
        
        # Print using tabulate
        print("\nFeature Importance by PCA Component:")
        print(tabulate(table_data, 
                      headers=['Feature', 'Component', 'Importance Score'],
                      tablefmt='grid'))

    def run_fault_detection(self, loaded_data_dict, n_components=None):
        """Run complete fault detection pipeline with dimension reduction"""
        # Initialize feature extractor with specified number of components
        self.feature_extractor = DimensionReductionFeatureExtractor(n_components=n_components)
        
        # Generate residuals
        self.generate_all_residuals(loaded_data_dict)
        
        # Fit PCA and transform all residuals
        transformed_features = self.feature_extractor.fit_transform(self.all_residuals)
        
        # Prepare labels
        labels = [res.condition for res in self.all_residuals]
        
        # Run fault detection
        results = self.fault_detector.train_and_evaluate(transformed_features, labels)
        
        # Plot results
        # Generate visualizations
        self.plot_confusion_matrix(results)
        self.plot_explained_variance()
        self.plot_feature_importance(results)
        
        # Print feature importance details
        self.print_feature_importance_by_component()
        
        return results
    
    def run_dimensionality_analysis(self, loaded_data_dict, max_components=None):
        """
        Run fault detection with different numbers of PCA components to analyze impact.
        
        Args:
            model: Trained prediction model
            loaded_data_dict: Dictionary mapping conditions to their loaded data
            max_components: Maximum number of components to try (default: None = use all)
        """
        results = []
        
        # If max_components not specified, use sequence length
        if max_components is None:
            max_components = self.horizon
            
        # Generate residuals once - we'll reuse these for each component count
        self.generate_all_residuals(loaded_data_dict)
        
        component_range = range(1, max_components + 1)
        
        for n_components in component_range:
            print(f"\nTesting with {n_components} components:")
            test_results = self.run_fault_detection(loaded_data_dict, n_components=n_components)
            results.append({
                'n_components': n_components,
                'accuracy': test_results['accuracy']
            })
        
        # Plot accuracy vs number of components
        plt.figure(figsize=(10, 6))
        accuracies = [r['accuracy'] for r in results]
        plt.plot(component_range, accuracies, marker='o')
        plt.xlabel('Number of PCA Components')
        plt.ylabel('Classification Accuracy')
        plt.title('Classification Accuracy vs Number of PCA Components')
        plt.grid(True)
        plt.savefig('Results/accuracy_vs_components.png')
        plt.close()
        
        # Print results table
        table_data = [[r['n_components'], f"{r['accuracy']:.4f}"] for r in results]
        print("\nAccuracy vs Number of Components:")
        print(tabulate(table_data,
                      headers=['Number of Components', 'Accuracy'],
                      tablefmt='grid'))
    
    def process_condition_aligned(self, loaded_data, condition: str):
        """
        Process a single condition following the progressive validation algorithm.
        For each target time t, compute residuals from predictions made at different origin times.
        """
        horizon = self.horizon  # This is H' in the algorithm
        start_idx = horizon  # Need enough history for first prediction
        end_idx = len(loaded_data.dataset)
        
        for target_t in range(start_idx, end_idx):
            # Step 1: Get the target observation
            _, target_obs, _, _ = loaded_data.dataset[target_t]
            
            # Step 2: Set origin time (t - 1 - H')
            origin_t = target_t - 1 - horizon
            
            # Skip if we don't have enough history
            if origin_t < 0:
                continue
                
            # Initialize residuals collector for this target time
            residuals_collector = {feature: {
                residual_type: [] for residual_type in ['raw', 'kl_divergence', 'cusum']
            } for feature in FEATURE_NAMES}
            
            # Step 3: Loop through prediction times
            for i in range(horizon):
                pred_t = origin_t + i  # prediction time
                offset = horizon - i    # offset into prediction
                
                # Skip if prediction time is out of range
                if pred_t >= len(loaded_data.predictions) or pred_t < 0:
                    continue
                    
                prediction = loaded_data.predictions[pred_t]
                if prediction is None or len(prediction) == 0:
                    continue
                    
                # Get the relevant prediction for the target time
                if offset > 0 and offset <= prediction['position_mean'].shape[0]:
                    current_prediction = {}
                    
                    # Extract predictions for each feature
                    for feature in ['position', 'velocity', 'steering', 'acceleration']:
                        mean_key = f'{feature}_mean'
                        var_key = f'{feature}_var'
                        
                        if mean_key in prediction and var_key in prediction:
                            current_prediction[mean_key] = prediction[mean_key][offset-1:offset]
                            current_prediction[var_key] = prediction[var_key][offset-1:offset]
                    
                    # Handle special features
                    for feature in ['object_distance', 'traffic_light_detected']:
                        if feature in prediction:
                            current_prediction[feature] = prediction[feature][offset-1:offset]
                    
                    # Compute residuals for this prediction
                    try:
                        residual_output = self.residual_generator.compute_residuals(
                            current_prediction,
                            target_obs,
                            target_t
                        )
                        
                        # Store residuals from this prediction time
                        for feature in FEATURE_NAMES:
                            for residual_type in residual_output.residuals[feature]:
                                residuals_collector[feature][residual_type].append(
                                    residual_output.residuals[feature][residual_type]
                                )
                                
                    except Exception as e:
                        print(f"Error computing residuals for target {target_t}, prediction {pred_t}: {str(e)}")
            
            # After collecting all residuals for this target time, combine them
            if any(any(len(r) > 0 for r in feature_residuals.values()) 
                for feature_residuals in residuals_collector.values()):
                
                # Stack residuals for each feature and type
                combined_residuals = {}
                for feature in FEATURE_NAMES:
                    combined_residuals[feature] = {}
                    for residual_type in residuals_collector[feature]:
                        if residuals_collector[feature][residual_type]:
                            # Stack along a new axis to preserve the sequence of residuals
                            combined_residuals[feature][residual_type] = np.stack(
                                residuals_collector[feature][residual_type], 
                                axis=0  # This creates a new dimension for the sequence
                            )
                        else:
                            # Handle case where no residuals were collected
                            combined_residuals[feature][residual_type] = np.array([])
                
                # Create ResidualFeatures object for this target time
                residual_features = ResidualFeatures(
                    time=target_t,
                    condition=condition,
                    residuals=combined_residuals
                )
                
                self.all_residuals.append(residual_features)