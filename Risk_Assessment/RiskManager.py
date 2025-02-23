from Risk_Assessment.DataLoader import DataLoader
from Risk_Assessment.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Risk_Assessment.FeatureExtractor import DimensionReductionFeatureExtractor, TemporalFeatureExtractor
from Risk_Assessment.FaultDetector import FaultDetector
from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES

from collections import defaultdict
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
        self.approach = None
        
        self.features_by_condition = {}
        self.labels_by_condition = {}
        self.all_residuals = []  # Store all residuals for fitting PCA
    
    def generate_all_residuals(self, loaded_data_dict):
        """Generate residuals for all conditions using pre-loaded data."""
        self.all_residuals = []  # Reset residuals list
        
        # Process all conditions using pre-loaded data
        for condition in self.config['conditions']:
            loaded_data = loaded_data_dict[condition]
            self.approach = 'Approach1'
            self.process_condition(loaded_data, condition)
            #self.process_condition_aligned(loaded_data, condition)

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
    
    def process_condition_aligned(self, loaded_data, condition: str):
        """
        Process a single condition with aligned predictions and observations.
        For each target time t, collects horizon H predictions and observations made between t-H and t-1,
        each predicting/observing the appropriate offset into the future to align with time t.
        
        Args:
            loaded_data: LoadedData object containing dataset and predictions
            condition: String identifier for the condition being processed
        """
        horizon = self.horizon
        start_idx = horizon  # Need enough history for first prediction
        end_idx = len(loaded_data.dataset)
        
        for target_t in range(start_idx, end_idx):
            # Initialize collected predictions and observations
            collected_predictions = {
                'position_mean': [],
                'position_var': [],
                'velocity_mean': [],
                'velocity_var': [],
                'steering_mean': [],
                'steering_var': [],
                'acceleration_mean': [],
                'acceleration_var': [],
                'object_distance': [],
                'traffic_light_detected': []
            }
            
            collected_observations = {
                'position': [],
                'velocity': [],
                'steering': [],
                'acceleration': [],
                'object_distance': [],
                'traffic_light_detected': []
            }
            
            # Set origin time (t - H)
            origin_t = target_t - horizon
            
            # Skip if we don't have enough history
            if origin_t < 0:
                continue
                
            valid_sequence = True
            # Collect predictions and observations made between origin_t and target_t-1
            for i in range(horizon):
                pred_t = origin_t + i  # prediction time
                offset = horizon - i    # offset into prediction
                
                # Skip if prediction is out of bounds
                if pred_t >= len(loaded_data.predictions) or pred_t < 0:
                    valid_sequence = False
                    break
                    
                # Get prediction and corresponding ground truth
                prediction = loaded_data.predictions[pred_t]
                _, future, _, _ = loaded_data.dataset[pred_t]
                
                if prediction is None or len(prediction) == 0:
                    valid_sequence = False
                    break
                    
                # Extract predictions for standard features
                for feature in ['position', 'velocity', 'steering', 'acceleration']:
                    mean_key = f'{feature}_mean'
                    var_key = f'{feature}_var'
                    
                    if mean_key in prediction and var_key in prediction:
                        if offset <= prediction[mean_key].shape[0]:
                            # Extract specific timestep prediction
                            collected_predictions[mean_key].append(prediction[mean_key][offset-1])
                            collected_predictions[var_key].append(prediction[var_key][offset-1])
                            
                            # Extract corresponding ground truth
                            if feature in future:
                                obs = future[feature][offset-1]
                                collected_observations[feature].append(obs.detach().cpu().numpy())
                        else:
                            valid_sequence = False
                            break
                            
                # Extract predictions and observations for special features
                for feature in ['object_distance', 'traffic_light_detected']:
                    if feature in prediction and offset <= prediction[feature].shape[0]:
                        collected_predictions[feature].append(prediction[feature][offset-1])
                        
                        # Extract corresponding ground truth
                        if feature in future:
                            obs = future[feature][offset-1]
                            collected_observations[feature].append(obs.detach().cpu().numpy())
                    else:
                        valid_sequence = False
                        break
                        
                if not valid_sequence:
                    break
            
            # Only process if we have a valid sequence
            if valid_sequence:
                # Stack collected predictions
                predictions_dict = {}
                for key in collected_predictions:
                    if collected_predictions[key]:
                        predictions_dict[key] = np.stack(collected_predictions[key])
                
                # Stack collected observations
                observations_dict = {}
                for key in collected_observations:
                    if collected_observations[key]:
                        observations_dict[key] = torch.tensor(
                            np.stack(collected_observations[key]),
                            dtype=torch.float32
                        )
                
                # Structure predictions in the format expected by compute_residuals
                final_predictions = {}
                for feature in ['position', 'velocity', 'steering', 'acceleration']:
                    mean_key = f'{feature}_mean'
                    var_key = f'{feature}_var'
                    if mean_key in predictions_dict and var_key in predictions_dict:
                        final_predictions[mean_key] = predictions_dict[mean_key]
                        final_predictions[var_key] = predictions_dict[var_key]
                
                for feature in ['object_distance', 'traffic_light_detected']:
                    if feature in predictions_dict:
                        final_predictions[feature] = predictions_dict[feature]
                
                try:
                    # Compute residuals using the collected sequence
                    residual_output = self.residual_generator.compute_residuals(
                        final_predictions,
                        observations_dict,  # Now using collected observations
                        target_t
                    )
                    
                    # Store the residual features
                    residual_features = ResidualFeatures(
                        time=target_t,
                        condition=condition,
                        residuals=residual_output.residuals
                    )
                    
                    self.all_residuals.append(residual_features)
                    
                except Exception as e:
                    print(f"Error computing residuals for target {target_t}: {str(e)}")

    def plot_confusion_matrix(self, results: Dict[str, Any]):
        """Plot and save confusion matrix visualization."""
        import os
        save_path = f'Results/{self.approach}/confusion_matrix.png'
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

    def plot_explained_variance(self):
        """Plot cumulative explained variance for each feature-residual type combination."""
        save_path = f'Results/{self.approach}/explained_variance.png'
        
        if not self.feature_extractor or not self.feature_extractor.is_fitted:
            raise ValueError("Feature extractor not fitted yet.")
        
        cumulative_variance = self.feature_extractor.get_cumulative_explained_variance()
        
        plt.figure(figsize=(15, 10))
        
        line_styles = ['-', '--', ':', '-.']
        colors = plt.cm.tab10(np.linspace(0, 1, len(cumulative_variance)))
        
        for feature_idx, (feature, residual_dict) in enumerate(cumulative_variance.items()):
            for residual_idx, (residual_type, variance) in enumerate(residual_dict.items()):
                label = f"{feature}-{residual_type}"
                style = line_styles[residual_idx % len(line_styles)]
                color = colors[feature_idx]
                
                plt.plot(range(1, len(variance) + 1), variance, 
                        marker='o', linestyle=style, color=color, label=label)
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Explained variance plot saved to {save_path}")

    def plot_feature_importance(self, results: Dict[str, Any]):
        """Plot feature importance by PCA component and residual type."""
        save_path = f'Results/{self.approach}/feature_importance.png'
        
        if not self.feature_extractor:
            raise ValueError("Feature extractor not initialized.")
        
        importance_by_component = self.feature_extractor.get_feature_importance_by_component(
            self.fault_detector.pipeline.named_steps['classifier']
        )
        
        # First, create a complete matrix of all possible combinations
        all_features = list(importance_by_component.keys())
        all_residual_types = set()
        max_components = {}
        
        # Find all residual types and max components for each feature
        for feature, residual_dict in importance_by_component.items():
            for residual_type in residual_dict.keys():
                all_residual_types.add(residual_type)
            for residual_type, component_dict in residual_dict.items():
                if feature not in max_components:
                    max_components[feature] = 0
                max_components[feature] = max(max_components[feature], max(component_dict.keys()))
        
        all_residual_types = list(all_residual_types)
        
        # Create figure with adjusted size
        plt.figure(figsize=(max(15, len(all_features) * 3), 10))
        
        # Set up the bar positions
        n_features = len(all_features)
        n_residuals = len(all_residual_types)
        bar_width = 0.15  # Adjust this to change bar width
        
        # Create color map for residual types
        colors = plt.cm.Set3(np.linspace(0, 1, n_residuals))
        
        # Plot each residual type
        for i, residual_type in enumerate(all_residual_types):
            positions = []
            values = []
            labels = []
            
            for j, feature in enumerate(all_features):
                if residual_type in importance_by_component[feature]:
                    component_scores = importance_by_component[feature][residual_type]
                    for component, score in component_scores.items():
                        # Calculate x position for this bar
                        x_pos = j + (i - n_residuals/2) * bar_width
                        positions.append(x_pos)
                        values.append(score)
                        labels.append(f"PC{component}")
            
            if positions:  # Only plot if we have data for this residual type
                plt.bar(positions, values, bar_width, 
                    label=residual_type, color=colors[i], alpha=0.8)
        
        # Customize plot
        plt.xlabel('Feature')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance by Residual Type and PCA Component')
        
        # Set x-ticks at feature positions
        plt.xticks(range(len(all_features)), all_features, rotation=45, ha='right')
        
        # Add legend
        plt.legend(title='Residual Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")

    def print_feature_importance_by_component(self):
        """Print detailed feature importance breakdown by PCA component."""
        if not self.feature_extractor:
            raise ValueError("Feature extractor not initialized.")
            
        importance_by_component = self.feature_extractor.get_feature_importance_by_component(
            self.fault_detector.pipeline.named_steps['classifier']
        )
        
        # Prepare table data with better organization
        table_data = []
        
        # Track total importance for each feature and residual type
        feature_totals = defaultdict(float)
        residual_totals = defaultdict(float)
        
        # Collect data and calculate totals
        for feature, residual_dict in importance_by_component.items():
            for residual_type, component_scores in residual_dict.items():
                feature_sum = sum(component_scores.values())
                feature_totals[feature] += feature_sum
                residual_totals[residual_type] += feature_sum
                
                for component, score in component_scores.items():
                    # Safe division to handle zero feature_sum
                    percentage = (score/feature_sum)*100 if feature_sum > 0 else 0.0
                    table_data.append([
                        feature,
                        residual_type,
                        f"PC{component}",
                        f"{score:.4f}",
                        f"{percentage:.1f}%"
                    ])
        
        # Sort by raw importance score (index 3, removing the % sign)
        table_data.sort(key=lambda x: float(x[3]), reverse=True)
        
        # Print main table
        print("\nFeature Importance by Residual Type and PCA Component:")
        print(tabulate(table_data, 
                    headers=['Feature', 'Residual Type', 'Component', 
                            'Importance Score', '% of Feature'],
                    tablefmt='grid'))
        
        # Print summary tables
        print("\nTotal Importance by Feature:")
        feature_summary = [[feature, f"{total:.4f}"] 
                        for feature, total in sorted(feature_totals.items(), 
                                                    key=lambda x: x[1], reverse=True)]
        print(tabulate(feature_summary, 
                    headers=['Feature', 'Total Importance'],
                    tablefmt='grid'))
        
        print("\nTotal Importance by Residual Type:")
        residual_summary = [[res_type, f"{total:.4f}"] 
                        for res_type, total in sorted(residual_totals.items(), 
                                                    key=lambda x: x[1], reverse=True)]
        print(tabulate(residual_summary, 
                    headers=['Residual Type', 'Total Importance'],
                    tablefmt='grid'))

    def _print_metrics_table(self, metrics: dict):
        """Print a formatted table of metrics."""
        from tabulate import tabulate
        
        # Prepare table data
        table_data = []
        for i in range(len(metrics['n_components'])):
            row = [
                metrics['n_components'][i],
                f"{metrics['accuracy'][i]:.4f}",
                f"{metrics['precision_macro'][i]:.4f}",
                f"{metrics['recall_macro'][i]:.4f}",
                f"{metrics['f1_macro'][i]:.4f}",
                f"{metrics['execution_time'][i]:.2f}"
            ]
            table_data.append(row)
        
        # Print table
        print("\nMetrics by Number of Components:")
        print(tabulate(table_data,
                    headers=['Components', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time (s)'],
                    tablefmt='grid'))

    def run_fault_detection(self, loaded_data_dict, n_components=None):
        """Run complete fault detection pipeline with dimension reduction"""
        # Initialize feature extractor with specified number of components
        #self.feature_extractor = DimensionReductionFeatureExtractor(n_components=n_components)
        self.feature_extractor = TemporalFeatureExtractor(n_components=n_components)
        
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
        
        # Initialize metrics storage
        metrics_by_component = {
            'n_components': [],
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_weighted': [],
            'recall_weighted': [],
            'f1_weighted': [],
            'execution_time': []
        }
        
        import time
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        for n_components in component_range:
            print(f"\nTesting with {n_components} components:")
            
            # Time the execution
            start_time = time.time()
            test_results = self.run_fault_detection(loaded_data_dict, n_components=n_components)
            execution_time = time.time() - start_time
            
            # Extract true and predicted labels from confusion matrix
            true_labels = []
            pred_labels = []
            for i in range(len(test_results['confusion_matrix'])):
                for j in range(len(test_results['confusion_matrix'])):
                    true_labels.extend([i] * test_results['confusion_matrix'][i][j])
                    pred_labels.extend([j] * test_results['confusion_matrix'][i][j])
            
            # Store all metrics
            metrics_by_component['n_components'].append(n_components)
            metrics_by_component['accuracy'].append(test_results['accuracy'])
            metrics_by_component['precision_macro'].append(precision_score(true_labels, pred_labels, average='macro'))
            metrics_by_component['recall_macro'].append(recall_score(true_labels, pred_labels, average='macro'))
            metrics_by_component['f1_macro'].append(f1_score(true_labels, pred_labels, average='macro'))
            metrics_by_component['precision_weighted'].append(precision_score(true_labels, pred_labels, average='weighted'))
            metrics_by_component['recall_weighted'].append(recall_score(true_labels, pred_labels, average='weighted'))
            metrics_by_component['f1_weighted'].append(f1_score(true_labels, pred_labels, average='weighted'))
            metrics_by_component['execution_time'].append(execution_time)
        
        # Print results table
        self._print_metrics_table(metrics_by_component)
