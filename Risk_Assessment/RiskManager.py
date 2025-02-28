from Risk_Assessment.DataLoader import DataLoader
from Risk_Assessment.ResidualGenerator import ResidualGenerator, ResidualFeatures
from Risk_Assessment.FeatureExtractor import TemporalFeatureExtractor
from Risk_Assessment.FaultDetector import FaultDetector
from Risk_Assessment.FaultDetectionConfig import FEATURE_NAMES

from collections import defaultdict
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import torch
import os
import time 

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

        # Tracking for residual generation time
        self.total_residual_generation_time = 0.0
        self.total_residual_samples = 0
    
    def generate_all_residuals(self, loaded_data_dict):
        """Generate residuals for all conditions using pre-loaded data."""
        self.all_residuals = []  # Reset residuals list
        
        # Reset timing metrics
        self.total_residual_generation_time = 0.0
        self.total_residual_samples = 0
        
        # Process all conditions using pre-loaded data
        for condition in self.config['conditions']:
            loaded_data = loaded_data_dict[condition]
            self.approach = 'Approach1'
            self.process_condition(loaded_data, condition)

    def process_condition(self, loaded_data, condition: str):        
        for t in range(len(loaded_data.dataset)):
            past, future, _, _ = loaded_data.dataset[t]
            if t < len(loaded_data.predictions):
                # Measure residual generation time
                start_time = time.time()
                
                residual_output = self.residual_generator.compute_residuals(
                    loaded_data.predictions[t],
                    future,
                    t
                )
                
                # Calculate and accumulate time
                residual_time = time.time() - start_time
                self.total_residual_generation_time += residual_time
                self.total_residual_samples += 1
                
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
                    # Measure residual generation time
                    start_time = time.time()

                    # Compute residuals using the collected sequence
                    residual_output = self.residual_generator.compute_residuals(
                        final_predictions,
                        observations_dict,  # Now using collected observations
                        target_t
                    )
                    # Calculate and accumulate time
                    residual_time = time.time() - start_time
                    self.total_residual_generation_time += residual_time
                    self.total_residual_samples += 1

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
        """Plot and save confusion matrix visualization with custom labels."""
        save_path = f'Results/{self.approach}/confusion_matrix.png'
        # Create Results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get confusion matrix and labels
        conf_matrix = results['confusion_matrix']
        report_lines = results['classification_report'].split('\n')
        class_labels = [line.split()[0] for line in report_lines[1:-5] if line.strip()]
        
        # Define custom labels mapping
        custom_labels = {
            'Normal': 'Normal',
            'Noisy_Camera': 'Camera Fault',
            'Noisy_Lidar': 'LiDAR Fault',
            'Noisy_IMU': 'IMU Fault'
            # Add more mappings as needed for other conditions
        }
        
        # Map original labels to custom labels
        display_labels = [custom_labels.get(label, label) for label in class_labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Confusion matrix plot saved to {save_path}")

    def plot_explained_variance(self, threshold: float = 0.95) -> int:
        """
        Plot cumulative explained variance for each feature-residual type combination,
        as well as a simplified plot per feature. Calculate and return the optimal 
        number of components based on the given threshold.
        
        Args:
            threshold: Variance threshold to determine optimal number of components (default: 0.95)
            
        Returns:
            int: Recommended number of components to use based on threshold
        """
        # Create Results directory if it doesn't exist
        os.makedirs(f'Results/{self.approach}', exist_ok=True)
        
        if not self.feature_extractor or not self.feature_extractor.is_fitted:
            raise ValueError("Feature extractor not fitted yet.")
        
        cumulative_variance = self.feature_extractor.get_cumulative_explained_variance()
        
        # Calculate recommended components for each feature-residual combination
        feature_component_counts = {}
        max_components = 0
        
        for feature, residual_dict in cumulative_variance.items():
            feature_component_counts[feature] = {}
            
            for residual_type, variance in residual_dict.items():
                # Track maximum component count
                max_components = max(max_components, len(variance))
                
                # Calculate recommended components based on threshold
                recommended_components = np.argmax(variance >= threshold) + 1
                if recommended_components > len(variance) or np.all(variance < threshold):
                    # In case threshold never reached
                    recommended_components = len(variance)
                    
                feature_component_counts[feature][residual_type] = recommended_components
        
        # Calculate the global recommended component count 
        # (take max across all feature-residual combinations)
        recommended_components = max(
            max(counts.values()) for counts in feature_component_counts.values()
        )
        
        # Create a larger figure to accommodate the bigger legend
        plt.figure(figsize=(20, 14))
        
        # Use a more distinct and colorblind-friendly color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(cumulative_variance)))
        
        # Set better fonts and sizes
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 20,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16,  # Increased legend font size
            'legend.title_fontsize': 18  # Added larger legend title font size
        })
        
        # Define distinct markers for different residual types
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
        line_styles = ['-', '--', '-.', ':']
        
        # Plot each feature-residual combination
        for feature_idx, (feature, residual_dict) in enumerate(cumulative_variance.items()):
            for residual_idx, (residual_type, variance) in enumerate(residual_dict.items()):
                # Plot the curve with formatted labels
                # Capitalize feature name and format residual type
                feature_display = feature.capitalize()
                # Rename specific features
                if feature == "object_distance":
                    feature_display = "Object Detection Flag"
                elif feature == "traffic_light_detected":
                    feature_display = "Traffic Light Status Flag"
                    
                # Format residual type with spaces
                residual_display = residual_type.replace("_", " ")
                
                label = f"{feature_display} - {residual_display}"
                style = line_styles[residual_idx % len(line_styles)]
                color = colors[feature_idx]
                marker = markers[residual_idx % len(markers)]
                
                plt.plot(range(1, len(variance) + 1), variance, 
                        marker=marker, linestyle=style, color=color, label=label,
                        linewidth=2, markersize=8)
                
                # Highlight recommended component count
                rec_comp = feature_component_counts[feature][residual_type]
                plt.plot(rec_comp, variance[rec_comp-1], 
                        marker='D', color=color, markersize=12, markeredgecolor='black')
        
        # Add threshold line - Make it darker and thicker
        plt.axhline(y=threshold, color='darkred', linestyle='-', alpha=0.7, linewidth=3,
                    label=f'Threshold ({threshold})')
        
        # Add only one vertical line for the final recommended component count
        plt.axvline(x=recommended_components, color='black', linestyle='-', alpha=0.7, linewidth=3,
                    label=f'Recommended Components ({recommended_components})')
        
        plt.xlabel('Number of Components', fontweight='bold')
        plt.ylabel('Cumulative Explained Variance Ratio', fontweight='bold')
        plt.title('Explained Variance Ratio by Feature and Residual Type', fontweight='bold', pad=20)
        
        # MODIFIED: Move legend to the bottom right inside the plot
        plt.legend(
            loc='lower right',  # Position in bottom right corner
            frameon=True,
            edgecolor='black',
            fancybox=False,
            markerscale=1.5,  # Slightly smaller markers to fit inside
            handlelength=2.5,  # Slightly shorter line segments
            handleheight=1.5,  # Keep legend lines thick
            labelspacing=0.7,  # Slightly tighter spacing to save space
            title='Feature-Residual Type Combinations',
            title_fontsize=18,
            ncol=2  # Organize legend in 2 columns for better readability
        )
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure axis starts at 0.4
        plt.ylim(0.4, 1.05)
        plt.xlim(0, max_components + 1)
        
        # Add text annotation for the recommended components
        plt.text(recommended_components + 0.2, 0.5, 
                f'Recommended: {recommended_components}',
                rotation=90, verticalalignment='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'Results/{self.approach}/explained_variance_detailed.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Recommended number of components based on {threshold} variance threshold: {recommended_components}")
        
        return recommended_components

    def plot_feature_importance(self, results: Dict[str, Any]):
        """Plot feature importance by PCA component and residual type."""
        save_path = f'Results/{self.approach}/feature_importance.png'
        features_save_path = f'Results/{self.approach}/feature_importance_by_features.png'
        
        if not self.feature_extractor:
            raise ValueError("Feature extractor not initialized.")
        
        importance_by_component = self.feature_extractor.get_feature_importance_by_component(
            self.fault_detector.pipeline.named_steps['classifier']
        )
        
        # First, create a complete matrix of all possible combinations
        all_features = list(importance_by_component.keys())
        all_residual_types = set()
        
        # Find all residual types
        for feature, residual_dict in importance_by_component.items():
            for residual_type in residual_dict.keys():
                all_residual_types.add(residual_type)
        
        # Define the specific order and pretty labels for residual types
        residual_order = ['raw', 'kl_divergence', 'cusum']
        residual_labels = {
            'raw': 'Raw',
            'kl_divergence': 'KL Divergence',
            'cusum': 'CUSUM'
        }
        
        # Filter residual types to only include those in our desired order and that exist in the data
        ordered_residual_types = [rt for rt in residual_order if rt in all_residual_types]
        
        # Define pretty labels for features
        feature_labels = {}
        for feature in all_features:
            if feature == 'object_distance':
                feature_labels[feature] = 'Object Detection Flag'
            elif feature == 'traffic_light_detected':
                feature_labels[feature] = 'Traffic Light Status Flag'
            else:
                # Capitalize the first letter of each feature
                feature_labels[feature] = feature.capitalize()
        
        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 22,  # Increased from 18
            'legend.title_fontsize': 24  # Increased from 20
        })
        
        # Create figure with adjusted size
        plt.figure(figsize=(max(16, len(all_features) * 3.5), 12))
        
        # Set up the bar positions
        n_features = len(all_features)
        n_residuals = len(ordered_residual_types)
        bar_width = 0.25  # Wider bars
        
        # Calculate total importance for percentage scaling
        total_importance = 0
        for feature, residual_dict in importance_by_component.items():
            for residual_type, component_dict in residual_dict.items():
                total_importance += sum(component_dict.values())
        
        # Create color map for residual types - use a more distinct colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_residuals))
        
        # Plot each residual type
        for i, residual_type in enumerate(ordered_residual_types):
            positions = []
            values = []
            
            for j, feature in enumerate(all_features):
                if residual_type in importance_by_component[feature]:
                    component_scores = importance_by_component[feature][residual_type]
                    # Combine all component scores for this feature/residual type
                    total_score = sum(component_scores.values())
                    # Convert to percentage if total_importance is not zero
                    if total_importance > 0:
                        total_score = (total_score / total_importance) * 100
                    
                    x_pos = j + (i - n_residuals/2) * bar_width
                    positions.append(x_pos)
                    values.append(total_score)
            
            if positions:  # Only plot if we have data for this residual type
                plt.bar(positions, values, bar_width, 
                    label=residual_labels[residual_type], color=colors[i], alpha=0.9,
                    edgecolor='black', linewidth=0.5)  # Add border for better definition
        
        # Customize plot
        plt.xlabel('Feature', fontweight='bold')
        plt.ylabel('Importance Score (%)', fontweight='bold')
        plt.title('Feature Importance by Residual Types', 
                fontweight='bold', pad=20)
        
        # Set x-ticks at feature positions with pretty labels
        pretty_feature_labels = [feature_labels[feature] for feature in all_features]
        plt.xticks(range(len(all_features)), pretty_feature_labels, rotation=0, ha='center')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend with improved styling - now inside the plot at top left
        plt.legend(title='Residual Type', 
                loc='upper left',  # Changed from outside to inside
                frameon=True,
                fancybox=True,
                shadow=True,
                markerscale=2.0,
                handlelength=3.0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
        
        # Create a plot showing overall importance by features
        plt.figure(figsize=(max(12, len(all_features) * 2.5), 10))
        
        # Calculate total importance for each feature across all residual types
        feature_importance = {}
        for feature in all_features:
            feature_importance[feature] = 0
            for residual_type in all_residual_types:
                if residual_type in importance_by_component[feature]:
                    component_scores = importance_by_component[feature][residual_type]
                    feature_importance[feature] += sum(component_scores.values())
        
        # Convert to percentages
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] = (feature_importance[feature] / total_importance) * 100
        
        # Sort features by importance
        sorted_features = sorted(all_features, key=lambda x: feature_importance[x], reverse=True)
        sorted_values = [feature_importance[feature] for feature in sorted_features]
        sorted_labels = [feature_labels[feature] for feature in sorted_features]
        
        # Plot bars with a blue color palette
        bar_colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(sorted_features)))
        
        plt.bar(range(len(sorted_features)), sorted_values, color=bar_colors, 
                alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_values):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=14, fontweight='bold')
        
        # Customize plot
        plt.xlabel('Feature', fontweight='bold')
        plt.ylabel('Total Importance Score (%)', fontweight='bold')
        plt.title('Overall Feature Importance', fontweight='bold', pad=20)
        
        # Set x-ticks with sorted feature labels
        plt.xticks(range(len(sorted_features)), sorted_labels, rotation=30, ha='right')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save feature importance plot
        plt.savefig(features_save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Feature importance by features plot saved to {features_save_path}")

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

    def _plot_timing_breakdown(self, summary_df):
        """Create a stacked bar chart showing the breakdown of prediction time components"""
        if not all(col in summary_df.columns for col in 
                ['avg_feature_extraction_time_per_sample', 
                'avg_classifier_prediction_time_per_sample',
                'avg_residual_generation_time_per_sample']):
            # Skip if we don't have all the detailed timing metrics
            return
        
        # Create Results directory if it doesn't exist
        os.makedirs(f'Results/{self.approach}', exist_ok=True)
        
        # Convert all times to milliseconds
        fe_time_ms = summary_df['avg_feature_extraction_time_per_sample'] * 1000
        clf_time_ms = summary_df['avg_classifier_prediction_time_per_sample'] * 1000
        res_gen_time_ms = summary_df['avg_residual_generation_time_per_sample'] * 1000
        
        # Find best n_estimators based on CV accuracy
        best_est = summary_df.loc[summary_df['test_accuracy'].idxmax(), 'n_estimators']
        
        # Create a bar chart for a few selected n_estimators values
        # Choose a representative sample of n_estimators values
        if len(summary_df) > 10:
            indices = np.linspace(0, len(summary_df)-1, 10, dtype=int)
            selected_df = summary_df.iloc[indices].copy()
        else:
            selected_df = summary_df.copy()
        
        # Ensure the best n_estimators value is included
        if best_est not in selected_df['n_estimators'].values:
            best_row = summary_df[summary_df['n_estimators'] == best_est]
            if not best_row.empty:
                selected_df = pd.concat([selected_df, best_row])
                selected_df = selected_df.sort_values('n_estimators')
        
        # Extract timing data for the selected n_estimators values
        n_estimators = selected_df['n_estimators']
        fe_selected = selected_df['avg_feature_extraction_time_per_sample'] * 1000
        clf_selected = selected_df['avg_classifier_prediction_time_per_sample'] * 1000
        res_selected = selected_df['avg_residual_generation_time_per_sample'] * 1000
        
        # Create figure
        plt.figure(figsize=(12, 7))
        
        # Create stacked bar chart
        width = 0.7
        bottom_vals = np.zeros(len(n_estimators))
        
        # First bar (residual generation time)
        p1 = plt.bar(n_estimators, res_selected, width, label='Residual Generation', 
                    color='#1f77b4', edgecolor='black', linewidth=0.5)
        bottom_vals += res_selected
        
        # Second bar (feature extraction time)
        p2 = plt.bar(n_estimators, fe_selected, width, bottom=bottom_vals, 
                    label='Feature Extraction', color='#ff7f0e', 
                    edgecolor='black', linewidth=0.5)
        bottom_vals += fe_selected
        
        # Third bar (classifier prediction time)
        p3 = plt.bar(n_estimators, clf_selected, width, bottom=bottom_vals, 
                    label='Classifier Prediction', color='#2ca02c', 
                    edgecolor='black', linewidth=0.5)
        
        # Highlight the best n_estimators bar
        if best_est in n_estimators.values:
            # Find the position of best_est in the selected dataframe
            best_idx = n_estimators.values.tolist().index(best_est)
            
            # Draw an outline around the best bar
            plt.axvline(x=best_est, color='red', linestyle='--', alpha=0.7)
            plt.text(best_est, bottom_vals[best_idx] + 1, f'Best\n{best_est}', 
                    ha='center', va='bottom', color='red', fontweight='bold')
        else:
            print(f"Warning: Best n_estimators ({best_est}) not found in selected subset")
                
        # Add total time labels on top of each bar
        for i, (n_est, total) in enumerate(zip(n_estimators, bottom_vals)):
            plt.text(n_est, total + 0.5, f'{total:.1f} ms', ha='center', va='bottom')
        
        # Add percentage labels within each segment
        for i, (n_est, res, fe, clf) in enumerate(zip(n_estimators, res_selected, fe_selected, clf_selected)):
            total = res + fe + clf
            
            # Only show percentages if the segment is large enough
            if res/total > 0.05:  # Only show if > 5%
                plt.text(n_est, res/2, f'{res/total*100:.0f}%', 
                        ha='center', va='center', color='black', fontweight='bold')
            
            if fe/total > 0.05:
                plt.text(n_est, res + fe/2, f'{fe/total*100:.0f}%', 
                        ha='center', va='center', color='black', fontweight='bold')
            
            if clf/total > 0.05:
                plt.text(n_est, res + fe + clf/2, f'{clf/total*100:.0f}%', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        # Add labels and title
        plt.xlabel('Number of Trees (Estimators)', fontweight='bold')
        plt.ylabel('Time per Sample (milliseconds)', fontweight='bold')
        plt.title('Breakdown of Prediction Time Components', fontweight='bold')
        plt.legend(loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'Results/{self.approach}/timing_breakdown.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_n_estimators_comparison(self, summary_df):
        """Plot comparison of metrics across different n_estimators values"""
        # Check which columns are available in the DataFrame
        available_columns = set(summary_df.columns)
        
        # Create Results directory if it doesn't exist
        os.makedirs(f'Results/{self.approach}', exist_ok=True)
        
        # Plot accuracy metrics if available
        if 'test_accuracy' in available_columns:
            plt.figure(figsize=(10, 6))
            plt.plot(summary_df['n_estimators'], summary_df['test_accuracy'], 'o-', label='Test Accuracy', color='#1f77b4')
            
            if 'train_accuracy' in available_columns:
                plt.plot(summary_df['n_estimators'], summary_df['train_accuracy'], 's-', label='Train Accuracy', color='#ff7f0e')
            
            if 'cv_accuracy' in available_columns:
                plt.plot(summary_df['n_estimators'], summary_df['cv_accuracy'], '^-', label='CV Accuracy', color='#2ca02c')
            
            # Find best n_estimators based on CV accuracy
            best_est = summary_df.loc[summary_df['test_accuracy'].idxmax(), 'n_estimators']
            best_acc = summary_df.loc[summary_df['test_accuracy'].idxmax(), 'test_accuracy']
            
            # Add vertical line and annotation for best n_estimators
            plt.axvline(x=best_est, color='green', linestyle='--', alpha=0.7,
                    label=f'Best #Estimators = {best_est}')
            plt.scatter([best_est], [best_acc], color='green', s=100, zorder=5)
            plt.annotate(f'Best: {best_acc:.4f}', 
                        (best_est, best_acc),
                        xytext=(10, -30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'))
            
            plt.title('Accuracy vs. Number of Trees')
            plt.xlabel('Number of Trees (Estimators)')
            plt.ylabel('Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'Results/{self.approach}/n_estimators_accuracy.png', bbox_inches='tight', dpi=300)
            plt.close()
        
        # Plot prediction timing metrics separately - with combined times for feature extraction and classification
        if 'avg_prediction_time_per_sample' in available_columns:
            # Determine if we have detailed timing metrics
            has_detailed_timing = all(col in available_columns for col in 
                                    ['avg_feature_extraction_time_per_sample', 
                                    'avg_classifier_prediction_time_per_sample',
                                    'avg_residual_generation_time_per_sample'])
            
            # Create more compact figure
            plt.figure(figsize=(10, 5))
            
            # Convert time to milliseconds for better readability
            per_sample_time_ms = summary_df['avg_prediction_time_per_sample'] * 1000
            
            # Calculate combined feature extraction and classifier time (excluding residual generation)
            if has_detailed_timing:
                fe_time_ms = summary_df['avg_feature_extraction_time_per_sample'] * 1000
                clf_time_ms = summary_df['avg_classifier_prediction_time_per_sample'] * 1000
                res_gen_time_ms = summary_df['avg_residual_generation_time_per_sample'] * 1000
                combined_time_ms = fe_time_ms + clf_time_ms + res_gen_time_ms
                
                # Plot the combined timing metric (excluding residual generation)
                plt.plot(summary_df['n_estimators'], combined_time_ms, 'o-',
                        label='Total Time per Sample', color='#9467bd', linewidth=2)
            else:
                # If detailed metrics not available, just plot total time
                plt.plot(summary_df['n_estimators'], per_sample_time_ms, 'o-',
                        label='Total Time per Sample', color='#9467bd', linewidth=2)
            
            # Find best n_estimators (from cv_accuracy)
            best_est = summary_df.loc[summary_df['test_accuracy'].idxmax(), 'n_estimators']
            
            # Get corresponding timing for best n_estimators
            best_row = summary_df[summary_df['n_estimators'] == best_est]
            if not best_row.empty and has_detailed_timing:
                best_row = best_row.iloc[0]
                best_fe_time_ms = best_row['avg_feature_extraction_time_per_sample'] * 1000
                best_clf_time_ms = best_row['avg_classifier_prediction_time_per_sample'] * 1000
                best_res_gen_time_ms = best_row['avg_residual_generation_time_per_sample'] * 1000
                best_combined_time_ms = best_fe_time_ms + best_clf_time_ms + best_res_gen_time_ms
                
                # Add vertical line at best n_estimators
                plt.axvline(x=best_est, color='green', linestyle='--', alpha=0.7,
                        label=f'Best #Estimators = {best_est}')
                
                # Add annotation for time at best n_estimators
                plt.scatter([best_est], [best_combined_time_ms], color='green', s=80, zorder=5)
                plt.annotate(f'{best_combined_time_ms:.2f} ms', 
                            (best_est, best_combined_time_ms),
                            xytext=(10, 10),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='green'))
            
            plt.title('Risk Assessment Time per Sample vs. Classifier Size')
            plt.xlabel('Number of Trees (Estimators)')
            plt.ylabel('Time (milliseconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'Results/{self.approach}/n_estimators_prediction_time.png', 
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            # Plot Time vs Accuracy tradeoff
            if 'test_accuracy' in available_columns and has_detailed_timing:
                plt.figure(figsize=(10, 5))
                
                # Use combined timing without residual generation for the tradeoff plot
                combined_time_ms = fe_time_ms + clf_time_ms + res_gen_time_ms
                
                # Create scatter plot with n_estimators as size
                sizes = summary_df['n_estimators'] / 3 + 20  # Scale for visibility
                
                # Add color gradient based on n_estimators
                norm = plt.Normalize(summary_df['n_estimators'].min(), 
                                summary_df['n_estimators'].max())
                colors = plt.cm.viridis(norm(summary_df['n_estimators']))
                
                scatter = plt.scatter(combined_time_ms, 
                                summary_df['test_accuracy'], 
                                s=sizes, 
                                c=summary_df['n_estimators'],
                                cmap='viridis',
                                alpha=0.7)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Number of Trees')
                
                # Find best n_estimators again
                best_idx = summary_df['test_accuracy'].idxmax()
                best_est = summary_df.loc[best_idx, 'n_estimators']
                best_fe_time_ms = summary_df.loc[best_idx, 'avg_feature_extraction_time_per_sample'] * 1000
                best_clf_time_ms = summary_df.loc[best_idx, 'avg_classifier_prediction_time_per_sample'] * 1000
                best_combined_time_ms = best_fe_time_ms + best_clf_time_ms + best_res_gen_time_ms
                best_acc = summary_df.loc[best_idx, 'test_accuracy']
                
                # Highlight the best point
                plt.scatter([best_combined_time_ms], [best_acc], 
                        s=200, edgecolor='red', facecolor='none', 
                        linewidth=2, zorder=10)
                
                plt.annotate(f'Best: n={best_est}\n{best_acc:.4f}, {best_combined_time_ms:.2f}ms', 
                            (best_combined_time_ms, best_acc),
                            xytext=(20, -50),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'))
                
                plt.title('Accuracy vs. Risk Assessment Time Tradeoff')
                plt.xlabel('Total Time per Sample (ms)')
                plt.ylabel('Accuracy')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(f'Results/{self.approach}/accuracy_vs_time_tradeoff.png', 
                        bbox_inches='tight', dpi=300)
                plt.close()
    
    def run_fault_detection(self, loaded_data_dict, n_components=None, variance_threshold=0.95):
        """
        Run complete fault detection pipeline with dimension reduction and evaluate 
        model performance with different n_estimators values.
        
        Args:
            loaded_data_dict: Dictionary with loaded data for each condition
            n_components: Number of components to use (if None, will be determined automatically)
            variance_threshold: Threshold for explained variance when determining optimal components
            
        Returns:
            Dictionary with classification results for different n_estimators values
        """
        # Initialize feature extractor with specified number of components
        self.feature_extractor = TemporalFeatureExtractor(n_components=n_components)
        
        # Create results directory if it doesn't exist
        os.makedirs(f'Results/{self.approach}', exist_ok=True)
        
        # Generate residuals
        print("Generating residuals...")
        self.generate_all_residuals(loaded_data_dict)

        # Calculate average residual generation time per sample
        avg_residual_time_per_sample = 0.0
        if self.total_residual_samples > 0:
            avg_residual_time_per_sample = self.total_residual_generation_time / self.total_residual_samples
    
        print(f"Average residual generation time per sample: {avg_residual_time_per_sample*1000:.2f} ms")
        
        # Fit feature extractor without transform to analyze explained variance
        print("Fitting feature extractor for variance analysis...")
        self.feature_extractor.fit(self.all_residuals)
        
        # Plot explained variance and get recommended component count
        print("Analyzing explained variance...")
        recommended_components = self.plot_explained_variance(threshold=variance_threshold)
        
        # If n_components is None, use the recommended count
        if n_components is None:
            print(f"Using recommended component count: {recommended_components}")
            self.feature_extractor.n_components = recommended_components
            
            # Refit with proper component count if needed
            if recommended_components != self.feature_extractor.n_components:
                self.feature_extractor.fit(self.all_residuals)
        
        # Transform all residuals using the fitted extractor
        print("Transforming features...")
        transformed_features = self.feature_extractor.transform_all(self.all_residuals)
        
        # Prepare labels
        labels = [res.condition for res in self.all_residuals]
        
        # Define n_estimators values to evaluate
        n_estimators_values = list(range(1,25,5))
        n_estimators_values.extend(list(range(50,501,50)))
        
        # Dictionary to store results for each n_estimators value
        all_results = {}
        best_accuracy = 0
        best_n_estimators = 0
        best_results = None
        
        print("\n===== Evaluating Different n_estimators Values =====")
        print(f"{'n_estimators':<12} {'Test Acc':<10} {'Train Acc':<10} {'CV Acc':<10} {'Time (s)':<10}")
        print("-" * 60)
        
        # Run fault detection with different n_estimators values
        for n_est in n_estimators_values:
            print(f"Training model with n_estimators={n_est}...")
            
            # Create new fault detector with current n_estimators
            self.fault_detector = FaultDetector(
                test_size=0.2, 
                random_state=47,
                n_estimators=n_est
            )
            
            # Train and evaluate
            results = self.fault_detector.train_and_evaluate(
                transformed_features, 
                labels, 
                avg_residual_time_per_sample
            )
            
            # Store results
            all_results[n_est] = results
            
            # Print summary results
            print(f"{n_est:<12} {results['test_accuracy']:.4f}     {results['train_accuracy']:.4f}     {results['cv_test_accuracy']:.4f}     {results['train_time_seconds']:.2f}")
            
            # Track best model based on CV accuracy
            if results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_n_estimators = n_est
                best_results = results
        
        print("\n===== Best Model =====")
        print(f"Best n_estimators: {best_n_estimators} (Test Accuracy: {best_accuracy:.4f})")
        
        # Generate visualizations for the best model
        print("\nGenerating visualizations for the best model...")
        self.plot_confusion_matrix(best_results)
        self.plot_feature_importance(best_results)
        
        # Print feature importance details for the best model
        #self.print_feature_importance_by_component()
        
        # Create a summary DataFrame of all results
        print("\n===== Detailed Results Summary =====")
        summary_data = []
        for n_est, results in all_results.items():
            summary_data.append({
                'n_estimators': n_est,
                'test_accuracy': results['test_accuracy'],
                'test_precision': results['test_precision'],
                'test_recall': results['test_recall'],
                'train_accuracy': results['train_accuracy'],
                'train_precision': results['train_precision'],
                'train_recall': results['train_recall'],
                'cv_accuracy': results['cv_test_accuracy'],
                'prediction_time_seconds': results['prediction_time_seconds'],
                'avg_prediction_time_per_sample': results['avg_prediction_time_per_sample'],
                'avg_feature_extraction_time_per_sample': results['avg_feature_extraction_time_per_sample'],
                'avg_classifier_prediction_time_per_sample': results['avg_classifier_prediction_time_per_sample'],
                'avg_residual_generation_time_per_sample': results['avg_residual_generation_time_per_sample']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save results summary to CSV
        results_path = f'Results/{self.approach}/n_estimators_comparison.csv'
        summary_df.to_csv(results_path, index=False)
        print(f"\nResults summary saved to {results_path}")
        
        # Plot accuracy vs. n_estimators
        self._plot_n_estimators_comparison(summary_df)

        #self._plot_timing_breakdown(summary_df)
        
        return {
            'all_results': all_results,
            'best_results': best_results,
            'best_n_estimators': best_n_estimators,
            'summary_df': summary_df
        }
    
    