from Risk_Assessment.ResidualClassifier import ResidualClassifier
from Risk_Assessment.ResidualDataset import ResidualDataset
from Risk_Assessment.Residuals import *
from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import make_predictions

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class RiskAssessmentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets_by_condition = {}
        self.predictions_by_condition = {}
        self.all_residuals = None  # Will store all calculated residuals
        self.all_labels = None     # Will store all labels
        self.classification_results = {}
        
    def load_data_and_predictions(self, model):
        """Load all datasets and generate predictions once"""
        for condition in self.config['conditions']:
            data_folder = os.path.join(self.config['test_data_folder'], condition)
            dataset = TrajectoryDataset(
                data_folder,
                position_scaling_factor=self.config['position_scaling_factor'],
                velocity_scaling_factor=self.config['velocity_scaling_factor'],
                steering_scaling_factor=self.config['steering_scaling_factor'],
                acceleration_scaling_factor=self.config['acceleration_scaling_factor']
            )
            
            predictions = make_predictions(model, dataset, self.config)
            
            self.datasets_by_condition[condition] = dataset
            self.predictions_by_condition[condition] = predictions
            
    def calculate_all_residuals(self) -> None:
        """Calculate all possible residuals once and store them"""
        if self.all_residuals is not None:
            return  # Residuals already calculated
            
        # Initialize dataset processor with all possible residual types
        dataset_processor = ResidualDataset(horizon=self.config['output_seq_len'])
        all_residual_types = [
            'raw', 'kl_divergence', 'cusum'
        ]
        dataset_processor.residual_generator.residual_types = all_residual_types
        
        # Initialize all residual calculators
        residual_class_map = {
            'raw': RawResidual(),
            #'normalized': NormalizedResidual(),
            #'uncertainty': UncertaintyResidual(),
            'kl_divergence': KLDivergenceResidual(),
            #'shewhart': ShewartResidual(),
            'cusum': CUSUMResidual(),
            #'sprt': SPRTResidual()
        }
        dataset_processor.residual_generator.residual_calculators = residual_class_map
        
        # Process all sequences
        for condition, dataset in self.datasets_by_condition.items():
            dataset_processor.process_sequence(
                dataset=dataset,
                predictions=self.predictions_by_condition[condition],
                condition=condition
            )
        
        # Store all residuals and labels
        self.all_residuals = dataset_processor.features
        self.all_labels = dataset_processor.labels
            
    def filter_features_by_residual_types(self, 
                                        features: List[Dict[str, float]], 
                                        residual_types: List[str]) -> List[Dict[str, float]]:
        """Filter features to only include specified residual types"""
        filtered_features = []
        for feature_dict in features:
            filtered_dict = {}
            for key, value in feature_dict.items():
                # Check if any of the requested residual types are in the feature key
                if any(res_type in key for res_type in residual_types):
                    filtered_dict[key] = value
            filtered_features.append(filtered_dict)
        return filtered_features
        
    def get_residual_subset(self, residual_types: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
        """Get a subset of pre-calculated residuals for specific residual types"""
        # Calculate all residuals if not already done
        self.calculate_all_residuals()
        
        # Filter features to only include requested residual types
        filtered_features = self.filter_features_by_residual_types(
            self.all_residuals, residual_types
        )
        
        return filtered_features, self.all_labels
        
    def run_classification(self, residual_combinations: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Run classification for all residual combinations"""
        results = {}
        
        # Create directory for confusion matrices
        os.makedirs('predictions/confusion_matrices', exist_ok=True)
        
        for residual_types in residual_combinations:
            features, labels = self.get_residual_subset(residual_types)
            classifier = ResidualClassifier(test_size=0.2)
            
            classification_results = classifier.train_and_evaluate(
                features=features,
                labels=labels
            )
            
            combo_name = '+'.join(residual_types)
            self.classification_results[combo_name] = classification_results
            
            # Save confusion matrices
            self._save_confusion_matrices(combo_name, classification_results)
            
            results[combo_name] = {
                'accuracy': classification_results['multi_class']['cv_results']['mean_accuracy'],
                'f1': classification_results['multi_class']['cv_results']['mean_f1'],
                'precision': classification_results['multi_class']['cv_results']['mean_precision'],
                'recall': classification_results['multi_class']['cv_results']['mean_recall']
            }
            
        return results
    
    def _save_confusion_matrices(self, combo_name: str, results: Dict[str, Any]):
        """Save confusion matrices for both multi-class and binary classification"""
        for classification_type in ['multi_class', 'binary']:
            confusion_mat = results[classification_type]['test_results']['confusion_matrix']
            
            # Get class labels
            report_lines = results[classification_type]['test_results']['classification_report'].split('\n')
            class_labels = [line.split()[0] for line in report_lines[1:-5] if line.strip()]
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels,
                       yticklabels=class_labels)
            
            plt.title(f'{classification_type.title()} Classification\nResidual Types: {combo_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            filename = f'predictions/confusion_matrices/{combo_name}_{classification_type}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
    
    def display_dataset_statistics(self, features: List[Dict[str, float]], labels: List[str]):
        """Display dataset statistics"""
        train_size = int(0.8 * len(features))
        test_size = len(features) - train_size
        
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
            
        print("\n=== Dataset Statistics ===")
        basic_stats = [
            ["Total Samples", len(features)],
            ["Training Samples", train_size],
            ["Test Samples", test_size],
            ["Number of Features", len(features[0])]
        ]
        print(tabulate(basic_stats, headers=["Metric", "Value"], tablefmt="grid"))
        
        print("\nLabel Distribution:")
        distribution_data = [[label, count] for label, count in label_counts.items()]
        print(tabulate(distribution_data, headers=["Label", "Count"], tablefmt="grid"))
        
        plt.figure(figsize=(8, 6))
        plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%')
        plt.title("Dataset Label Distribution")
        plt.savefig('predictions/dataset_distribution.png')
        plt.close()