from Model_Testing.ResidualClassifier import ResidualClassifier
from Model_Testing.ResidualDataset import ResidualDataset
from Model_Testing.Residuals import *
from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import load_model, make_predictions

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def merge_dataset_statistics(stats_by_condition: Dict[str, Any]) -> Dict[str, Any]:
    """Merge statistics from all conditions into a single summary."""
    total_samples = 0
    combined_distribution = defaultdict(int)
    num_features = None
    
    for condition, stats in stats_by_condition.items():
        if "error" in stats:
            continue
            
        total_samples += stats["total_samples"]
        for label, count in stats["label_distribution"].items():
            combined_distribution[label] += count
            
        if num_features is None:
            num_features = stats["num_features"]
    
    return {
        "total_samples": total_samples,
        "label_distribution": dict(combined_distribution),
        "num_features": num_features
    }


def display_merged_dataset_statistics(train_stats: Dict[str, Any], test_stats: Dict[str, Any]) -> None:
    """Display merged statistics for both training and test sets."""
    print("\n=== Dataset Statistics ===")
    
    # Basic stats table
    basic_stats = [
        ["Total Training Samples", train_stats["total_samples"]],
        ["Total Test Samples", test_stats["total_samples"]],
        ["Number of Features", train_stats["num_features"]]
    ]
    print(tabulate(basic_stats, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Create side-by-side pie charts for train/test distribution
    if train_stats["label_distribution"] and test_stats["label_distribution"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set distribution
        train_labels = list(train_stats["label_distribution"].keys())
        train_sizes = list(train_stats["label_distribution"].values())
        ax1.pie(train_sizes, labels=train_labels, autopct='%1.1f%%')
        ax1.set_title("Training Set Distribution")
        
        # Test set distribution
        test_labels = list(test_stats["label_distribution"].keys())
        test_sizes = list(test_stats["label_distribution"].values())
        ax2.pie(test_sizes, labels=test_labels, autopct='%1.1f%%')
        ax2.set_title("Test Set Distribution")
        
        plt.savefig('predictions/dataset_distributions.png')
        plt.close()
    
    # Print label distribution in table format
    print("\nLabel Distribution:")
    distribution_data = []
    for label in train_stats["label_distribution"]:
        distribution_data.append([
            label,
            train_stats["label_distribution"][label],
            test_stats["label_distribution"][label],
            train_stats["label_distribution"][label] + test_stats["label_distribution"][label]
        ])
    
    print(tabulate(distribution_data, 
                  headers=["Label", "Train Count", "Test Count", "Total"],
                  tablefmt="grid"))

def display_classification_results(results: Dict[str, Any], classification_type: str) -> None:
    """Display classification results with enhanced visualization."""
    print(f"\n{'='*20} {classification_type} Classification Results {'='*20}")
    
    # Cross-validation results
    cv_metrics = []
    for metric, value in results['cv_results'].items():
        if metric.startswith('mean_'):
            base_metric = metric[5:]
            std_metric = f'std_{base_metric}'
            if std_metric in results['cv_results']:
                formatted_value = f"{value:.3f} ± {results['cv_results'][std_metric]:.3f}"
                cv_metrics.append([base_metric.replace('_', ' ').title(), formatted_value])
    
    print("\nCross-validation Results (Training Set):")
    print(tabulate(cv_metrics, headers=["Metric", "Value (mean ± std)"], tablefmt="grid"))
    
    # Data split information
    split_data = [
        ["Train set size", results['data_split']['train_size']],
        ["Test set size", results['data_split']['test_size']]
    ]
    print("\nData Split Information:")
    print(tabulate(split_data, headers=["Split", "Size"], tablefmt="grid"))
    
    # Test set results
    print("\nTest Set Results:")
    print(results['test_results']['classification_report'])

    # Get class labels from classification report
    report_lines = results['test_results']['classification_report'].split('\n')
    class_labels = [line.split()[0] for line in report_lines[1:-5] if line.strip()]
    
    # Confusion Matrix Visualization
    confusion_mat = results['test_results']['confusion_matrix']
    
    # Adjust figure size based on number of classes
    n_classes = len(class_labels)
    figsize = (8, 6) if n_classes <= 2 else (12, 10)
    
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    
    # Increase font size
    plt.title(f'{classification_type} Classification Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f'predictions/{classification_type.lower()}_confusion_matrix.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Feature importance analysis
    print("\nTop 10 Most Important Features:")
    print(tabulate(
        results['feature_importance'].head(10).values,
        headers=results['feature_importance'].columns,
        tablefmt="grid"
    ))

def analyze_residual_impact(predictions_by_condition: Dict[str, List], datasets_by_condition: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Analyze the impact of different residual types on classification performance."""
    results = {}
    
    residual_combinations = [
        ['raw'],
        ['normalized'],
        ['kl_divergence'],
        ['raw', 'normalized'],
        ['raw', 'kl_divergence'],
        ['normalized', 'kl_divergence'],
        ['raw', 'normalized', 'kl_divergence']
    ]
    
    for residual_types in residual_combinations:
        dataset_processor = ResidualDataset(horizon=config['output_seq_len'])
        dataset_processor.residual_generator.residual_types = residual_types
        
        residual_class_map = {
            'raw': RawResidual,
            'normalized': NormalizedResidual,
            'uncertainty': UncertaintyResidual,
            'kl_divergence': KLDivergenceResidual,
            'shewhart': ShewartResidual,
            'cusum': CUSUMResidual,
            'sprt': SPRTResidual
        }
        
        dataset_processor.residual_generator.residual_calculators = {
            residual_type: residual_class_map[residual_type]()
            for residual_type in residual_types
            if residual_type in residual_class_map
        }
        
        classifier = ResidualClassifier(test_size=0.2)
        
        for condition, dataset in datasets_by_condition.items():
            dataset_processor.process_sequence(
                dataset=dataset, 
                predictions=predictions_by_condition[condition], 
                condition=condition
            )
        
        classification_results = classifier.train_and_evaluate(
            features=dataset_processor.features,
            labels=dataset_processor.labels
        )
        
        combo_name = '+'.join(residual_types)
        results[combo_name] = {
            'accuracy': classification_results['multi_class']['cv_results']['mean_accuracy'],
            'f1': classification_results['multi_class']['cv_results']['mean_f1'],
            'precision': classification_results['multi_class']['cv_results']['mean_precision'],
            'recall': classification_results['multi_class']['cv_results']['mean_recall']
        }
    
    # Visualization
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    combinations = list(results.keys())
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(combinations))
    width = 0.2
    multiplier = 0
    
    for metric in metrics:
        metric_values = [results[combo][metric] for combo in combinations]
        offset = width * multiplier
        plt.bar(x + offset, metric_values, width, label=metric.title())
        multiplier += 1
    
    plt.xlabel('Residual Type Combinations', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Impact of Residual Types on Classification Performance', fontsize=14)
    plt.xticks(x + width * 1.5, combinations, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('predictions/residual_impact_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return results