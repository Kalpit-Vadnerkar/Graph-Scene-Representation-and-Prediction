from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.DLModels import GraphTrajectoryLSTM, TrajectoryLSTM, GraphAttentionLSTM
from Prediction_Model.Trainer import Trainer
from Prediction_Model.model_utils import load_model, make_predictions, make_limited_predictions, print_model_summary
from Visualization.visualizer import visualize_predictions, plot_vel_distributions_by_timestep, plot_steer_distributions_by_timestep, plot_pos_distributions_by_timestep, plot_acceleration_distributions_by_timestep
from Visualization.probability_viz import plot_probabilities, plot_probabilities2
from Visualization.trajectory_results import position_result_metrics
from Model_Testing.ResidualDataset import ResidualDataset
from Model_Testing.ResidualClassifier import ResidualClassifier
from model_config import CONFIG

import argparse
import torch
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict

#import warnings

# Suppress the specific CuDNN warning
#warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")

def create_data_loaders(dataset, batch_size, train_ratio=0.8):
    """Create train and test data loaders."""
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, test_loader

def collate_fn(batch):
    """Collate function for DataLoader."""
    past_batch = {k: torch.stack([item[0][k] for item in batch]) for k in batch[0][0].keys()}
    future_batch = {k: torch.stack([item[1][k] for item in batch]) for k in batch[0][1].keys()}
    graph_batch = {
        'node_features': torch.stack([item[2]['node_features'] for item in batch]),
        'adj_matrix': torch.stack([item[2]['adj_matrix'] for item in batch])
    }
    
    # Ensure all tensors have 3 dimensions
    for key in ['steering', 'acceleration', 'object_distance', 'traffic_light_detected']:
        if past_batch[key].dim() == 3:
            past_batch[key] = past_batch[key].squeeze(-1)
        if future_batch[key].dim() == 3:
            future_batch[key] = future_batch[key].squeeze(-1)
    
    return past_batch, future_batch, graph_batch

def train(config):
    """Train the model."""
    device = config['device']
    dataset = TrajectoryDataset(config['train_data_folder'], 
                                position_scaling_factor=config['position_scaling_factor'], 
                                velocity_scaling_factor=config['velocity_scaling_factor'], 
                                steering_scaling_factor=config['steering_scaling_factor'], 
                                acceleration_scaling_factor=config['acceleration_scaling_factor'])
    
    train_loader, test_loader = create_data_loaders(dataset, config['batch_size'])
    
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    model = GraphAttentionLSTM(config).to(device)

    #model = GraphTrajectoryLSTM(config).to(device)

    #model = TrajectoryLSTM(config).to(device)

    # load model for retraining
    model = load_model(config).to(device)
    
    trainer = Trainer(model, train_loader, test_loader, config['learning_rate'], device)
    trained_model = trainer.train(config['num_epochs'])
    
    torch.save(trained_model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}")

def visualize(config):
    """Visualize predictions from a trained model."""
    device = config['device']
    model = load_model(config)
    for condition in config['conditions']:
        data_folder = os.path.join(config['test_data_folder'], condition)
        dataset = TrajectoryDataset(data_folder, 
                                    position_scaling_factor=config['position_scaling_factor'], 
                                    velocity_scaling_factor=config['velocity_scaling_factor'], 
                                    steering_scaling_factor=config['steering_scaling_factor'], 
                                    acceleration_scaling_factor=config['acceleration_scaling_factor'])



        # Extract past and future positions
        past_positions = []
        future_positions = []
        past_velocities = []
        future_velocities = []
        past_steering = []
        future_steering = []
        past_acceleration = []
        future_acceleration = []
        all_graph_bounds = []
        #for i in range(CONFIG['sample_start_index'], CONFIG['sample_start_index'] + CONFIG['num_samples']):
        for i in range(len(dataset)):
            past, future, graph, graph_bounds = dataset[i]
            
            past_positions.append(past['position'].numpy())
            future_positions.append(future['position'].numpy())
            
            past_velocities.append(past['velocity'].numpy())
            future_velocities.append(future['velocity'].numpy())
            
            past_steering.append(past['steering'].numpy())
            future_steering.append(future['steering'].numpy())

            past_acceleration.append(past['acceleration'].numpy())
            future_acceleration.append(future['acceleration'].numpy())
            
            all_graph_bounds.append(graph_bounds)

        predictions, sampled_indices = make_limited_predictions(model, dataset, config)

        #visualize_predictions(dataset, config['position_scaling_factor'], predictions, sampled_indices, condition)
        
        predictions = make_predictions(model, dataset, config)

        position_result_metrics(dataset, config['position_scaling_factor'], predictions, condition)

        #residuals(dataset, predictions, config, condition)

        #plot_vel_distributions_by_timestep(predictions, past_velocities, future_velocities, condition)
        plot_steer_distributions_by_timestep(predictions, past_steering, future_steering, condition)
        plot_acceleration_distributions_by_timestep(predictions, past_acceleration, future_acceleration, condition)
        #plot_pos_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, condition)
        
        #plot_probabilities(config, predictions, future_steering, condition, 'steering', scale=(0, 100))
        #plot_probabilities(config, predictions, future_acceleration, condition, 'acceleration', scale=(0, 100))

        #plot_probabilities2(config, predictions, future_positions, condition, 'position', scale=(0, 10))
        #plot_probabilities2(config, predictions, future_velocities, condition, 'velocity', scale=(0, 10))

        print(f"Visualization complete for {condition}. Check the 'predictions' folder for output.")


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

def display_classification_results(results: Dict[str, Any], classification_type: str) -> None:
    """Display classification results with improved formatting."""
    print(f"\n{'='*20} {classification_type} Classification Results {'='*20}")
    
    # Format cross-validation results with ± for standard deviations
    cv_metrics = []
    for metric, value in results['cv_results'].items():
        if metric.startswith('mean_'):
            base_metric = metric[5:]  # Remove 'mean_' prefix
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
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    print(tabulate(
        results['feature_importance'].head(10).values,
        headers=results['feature_importance'].columns,
        tablefmt="grid"
    ))

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


def run_fault_detection(config: Dict[str, Any]) -> Dict[str, Any]:
    """Modified fault detection function with improved result visualization."""
    dataset_processor = ResidualDataset(horizon=config['output_seq_len'])
    classifier = ResidualClassifier(test_size=0.2)
    model = load_model(config).to(config['device'])
    
    combined_stats = {}
    
    # Process each condition
    for condition in config['conditions']:
        print(f"\nProcessing condition: {condition}")
        
        data_folder = os.path.join(config['test_data_folder'], condition)
        dataset = TrajectoryDataset(
            data_folder,
            position_scaling_factor=config['position_scaling_factor'],
            velocity_scaling_factor=config['velocity_scaling_factor'],
            steering_scaling_factor=config['steering_scaling_factor'],
            acceleration_scaling_factor=config['acceleration_scaling_factor']
        )
        
        predictions = make_predictions(model, dataset, config)
        dataset_processor.process_sequence(dataset=dataset, predictions=predictions, condition=condition)
    
    # Get dataset statistics before classification
    stats = dataset_processor.get_dataset_statistics()
    total_features = len(dataset_processor.features)
    num_features = stats["num_features"] if "num_features" in stats else len(dataset_processor.features[0].keys())
    
    print(f"\nTotal features collected across all conditions: {total_features}")
    
    if not dataset_processor.features:
        raise ValueError("No features were generated from the sequences")
    
    # Train and evaluate classifier
    classification_results = classifier.train_and_evaluate(
        features=dataset_processor.features,
        labels=dataset_processor.labels,
    )
    
    # Create train/test statistics with proper structure
    train_stats = {
        "total_samples": classification_results['multi_class']['data_split']['train_size'],
        "num_features": num_features,
        "label_distribution": {}
    }
    
    test_stats = {
        "total_samples": classification_results['multi_class']['data_split']['test_size'],
        "num_features": num_features,
        "label_distribution": {}
    }
    
    # Extract label distribution from classification report
    report_lines = classification_results['multi_class']['test_results']['classification_report'].split('\n')
    for line in report_lines[1:-5]:  # Skip header and footer lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:  # Ensure line has enough parts
                label = parts[0]
                support = int(parts[4])
                total = train_stats["total_samples"] + test_stats["total_samples"]
                train_count = int(support * train_stats["total_samples"] / total)
                test_count = int(support * test_stats["total_samples"] / total)
                
                train_stats["label_distribution"][label] = train_count
                test_stats["label_distribution"][label] = test_count
    
    # Display merged statistics and classification results
    display_merged_dataset_statistics(train_stats, test_stats)
    display_classification_results(classification_results['multi_class'], "Multi-class")
    display_classification_results(classification_results['binary'], "Binary")
    
    return classification_results, (train_stats, test_stats)


def main():
    parser = argparse.ArgumentParser(description="Train or visualize trajectory prediction model")
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'visualize', 'evaluate', 'summary'], 
                       required=True,
                       help='Mode of operation: train, visualize, evaluate, or show model summary')
    args = parser.parse_args()

    if args.mode == 'summary':
        model = load_model(CONFIG)
        print_model_summary(model, CONFIG)
    elif args.mode == 'train':
        train(CONFIG)
    elif args.mode == 'visualize':
        visualize(CONFIG)
    elif args.mode == 'evaluate':
       results, (train_stats, test_stats) = run_fault_detection(CONFIG)

if __name__ == "__main__":
    main()