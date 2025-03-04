from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.DLModels import GraphTrajectoryLSTM, TrajectoryLSTM, GraphAttentionLSTM
from Prediction_Model.Trainer import Trainer
from Prediction_Model.model_utils import load_model, make_predictions, make_limited_predictions, print_model_summary
from Visualization.visualizer import visualize_predictions, plot_vel_distributions_by_timestep, plot_steer_distributions_by_timestep, plot_pos_distributions_by_timestep, plot_acceleration_distributions_by_timestep
from Visualization.probability_viz import plot_probabilities, plot_probabilities2
from Visualization.trajectory_results import position_result_metrics
from Risk_Assessment.RiskManager import RiskAssessmentManager
from model_config import CONFIG


import argparse
import torch
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tabulate import tabulate
import random

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

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    #model = load_model(config).to(device)
    
    trainer = Trainer(model, train_loader, test_loader, config['learning_rate'], device)
    trained_model = trainer.train(config['num_epochs'])
    
    torch.save(trained_model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}")

def train_ensemble(config, num_models=5):
    """Train an ensemble of models with different random initializations."""
    device = config['device']
    
    # Create directory for ensemble models if it doesn't exist
    ensemble_dir = os.path.join(os.path.dirname(config['model_path']), 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)
    
    print(f"Training ensemble of {num_models} models...")
    
    # Load dataset once
    dataset = TrajectoryDataset(config['train_data_folder'], 
                               position_scaling_factor=config['position_scaling_factor'], 
                               velocity_scaling_factor=config['velocity_scaling_factor'], 
                               steering_scaling_factor=config['steering_scaling_factor'], 
                               acceleration_scaling_factor=config['acceleration_scaling_factor'])
    
    train_loader, test_loader = create_data_loaders(dataset, config['batch_size'])
    
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Train ensemble of models
    for i in range(num_models):
        # Set a different seed for each model to ensure different initializations
        seed = 42 + i
        set_seed(seed)
        
        print(f"\n=======================================")
        print(f"Training model {i+1}/{num_models} with seed {seed}")
        print(f"=======================================\n")
        
        # Create a new model with different random initialization
        model = GraphAttentionLSTM(config).to(device)
        
        # Create a trainer instance
        trainer = Trainer(model, train_loader, test_loader, config['learning_rate'], device)
        
        # Train the model
        trained_model = trainer.train(config['num_epochs'], seed=seed)
        
        # Save the model with a unique name
        ensemble_model_path = os.path.join(ensemble_dir, f'ensemble_model_{i+1}.pth')
        torch.save(trained_model.state_dict(), ensemble_model_path)
        print(f"Model {i+1} saved to {ensemble_model_path}")
    
    print(f"\nEnsemble training complete. All models saved to {ensemble_dir}.")


def visualize(config):
    """Visualize predictions from a trained model."""
    model = load_model(config).to(config['device'])
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
        for i in range(CONFIG['sample_start_index'], CONFIG['sample_start_index'] + CONFIG['num_samples']):
        #for i in range(len(dataset)):
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

        visualize_predictions(dataset, config['position_scaling_factor'], predictions, sampled_indices, condition)
        
        predictions = make_predictions(model, dataset, config)

        position_result_metrics(dataset, config['position_scaling_factor'], predictions, condition)

        #residuals(dataset, predictions, config, condition)

        #plot_vel_distributions_by_timestep(predictions, past_velocities, future_velocities, condition)
        #plot_steer_distributions_by_timestep(predictions, past_steering, future_steering, condition)
        #plot_acceleration_distributions_by_timestep(predictions, past_acceleration, future_acceleration, condition)
        #plot_pos_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, condition)
        
        #plot_probabilities(config, predictions, future_steering, condition, 'steering', scale=(0, 100))
        #plot_probabilities(config, predictions, future_acceleration, condition, 'acceleration', scale=(0, 100))

        #plot_probabilities2(config, predictions, future_positions, condition, 'position', scale=(0, 10))
        #plot_probabilities2(config, predictions, future_velocities, condition, 'velocity', scale=(0, 10))

        print(f"Visualization complete for {condition}. Check the 'predictions' folder for output.")

def evaluate(config):
    # Initialize enhanced manager
    manager = RiskAssessmentManager(config)

    # Load single model (with out ensemble)
    manager.approach = 'single'
    model = load_model(config)
    #manager.approach = 'ensemble'
    #model = None

    # Load data for all conditions once
    loaded_data_dict = {}
    
    for condition in config['conditions']:
        loaded_data_dict[condition] = manager.data_loader.load_data_and_predictions(model, condition)
    
    # Run complete pipeline
    results = manager.run_fault_detection(loaded_data_dict, n_components=None)
        


def main():
    parser = argparse.ArgumentParser(description="Train or visualize trajectory prediction model")
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'train_ensemble', 'visualize', 'evaluate', 'summary', 'dim_analysis'], 
                       required=True,
                       help='Mode of operation: train, train_ensemble, visualize, evaluate, dim_analysis, or show model summary')
    parser.add_argument('--components', type=int, default=None,
                       help='Number of PCA components to use (default: None = use all)')
    parser.add_argument('--num_models', type=int, default=10,
                       help='Number of models in the ensemble (default: 5)')
    args = parser.parse_args()

    if args.mode == 'summary':
        model = load_model(CONFIG)
        print_model_summary(model, CONFIG)
    elif args.mode == 'train':
        train(CONFIG)
    elif args.mode == 'train_ensemble':
        train_ensemble(CONFIG, num_models=args.num_models)
    elif args.mode == 'visualize':
        visualize(CONFIG)
    elif args.mode in ['evaluate', 'dim_analysis']:
        evaluate(CONFIG)

if __name__ == "__main__":
    main()