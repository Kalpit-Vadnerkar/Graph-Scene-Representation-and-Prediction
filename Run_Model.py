import argparse
import torch
from torch.utils.data import DataLoader, random_split
from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.DLModels import GraphTrajectoryLSTM
from Prediction_Model.Trainer import Trainer
from Prediction_Model.model_utils import load_model, make_predictions
from Visualization.visualizer import visualize_predictions, plot_vel_distributions_by_timestep, plot_steer_distributions_by_timestep, plot_pos_distributions_by_timestep, plot_acceleration_distributions_by_timestep
from model_config import CONFIG

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
    
    model = GraphTrajectoryLSTM(config).to(device)

    model = load_model(config).to(device)
    
    trainer = Trainer(model, train_loader, test_loader, config['learning_rate'], device)
    trained_model = trainer.train(config['num_epochs'])
    
    torch.save(trained_model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}")

def visualize(config):
    """Visualize predictions from a trained model."""
    device = config['device']
    dataset = TrajectoryDataset(config['test_data_folder'], 
                                position_scaling_factor=config['position_scaling_factor'], 
                                velocity_scaling_factor=config['velocity_scaling_factor'], 
                                steering_scaling_factor=config['steering_scaling_factor'], 
                                acceleration_scaling_factor=config['acceleration_scaling_factor'])
    
    model = load_model(config)
    predictions, sampled_indices = make_predictions(model, dataset, config)

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

    #visualize_predictions(dataset, config['position_scaling_factor'], predictions, sampled_indices, "Visualization")
    #plot_vel_distributions_by_timestep(predictions, past_velocities, future_velocities, "Visualization")
    #plot_steer_distributions_by_timestep(predictions, past_steering, future_steering, "Visualization")
    #plot_acceleration_distributions_by_timestep(predictions, past_acceleration, future_acceleration, "Visualization")
    plot_pos_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, "Visualization")
    print("Visualization complete. Check the 'predictions' folder for output.")

def main():
    parser = argparse.ArgumentParser(description="Train or visualize trajectory prediction model")
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True,
                        help='Mode of operation: train a new model or visualize a saved model')
    args = parser.parse_args()

    if args.mode == 'train':
        train(CONFIG)
    elif args.mode == 'visualize':
        visualize(CONFIG)

if __name__ == "__main__":
    main()