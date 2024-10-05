import argparse
import torch
from torch.utils.data import DataLoader, random_split
from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.DLModels import GraphTrajectoryLSTM
from Prediction_Model.Trainer import Trainer
from Prediction_Model.model_utils import load_model, make_predictions
from Visualization.visualizer import visualize_predictions
from model_config import CONFIG

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
                                velocity_scaling_factor=100, 
                                steering_scaling_factor=100, 
                                acceleration_scaling_factor=100)
    
    train_loader, test_loader = create_data_loaders(dataset, config['batch_size'])
    
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    model = GraphTrajectoryLSTM(config['input_sizes'], 
                                config['hidden_size'], 
                                config['num_layers'], 
                                config['input_seq_len'], 
                                config['output_seq_len']).to(device)
    
    trainer = Trainer(model, train_loader, test_loader, config['learning_rate'], device)
    trained_model = trainer.train(config['num_epochs'])
    
    torch.save(trained_model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}")

def visualize(config):
    """Visualize predictions from a trained model."""
    device = config['device']
    dataset = TrajectoryDataset(config['test_data_folder'], 
                                position_scaling_factor=config['position_scaling_factor'], 
                                velocity_scaling_factor=100, 
                                steering_scaling_factor=100)
    
    model = load_model(config)
    predictions, sampled_indices = make_predictions(model, dataset, config)
    visualize_predictions(dataset, config['position_scaling_factor'], predictions, sampled_indices, "Visualization")
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