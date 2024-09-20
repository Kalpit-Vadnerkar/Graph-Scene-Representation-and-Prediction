import torch
from torch.utils.data import DataLoader, random_split
from TrajectoryDataset import TrajectoryDataset
from DLModels import GraphTrajectoryLSTM
from Trainer import Trainer
from visualizer import visualize_predictions
import pickle
import os

def load_sequences(folder_path):
    all_sequences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                sequences = pickle.load(f)
                all_sequences.extend(sequences)
    print(f"Loaded {len(all_sequences)} sequences")
    return all_sequences

def train():
    # Hyperparameters
    input_sizes = {
        'node_features': 4,
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'object_in_path': 1,
        'traffic_light_detected': 1
    }
    hidden_size = 64
    num_layers = 2
    input_seq_len = 3  # past trajectory length
    output_seq_len = 3  # future prediction length
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.0001

    model_path = 'graph_trajectory_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Scaling for distributions
    scaling_factor = 10
    
    # Data loading
    data_folder = "Dataset/Sequence_Dataset"
    #data_folder = "Test_Dataset/Sequence_Dataset"
    dataset = TrajectoryDataset(data_folder, position_scaling_factor=10, velocity_scaling_factor=100, steering_scaling_factor=100)
    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Model initialization
    #model = GraphTrajectoryLSTM(input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len)
    model = load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device)
    
    # Training
    
    trainer = Trainer(model, train_loader, test_loader, learning_rate, device)
    trained_model = trainer.train(num_epochs)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), model_path)
    
    visualize_predictions(trained_model, dataset, scaling_factor, device)

def collate_fn(batch):
    past_batch = {k: torch.stack([item[0][k] for item in batch]) for k in batch[0][0].keys()}
    future_batch = {k: torch.stack([item[1][k] for item in batch]) for k in batch[0][1].keys()}
    graph_batch = {
        'node_features': torch.stack([item[2]['node_features'] for item in batch]),
        'adj_matrix': torch.stack([item[2]['adj_matrix'] for item in batch])
    }
    
    # Ensure all tensors have 3 dimensions
    for key in ['steering', 'object_in_path', 'traffic_light_detected']:
        if past_batch[key].dim() == 3:
            past_batch[key] = past_batch[key].squeeze(-1)
        if future_batch[key].dim() == 3:
            future_batch[key] = future_batch[key].squeeze(-1)
    
    return past_batch, future_batch, graph_batch

def load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device):
    model = GraphTrajectoryLSTM(input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_loaded_model(model_path):
    # Hyperparameters (ensure these match the values used during training)
    input_sizes = {
        'node_features': 4,
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'object_in_path': 1,
        'traffic_light_detected': 1
    }
    hidden_size = 64
    num_layers = 2
    input_seq_len = 3  # past trajectory length
    output_seq_len = 3  # future prediction length

    # Scaling for distributions
    scaling_factor = 10
    
    # Data loading
    data_folder = "Test_Dataset/Sequence_Dataset"
    dataset = TrajectoryDataset(data_folder, position_scaling_factor=10, velocity_scaling_factor=100, steering_scaling_factor=100)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device)

    # Visualization
    visualize_predictions(model, dataset, scaling_factor, device)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train or visualize trajectory prediction model")
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True,
                        help='Mode of operation: train a new model or visualize a saved model')
    parser.add_argument('--model_path', type=str, default='graph_trajectory_model.pth',
                        help='Path to save/load the model')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'visualize':
        visualize_loaded_model(args.model_path)

if __name__ == "__main__":
    main()