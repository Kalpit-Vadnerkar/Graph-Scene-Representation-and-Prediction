import torch
from torch.utils.data import DataLoader, random_split
from TrajectoryDataset import TrajectoryDataset
from DLModels import TrajectoryLSTM
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

def main():
    # Hyperparameters
    input_sizes = {
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
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Data loading
    data_folder = "Dataset/Sequence_Dataset"
    dataset = TrajectoryDataset(data_folder)
    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model initialization
    model = TrajectoryLSTM(input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len)
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, train_loader, test_loader, learning_rate, device)
    trained_model = trainer.train(num_epochs)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "improved_trajectory_model.pth")
    
    # Evaluate on test set
    test_loss = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")

    # Visualization
    all_sequences = load_sequences(data_folder)
    visualize_predictions(trained_model, dataset, device, all_sequences)

if __name__ == "__main__":
    main()