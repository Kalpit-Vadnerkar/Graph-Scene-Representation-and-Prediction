import torch

CONFIG = {
    'input_sizes': {
        'node_features': 4,
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'acceleration': 1,
        'object_distance': 1,
        'traffic_light_detected': 1
    },
    'num_epochs': 2,
    'batch_size': 128,
    'hidden_size': 64,
    'num_layers': 2,
    'learning_rate': 0.0001,
    'input_seq_len': 20,
    'output_seq_len': 20,
    'position_scaling_factor': 10,
    'train_data_folder': "Dataset/Sequence_Dataset", 
    'test_data_folder': "Test_Dataset/Sequence_Dataset/Nominal",
    'model_path': 'models/graph_trajectory_model.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_samples': 2,
    'sample_start_index': 40,
    #'conditions': ['Nominal', 'Noisy_Camera', 'Noisy_GNSS', 'Noisy_IMU', 'Noisy_Lidar']
    'conditions': ['Nominal']
}