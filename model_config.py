import torch

CONFIG = {
    'graph_sizes': {
        'node_features': 4,
        'number_of_nodes': 150
    },

    'feature_sizes': {
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'acceleration': 1,
        'object_distance': 1,
        'traffic_light_detected': 1
    },
    'num_epochs': 100,
    'batch_size': 128,
    'hidden_size': 128,
    'num_layers': 2,
    'learning_rate': 0.001,
    'input_seq_len': 20,
    'output_seq_len': 20,
    'dropout_rate': 0.2,
    'position_scaling_factor': 10,
    'velocity_scaling_factor':10, 
    'steering_scaling_factor':10, 
    'acceleration_scaling_factor':10,
    'train_data_folder': "Dataset/Sequence_Dataset", 
    'test_data_folder': "Test_Dataset/Sequence_Dataset/Nominal",
    'model_path': 'models/graph_trajectory_model(20_steps).pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_samples': 10,
    'sample_start_index': 275,
    #'conditions': ['Nominal', 'Noisy_Camera', 'Noisy_GNSS', 'Noisy_IMU', 'Noisy_Lidar']
    'conditions': ['Nominal']
}