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
    'num_epochs': 30,
    'batch_size': 128,
    'hidden_size': 256,
    'num_layers': 2,
    'learning_rate': 0.001,
    'input_seq_len': 30,
    'output_seq_len': 30,
    'dropout_rate': 0.2,
    'position_scaling_factor': 10,
    'velocity_scaling_factor': 10, 
    'steering_scaling_factor': 100, 
    'acceleration_scaling_factor': 100,
    'train_data_folder': "Train_Dataset/Sequence_Dataset", 
    'test_data_folder': "Test_Dataset/Sequence_Dataset",
    'model_path': 'models/graph_trajectory_model.pth',
    #'model_path': 'models/lstm_trajectory_model.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_samples': 1,
    'sample_start_index': 125,
    #'conditions': ['Nominal', 'Noisy_Camera', 'Noisy_IMU', 'Noisy_Lidar']
    'conditions': ['Nominal']
}