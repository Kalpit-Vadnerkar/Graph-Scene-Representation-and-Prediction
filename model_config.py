import torch

CONFIG = {
    'graph_sizes': {
        'node_features': 4,
        'number_of_nodes': 256
    },

    'feature_sizes': {
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'acceleration': 1,
        'object_distance': 1,
        'traffic_light_detected': 1
    },
    'num_epochs': 3,
    'batch_size': 128,
    #'hidden_size': 128, # LSTM
    'hidden_size': 256, # GAT
    'num_layers': 2,
    'learning_rate': 0.0004,
    'input_seq_len': 30,
    'output_seq_len': 30,
    'dropout_rate': 0.2,
    'position_scaling_factor': 10,
    'velocity_scaling_factor': 10, 
    'steering_scaling_factor': 10, 
    'acceleration_scaling_factor': 10,
    'train_data_folder': "Train_Dataset/Sequence_Dataset", 
    'test_data_folder': "Test_Dataset/Sequence_Dataset",
    'model_path': 'models/graph_attention_network_test.pth', #GAT
    #'model_path': 'models/lstm_trajectory_model.pth', #LSTM
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_samples': 50,
    'sample_start_index': 325,
    'conditions': ['Nominal', 'Noisy_Camera', 'Noisy_IMU', 'Noisy_Lidar']
    #'conditions': ['Nominal']
}