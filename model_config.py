import torch

CONFIG = {
    'input_sizes': {
        'node_features': 4,
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'object_in_path': 1,
        'traffic_light_detected': 1
    },
    'hidden_size': 64,
    'num_layers': 2,
    'input_seq_len': 3,
    'output_seq_len': 3,
    'position_scaling_factor': 10,
    'data_folder': "Test_Dataset/Sequence_Dataset",
    'model_path': 'models/graph_trajectory_model.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_samples': 9,
    'sample_start_index': 10
}