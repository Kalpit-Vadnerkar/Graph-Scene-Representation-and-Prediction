import torch
from torch.utils.data import Dataset
import pickle
import os
import networkx as nx
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_folder, position_scaling_factor=10, velocity_scaling_factor=100, steering_scaling_factor=100):
        self.data = []
        self.position_scaling_factor = position_scaling_factor
        self.velocity_scaling_factor = velocity_scaling_factor
        self.steering_scaling_factor = steering_scaling_factor
        for filename in os.listdir(data_folder):
            if filename.endswith('.pkl'):
                with open(os.path.join(data_folder, filename), 'rb') as f:
                    self.data.extend(pickle.load(f))
        print(f"Loaded {len(self.data)} sequences")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        past_features = {
            'position': [],
            'velocity': [],
            'steering': [],
            'object_in_path': [],
            'traffic_light_detected': []
        }
        
        future_features = {
            'position': [],
            'velocity': [],
            'steering': [],
            'object_in_path': [],
            'traffic_light_detected': []
        }
        
        for step in sequence['past']:
            past_features['position'].append([i * self.position_scaling_factor for i in step['position']])
            #past_features['position'].append(step['position'])
            past_features['velocity'].append([i * self.velocity_scaling_factor for i in step['velocity']])
            #past_features['velocity'].append(step['velocity'])
            past_features['steering'].append([step['steering'] * self.steering_scaling_factor])  # Wrap in list to create 2D tensor
            past_features['object_in_path'].append([step['object_in_path']])
            past_features['traffic_light_detected'].append([step['traffic_light_detected']])
        
        for step in sequence['future']:
            future_features['position'].append([i * self.position_scaling_factor for i in step['position']])
            #future_features['position'].append(step['position'])
            future_features['velocity'].append([i * self.velocity_scaling_factor for i in step['velocity']])
            #future_features['velocity'].append(step['velocity'])
            future_features['steering'].append([step['steering'] * self.steering_scaling_factor])  # Wrap in list to create 2D tensor
            future_features['object_in_path'].append([step['object_in_path']])
            future_features['traffic_light_detected'].append([step['traffic_light_detected']])
        
        past_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in past_features.items()}
        future_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in future_features.items()}

        # Ensure all tensors have 3 dimensions [sequence_length, 1] for scalar values
        for key in ['steering', 'object_in_path', 'traffic_light_detected']:
            if past_tensor[key].dim() == 1:
                past_tensor[key] = past_tensor[key].unsqueeze(-1)
            if future_tensor[key].dim() == 1:
                future_tensor[key] = future_tensor[key].unsqueeze(-1)
        
        # Print shapes for debugging
        #print(f"Shapes for sample {idx}:")
        #for key, value in past_tensor.items():
        #    print(f"past_{key}: {value.shape}")
        #for key, value in future_tensor.items():
        #    print(f"future_{key}: {value.shape}")
            
        # Process graph data
        G = sequence['graph']
        node_features = torch.zeros((150, 4), dtype=torch.float32)
        for node, data in G.nodes(data=True):
            if node < 150:  # Ensure we don't exceed 150 nodes
                node_features[node] = torch.tensor([
                    data['x'] * self.position_scaling_factor,
                    data['y'] * self.position_scaling_factor,
                    float(data['traffic_light_detection_node']),
                    float(data['path_node'])
                ])
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        adj_matrix = adj_matrix[:150, :150]  # Ensure we don't exceed 150 nodes
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        graph_tensor = {
            'node_features': node_features,
            'adj_matrix': adj_matrix
        }

        graph_bounds = sequence['graph_bounds']
        
        return past_tensor, future_tensor, graph_tensor, graph_bounds