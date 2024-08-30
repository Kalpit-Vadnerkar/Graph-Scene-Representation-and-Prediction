import torch
from torch.utils.data import Dataset
import pickle
import os
import networkx as nx
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
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
            past_features['position'].append(step['position'])
            past_features['velocity'].append(step['velocity'])
            past_features['steering'].append([step['steering']])
            past_features['object_in_path'].append([step['object_in_path']])
            past_features['traffic_light_detected'].append([step['traffic_light_detected']])
        
        for step in sequence['future']:
            future_features['position'].append(step['position'])
            future_features['velocity'].append(step['velocity'])
            future_features['steering'].append([step['steering']])
            future_features['object_in_path'].append([step['object_in_path']])
            future_features['traffic_light_detected'].append([step['traffic_light_detected']])
        
        past_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in past_features.items()}
        future_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in future_features.items()}
        
        # Process graph data
        G = sequence['graph']
        node_features = torch.zeros((200, 4), dtype=torch.float32)
        for node, data in G.nodes(data=True):
            if node < 200:  # Ensure we don't exceed 200 nodes
                node_features[node] = torch.tensor([
                    data['x'],
                    data['y'],
                    float(data['traffic_light_detection_node']),
                    float(data['path_node'])
                ])
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        adj_matrix = adj_matrix[:200, :200]  # Ensure we don't exceed 200 nodes
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        graph_tensor = {
            'node_features': node_features,
            'adj_matrix': adj_matrix
        }
        
        return past_tensor, future_tensor, graph_tensor