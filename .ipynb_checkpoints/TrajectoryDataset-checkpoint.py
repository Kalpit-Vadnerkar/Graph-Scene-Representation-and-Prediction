import torch
from torch.utils.data import Dataset
import pickle
import os

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
        
        return past_tensor, future_tensor