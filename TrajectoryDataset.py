import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import os
from tqdm import tqdm

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
        
        past_features = []
        future_features = []
        
        for step in sequence['past']:
            past_features.append([
                *step['position'],
                *step['velocity'],
                step['steering'],
                step['object_in_path'],
                step['traffic_light_detected']
            ])
        
        for step in sequence['future']:
            future_features.append([
                *step['position'],
                *step['velocity'],
                step['steering'],
                step['object_in_path'],
                step['traffic_light_detected']
            ])
        
        past_tensor = torch.tensor(past_features, dtype=torch.float32)
        future_tensor = torch.tensor(future_features, dtype=torch.float32)
        
        return past_tensor, future_tensor