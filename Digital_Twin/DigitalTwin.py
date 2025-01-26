from Prediction_Model.DLModels import GraphAttentionLSTM

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

class DigitalTwin:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model(config)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion() # Enable Interactive Mode
        self.scaling_factors = {
            'position': 10,
            'velocity': 10,
            'steering': 10,
            'acceleration': 10
        }

    def load_model(config):
        model = GraphAttentionLSTM(config)
        model.load_state_dict(torch.load(config['model_path'], map_location='cpu', weights_only=True))
        model.eval()
        return model

    def update_state(self, state):
        self.current_state = state
        predictions = self.predict_future_state()
        if predictions:
            self.visualize_predictions(predictions)
        return predictions
        
    def predict_future_state(self):
        if not self.current_state:
            return None
            
        # Extract and scale features like TrajectoryDataset
        past_features = {
            'position': [],
            'velocity': [],
            'steering': [],
            'acceleration': [],
            'object_distance': [],
            'traffic_light_detected': []
        }

        for step in self.current_state['past']:
            past_features['position'].append([p * self.scaling_factors['position'] for p in step['position']])
            past_features['velocity'].append([v * self.scaling_factors['velocity'] for v in step['velocity']])
            past_features['steering'].append([step['steering'] * self.scaling_factors['steering']])
            past_features['acceleration'].append([step['acceleration'] * self.scaling_factors['acceleration']])
            past_features['object_distance'].append([step['object_distance']])
            past_features['traffic_light_detected'].append([step['traffic_light_detected']])

        # Convert to tensors
        past = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.config['device']) 
               for k, v in past_features.items()}

        # Process graph
        G = self.current_state['graph']
        node_features = torch.zeros((150, 4), dtype=torch.float32)
        for node, data in G.nodes(data=True):
            if node < 150:
                node_features[node] = torch.tensor([
                    data['x'] * self.scaling_factors['position'],
                    data['y'] * self.scaling_factors['position'],
                    float(data['traffic_light_detection_node']),
                    float(data['path_node'])
                ])

        adj_matrix = nx.to_numpy_array(G)[:150, :150]
        
        graph = {
            'node_features': node_features.unsqueeze(0).to(self.config['device']),
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).to(self.config['device'])
        }

        with torch.no_grad():
            predictions = self.model(past, graph)
            predictions = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}

        return predictions

    def visualize_predictions(self, predictions):
        self.ax.clear()
        
        # Plot graph nodes
        node_pos = {i: (data['x']*10, data['y']*10) for i, data in self.current_state['graph'].nodes(data=True)}
        nx.draw(self.current_state['graph'], pos=node_pos, node_color='black', node_size=1, ax=self.ax)
        
        # Plot past trajectory
        past_positions = np.array([step['position'] for step in self.current_state['past']])
        #self.ax.plot(past_positions[:, 0]*10, past_positions[:, 1]*10, 'b-', label='Past')
        self.ax.scatter(past_positions[:, 0]*10, past_positions[:, 1]*10, c='darkgreen', s=10, label='Past positions')

        # Plot predicted trajectory with uncertainty
        pos_mean = predictions['position_mean']
        pos_var = predictions['position_var']
        
        self.ax.plot(pos_mean[:, 0], pos_mean[:, 1], 'r-', label='Predicted')

        
        # Plot uncertainty ellipses
        for i in range(len(pos_mean)):
            confidence_ellipse(
                pos_mean[i, 0], pos_mean[i, 1],
                pos_var[i, 0], pos_var[i, 1],
                self.ax, n_std=2, alpha=0.1
            )
        
        self.ax.set_title('Real-time Trajectory Prediction')
        self.ax.legend()
        self.ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)  # Short pause to update display

def confidence_ellipse(mean_x, mean_y, var_x, var_y, ax, n_std=2, **kwargs):
    """Draw confidence ellipse around prediction points"""
    pearson = 0  # Assuming independence between x and y
    ell_radius_x = np.sqrt(var_x) * n_std
    ell_radius_y = np.sqrt(var_y) * n_std
    
    circle = plt.Circle((mean_x, mean_y), np.max([ell_radius_x, ell_radius_y]), **kwargs)
    ax.add_patch(circle)