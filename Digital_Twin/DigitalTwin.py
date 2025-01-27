from Prediction_Model.DLModels import GraphAttentionLSTM
from Visualization.Rescaler import GraphBoundsScaler
from Data_Curator.config import config

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import networkx as nx
import numpy as np
import torch

class DigitalTwin:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = self.load_model(model_config).to(config.DEVICE) 
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion() # Enable Interactive Mode
        self.scaling_factors = {
            'position': 10,
            'velocity': 10,
            'steering': 10,
            'acceleration': 10
        }

    def load_model(self, model_config):
        model = GraphAttentionLSTM(model_config)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True))
        #model.load_state_dict(torch.load(config['model_path'], map_location='cpu', weights_only=True))
        model.eval()
        return model

    def update_state(self, state):
        self.current_state = state
        predictions = self.predict_future_state()
        graph_bounds = state['graph_bounds']
        if predictions:
            self.visualize_predictions(predictions, graph_bounds)
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
        #past = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) 
        #       for k, v in past_features.items()}
        past = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(config.DEVICE) 
               for k, v in past_features.items()}

        # Process graph
        G = self.current_state['graph']
        node_features = torch.zeros((config.MAX_GRAPH_NODES, config.NODE_FEATURES), dtype=torch.float32)
        for node, data in G.nodes(data=True):
            if node < config.MAX_GRAPH_NODES:
                node_features[node] = torch.tensor([
                    data['x'] * self.scaling_factors['position'],
                    data['y'] * self.scaling_factors['position'],
                    float(data['traffic_light_detection_node']),
                    float(data['path_node'])
                ])

        adj_matrix = nx.to_numpy_array(G)[:config.MAX_GRAPH_NODES, :config.MAX_GRAPH_NODES]
        
        #graph = {
        #    'node_features': node_features.unsqueeze(0),
        #    'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
        #}
        graph = {
            'node_features': node_features.unsqueeze(0).to(config.DEVICE),
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        }

        with torch.no_grad():
            predictions = self.model(past, graph)
            predictions = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}

        return predictions

    def visualize_predictions(self, predictions, graph_bounds):
        self.ax.clear()

        G = self.current_state['graph']
        scaler = GraphBoundsScaler(graph_bounds)
        
        # Plot graph nodes
        #node_pos = {i: (data['x']*10, data['y']*10) for i, data in self.current_state['graph'].nodes(data=True)}
        #nx.draw(self.current_state['graph'], pos=node_pos, node_color='black', node_size=1, ax=self.ax)
        # Plot graph nodes
        pos = {node: scaler.restore_position(data['x'] * 10, data['y'] * 10) for node, data in G.nodes(data=True)}
        # Plot regular map nodes
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='black', node_size=1, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 0 and d['path_node'] == 0])
        # Plot traffic light nodes
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='red', node_size=2, 
                            nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 1])

        # Plot path nodes
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='yellow', node_size=2, 
                            nodelist=[n for n, d in G.nodes(data=True) if d['path_node'] == 1])

        # Plot edges
        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color='gray', width=1)
               
        # Plot past trajectory
        #past_positions = np.array([step['position'] for step in self.current_state['past']])
        #self.ax.plot(past_positions[:, 0]*10, past_positions[:, 1]*10, 'b-', label='Past')
        #self.ax.scatter(past_positions[:, 0]*10, past_positions[:, 1]*10, c='darkgreen', s=10, label='Past positions')

        # Plot past trajectory
        past_positions = np.array([scaler.restore_position(step['position'][0] * 10, step['position'][1] * 10) for step in self.current_state['past']])
        self.ax.scatter(past_positions[:, 0], past_positions[:, 1], c='darkgreen', s=10, label='Past positions')

        # Plot predicted trajectory with uncertainty
        #pos_mean = predictions['position_mean']
        #pos_var = predictions['position_var']
        pred_positions = np.array([scaler.restore_mean(x, y) for x, y in predictions['position_mean']])
        pred_variances = np.array([scaler.restore_variance(x, y) for x, y in predictions['position_var']])
        
        #self.ax.plot(pos_mean[:, 0], pos_mean[:, 1], 'r-', label='Predicted')
        self.ax.scatter(pred_positions[:, 0], pred_positions[:, 1], c='red', s=10, label='Predicted')
    

        # Visualize uncertainty as distributions
        for i in range(len(pred_positions)):
            x, y = np.mgrid[pred_positions[i, 0] - 3*np.sqrt(pred_variances[i, 0]):pred_positions[i, 0] + 3*np.sqrt(pred_variances[i, 0]):0.1, 
                        pred_positions[i, 1] - 3*np.sqrt(pred_variances[i, 1]):pred_positions[i, 1] + 3*np.sqrt(pred_variances[i, 1]):0.1]
            pos = np.dstack((x, y))
        
            rv = multivariate_normal([pred_positions[i, 0], pred_positions[i, 1]], [[pred_variances[i, 0], 0], [0, pred_variances[i, 1]]])
            self.ax.contour(x, y, rv.pdf(pos), cmap="RdYlGn", alpha=0.1)
        # Plot uncertainty ellipses
        #for i in range(len(pos_mean)):
        #    confidence_ellipse(
        #        pos_mean[i, 0], pos_mean[i, 1],
        #        pos_var[i, 0], pos_var[i, 1],
        #        self.ax, n_std=2, alpha=0.1
        #    )
        
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