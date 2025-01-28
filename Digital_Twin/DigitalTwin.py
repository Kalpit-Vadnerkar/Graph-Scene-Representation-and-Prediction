from Prediction_Model.DLModels import GraphAttentionLSTM
from Visualization.Rescaler import GraphBoundsScaler
from Data_Curator.config import config

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import networkx as nx
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.animation as animation
import cv2
from tqdm import tqdm
import tempfile
import os


@dataclass
class TrajectoryState:
    timestamp: float
    past_positions: np.ndarray
    predicted_positions: np.ndarray
    predicted_variances: np.ndarray
    actual_future_positions: Optional[np.ndarray] = None
    graph: Optional[nx.Graph] = None
    graph_bounds: Optional[List] = None

class DigitalTwin:
    def __init__(self, model_config, max_history=50):
        self.model_config = model_config
        self.model = self.load_model(model_config).to(config.DEVICE)
        
        # Initialize figure for video creation
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        
        # Store states instead of visualizing immediately
        self.state_history = []
        
        # Scaling factors
        self.scaling_factors = {
            'position': 10,
            'velocity': 10,
            'steering': 10,
            'acceleration': 10
        }
        
        # Visualization settings
        self.viz_settings = {
            'past_color': 'darkgreen',
            'past_size': 10,
            'pred_color': 'red',
            'pred_size': 10,
            'actual_color': 'blue',
            'actual_size': 10,
            'uncertainty_alpha': 0.1,
            'node_colors': {
                'default': 'black',
                'traffic_light': 'red',
                'path': 'yellow'
            }
        }

    def load_model(self, model_config):
        model = GraphAttentionLSTM(model_config)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True))
        model.eval()
        return model

    def update_state(self, state, future_state=None):
        """Process state and store it for later visualization"""
        # Extract and scale features for prediction
        past_features = self._prepare_past_features(state)
        graph_data = self._prepare_graph_features(state)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(past_features, graph_data)
            predictions = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}
        
        if predictions and state['graph_bounds']:
            # Create trajectory state object
            trajectory_state = TrajectoryState(
                timestamp=state.get('timestamp', 0),
                past_positions=np.array([step['position'] for step in state['past']]),
                predicted_positions=predictions['position_mean'],
                predicted_variances=predictions['position_var'],
                actual_future_positions=np.array([step['position'] for step in future_state['future']]) if future_state else None,
                graph=state['graph'],
                graph_bounds=state['graph_bounds']
            )
            
            # Add to history
            self.state_history.append(trajectory_state)
            
        return predictions

    def _prepare_past_features(self, state):
        past_features = {
            'position': [],
            'velocity': [],
            'steering': [],
            'acceleration': [],
            'object_distance': [],
            'traffic_light_detected': []
        }

        for step in state['past']:
            past_features['position'].append([p * self.scaling_factors['position'] for p in step['position']])
            past_features['velocity'].append([v * self.scaling_factors['velocity'] for v in step['velocity']])
            past_features['steering'].append([step['steering'] * self.scaling_factors['steering']])
            past_features['acceleration'].append([step['acceleration'] * self.scaling_factors['acceleration']])
            past_features['object_distance'].append([step['object_distance']])
            past_features['traffic_light_detected'].append([step['traffic_light_detected']])

        return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(config.DEVICE) 
                for k, v in past_features.items()}

    def _prepare_graph_features(self, state):
        G = state['graph']
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
        
        return {
            'node_features': node_features.unsqueeze(0).to(config.DEVICE),
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        }

    def _visualize_state(self, trajectory_state: TrajectoryState):
        """Process a single state for visualization"""
        self.ax.clear()
        scaler = GraphBoundsScaler(trajectory_state.graph_bounds)
        
        # Draw graph elements
        self._draw_graph(trajectory_state.graph, scaler)
        
        # Draw trajectories
        self._draw_trajectories(trajectory_state, scaler)
        
        # Draw uncertainty ellipses
        self._draw_uncertainties(trajectory_state, scaler)
        
        # Set plot properties
        self._set_plot_properties()
        
        # Update plot limits based on data
        self.ax.relim()
        self.ax.autoscale_view()

    def _draw_graph(self, G, scaler):
        pos = {node: scaler.restore_position(data['x'] * 10, data['y'] * 10) 
               for node, data in G.nodes(data=True)}
        
        node_types = {
            'default': [n for n, d in G.nodes(data=True) 
                       if d['traffic_light_detection_node'] == 0 and d['path_node'] == 0],
            'traffic_light': [n for n, d in G.nodes(data=True) 
                            if d['traffic_light_detection_node'] == 1],
            'path': [n for n, d in G.nodes(data=True) 
                    if d['path_node'] == 1]
        }
        
        for node_type, nodes in node_types.items():
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos, ax=self.ax,
                    node_color=self.viz_settings['node_colors'][node_type],
                    node_size=2 if node_type != 'default' else 1,
                    nodelist=nodes
                )
        
        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color='gray', width=1)

    def _draw_trajectories(self, trajectory_state: TrajectoryState, scaler):
        past_positions = np.array([scaler.restore_position(p[0] * 10, p[1] * 10) 
                                 for p in trajectory_state.past_positions])
        self.ax.scatter(
            past_positions[:, 0], past_positions[:, 1],
            c=self.viz_settings['past_color'],
            s=self.viz_settings['past_size'],
            label='Past positions'
        )
        
        pred_positions = np.array([scaler.restore_mean(x, y) 
                                 for x, y in trajectory_state.predicted_positions])
        self.ax.scatter(
            pred_positions[:, 0], pred_positions[:, 1],
            c=self.viz_settings['pred_color'],
            s=self.viz_settings['pred_size'],
            label='Predicted'
        )
        
        if trajectory_state.actual_future_positions is not None:
            actual_positions = np.array([scaler.restore_position(p[0] * 10, p[1] * 10) 
                                      for p in trajectory_state.actual_future_positions])
            self.ax.scatter(
                actual_positions[:, 0], actual_positions[:, 1],
                c=self.viz_settings['actual_color'],
                s=self.viz_settings['actual_size'],
                label='Actual Future'
            )

    def _draw_uncertainties(self, trajectory_state: TrajectoryState, scaler):
        pred_positions = np.array([scaler.restore_mean(x, y) 
                                 for x, y in trajectory_state.predicted_positions])
        pred_variances = np.array([scaler.restore_variance(x, y) 
                                 for x, y in trajectory_state.predicted_variances])
        
        for i in range(len(pred_positions)):
            x, y = np.mgrid[
                pred_positions[i, 0] - 3*np.sqrt(pred_variances[i, 0]):
                pred_positions[i, 0] + 3*np.sqrt(pred_variances[i, 0]):0.1,
                pred_positions[i, 1] - 3*np.sqrt(pred_variances[i, 1]):
                pred_positions[i, 1] + 3*np.sqrt(pred_variances[i, 1]):0.1
            ]
            pos = np.dstack((x, y))
            
            rv = multivariate_normal(
                [pred_positions[i, 0], pred_positions[i, 1]],
                [[pred_variances[i, 0], 0], [0, pred_variances[i, 1]]]
            )
            self.ax.contour(
                x, y, rv.pdf(pos),
                cmap="RdYlGn",
                alpha=self.viz_settings['uncertainty_alpha']
            )

    def _set_plot_properties(self):
        self.ax.set_title('Trajectory Prediction')
        self.ax.legend()
        self.ax.set_aspect('equal')

    def create_video(self, output_path, fps=10):
        """Create a video from stored states with progress bar"""
        if not self.state_history:
            print("No states to visualize!")
            return

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save individual frames
            total_frames = len(self.state_history)
            frame_files = []
            
            print("Generating frames...")
            for i, state in enumerate(tqdm(self.state_history, desc="Creating frames", unit="frame")):
                self._visualize_state(state)
                frame_file = os.path.join(temp_dir, f'frame_{i:04d}.png')
                self.fig.savefig(frame_file, dpi=100, bbox_inches='tight')
                frame_files.append(frame_file)
                self.ax.clear()  # Clear figure for next frame

            # Create video from frames
            if frame_files:
                print("Combining frames into video...")
                frame = cv2.imread(frame_files[0])
                height, width, _ = frame.shape
                
                # Sort frame files to ensure correct order
                frame_files.sort()  # Ensure frames are in correct order
                
                # Use H264 codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not video_writer.isOpened():
                    print("Failed to open video writer. Trying MP4V codec instead...")
                    video_writer.release()
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame_file in tqdm(frame_files, desc="Writing video", unit="frame"):
                    frame = cv2.imread(frame_file)
                    if frame is None:
                        print(f"Failed to read frame: {frame_file}")
                        continue
                    video_writer.write(frame)
                
                video_writer.release()
                
                print(f"\nVideo saved to {output_path} with {len(frame_files)} frames")
                print(f"First frame: {frame_files[0]}")
                print(f"Last frame: {frame_files[-1]}")
            else:
                print("No frames were generated!")