from Prediction_Model.DLModels import GraphAttentionLSTM
from Visualization.Rescaler import GraphBoundsScaler
from Data_Curator.config import config

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import networkx as nx
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
import cv2
from tqdm import tqdm
import tempfile
import os
from collections import defaultdict

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
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = self.load_model(model_config).to(config.DEVICE)

        # Fixed figure dimensions
        self.fig_width = 10
        self.fig_height = 10
        self.dpi = 100
        
        # Initialize figure with fixed size
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        
        # Store states and timestamps
        self.state_history = []
        self.timestamps = []
        
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
        """Process state and store it with timestamp"""
        past_features = self._prepare_past_features(state)
        graph_data = self._prepare_graph_features(state)
        
        with torch.no_grad():
            predictions = self.model(past_features, graph_data)
            predictions = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}
        
        if predictions and state['graph_bounds']:
            trajectory_state = TrajectoryState(
                timestamp=state.get('timestamp', 0),
                past_positions=np.array([step['position'] for step in state['past']]),
                predicted_positions=predictions['position_mean'],
                predicted_variances=predictions['position_var'],
                actual_future_positions=np.array([step['position'] for step in future_state['future']]) if future_state else None,
                graph=state['graph'],
                graph_bounds=state['graph_bounds']
            )
            
            self.state_history.append(trajectory_state)
            self.timestamps.append(state.get('timestamp', 0))
            
        return predictions

    def _normalize_timestamps(self):
        """Process timestamps to ensure consistent frame timing"""
        if not self.timestamps:
            return [], 0
            
        # Count states per timestamp
        timestamp_counts = defaultdict(int)
        for ts in self.timestamps:
            timestamp_counts[ts] += 1
            
        # Find the mode (most common count) of states per second
        counts = list(timestamp_counts.values())
        mode_count = max(set(counts), key=counts.count)
        target_fps = mode_count  # Use the most frequent state count as target FPS
        
        # First, collect all indices for each timestamp
        timestamp_map = defaultdict(list)
        for idx, ts in enumerate(self.timestamps):
            timestamp_map[ts].append(idx)
            
        # Then normalize the number of states for each timestamp
        final_map = defaultdict(list)
        for ts, indices in timestamp_map.items():
            current_count = len(indices)
            
            if current_count > target_fps:
                # Downsample
                selected_indices = np.linspace(0, current_count - 1, target_fps, dtype=int)
                final_map[ts] = [indices[i] for i in selected_indices]
            elif current_count < target_fps:
                # Upsample
                duplicates_needed = target_fps - current_count
                if ts == max(timestamp_map.keys()):
                    # For the last timestamp, extend state_history
                    self.state_history.extend([self.state_history[indices[-1]]] * duplicates_needed)
                    new_indices = range(len(self.state_history) - duplicates_needed, len(self.state_history))
                    final_map[ts] = indices + list(new_indices)
                else:
                    # For other timestamps, duplicate the last index
                    final_map[ts] = indices + [indices[-1]] * duplicates_needed
            else:
                final_map[ts] = indices
                
        return final_map, target_fps

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

    def _visualize_state(self, trajectory_state: TrajectoryState):
        """Visualize state with fixed dimensions"""
        self.ax.clear()
        scaler = GraphBoundsScaler(trajectory_state.graph_bounds)
        
        # Draw components
        self._draw_graph(trajectory_state.graph, scaler)
        self._draw_trajectories(trajectory_state, scaler)
        self._draw_uncertainties(trajectory_state, scaler)
        
        # Set fixed plot properties
        self._set_plot_properties()
        
        # Enforce consistent dimensions
        self.fig.set_size_inches(self.fig_width, self.fig_height)
        plt.tight_layout()

    def create_video(self, output_path, requested_fps=None):
        if not self.state_history:
            print("No states to visualize!")
            return
            
        # Get normalized timestamps and calculated FPS
        timestamp_map, calculated_fps = self._normalize_timestamps()
        fps = requested_fps if requested_fps else calculated_fps
        print(f"Using frame rate: {fps} fps")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []
            frame_count = 0
            
            # Generate frames
            print("\nGenerating frames...")
            for norm_ts in tqdm(sorted(timestamp_map.keys()), desc="Processing timestamps", unit="timestamp"):
                indices = timestamp_map[norm_ts]
                for idx in tqdm(indices, desc=f"Generating frames for timestamp {norm_ts:.2f}", unit="frame", leave=False):
                    state = self.state_history[idx]
                    self._visualize_state(state)
                    
                    frame_file = os.path.join(temp_dir, f'frame_{frame_count:04d}.png')
                    self.fig.savefig(frame_file, dpi=self.dpi, bbox_inches='tight')
                    frame_files.append(frame_file)
                    frame_count += 1
                    self.ax.clear()
            
            if frame_files:
                print("\nCombining frames into video...")
                # Read and standardize first frame
                frame = cv2.imread(frame_files[0])
                if frame is None:
                    raise RuntimeError("Failed to read first frame")
                    
                # Standardize dimensions based on the first frame
                target_height, target_width = frame.shape[:2]
                print(f"\nVideo parameters:")
                print(f"- Frame size: {target_width}x{target_height}")
                print(f"- Total frames: {len(frame_files)}")
                print(f"- FPS: {fps}")
                print(f"- Expected duration: {len(frame_files)/fps:.1f} seconds")
                
                # Try different codecs
                codecs = ['avc1', 'mp4v']
                video_writer = None
                
                for codec in codecs:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            float(fps),
                            (target_width, target_height),
                            isColor=True
                        )
                        if video_writer.isOpened():
                            print(f"Using codec: {codec}")
                            break
                        video_writer.release()
                    except Exception as e:
                        print(f"Failed with codec {codec}: {str(e)}")
                
                if not video_writer or not video_writer.isOpened():
                    raise RuntimeError("Failed to initialize video writer with any codec")
                
                # Write frames
                print("\nWriting frames to video...")
                frames_written = 0
                for frame_file in tqdm(frame_files, desc="Writing video", unit="frame"):
                    try:
                        frame = cv2.imread(frame_file)
                        if frame is None:
                            print(f"Failed to read frame: {frame_file}")
                            continue
                            
                        # Ensure consistent dimensions
                        if frame.shape[:2] != (target_height, target_width):
                            frame = cv2.resize(frame, (target_width, target_height))
                        
                        success = video_writer.write(frame)
                        frames_written += 1
                        #if success:
                        #    frames_written += 1
                        #else:
                        #    print(f"Failed to write frame {frame_file}")
                        #    print(f"Frame properties: dtype={frame.dtype}, shape={frame.shape}")
                    except Exception as e:
                        print(f"Error processing frame {frame_file}: {str(e)}")
                
                video_writer.release()
                print(f"\nFrames successfully written: {frames_written}/{len(frame_files)}")
                
                # Verify final video
                cap = cv2.VideoCapture(output_path)
                if not cap.isOpened():
                    print("Warning: Could not verify final video")
                else:
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    actual_duration = frame_count / actual_fps
                    cap.release()
                    
                    print(f"\nFinal video statistics:")
                    print(f"- Actual FPS: {actual_fps:.1f}")
                    print(f"- Actual frame count: {frame_count}")
                    print(f"- Actual duration: {actual_duration:.1f} seconds")
                    
                    if abs(actual_duration - len(frame_files)/fps) > 1.0:
                        print("\nWarning: Significant difference between expected and actual duration")