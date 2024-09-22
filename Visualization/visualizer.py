import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal

from Visualization.Rescaler import GraphBoundsScaler

def plot_graph_and_trajectories(sequence, scaling_factor, predicted_future, ax):
    # Extract the graph
    G = sequence['graph']

    scaler = GraphBoundsScaler(sequence['graph_bounds'])
    
    # Plot graph nodes
    pos = {node: scaler.restore_position(data['x'] * scaling_factor, data['y'] * scaling_factor) for node, data in G.nodes(data=True)}

    # Plot regular map nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='black', node_size=1, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 0 and d['path_node'] == 0])

    # Plot traffic light nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='green', node_size=2, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 1])

    # Plot path nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='yellow', node_size=2, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['path_node'] == 1])

    # Plot edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1)

    # Plot past trajectory
    past_positions = np.array([scaler.restore_position(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) for step in sequence['past']])
    ax.scatter(past_positions[:, 0], past_positions[:, 1], c='blue', s=30, label='Past positions')

    # Plot actual future trajectory
    future_positions = np.array([scaler.restore_position(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) for step in sequence['future']])
    ax.scatter(future_positions[:, 0], future_positions[:, 1], c='green', s=30, label='Actual future')

    # Plot predicted future trajectory with uncertainty
    pred_positions = np.array([scaler.restore_mean(x, y) for x, y in predicted_future['position_mean']])
    pred_variances = np.array([scaler.restore_variance(x, y) for x, y in predicted_future['position_var']])

    ax.scatter(pred_positions[:, 0], pred_positions[:, 1], c='red', s=30, label='Predicted future')
    
    # Visualize uncertainty as distributions
    for i in range(len(pred_positions)):
        x, y = np.mgrid[pred_positions[i, 0] - 3*np.sqrt(pred_variances[i, 0]):pred_positions[i, 0] + 3*np.sqrt(pred_variances[i, 0]):0.1, 
                        pred_positions[i, 1] - 3*np.sqrt(pred_variances[i, 1]):pred_positions[i, 1] + 3*np.sqrt(pred_variances[i, 1]):0.1]
        pos = np.dstack((x, y))

        rv = multivariate_normal([pred_positions[i, 0], pred_positions[i, 1]], [[pred_variances[i, 0], 0], [0, pred_variances[i, 1]]])
        ax.contour(x, y, rv.pdf(pos), cmap="Reds", alpha=0.5)

    ax.legend()
    ax.set_aspect('equal')

def plot_3d_distributions(predictions, past_positions, future_positions, all_graph_bounds, condition):
    num_samples = len(predictions)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    colors = ['viridis', 'plasma', 'inferno']
    
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        scaler = GraphBoundsScaler(all_graph_bounds[i])
        
        future_pos = np.array([scaler.restore_position(x, y) for x, y in future_positions[i]])
        pred_pos = np.array([scaler.restore_mean(x, y) for x, y in predictions[i]['position_mean']])
        pred_var = np.array([scaler.restore_variance(x, y) for x, y in predictions[i]['position_var']])
        
        ax.scatter(future_pos[:, 0], future_pos[:, 1], np.zeros_like(future_pos[:, 0]), c='g', marker='o', label='Ground Truth')
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], np.zeros_like(pred_pos[:, 0]), c='r', marker='x', label='Predicted Mean')
        
        padding = 50
        x = np.linspace(min(pred_pos[:, 0]) - padding, max(pred_pos[:, 0]) + padding, 100)
        y = np.linspace(min(pred_pos[:, 1]) - padding, max(pred_pos[:, 1]) + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        for t in range(len(pred_pos)):
            mean = pred_pos[t]
            cov = np.diag(pred_var[t])
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
            ax.plot_surface(X, Y, Z, cmap=colors[t], alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Sequence {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"predictions/sequence_pdf_{condition}.png")
    plt.close()

def plot_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, condition):
    num_samples = len(predictions)
    num_timesteps = num_samples + 2
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    colors = ['viridis', 'plasma', 'inferno']

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(rows, cols, timestep + 1, projection='3d')
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + 3)]

        future_pos_for_timestep = []
        for seq_idx in relevant_sequences:
            scaler = GraphBoundsScaler(all_graph_bounds[seq_idx])
            future_pos = np.array([scaler.restore_position(x, y) for x, y in future_positions[seq_idx]])
            future_pos_for_timestep.append(future_pos[timestep - seq_idx])

        if future_pos_for_timestep:
            mean_future_pos = np.mean(future_pos_for_timestep, axis=0)
            padding = 100
            x = np.linspace(mean_future_pos[0] - padding, mean_future_pos[0] + padding, 100)
            y = np.linspace(mean_future_pos[1] - padding, mean_future_pos[1] + padding, 100)
            X, Y = np.meshgrid(x, y)
        
            for seq_idx in relevant_sequences:
                scaler = GraphBoundsScaler(all_graph_bounds[seq_idx])
                pred_pos = np.array([scaler.restore_mean(x, y) for x, y in predictions[seq_idx]['position_mean']])
                pred_var = np.array([scaler.restore_variance(x, y) for x, y in predictions[seq_idx]['position_var']])

                t = timestep - seq_idx
                if 0 <= t < len(pred_pos):
                    mean = pred_pos[t]
                    cov = np.diag(pred_var[t])

                    original_future_pos = scaler.restore_position(future_positions[seq_idx][t][0], future_positions[seq_idx][t][1])
                    mean_shift = mean_future_pos - original_future_pos
                    mean += mean_shift

                    Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
                    ax.plot_surface(X, Y, Z, cmap=colors[seq_idx % 3], alpha=0.3)
                    ax.scatter(mean[0], mean[1], 0, c='r', marker='x',
                               label='Predicted Mean' if seq_idx == relevant_sequences[0] else "")

            ax.scatter(mean_future_pos[0], mean_future_pos[1], 0, c='g', marker='o', label='Mean Ground Truth')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.savefig(f"predictions/timestep_pdf_{condition}.png")
    plt.close()

def visualize_predictions(dataset, scaling_factor, predictions, sampled_indices, condition):
    num_samples = len(predictions)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f"Trajectory Predictions - {condition}", fontsize=16)

    for i, (pred, idx) in enumerate(zip(predictions, sampled_indices)):
        ax = axes[i // cols, i % cols] if num_samples > 1 else axes
        sequence = dataset.data[idx]
        plot_graph_and_trajectories(sequence, scaling_factor, pred, ax)
        ax.set_title(f"Sample {i+1}")

    # Hide any unused subplots
    for i in range(num_samples, rows * cols):
        axes.flatten()[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"predictions/trajectory_prediction_{condition}.png")
    plt.close()