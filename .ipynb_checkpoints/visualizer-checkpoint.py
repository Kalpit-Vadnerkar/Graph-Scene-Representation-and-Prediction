import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import matplotlib.patches as patches

def plot_graph_and_trajectories(sequence, actual_future, predicted_future, ax):
    # Extract the graph
    G = sequence['graph']

    # Plot graph nodes
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

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
    past_positions = [step['position'] for step in sequence['past']]
    x_past, y_past = zip(*past_positions)
    ax.scatter(x_past, y_past, c='blue', s=30, label='Past positions')

    # Plot actual future trajectory
    x_actual, y_actual = zip(*[pos.tolist() for pos in actual_future['position']])
    ax.scatter(x_actual, y_actual, c='green', s=30, label='Actual future')

    # Plot predicted future trajectory (mean and confidence ellipse)
    position_mean, position_log_var = predicted_future['position']
    
    # Convert to numpy if they're PyTorch tensors
    if isinstance(position_mean, torch.Tensor):
        position_mean = position_mean.numpy()
    if isinstance(position_log_var, torch.Tensor):
        position_log_var = position_log_var.numpy()
    
    x_pred, y_pred = position_mean[:, 0], position_mean[:, 1]
    position_var = np.exp(position_log_var)
    x_std, y_std = np.sqrt(position_var[:, 0]), np.sqrt(position_var[:, 1])

    ax.scatter(x_pred, y_pred, c='red', s=30, label='Predicted future (mean)')

    for x, y, std_x, std_y in zip(x_pred, y_pred, x_std, y_std):
        ellipse = patches.Ellipse((x, y), width=2*std_x, height=2*std_y, angle=0, edgecolor='r', fc='None', lw=1)
        ax.add_patch(ellipse)

    ax.legend()
    ax.set_aspect('equal')

def make_predictions(model, dataset, device, num_samples=9):
    model.eval()
    all_predictions = []
    all_actual = []
    sampled_sequences = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for idx in sampled_sequences:
            past, future, graph = dataset[idx]
            past = {k: v.unsqueeze(0).to(device) for k, v in past.items()}  # Add batch dimension
            graph = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}  # Add batch dimension

            predictions = model(past, graph)

            # Convert predictions to numpy arrays
            all_predictions.append({k: (v[0].squeeze().cpu().numpy(), v[1].squeeze().cpu().numpy()) 
                                    if isinstance(v, tuple) else v.squeeze().cpu().numpy() 
                                    for k, v in predictions.items()})
            all_actual.append({k: v.numpy() for k, v in future.items()})

    return all_predictions, all_actual, sampled_sequences

def visualize_predictions(model, dataset, device, all_sequences):
    # Make predictions
    num_samples = 9  # 3x3 grid
    predictions, actuals, sampled_indices = make_predictions(model, dataset, device, num_samples)

    # Create a grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle("Trajectory Predictions", fontsize=16)

    for i, (pred, actual, idx) in enumerate(zip(predictions, actuals, sampled_indices)):
        ax = axes[i // 3, i % 3]
        plot_graph_and_trajectories(all_sequences[idx], actual, pred, ax)
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()