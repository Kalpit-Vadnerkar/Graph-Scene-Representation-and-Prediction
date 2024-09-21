import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.stats import multivariate_normal

def plot_graph_and_trajectories(sequence, scaling_factor, predicted_future, ax):
    # Extract the graph
    G = sequence['graph']

    # Plot graph nodes
    pos = {node: (data['x'] * scaling_factor, data['y'] * scaling_factor) for node, data in G.nodes(data=True)}

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

    # Plot past trajectory (scaled up)
    past_positions = [(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) 
                      for step in sequence['past']]
    x_past, y_past = zip(*past_positions)
    ax.scatter(x_past, y_past, c='blue', s=30, label='Past positions')

    # Plot actual future trajectory (scaled up)
    future_positions = [(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) 
                        for step in sequence['future']]
    x_actual, y_actual = zip(*future_positions)
    ax.scatter(x_actual, y_actual, c='green', s=30, label='Actual future')

    # Plot predicted future trajectory with uncertainty
    x_pred = predicted_future['position_mean'][:, 0]
    y_pred = predicted_future['position_mean'][:, 1]
    x_var = predicted_future['position_var'][:, 0]
    y_var = predicted_future['position_var'][:, 1]
    #print(f"Variance X: {x_var}")
    #print(f"Variance Y: {y_var}")

    ax.scatter(x_pred, y_pred, c='red', s=30, label='Predicted future')
    
    #Visualize uncertainty as distributions
    for i in range(len(x_pred)):
        # Create a grid of points for the contour plot
        x, y = np.mgrid[x_pred[i] - 3*np.sqrt(x_var[i]):x_pred[i] + 3*np.sqrt(x_var[i]):0.1, 
                        y_pred[i] - 3*np.sqrt(y_var[i]):y_pred[i] + 3*np.sqrt(y_var[i]):0.1]
        pos = np.dstack((x, y))

        # Create a multivariate normal distribution
        rv = multivariate_normal([x_pred[i], y_pred[i]], [[x_var[i], 0], [0, y_var[i]]])

        # Plot the contour
        ax.contour(x, y, rv.pdf(pos), cmap="Reds", alpha=0.5)

    ax.legend()
    ax.set_aspect('equal')


def plot_velocity_and_steering(sequence, predicted_future, ax1, ax2):
    # Time steps
    time_steps = range(len(sequence['past']) + len(sequence['future']))
    past_steps = range(len(sequence['past']))
    future_steps = range(len(sequence['past']), len(sequence['past']) + len(sequence['future']))

    # Actual past and future velocity
    past_velocity = [np.linalg.norm(step['velocity']) for step in sequence['past']]
    future_velocity = [np.linalg.norm(step['velocity']) for step in sequence['future']]
    
    # Predicted future velocity
    pred_velocity_mean = np.linalg.norm(predicted_future['velocity_mean'], axis=1)
    pred_velocity_std = np.sqrt(np.sum(predicted_future['velocity_var'], axis=1))

    # Plot velocity
    ax1.plot(past_steps, past_velocity, 'b-', label='Past velocity')
    ax1.plot(future_steps, future_velocity, 'g-', label='Actual future velocity')
    ax1.plot(future_steps, pred_velocity_mean, 'r-', label='Predicted future velocity')
    ax1.fill_between(future_steps, 
                     pred_velocity_mean - 2*pred_velocity_std, 
                     pred_velocity_mean + 2*pred_velocity_std, 
                     color='r', alpha=0.2)
    ax1.set_ylabel('Velocity')
    ax1.legend()

    # Actual past and future steering
    past_steering = [step['steering'] for step in sequence['past']]
    future_steering = [step['steering'] for step in sequence['future']]
    
    # Predicted future steering
    pred_steering_mean = predicted_future['steering_mean']
    pred_steering_std = np.sqrt(predicted_future['steering_var'])

    # Plot steering
    ax2.plot(past_steps, past_steering, 'b-', label='Past steering')
    ax2.plot(future_steps, future_steering, 'g-', label='Actual future steering')
    ax2.plot(future_steps, pred_steering_mean, 'r-', label='Predicted future steering')
    ax2.fill_between(future_steps, 
                     pred_steering_mean - 2*pred_steering_std, 
                     pred_steering_mean + 2*pred_steering_std, 
                     color='r', alpha=0.2)
    ax2.set_ylabel('Steering')
    ax2.set_xlabel('Time steps')
    ax2.legend()

def make_predictions(model, dataset, device, num_samples=9):
    model.eval()
    all_predictions = []
    #sampled_sequences = random.sample(range(len(dataset)), num_samples)
    sampled_sequences = [i + 10 for i in range(num_samples)]

    with torch.no_grad():
        for idx in sampled_sequences:
            past, future, graph, graph_bounds = dataset[idx]
            
            # Ensure consistent dimensions
            past = {k: v.unsqueeze(0).to(device) for k, v in past.items()}  # Add batch dimension
            graph = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}  # Add batch dimension
            
            predictions = model(past, graph)

            all_predictions.append({k: v.squeeze().cpu().numpy() for k, v in predictions.items()})

    return all_predictions, sampled_sequences

def visualize_predictions(model, dataset, scaling_factor, device):
    # Make predictions
    num_samples = 9  # 3x3 grid
    predictions, sampled_indices = make_predictions(model, dataset, device, num_samples)

    # Create a grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle("Trajectory Predictions", fontsize=16)

    for i, (pred, idx) in enumerate(zip(predictions, sampled_indices)):
        ax = axes[i // 3, i % 3]
        sequence = dataset.data[idx]  # Get the full sequence from the dataset
        plot_graph_and_trajectories(sequence, scaling_factor, pred, ax)
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.savefig("predictions/model_visualization.png")
    plt.close()