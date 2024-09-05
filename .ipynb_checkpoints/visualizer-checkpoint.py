import torch
import matplotlib.pyplot as plt
import networkx as nx
import random

def plot_graph_and_trajectories(sequence, predicted_future, ax):
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
    future_positions = [step['position'] for step in sequence['future']]
    x_actual, y_actual = zip(*future_positions)
    ax.scatter(x_actual, y_actual, c='green', s=30, label='Actual future')

    # Plot predicted future trajectory
    x_pred, y_pred = zip(*[pos.tolist() for pos in predicted_future['position']])
    ax.scatter(x_pred, y_pred, c='red', s=30, label='Predicted future')

    ax.legend()
    ax.set_aspect('equal')

def make_predictions(model, dataset, device, num_samples=9):
    model.eval()
    all_predictions = []
    sampled_sequences = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for idx in sampled_sequences:
            past, future, graph = dataset[idx]
            
            # Ensure consistent dimensions
            past = {k: v.unsqueeze(0).to(device) for k, v in past.items()}  # Add batch dimension
            graph = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}  # Add batch dimension
            
            # Move tensors to device
            #past = {k: v.to(device) for k, v in past.items()}
            #graph = {k: v.to(device) for k, v in graph.items()}
            
            # Print shapes for debugging
            #print("Input shapes:")
            #for k, v in past.items():
            #    print(f"{k}: {v.shape}")
            #for k, v in graph.items():
            #    print(f"graph_{k}: {v.shape}")

            predictions = model(past, graph)

            # Print prediction shapes for debugging
            #print("Prediction shapes:")
            #for k, v in predictions.items():
            #    print(f"{k}: {v.shape}")

            all_predictions.append({k: v.squeeze().cpu().numpy() for k, v in predictions.items()})

    return all_predictions, sampled_sequences

def visualize_predictions(model, dataset, device):
    # Make predictions
    num_samples = 9  # 3x3 grid
    predictions, sampled_indices = make_predictions(model, dataset, device, num_samples)

    # Create a grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle("Trajectory Predictions", fontsize=16)

    for i, (pred, idx) in enumerate(zip(predictions, sampled_indices)):
        ax = axes[i // 3, i % 3]
        sequence = dataset.data[idx]  # Get the full sequence from the dataset
        plot_graph_and_trajectories(sequence, pred, ax)
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()