import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import random

def load_sequences(folder_path):
    all_sequences = []
    sequence_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                sequences = pickle.load(f)
                all_sequences.append(sequences)
                sequence_names.append(filename)
    print(f"loaded {len(all_sequences)} files")            
    return all_sequences, sequence_names

def plot_graph_and_sequence(sequence, ax):
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
    
    # Plot ego vehicle positions
    past_positions = [timestep['position'] for timestep in sequence['past']]
    future_positions = [timestep['position'] for timestep in sequence['future']]
    
    x_past, y_past = zip(*past_positions)
    x_future, y_future = zip(*future_positions)
    
    ax.scatter(x_past, y_past, c='blue', s=30, label='Past positions')
    ax.scatter(x_future, y_future, c='red', s=30, label='Future positions')
    
    ax.legend()
    ax.set_aspect('equal')

# Main execution
main_folder = input("Enter data folder name: ")
output_folder = os.path.join(main_folder, "Sequence_Dataset")  # Replace with your actual output folder path
plots_folder = "Test_plots"  # Folder where plots will be saved

# Create plots folder if it doesn't exist
os.makedirs(plots_folder, exist_ok=True)

all_sequences, sequence_names = load_sequences(output_folder)

selected_sequences = []

for sequence in all_sequences:
    # Randomly select 5 sequences
    selected_sequences.extend(random.sample(sequence, min(3, len(sequence))))

print(f"loaded {len(selected_sequences)} sequences")

# Create individual plots for each sequence
for i, sequence in enumerate(selected_sequences):
    plt.figure(figsize=(10, 8))
    plot_graph_and_sequence(sequence, plt.gca())
    plt.title(f"Sequence {i+1}")
    individual_plot_filename = os.path.join(plots_folder, f"{sequence_names[(i // 3)]}_sequence_{i + 1}.png")
    plt.savefig(individual_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {individual_plot_filename}")
    plt.close()

print("All plots have been saved.")
