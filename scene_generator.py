import lanelet2
import networkx as nx
from lanelet2.projection import LocalCartesianProjector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
from math import sqrt
import pickle
import os
from helper_functions import *
from map import lanelet2_to_graph, plot_graph_and_data

def load_lanelet_map(file):
    #projector = LocalCartesianProjector(lanelet2.io.Origin(49, 8.4))
    projector = LocalCartesianProjector(lanelet2.io.Origin(35.67, 139.65, 0))
    #projector = LocalCartesianProjector(lanelet2.io.Origin(0, 0))
    map_data, load_errors = lanelet2.io.loadRobust(file, projector)
    return map_data



def process_and_save_sequences(map_data, input_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over subfolders in input folder
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            # Read scene data
            data = read_scene_data(folder_path)

            # Process data and create sequence
            sequence = []
            window_size = 4

            for i in range(len(data) - window_size + 1):
                window_sequence = []
                for j in range(i, i + window_size):
                    timestamp, data_dict = list(data.items())[j]
                    dynamic_object_positions = []
                    dynamic_object_velocities = []
                    for obj in data_dict['objects']:
                        dynamic_object_positions.append(convert_coordinate_frame(obj['position']["x"], obj['position']["y"]))
                        dynamic_object_velocities.append(Point(obj['velocity']["Vx"], obj['velocity']["Vy"]))

                    G = lanelet2_to_graph(map_data, dynamic_object_positions, dynamic_object_velocities)
                    #print(f"Number of nodes: {G.number_of_nodes()}")
                    #print(f"Number of edges: {G.number_of_edges()}")
                    window_sequence.append(G)
                sequence.append(window_sequence)
                

            # Save sequence to pickle file
            output_file = os.path.join(output_folder, f"{folder_name}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(sequence, f)
            print("done!")
            

def load_sequence_data(file_name):
    # Load sequence from file
    with open(file_name, 'rb') as f:
        sequence = pickle.load(f)
    return sequence

def plot_sequence_data(sequence):
    # Loop through the sequences and graphs
    for i, window_sequence in enumerate(sequence):
        print(f"Sequence {i+1}:")
        for j, G in enumerate(window_sequence):
            print(f"  Graph {j+1}:")
            filename = 'plots/sequence_' + str(i+1) + '_Graph_' + str(j+1) +'.png'
            plot_graph_and_data(G, filename)
            #print(f"    Number of nodes: {G.number_of_nodes()}")
            #print(f"    Number of edges: {G.number_of_edges()}")

# Call the function
input_folder = "Cleaned_Data_set"
output_folder = "Sequence_Dataset"
map_file = "lanelet2_map.osm"
map_data = load_lanelet_map(map_file)
process_and_save_sequences(map_data, input_folder, output_folder)
for file_name in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file_name)
    plot_sequence_data(load_sequence_data(file_path))
    break















































def plot_graph_and_data_distributions(G, filename):
    # Create a combined dictionary with x and y coordinates
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

    # Separate nodes by type
    map_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'map_node']
    traffic_light_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'traffic_light_node']
    dynamic_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'dynamic_object_node']

    # Draw the graph
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw map nodes
    nx.draw_networkx_nodes(G, pos, nodelist=map_nodes, node_color='black', node_size=3, label='Map Nodes')

    
    # Create a custom colormap for the Gaussian distribution
    cmap = colors.LinearSegmentedColormap.from_list('gaussian', ['white', 'black'], N=256)
    norm = colors.Normalize(vmin=0, vmax=1)
    
    # Draw dynamic object nodes as ellipses with Gaussian distribution fill color
    for node in dynamic_nodes:
        x, y = pos[node]
        # Create a grid of values for the Gaussian distribution
        xx, yy = np.meshgrid(np.linspace(x-1, x+1, 100), np.linspace(y-1, y+1, 100))
        zz = np.exp(-((xx-x)**2/(2*0.5**2) + (yy-y)**2/(2*0.5**2)))
        # Create an image of the Gaussian distribution and add it to the plot
        ax.imshow(zz, extent=[x-1, x+1, y-1, y+1], origin='lower', cmap=cmap, norm=norm, alpha=1)
    
    # Draw traffic light nodes
    nx.draw_networkx_nodes(G, pos, nodelist=traffic_light_nodes, node_color='r', node_size=10, label='Traffic Light Nodes')

    # Create a custom colormap for the Gaussian distribution
    cmap = colors.LinearSegmentedColormap.from_list('gaussian', ['white', 'yellow', 'red'], N=256)
    norm = colors.Normalize(vmin=0, vmax=1)

    # Draw dynamic object nodes as ellipses with Gaussian distribution fill color
    for node in dynamic_nodes:
        x, y = pos[node]
        # Create a grid of values for the Gaussian distribution
        xx, yy = np.meshgrid(np.linspace(x-1, x+1, 100), np.linspace(y-1, y+1, 100))
        zz = np.exp(-((xx-x)**2/(2*0.5**2) + (yy-y)**2/(2*0.5**2)))
        # Create an image of the Gaussian distribution and add it to the plot
        ax.imshow(zz, extent=[x-1, x+1, y-1, y+1], origin='lower', cmap=cmap, norm=norm, alpha=1)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # Set axis limits to fit the data
    x_values = [data['x'] for node, data in G.nodes(data=True)]
    y_values = [data['y'] for node, data in G.nodes(data=True)]
    plt.xlim(min(x_values) - 10, max(x_values) + 10)
    plt.ylim(min(y_values) - 10, max(y_values) + 10)

    # Add a legend
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

#lanelet2_to_graph_debug(map_data)
