import lanelet2
import networkx as nx
from lanelet2.projection import LocalCartesianProjector, UtmProjector, GeocentricProjector
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from helper_functions import *

#path = "lanelet2_map.osm"
#projector = LocalCartesianProjector(lanelet2.io.Origin(49, 8.4))
#projector = LocalCartesianProjector(lanelet2.io.Origin(35.67, 139.65, 0))
#projector = LocalCartesianProjector(lanelet2.io.Origin(0, 0))
#map_data, load_errors = lanelet2.io.loadRobust(path, projector)



def lanelet2_to_graph(map_data, dynamic_object_positions, dynamic_object_velocities):
    max_distance = 45 # Radius of the complete graph
    min_dist_between_node = 1 # minmum distance between two connected nodes
    connection_threshold = 1 # minimum distance within which 2 connected components will be connected
    
    G = nx.Graph()
    node_ids = {}  # Dictionary to store node IDs for points
    Vehicel_X = dynamic_object_positions[0].x
    Vehicel_Y = dynamic_object_positions[0].y

    # Add map nodes (centerline points) within max_distance
    for ll in map_data.laneletLayer:
        centerline = ll.centerline
        if ll.attributes["subtype"] == "road":
            prev_point = None
            for point in centerline:
                distance = sqrt((Vehicel_X - point.x)**2 + (Vehicel_Y - point.y)**2)
                if distance <= max_distance:
                    # CHeck to maintain some minimum distance between nodes.
                    if prev_point is None or sqrt((prev_point.x - point.x)**2 + (prev_point.y - point.y)**2) >= min_dist_between_node:
                        node_id = len(node_ids) + 1
                        node_ids[point] = node_id
                        G.add_node(node_id, type="map_node", x=point.x, y=point.y,
                                    dynamic_object_exist_probability = 0, dynamic_object_position_X = 0, dynamic_object_position_Y = 0,
                                    dynamic_object_velocity_X = 0, dynamic_object_velocity_Y = 0, nearest_traffic_light_detection_probability = 0)
                        if prev_point is not None:
                            G.add_edge(node_ids[prev_point], node_id, type="lane_edge", lanelet_id=ll.id)
                        prev_point = point

    # Add dynamic object nodes and set dynamic_object_exist_probability
    for i, obj_pos in enumerate(dynamic_object_positions):
        min_distance = float('inf')
        nearest_node_id = None
        for node in G.nodes(data=True):
            distance = sqrt((obj_pos.x - node[1]['x'])**2 + (obj_pos.y - node[1]['y'])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node_id = node[0]
        G.nodes[nearest_node_id]['dynamic_object_exist_probability'] = 1
        G.nodes[nearest_node_id]['dynamic_object_position_X'] = obj_pos.x
        G.nodes[nearest_node_id]['dynamic_object_position_Y'] = obj_pos.y
        G.nodes[nearest_node_id]['dynamic_object_velocity_X'] = dynamic_object_velocities[i].x
        G.nodes[nearest_node_id]['dynamic_object_velocity_Y'] = dynamic_object_velocities[i].y 

    # Set nearest_traffic_light_detection_probability
    for ls in map_data.lineStringLayer:
        if ls.attributes["type"] == "traffic_light":
            mid_point = get_mid_point(ls)
            for node in G.nodes(data=True):
                distance = sqrt((mid_point.x - node[1]['x'])**2 + (mid_point.y - node[1]['y'])**2)
                if distance <= 10:
                    G.nodes[node[0]]['nearest_traffic_light_detection_probability'] = 1

    # Ensure the graph is connected
    components = list(nx.connected_components(G))
    if len(components) > 1:
        # Connect the components by adding edges between all pairs of nodes in different components that are within the connection threshold
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                for node1 in components[i]:
                    for node2 in components[j]:
                        distance = sqrt((G.nodes[node1]['x'] - G.nodes[node2]['x'])**2 + (G.nodes[node1]['y'] - G.nodes[node2]['y'])**2)
                        if distance <= connection_threshold:
                            G.add_edge(node1, node2, type="connection_edge")

    return G

def plot_graph_and_data(G, filename):
    # Create a combined dictionary with x and y coordinates
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    Dyna_pos = {node: (data['dynamic_object_position_X'], data['dynamic_object_position_Y']) for node, data in G.nodes(data=True)}

    # Separate nodes by type
    map_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'map_node']

    # Separate nodes by dynamic_object_exist_probability
    dynamic_object_nodes = [node for node, data in G.nodes(data=True) if data['dynamic_object_exist_probability'] == 1]

    # Separate nodes by nearest_traffic_light_detection_probability
    traffic_light_nodes = [node for node, data in G.nodes(data=True) if data['nearest_traffic_light_detection_probability'] == 1]

    # Separate nodes by dynamic object positions
    dynamic_object_positions = [node for node, data in G.nodes(data=True) if data['dynamic_object_position_X'] != 0 and data['dynamic_object_position_Y'] != 0]

    # Draw the graph
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw map nodes
    nx.draw_networkx_nodes(G, pos, nodelist=map_nodes, node_color='black', node_size=2, label='Map Nodes')

    # Draw dynamic object nodes
    nx.draw_networkx_nodes(G, pos, nodelist=dynamic_object_nodes, node_color='yellow', node_size=5, label='Dynamic Object Nodes')

    # Draw traffic light nodes
    nx.draw_networkx_nodes(G, pos, nodelist=traffic_light_nodes, node_color='green', node_size=2, label='Traffic Light Nodes')

    # Draw dynamic object positions
    nx.draw_networkx_nodes(G, Dyna_pos, nodelist=dynamic_object_positions, node_color='blue', node_size=5, label='Dynamic Object Positions')

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

