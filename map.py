import lanelet2
import networkx as nx
from lanelet2.projection import LocalCartesianProjector, UtmProjector, GeocentricProjector
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from helper_functions import *


# data
vehicle_pos = {"data": {"timestamp_sec": 72, "timestamp_ns": 549998378, "position": {"x": 81377.35044311438, "y": 49916.90360337597, "z": 41.20968763772856}, "orientation": {"x": 0.0008842478620083796, "y": -0.005101364535519776, "z": 0.3005808725146492, "w": 0.9537422782198168}}}
vehicle_velocity = {"data": {"timestamp_sec": 72, "timestamp_ns": 549998378, "longitudinal_velocity": 0.0, "lateral_velocity": -0.0, "yaw_rate": -0.0}}
vehicle_steering = {"data": {"timestamp_sec": 72, "timestamp_ns": 549998378, "steering_angle": -0.0}}
traffic_lights = {"data": {"timestamp_sec": 72, "timestamp_ns": 589998377, "num_lights": 6, "lights": [{"map_primitive_id": 1686, "color": 1, "status": 0, "confidence": 0.9951673746109009},
													   {"map_primitive_id": 1688, "color": 1, "status": 0, "confidence": 0.9388778805732727},
													   {"map_primitive_id": 1690, "color": 1, "status": 0, "confidence": 0.8974975943565369}, 
													   {"map_primitive_id": 1727, "color": 16, "status": 0, "confidence": 0.8633347749710083}, 
													   {"map_primitive_id": 1729, "color": 16, "status": 0, "confidence": 0.9355295300483704}, 
													   {"map_primitive_id": 1731, "color": 16, "status": 0, "confidence": 0.9358839392662048}]}}
													   
dynamic_objects = {"data": 
		   {"timestamp_sec": 72, "timestamp_ns": 609998377, "num_objects": 2, "is_map_frame": "true", "objects": 
		   [{"x": 81367.00616183136, "y": 49915.02361205871, "z": 41.99928283691406, 
		     "orientation_x": 0.005100877709306862, "orientation_y": 0.0009281646564563296, "orientation_z": -0.9555438159224402, "orientation_w": 0.29480355392930196, 
		     "linear_velocity_x": 0.9410437504631197, "linear_velocity_y": 0.0, "linear_velocity_z": 0.0, "classification": 7}, 
		    {"x": 81417.7068073648, "y": 49905.75280915967, "z": 43.10921096801758, 
		     "orientation_x": -0.002459241518556234, "orientation_y": -0.004575221328107636, "orientation_z": 0.800577187849788, "orientation_w": 0.5992071309451159, 
		     "linear_velocity_x": 0.006523147279652648, "linear_velocity_y": 0.0, "linear_velocity_z": 0.0, "classification": 2}]}}


path = "lanelet2_map.osm"
#projector = LocalCartesianProjector(lanelet2.io.Origin(49, 8.4))
projector = LocalCartesianProjector(lanelet2.io.Origin(35.67, 139.65, 0))
#projector = LocalCartesianProjector(lanelet2.io.Origin(0, 0))
map_data, load_errors = lanelet2.io.loadRobust(path, projector)



def lanelet2_to_graph(map_data, vehicle_pos, max_distance=75):
    G = nx.Graph()
    node_ids = {}  # Dictionary to store node IDs for centerline points
    lanelet_node_mapping = {}  # Dictionary to map lanelet IDs to centerline node IDs
    
    # Add vehicle node
    vehicle_node_id = 0
    G.add_node(vehicle_node_id, type="vehicle_node", x=vehicle_pos.x, y=vehicle_pos.y)
    
    # Add map nodes (centerline points) within max_distance
    for ll in map_data.laneletLayer:
        centerline = ll.centerline
        centerline_node_ids = []
        if ll.attributes["subtype"] == "road":
            for point in centerline:
                distance = sqrt((vehicle_pos.x - point.x)**2 + (vehicle_pos.y - point.y)**2)
                if distance <= max_distance:
                    node_id = len(node_ids) + 1
                    node_ids[point] = node_id
                    G.add_node(node_id, type="map_node", x=point.x, y=point.y)
                    centerline_node_ids.append(node_id)
            lanelet_node_mapping[ll.id] = centerline_node_ids
        
        # Add lane edges between consecutive centerline points
        for i in range(len(centerline_node_ids) - 1):
            node1_id = centerline_node_ids[i]
            node2_id = centerline_node_ids[i+1]
            G.add_edge(node1_id, node2_id, type="lane_edge", lanelet_id=ll.id, direction="forward")
            G.add_edge(node2_id, node1_id, type="lane_edge", lanelet_id=ll.id, direction="backward")
    
    # Add traffic light nodes within max_distance
    traffic_light_nodes = []
    for ls in map_data.lineStringLayer:
        if ls.attributes["type"] == "traffic_light":
            mid_point = get_mid_point(ls)
            distance = sqrt((vehicle_pos.x - mid_point.x)**2 + (vehicle_pos.y - mid_point.y)**2)
            if distance <= max_distance:
                node_id = len(node_ids) + 1
                node_ids[mid_point] = node_id
                G.add_node(node_id, type="traffic_light_node", x=mid_point.x, y=mid_point.y)
                traffic_light_nodes.append(node_id)
    
    # Connect traffic light nodes to nearest lanelets
    for traffic_light_node in traffic_light_nodes:
        traffic_light_pos = (G.nodes[traffic_light_node]['x'], G.nodes[traffic_light_node]['y'])
        nearest_lanelets = []
        min_distance = 50.0
        for lanelet_id, centerline_nodes in lanelet_node_mapping.items():
            for node_id in centerline_nodes:
                map_node_pos = (G.nodes[node_id]['x'], G.nodes[node_id]['y'])
                distance = sqrt((traffic_light_pos[0] - map_node_pos[0])**2 + (traffic_light_pos[1] - map_node_pos[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_lanelets = [lanelet_id]
                elif distance == min_distance:
                    nearest_lanelets.append(lanelet_id)
        
        # Add edges from traffic light node to centerline nodes of nearest lanelets
        for lanelet_id in nearest_lanelets:
            for node_id in lanelet_node_mapping[lanelet_id]:
                #break
                G.add_edge(traffic_light_node, node_id, type="traffic_light_edge")
            break
    
    return G


def plot_graph_and_data(G, vehicle_pos, dynamic_objects):


    # Create a combined dictionary with x and y coordinates
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

    # Separate nodes by type
    map_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'map_node']
    traffic_light_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'traffic_light_node']
    dynamic_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'vehicle_node']

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Draw map nodes
    nx.draw_networkx_nodes(G, pos, nodelist=map_nodes, node_color='b', node_size=10, label='Map Nodes')

    # Draw traffic light nodes
    nx.draw_networkx_nodes(G, pos, nodelist=traffic_light_nodes, node_color='r', node_size=20, label='Traffic Light Nodes')
    
    # Draw dynamic objetcs nodes
    nx.draw_networkx_nodes(G, pos, nodelist=dynamic_nodes, node_color='g', node_size=40, label='Dynamic Object Nodes')

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # Set axis limits to fit the data
    x_values = [data['x'] for node, data in G.nodes(data=True)]
    y_values = [data['y'] for node, data in G.nodes(data=True)]
    plt.xlim(min(x_values) - 10, max(x_values) + 10)
    plt.ylim(min(y_values) - 10, max(y_values) + 10)

    # Add a legend
    plt.legend(loc='upper right')
    
    plt.show()




#x=3535.7806390146916, y=1779.5797283311367
lanelet2_to_graph_debug(map_data)

vehicle_pos = {"position": {"x": 81377.35044311438, "y": 49916.90360337597}}

vehicle_pos = convert_coordinate_frame(vehicle_pos["position"]["x"], vehicle_pos["position"]["y"])

G = lanelet2_to_graph(map_data, vehicle_pos)
# Print some information about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

plot_graph_and_data(G, vehicle_pos, dynamic_objects)

