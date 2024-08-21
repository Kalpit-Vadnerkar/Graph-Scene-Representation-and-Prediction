from math import sqrt
import networkx as nx
from helper_functions import get_mid_point

class GraphBuilder:
    def __init__(self, map_data, route):
        self.map_data = map_data
        self.route = route

    def build_graph(self, center_position, max_distance):
        G = nx.Graph()
        min_dist_between_node = 2  # minimum distance between two connected nodes

        for ll in self.map_data.laneletLayer:
            if ll.attributes["subtype"] == "road":
                self._add_lanelet_to_graph(G, ll, center_position, max_distance, min_dist_between_node)

        self._ensure_graph_connectivity(G)
        return G

    def _add_lanelet_to_graph(self, G, lanelet, center_position, max_distance, min_dist_between_node):
        prev_point = None
        for point in lanelet.centerline:
            distance = sqrt((center_position.x - point.x)**2 + (center_position.y - point.y)**2)
            if distance <= max_distance:
                if prev_point is None or sqrt((prev_point.x - point.x)**2 + (prev_point.y - point.y)**2) >= min_dist_between_node:
                    node_id = G.number_of_nodes() + 1
                    G.add_node(node_id, 
                               type="map_node", 
                               x=point.x, 
                               y=point.y,
                               traffic_light_detection_node=0,
                               path_node=1 if lanelet.id in self.route else 0)
                    if prev_point is not None:
                        G.add_edge(G.number_of_nodes() - 1, node_id, type="lane_edge", lanelet_id=lanelet.id)
                    prev_point = point

    def _ensure_graph_connectivity(self, G):
        # Implement graph connectivity logic here
        pass

    def update_graph_with_objects(self, G, objects):
        for obj in objects:
            self._add_object_to_graph(G, obj)

    def _add_object_to_graph(self, G, obj):
        min_distance = float('inf')
        nearest_node_id = None
        for node in G.nodes(data=True):
            distance = sqrt((obj['position'].x - node[1]['x'])**2 + (obj['position'].y - node[1]['y'])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node_id = node[0]
        G.nodes[nearest_node_id]['dynamic_object_exist_probability'] = 1
        G.nodes[nearest_node_id]['dynamic_object_position_X'] = obj['position'].x
        G.nodes[nearest_node_id]['dynamic_object_position_Y'] = obj['position'].y
        G.nodes[nearest_node_id]['dynamic_object_velocity_X'] = obj['velocity'].x
        G.nodes[nearest_node_id]['dynamic_object_velocity_Y'] = obj['velocity'].y

    def update_traffic_lights(self, G):
        for ls in self.map_data.lineStringLayer:
            if ls.attributes["type"] == "traffic_light":
                mid_point = get_mid_point(ls)
                for node in G.nodes(data=True):
                    distance = sqrt((mid_point.x - node[1]['x'])**2 + (mid_point.y - node[1]['y'])**2)
                    if distance <= 10:
                        G.nodes[node[0]]['traffic_light_detection_node'] = 1