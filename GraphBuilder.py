import networkx as nx
from Point import Point
from math import sqrt

class GraphBuilder:
    def __init__(self, map_data, route):
        self.map_data = map_data
        self.route = route
        self.min_dist_between_node = 5  # minimum distance between two connected nodes
        self.connection_threshold = 5

    def build_graph(self, center_position, max_distance):
        G = nx.Graph()
        
        for ll in self.map_data.laneletLayer:
            if ll.attributes["subtype"] == "road":
                self._add_lanelet_to_graph(G, ll, center_position, max_distance, self.min_dist_between_node)

        self._ensure_graph_connectivity(G)
        self.update_traffic_lights(G)
        return G

    def _add_lanelet_to_graph(self, G, lanelet, center_position, max_distance, min_dist_between_node):
        prev_point = None
        for point in lanelet.centerline:
            current_point = Point(point.x, point.y)
            distance = Point.distance(center_position, current_point)
            if distance <= max_distance:
                if prev_point is None or Point.distance(prev_point, current_point) >= min_dist_between_node:
                    node_id = G.number_of_nodes() + 1
                    G.add_node(node_id, 
                               type="map_node", 
                               x=current_point.x, 
                               y=current_point.y,
                               traffic_light_detection_node=0,
                               path_node=1 if lanelet.id in self.route else 0)
                    if prev_point is not None:
                        G.add_edge(G.number_of_nodes() - 1, node_id, type="lane_edge", lanelet_id=lanelet.id)
                    prev_point = current_point

    def find_closest_components(self, G, components):
                min_distance = float('inf')
                closest_pair = None
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        for node1 in components[i]:
                            for node2 in components[j]:
                                distance = sqrt((G.nodes[node1]['x'] - G.nodes[node2]['x'])**2 + 
                                                (G.nodes[node1]['y'] - G.nodes[node2]['y'])**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_pair = (node1, node2, i, j)
                return closest_pair, min_distance
    
    def _ensure_graph_connectivity(self, G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            closest_pair, min_distance = self.find_closest_components(G, components)
            if min_distance <= self.connection_threshold:
                node1, node2, i, j = closest_pair
                G.add_edge(node1, node2, type="connection_edge")
                # Merge the two connected components
                new_component = components[i].union(components[j])
                components = [comp for k, comp in enumerate(components) if k not in (i, j)]
                components.append(new_component)
            else:
                # If no components are within the threshold, we stop connecting
                break
        return G

    def update_traffic_lights(self, G):
        for ls in self.map_data.lineStringLayer:
            if ls.attributes["type"] == "traffic_light":
                mid_point = Point.get_mid_point(Point(ls[0].x, ls[0].y), Point(ls[1].x, ls[1].y))
                for node in G.nodes(data=True):
                    node_point = Point(node[1]['x'], node[1]['y'])
                    distance = Point.distance(mid_point, node_point)
                    if distance <= 10:
                        G.nodes[node[0]]['traffic_light_detection_node'] = 1