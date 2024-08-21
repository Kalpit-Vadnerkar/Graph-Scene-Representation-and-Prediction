import lanelet2
import json
import os

class MapProcessor:
    def __init__(self, map_file):
        self.map_data = self.load_lanelet_map(map_file)
        self.route = None  # Initialize route as None

    def load_lanelet_map(self, file):
        projector = lanelet2.projection.LocalCartesianProjector(lanelet2.io.Origin(35.67, 139.65, 0))
        map_data, load_errors = lanelet2.io.loadRobust(file, projector)
        return map_data

    def load_route(self, folder_path):
        route_file = os.path.join(folder_path, 'route.json')
        with open(route_file, 'r') as f:
            route_json = json.load(f)
        self.route = route_json['data']['route_segments']

    def get_route(self):
        return self.route