import lanelet2
import json
import os
from Data_Curation.config import config

class MapProcessor:
    def __init__(self):
        self.map_data = self.load_lanelet_map(config.MAP_FILE)
        self.route = None  # Initialize route as None

    def lanelet2_to_graph_debug(self):
        # Display attributes of the LaneletMap object
        print("LaneletMap Attributes:")
        for attr in dir(self.map_data):
            if not attr.startswith("__") and not callable(getattr(self.map_data, attr)):
                print(f"  - {attr}: {getattr(self.map_data, attr)}")

        # Display attributes of the lineStringLayer object
        print("\nlineStringLayer Attributes:")
        for ls in self.map_data.lineStringLayer:
            if ls.attributes["type"] == "traffic_light" and ls.id == 1686:
                print("Attributes:")
                for attr in dir(ls):
                    if not attr.startswith("__") and not callable(getattr(ls, attr)) and attr != "parameters":
                        print(f"  - {attr}: {getattr(ls, attr)}")
                # Print the points of the traffic light LineString
                print("Points of Traffic Light LineString:")
                for i, point in enumerate(ls):
                    print(f"  - Point {i}: id={point.id}, x={point.x}, y={point.y}")
                break

        for ll in self.map_data.laneletLayer:
            if ll.id == 255:
                print("Lanelet Attributes:")
                for attr in dir(ll):
                    if not attr.startswith("__") and not callable(getattr(ll, attr)):
                        print(f"  - {attr}: {getattr(ll, attr)}")
                print(f"Lanelet ID: {ll.id}")

                print("Centerline:")
                centerline = ll.centerline
                print(f"  - Number of points: {len(centerline)}")
                for i, point in enumerate(centerline):
                    print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")

                print("Left and Right Bounds:")
                left_bound = ll.leftBound
                right_bound = ll.rightBound
                print(f"  - Left Bound: {len(left_bound)} points")
                for i, point in enumerate(left_bound):
                    print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")
                print(f"  - Right Bound: {len(right_bound)} points")
                for i, point in enumerate(right_bound):
                    print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")

                break  # Just print the first one to avoid cluttering the output
            


    def load_lanelet_map(self, file):
        projector = lanelet2.projection.LocalCartesianProjector(lanelet2.io.Origin(35.67, 139.65, 0))
        map_data, load_errors = lanelet2.io.loadRobust(file, projector)
        return map_data

    def load_route(self, folder_path):
        route_file = os.path.join(folder_path, 'route.json')
        with open(route_file, 'r') as f:
            route_json = json.load(f)
        
        # Extract lane IDs from route segments
        route_lane_ids = []
        for segment in route_json['data']['route_segments']:
            for primitive in segment['primitives']:
                if primitive['primitive_type'] == 'lane':
                    route_lane_ids.append(primitive['id'])
        
        self.route = route_lane_ids

    def get_route(self):
        return self.route