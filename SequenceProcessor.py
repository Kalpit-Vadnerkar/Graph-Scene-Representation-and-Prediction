from Point import Point
from config import config

class SequenceProcessor:
    def __init__(self, window_size, prediction_horizon, reference_points):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reference_points = reference_points

    def extract_ego_data(self, data_dict):
        
        ego_pos = Point.convert_coordinate_frame(
            data_dict['ego']['position']["x"], 
            data_dict['ego']['position']["y"], 
            self.reference_points
        )
        
        ego_vel = Point(data_dict['ego']['velocity']["longitudinal"], data_dict['ego']['velocity']["lateral"])
        ego_orientation = data_dict['ego']['orientation']
        ego_steering = data_dict['ego']['steering']
        ego_yaw_rate = data_dict['ego']['velocity']["angular"]
        
        return {
            'position': ego_pos,
            'velocity': ego_vel,
            'orientation': ego_orientation,
            'steering': ego_steering,
            'yaw_rate': ego_yaw_rate
        }
    
    def scale_graph(self, G):
        x_coords = [node[1]['x'] for node in G.nodes(data=True)]
        y_coords = [node[1]['y'] for node in G.nodes(data=True)]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        for node in G.nodes(data=True):
            scaled_point = Point(node[1]['x'], node[1]['y']).scale(x_min, x_max, y_min, y_max)
            node[1]['x'] = scaled_point.x
            node[1]['y'] = scaled_point.y
        
        return G, x_min, x_max, y_min, y_max

    def extract_object_data(self, data_dict):
        objects = []
        for obj in data_dict['objects']:
            pos = Point.convert_coordinate_frame(
                obj['position']["x"], 
                obj['position']["y"], 
                self.reference_points
            )
            vel = Point(obj['velocity']["x"], obj['velocity']["y"])
            objects.append({
                'position': pos,
                'velocity': vel,
                'orientation': obj['orientation'],
                'classification': obj['classification']
            })
        return objects
    
    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def scale_position(self, position, x_min, x_max, y_min, y_max):
        scaled_x = self.clamp((position.x - x_min) / (x_max - x_min), 0, 1)
        scaled_y = self.clamp((position.y - y_min) / (y_max - y_min), 0, 1)
        
        return [scaled_x, scaled_y]

    def scale_velocity(self, velocity):
        def scale_component(value, min_val, max_val):
            return max(0, min(1, (value - min_val) / (max_val - min_val)))
        
        return [
            scale_component(velocity.x, config.MIN_VELOCITY, config.MAX_VELOCITY),
            scale_component(velocity.y, config.MIN_VELOCITY, config.MAX_VELOCITY)
        ]
    
    def scale_object(self, obj, x_min, x_max, y_min, y_max):
        return {
            'position': self.scale_position(obj['position'], x_min, x_max, y_min, y_max),
            'velocity': self.scale_velocity(obj['velocity']),
            'classification': obj['classification']
        }

    def scale_steering(self, steering):
        return max(0, min(1, (steering - config.MIN_STEERING) / (config.MAX_STEERING - config.MIN_STEERING)))
    
    def check_object_in_path(self, G, ego_position, objects):
        for obj in objects:
            closest_node = min(G.nodes(data=True), 
                               key=lambda n: (n[1]['x'] - obj['position'][0])**2 + 
                                             (n[1]['y'] - obj['position'][1])**2)
            if closest_node[1]['path_node'] == 1:
                return 1
        return 0

    def check_traffic_light_detected(self, G, ego_position):
        closest_node = min(G.nodes(data=True), 
                           key=lambda n: (n[1]['x'] - ego_position[0])**2 + 
                                         (n[1]['y'] - ego_position[1])**2)
        return closest_node[1]['traffic_light_detection_node']
    
    def process_timestep(self, data_dict, G, x_min, x_max, y_min, y_max, is_past):
        ego_data = self.extract_ego_data(data_dict)
        objects = self.extract_object_data(data_dict)
        
        scaled_ego_position = self.scale_position(ego_data['position'], x_min, x_max, y_min, y_max)
        scaled_objects = [self.scale_object(obj, x_min, x_max, y_min, y_max) for obj in objects]
        
        object_in_path = self.check_object_in_path(G, scaled_ego_position, scaled_objects)
        traffic_light_detected = self.check_traffic_light_detected(G, scaled_ego_position)
        
        processed_data = {
            'position': scaled_ego_position,
            'velocity': self.scale_velocity(ego_data['velocity']),
            'steering': self.scale_steering(ego_data['steering']),
            'object_in_path': object_in_path,
            'traffic_light_detected': traffic_light_detected,
            'objects': scaled_objects
        }
        
        return processed_data
    
    def create_sequences(self, data, graph_builder, inner_pbar):
        sequences = []
        timestamps = list(data.keys())
        
        for i in range(len(timestamps) - self.window_size - self.prediction_horizon + 1):
            initial_timestamp = timestamps[i]
            final_timestamp = timestamps[i + self.window_size + self.prediction_horizon - 1]
            initial_position = self.extract_ego_data(data[initial_timestamp])['position']
            final_position = self.extract_ego_data(data[final_timestamp])['position']
            
            G = graph_builder.create_expanded_graph(initial_position, final_position)

            G, x_min, x_max, y_min, y_max = self.scale_graph(G)
            
            past_sequence = []
            future_sequence = []
            
            for j in range(i, i + self.window_size):
                timestamp = timestamps[j]
                past_sequence.append(self.process_timestep(data[timestamp], G, x_min, x_max, y_min, y_max, is_past=True))
            
            for j in range(i + self.window_size, i + self.window_size + self.prediction_horizon):
                timestamp = timestamps[j]
                future_sequence.append(self.process_timestep(data[timestamp], G, x_min, x_max, y_min, y_max, is_past=False))
            
            sequences.append({
                'past': past_sequence,
                'future': future_sequence,
                'graph': G
            })
            inner_pbar.update(1)
        return sequences