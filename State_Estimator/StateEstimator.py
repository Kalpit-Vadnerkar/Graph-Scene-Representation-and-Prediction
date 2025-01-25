from Data_Curator.Point import Point

class StateEstimator:
    def __init__(self, window_size, reference_points, graph_builder):
        self.window_size = window_size
        self.reference_points = reference_points
        self.graph_builder = graph_builder
        
    def extract_ego_data(self, data_dict):
        ego_pos = Point.convert_coordinate_frame(
            data_dict['ego']['position']["x"], 
            data_dict['ego']['position']["y"], 
            self.reference_points
        )
        
        ego_vel = Point(data_dict['ego']['velocity']["longitudinal"], 
                       data_dict['ego']['velocity']["lateral"])
        
        return {
            'position': ego_pos,
            'velocity': ego_vel,
            'orientation': data_dict['ego']['orientation'],
            'steering': data_dict['ego']['steering'],
            'acceleration': data_dict['ego']['acceleration'],
            'yaw_rate': data_dict['ego']['velocity']["yaw_rate"]
        }

    def process_timestep(self, data_dict, G, x_min, x_max, y_min, y_max):
        ego_data = self.extract_ego_data(data_dict)
        objects = self.extract_object_data(data_dict)
        
        scaled_ego_position = self.scale_position(ego_data['position'], x_min, x_max, y_min, y_max)
        scaled_objects = [self.scale_object(obj, x_min, x_max, y_min, y_max) for obj in objects]
        
        object_distance = self.closest_object_distance(scaled_ego_position, scaled_objects)
        traffic_light_detected = self.check_traffic_light_detected(G, scaled_ego_position)
        
        return {
            'position': scaled_ego_position,
            'velocity': self.scale_velocity(ego_data['velocity']),
            'steering': self.scale_steering(ego_data['steering']),
            'acceleration': self.scale_acceleration(ego_data['acceleration']),
            'object_distance': object_distance,
            'traffic_light_detected': traffic_light_detected,
            'objects': scaled_objects
        }

    def estimate_state(self, data_buffer, timestamp):
        if len(data_buffer) < self.window_size:
            return None
            
        initial_position = self.extract_ego_data(data_buffer[0])['position']
        final_position = self.extract_ego_data(data_buffer[-1])['position']
        
        G = self.graph_builder.create_expanded_graph(initial_position, final_position)
        G, x_min, x_max, y_min, y_max = self.scale_graph(G)
        
        past_sequence = []
        for data in data_buffer[-self.window_size:]:
            past_sequence.append(
                self.process_timestep(data, G, x_min, x_max, y_min, y_max)
            )
        
        return {
            'timestamp': timestamp,
            'past': past_sequence,
            'graph': G,
            'graph_bounds': [x_min, x_max, y_min, y_max]
        }