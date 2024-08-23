import os
import pickle
import json
import numpy as np
from MapProcessor import MapProcessor
from GraphBuilder import GraphBuilder
from Point import Point

class DataProcessor:
    def __init__(self, map_file, input_folder, output_folder, window_size, prediction_horizon):
        self.map_processor = MapProcessor(map_file)
        self.graph_builder = None
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.graph_size = 30
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reference_points = [
            ((81370.40, 49913.81), (3527.96, 1775.78)),
            ((81375.16, 49917.01), (3532.70, 1779.04)),
            ((81371.85, 49911.62), (3529.45, 1773.63)),
            ((81376.60, 49914.82), (3534.15, 1776.87)),
        ]

    def process_all_sequences(self):
        os.makedirs(self.output_folder, exist_ok=True)
        for folder_name in os.listdir(self.input_folder):
            folder_path = os.path.join(self.input_folder, folder_name)
            if os.path.isdir(folder_path):
                self.process_sequence(folder_name, folder_path)

    def process_sequence(self, folder_name, folder_path):
        print(f"Processing folder: {folder_name}")
        self.map_processor.load_route(folder_path)
        self.graph_builder = GraphBuilder(self.map_processor.map_data, self.map_processor.get_route())
        data = self.read_scene_data(folder_path)
        sequence = self.create_sequence(data)
        self.save_sequence(sequence, folder_name)

    def read_scene_data(self, folder_path):
        file_paths = {
            'tf': os.path.join(folder_path, '_tf.json'),
            'objects': os.path.join(folder_path, '_perception_object_recognition_tracking_objects.json'),
            'traffic_lights': os.path.join(folder_path, '_perception_traffic_light_recognition_traffic_signals.json'),
            'velocity': os.path.join(folder_path, '_vehicle_status_velocity_status.json'),
            'steering': os.path.join(folder_path, '_vehicle_status_steering_status.json')
        }

        data = {}

        with open(file_paths['tf'], 'r') as f_tf, \
             open(file_paths['objects'], 'r') as f_obj, \
             open(file_paths['traffic_lights'], 'r') as f_tl, \
             open(file_paths['velocity'], 'r') as f_vel, \
             open(file_paths['steering'], 'r') as f_steer:
            
            while True:
                lines = [f.readline().strip() for f in [f_tf, f_obj, f_tl, f_vel, f_steer]]
                if not all(lines):
                    break

                tf_data, obj_data, tl_data, vel_data, steer_data = map(json.loads, lines)

                timestamp_sec = tf_data['data']['timestamp_sec']
                
                if all(d['data']['timestamp_sec'] == timestamp_sec for d in [obj_data, tl_data, vel_data, steer_data]):
                    ego_position = tf_data['data']['position']
                    ego_orientation = tf_data['data']['orientation']
                    ego_velocity = {
                        'longitudinal': vel_data['data']['longitudinal_velocity'],
                        'lateral': vel_data['data']['lateral_velocity'],
                        'angular': vel_data['data']['yaw_rate']
                    }
                    ego_steering = steer_data['data']['steering_angle']

                    objects = [{
                        'position': {'x': obj['x'], 'y': obj['y'], 'z': obj['z']},
                        'orientation': {'x': obj['orientation_x'], 'y': obj['orientation_y'], 'z': obj['orientation_z'], 'w': obj['orientation_w']},
                        'velocity': {'x': obj['linear_velocity_x'], 'y': obj['linear_velocity_y'], 'z': obj['linear_velocity_z']},
                        'classification': obj['classification']
                    } for obj in obj_data['data']['objects']]

                    traffic_lights = [{
                        'id': light['map_primitive_id'],
                        'color': light['color'],
                        'confidence': light['confidence']
                    } for light in tl_data['data']['lights'] if light['color'] <= 3]

                    data[timestamp_sec] = {
                        'ego': {
                            'position': ego_position,
                            'orientation': ego_orientation,
                            'velocity': ego_velocity,
                            'steering': ego_steering
                        },
                        'objects': objects,
                        'traffic_lights': traffic_lights
                    }

        return data

    def create_sequence(self, data):
        sequence = []
        timestamps = list(data.keys())
        
        for i in range(len(timestamps) - self.window_size - self.prediction_horizon + 1):
            # Get initial and final positions
            initial_timestamp = timestamps[i]
            final_timestamp = timestamps[i + self.window_size + self.prediction_horizon - 1]
            initial_position = self.extract_ego_data(data[initial_timestamp])['position']
            final_position = self.extract_ego_data(data[final_timestamp])['position']
            
            # Create a graph that covers both initial and final positions
            G = self.create_expanded_graph(initial_position, final_position)
            
            # Scale the graph and get scaling factors
            G, x_min, x_max, y_min, y_max = self.scale_graph(G)
            
            past_sequence = []
            future_sequence = []
            
            for j in range(i, i + self.window_size):
                timestamp = timestamps[j]
                past_sequence.append(self.process_timestep(data[timestamp], G, x_min, x_max, y_min, y_max, is_past=True))
            
            for j in range(i + self.window_size, i + self.window_size + self.prediction_horizon):
                timestamp = timestamps[j]
                future_sequence.append(self.process_timestep(data[timestamp], G, x_min, x_max, y_min, y_max, is_past=False))
            
            sequence.append({
                'past': past_sequence,
                'future': future_sequence,
                'graph': G  # Store the graph once for the entire sequence
            })
        
        return sequence

    def create_expanded_graph(self, initial_position, final_position):
        center_x = (initial_position.x + final_position.x) / 2
        center_y = (initial_position.y + final_position.y) / 2
        center_position = Point(center_x, center_y)

        distance = Point.distance(initial_position, final_position)

        max_distance = max(distance / 2, self.graph_size)  # 45 is the original max_distance from GraphBuilder

        return self.graph_builder.build_graph(center_position, max_distance)

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

    def extract_ego_data(self, data_dict):
        ego_pos = Point.convert_coordinate_frame(
            data_dict['ego']['position']["x"], 
            data_dict['ego']['position']["y"], 
            self.reference_points
        )
        ego_vel = Point(data_dict['ego']['velocity']["longitudinal"], data_dict['ego']['velocity']["lateral"])
        ego_orientation = data_dict['ego']['orientation']
        ego_steering = data_dict['ego']['steering']
        return {
            'position': ego_pos,
            'velocity': ego_vel,
            'orientation': ego_orientation,
            'steering': ego_steering,
            'yaw_rate': data_dict['ego']['velocity']["angular"]
        }

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
    
    def scale_position(self, position, x_min, x_max, y_min, y_max):
        return [
            (position.x - x_min) / (x_max - x_min),
            (position.y - y_min) / (y_max - y_min)
        ]

    def scale_object(self, obj, x_min, x_max, y_min, y_max):
        return {
            'position': self.scale_position(obj['position'], x_min, x_max, y_min, y_max),
            'velocity': self.scale_velocity(obj['velocity']),
            'classification': obj['classification']
        }

    def scale_velocity(self, velocity):
        scaling_factor = 10
        return [abs(velocity.x) / scaling_factor, abs(velocity.y) / scaling_factor]

    def scale_steering(self, steering):
        return steering + 0.5

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

    def save_sequence(self, sequence, folder_name):
        output_file = os.path.join(self.output_folder, f"{folder_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(sequence, f)
        print(f"Saved sequence for {folder_name}")