import os
import json

class DataReader:
    def __init__(self, folder_path = None):
        if folder_path:
            self.folder_path = folder_path
            self.file_paths = {
            'tf': os.path.join(folder_path, '_tf.json'),
            'objects': os.path.join(folder_path, '_perception_object_recognition_tracking_objects.json'),
            'traffic_lights': os.path.join(folder_path, '_perception_traffic_light_recognition_traffic_signals.json'),
            'velocity': os.path.join(folder_path, '_vehicle_status_velocity_status.json'),
            #'steering': os.path.join(folder_path, '_vehicle_status_steering_status.json'),
            'control': os.path.join(folder_path, '_system_emergency_control_cmd.json')
            }

    def _process_objects(self, objects_data):
        return [{
            'position': {'x': obj['x'], 'y': obj['y'], 'z': obj['z']},
            'orientation': {'x': obj['orientation_x'], 'y': obj['orientation_y'], 'z': obj['orientation_z'], 'w': obj['orientation_w']},
            'velocity': {'x': obj['linear_velocity_x'], 'y': obj['linear_velocity_y'], 'z': obj['linear_velocity_z']},
            'classification': obj['classification']
        } for obj in objects_data]

    def _process_traffic_lights(self, lights_data):
        return [{
            'id': light['map_primitive_id'],
            'color': light['color'],
            'confidence': light['confidence']
        } for light in lights_data if light['color'] <= 3]
    
    def _process_timestamp_data(self, tf_data, obj_data, tl_data, vel_data, control_data):
        ego_position = tf_data['data']['position']
        ego_orientation = tf_data['data']['orientation']
        ego_velocity = {
            'longitudinal': vel_data['data']['longitudinal_velocity'],
            'lateral': vel_data['data']['lateral_velocity'],
            'yaw_rate': vel_data['data']['yaw_rate']
        }
        ego_steering = control_data['data']['steering_angle']
        ego_acceleration = control_data['data']['acceleration']

        objects = self._process_objects(obj_data['data']['objects'])
        traffic_lights = self._process_traffic_lights(tl_data['data']['lights'])

        return {
            'ego': {
                'position': ego_position,
                'orientation': ego_orientation,
                'velocity': ego_velocity,
                'steering': ego_steering,
                'acceleration': ego_acceleration
            },
            'objects': objects,
            'traffic_lights': traffic_lights
        }
    
    def read_scene_data(self):
        data = {}
        with open(self.file_paths['tf'], 'r') as f_tf, \
             open(self.file_paths['objects'], 'r') as f_obj, \
             open(self.file_paths['traffic_lights'], 'r') as f_tl, \
             open(self.file_paths['velocity'], 'r') as f_vel, \
             open(self.file_paths['control'], 'r') as f_control:
            
            count = 1
            while True:
                lines = [f.readline().strip() for f in [f_tf, f_obj, f_tl, f_vel, f_control]]
                if not all(lines):
                    #print(f"Problems in {self.folder_path}")
                    break

                tf_data, obj_data, tl_data, vel_data, control_data = map(json.loads, lines)

                #timestamp_sec = tf_data['data']['timestamp_sec']
                
                #if all(d['data']['timestamp_sec'] == timestamp_sec for d in [obj_data, tl_data, vel_data, steer_data]):
                #    data[timestamp_sec] = self._process_timestamp_data(tf_data, obj_data, tl_data, vel_data, steer_data)

                data[count] = self._process_timestamp_data(tf_data, obj_data, tl_data, vel_data, control_data)
                
                count += 1

        return data