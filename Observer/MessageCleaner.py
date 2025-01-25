import json
import os
from collections import defaultdict
import numpy as np
from typing import Dict, Set, List, Any

class MessageCleaner:
    def __init__(self, data_points_per_second: int = 10, max_stopped_duration: int = 3, stream_mode: bool = False):
        self.data_points_per_second = data_points_per_second
        self.max_stopped_duration = max_stopped_duration
        self.stream_mode = stream_mode
        
        # These fields are only used in batch mode
        if not stream_mode:
            self.files_to_clean = {
                "_perception_object_recognition_tracking_objects.json",
                "_perception_traffic_light_recognition_traffic_signals.json",
                "_vehicle_status_steering_status.json",
                "_tf.json",
                "_system_emergency_control_cmd.json",
                "_vehicle_status_velocity_status.json"
            }
            self.tf_file = "_tf.json"
            self.metrics = self._initialize_metrics()

        
    def _initialize_metrics(self) -> Dict:
        metrics_fields = [
            'longitudinal_velocity', 'lateral_velocity', 'steering_angle',
            'speed', 'acceleration', 'yaw_rate', 'orientation_X', 'orientation_Y'
        ]
        return {field: {'min': float('inf'), 'max': float('-inf')} 
                for field in metrics_fields}

    def extract_timestamp(self, data: Dict) -> int:
        if isinstance(data, dict):
            return data.get('timestamp_sec')
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data[0].get('timestamp_sec')
        return None

    def update_metrics(self, data: Dict) -> None:
        for metric, value in data.items():
            if value is not None and metric in self.metrics:
                self.metrics[metric]['min'] = min(self.metrics[metric]['min'], value)
                self.metrics[metric]['max'] = max(self.metrics[metric]['max'], value)

    def process_velocity_data(self, data_points: Dict) -> Set[int]:
        stopped_threshold = 0.001  # m/s
        timestamps = sorted(data_points.keys())
        keep_timestamps = set()
        stopped_start = None

        for timestamp in timestamps:
            speed = data_points[timestamp].get('speed')
            if speed is None:
                continue

            if speed <= stopped_threshold:
                if stopped_start is None:
                    stopped_start = timestamp
            else:
                if stopped_start is not None:
                    stopped_duration = timestamp - stopped_start
                    if stopped_duration <= self.max_stopped_duration:
                        keep_timestamps.update(
                            ts for ts in timestamps 
                            if stopped_start <= ts < timestamp
                        )
                    stopped_start = None
                keep_timestamps.add(timestamp)

        if stopped_start is not None:
            stopped_duration = timestamps[-1] - stopped_start
            if stopped_duration <= self.max_stopped_duration:
                keep_timestamps.update(
                    ts for ts in timestamps 
                    if ts >= stopped_start
                )

        return keep_timestamps

    def load_data(self, input_folder: str) -> Dict:
        data_by_timestamp = defaultdict(lambda: defaultdict(list))
        
        for filename in self.files_to_clean:
            file_path = os.path.join(input_folder, filename)
            if not os.path.isfile(file_path):
                print(f"File: {file_path} not found!")
                continue

            with open(file_path, 'r') as file:
                print(f"Processing: {file_path}")
                for line in file:
                    try:
                        data = json.loads(line)
                        timestamp_sec = self.extract_timestamp(data['data'])
                        if timestamp_sec is not None:
                            data_by_timestamp[filename][timestamp_sec].append(data)
                            self._update_file_metrics(filename, data['data'])
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        print(f"Error in file {filename}: {str(e)}")
                        continue

        return data_by_timestamp

    def _update_file_metrics(self, filename: str, data: Dict) -> None:
        if filename == "_vehicle_status_velocity_status.json":
            self.update_metrics({
                'longitudinal_velocity': data.get('longitudinal_velocity'),
                'lateral_velocity': data.get('lateral_velocity'),
                'yaw_rate': data.get('yaw_rate')
            })
        elif filename == "_tf.json":
            self.update_metrics({
                'orientation_X': data['orientation'].get('x'),
                'orientation_Y': data['orientation'].get('y')
            })
        elif filename == "_system_emergency_control_cmd.json":
            self.update_metrics({
                'steering_angle': data.get('steering_angle'),
                'speed': data.get('speed'),
                'acceleration': data.get('acceleration')
            })

    def get_common_timestamps(self, data_by_timestamp: Dict) -> List[int]:
        all_timestamps = set.intersection(
            *[set(data_by_timestamp[filename].keys()) 
              for filename in self.files_to_clean]
        )
        return sorted(all_timestamps)

    def process_data(self, data_by_timestamp: Dict, common_timestamps: List[int]) -> Dict:
        processed_data = defaultdict(list)
        
        for timestamp in common_timestamps:
            for filename in self.files_to_clean:
                file_data = data_by_timestamp[filename][timestamp]
                
                if len(file_data) < self.data_points_per_second:
                    processed_points = (
                        file_data + 
                        [file_data[-1]] * (self.data_points_per_second - len(file_data))
                    )
                else:
                    indices = np.linspace(
                        0, len(file_data) - 1, 
                        self.data_points_per_second, dtype=int
                    )
                    processed_points = [file_data[i] for i in indices]
                
                processed_data[filename].extend(processed_points)

        return processed_data

    def write_output(self, output_folder: str, processed_data: Dict) -> None:
        for filename in self.files_to_clean:
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w') as outfile:
                for data_point in processed_data[filename]:
                    outfile.write(json.dumps(data_point) + '\n')

    def copy_route_file(self, input_folder: str, output_folder: str) -> None:
        route_file = "route.json"
        input_path = os.path.join(input_folder, route_file)
        if os.path.isfile(input_path):
            with open(input_path, 'r') as infile, \
                 open(os.path.join(output_folder, route_file), 'w') as outfile:
                outfile.write(infile.read())

    def clean_stream_data(self, current_data: dict) -> dict:
        """Process a single timestep of streaming data."""
        required_fields = {
            '/tf',
            '/perception/object_recognition/tracking/objects',
            '/perception/traffic_light_recognition/traffic_signals',
            '/vehicle/status/velocity_status',
            '/system/emergency/control_cmd'
        }
        
        if not all(field in current_data for field in required_fields):
            return None

        # Get reference timestamp from TF data
        reference_timestamp = current_data['/tf'].get('timestamp_sec')
        if reference_timestamp is None:
            return None
            
        # Verify all topics have matching timestamps
        for topic, data in current_data.items():
            timestamp = data.get('timestamp_sec')
            if timestamp is None or timestamp != reference_timestamp:
                return None

        return current_data


    def clean_batch_data(self, input_folder: str, output_folder: str) -> None:
        """Original batch processing logic"""
        os.makedirs(output_folder, exist_ok=True)
        
        data_by_timestamp = self.load_data(input_folder)
        common_timestamps = self.get_common_timestamps(data_by_timestamp)

        velocity_data_points = {
            ts: {'speed': data[0]['data'].get('speed')}
            for ts, data in data_by_timestamp["_system_emergency_control_cmd.json"].items()
            if ts in common_timestamps
        }

        timestamps_to_keep = self.process_velocity_data(velocity_data_points)
        filtered_timestamps = [
            ts for ts in common_timestamps 
            if ts in timestamps_to_keep
        ]

        processed_data = self.process_data(data_by_timestamp, filtered_timestamps)
        self.write_output(output_folder, processed_data)
        self.copy_route_file(input_folder, output_folder)

        return self.metrics

    def clean_data(self, input_data, output_folder: str = None) -> dict:
        """Unified interface for both streaming and batch modes"""
        if self.stream_mode:
            return self.clean_stream_data(input_data)
        else:
            return self.clean_batch_data(input_data, output_folder)