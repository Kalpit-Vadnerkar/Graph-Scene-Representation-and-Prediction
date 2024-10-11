import json
import os
from collections import defaultdict
import numpy as np

files_to_clean = {
    "_perception_object_recognition_tracking_objects.json",
    "_perception_traffic_light_recognition_traffic_signals.json",
    "_vehicle_status_steering_status.json",
    "_tf.json",
    "_system_emergency_control_cmd.json",
    "_vehicle_status_velocity_status.json",
}

tf_file = "_tf.json"

main_folder = input("Enter data folder name: ")
raw_data_folder = os.path.join(main_folder, "Raw_Dataset")
cleaned_data_folder = os.path.join(main_folder, "Cleaned_Dataset")

data_points_per_second = 10
max_stopped_duration = 3  # seconds

# Initialize metrics
metrics = {
    'longitudinal_velocity': {'min': float('inf'), 'max': float('-inf')},
    'lateral_velocity': {'min': float('inf'), 'max': float('-inf')},
    'steering_angle': {'min': float('inf'), 'max': float('-inf')},
    'speed': {'min': float('inf'), 'max': float('-inf')},
    'acceleration': {'min': float('inf'), 'max': float('-inf')},
    'yaw_rate': {'min': float('inf'), 'max': float('-inf')},
    'orientation_X': {'min': float('inf'), 'max': float('-inf')},
    'orientation_Y': {'min': float('inf'), 'max': float('-inf')},
}

def extract_timestamp(data):
    if isinstance(data, dict):
        return data.get('timestamp_sec')
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get('timestamp_sec')
    return None

def update_metrics(data):
    global metrics
    for metric, value in data.items():
        if value is not None and metric in metrics:
            metrics[metric]['min'] = min(metrics[metric]['min'], value)
            metrics[metric]['max'] = max(metrics[metric]['max'], value)

def process_velocity_data(data_points):
    stopped_threshold = 0.001  # m/s, threshold to consider the vehicle stopped

    timestamps = sorted(data_points.keys())
    keep_timestamps = set()
    stopped_start = None

    for i, timestamp in enumerate(timestamps):
        speed = data_points[timestamp].get('speed')
        
        if speed is None:
            continue  # Skip if speed data is missing

        if speed <= stopped_threshold:
            if stopped_start is None:
                stopped_start = timestamp
        else:
            if stopped_start is not None:
                stopped_duration = timestamp - stopped_start
                if stopped_duration <= max_stopped_duration:
                    # If stopped for 3 seconds or less, keep all timestamps in this interval
                    keep_timestamps.update(ts for ts in timestamps if stopped_start <= ts < timestamp)
                stopped_start = None
            keep_timestamps.add(timestamp)

    # Handle case where data ends while vehicle is stopped
    if stopped_start is not None:
        stopped_duration = timestamps[-1] - stopped_start
        if stopped_duration <= max_stopped_duration:
            keep_timestamps.update(ts for ts in timestamps if ts >= stopped_start)

    return keep_timestamps

def load_data(input_folder):
    data_by_timestamp = defaultdict(lambda: defaultdict(list))
    for filename in files_to_clean:
        file_path = os.path.join(input_folder, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    timestamp_sec = extract_timestamp(data['data'])
                    if timestamp_sec is not None:
                        data_by_timestamp[filename][timestamp_sec].append(data)
                        
                        # Update metrics based on the file type
                        if filename == "_vehicle_status_velocity_status.json":
                            update_metrics({
                                'longitudinal_velocity': data['data'].get('longitudinal_velocity'),
                                'lateral_velocity': data['data'].get('lateral_velocity'),
                                'yaw_rate' : data['data'].get('yaw_rate'),
                            })
                        elif filename == "_tf.json":
                            update_metrics({
                                'orientation_X': data['data']['orientation'].get('x'),
                                'orientation_Y': data['data']['orientation'].get('y'),
                            })
                        elif filename == "_system_emergency_control_cmd.json":
                            update_metrics({
                                'steering_angle': data['data'].get('steering_angle'),
                                'speed': data['data'].get('speed'),
                                'acceleration': data['data'].get('acceleration')
                            })
                        
                except json.JSONDecodeError:
                    continue
                except KeyError as e:
                    print(f"KeyError in file {filename}: {e}")
                except Exception as e:
                    print(f"Unexpected error in file {filename}: {e}")

    return data_by_timestamp

def get_common_timestamps(data_by_timestamp):
    all_timestamps = set.intersection(*[set(data_by_timestamp[filename].keys()) for filename in files_to_clean])
    return sorted(all_timestamps)

def process_data(data_by_timestamp, common_timestamps):
    processed_data = defaultdict(list)
    for timestamp in common_timestamps:
        min_data_points = len(data_by_timestamp[tf_file][timestamp])
        
        for filename in files_to_clean:
            file_data = data_by_timestamp[filename][timestamp]
            
            if len(file_data) < data_points_per_second:
                # If we have fewer than 'data_points_per_second' points, use all available points and pad if necessary
                processed_points = file_data + [file_data[-1]] * (data_points_per_second - len(file_data))
            else:
                # Select 'data_points_per_second' evenly spaced points
                indices = np.linspace(0, len(file_data) - 1, data_points_per_second, dtype=int)
                processed_points = [file_data[i] for i in indices]
            
            processed_data[filename].extend(processed_points)

    return processed_data

def write_output(output_folder, processed_data):
    for filename in files_to_clean:
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w') as outfile:
            for data_point in processed_data[filename]:
                outfile.write(json.dumps(data_point) + '\n')

def copy_route_file(input_folder, output_folder):
    route_file = "route.json"
    if os.path.isfile(os.path.join(input_folder, route_file)):
        with open(os.path.join(input_folder, route_file), 'r') as infile, \
             open(os.path.join(output_folder, route_file), 'w') as outfile:
            outfile.write(infile.read())

def main():
    os.makedirs(cleaned_data_folder, exist_ok=True)

    for subfolder in os.listdir(raw_data_folder):
        input_folder = os.path.join(raw_data_folder, subfolder)
        if not os.path.isdir(input_folder):
            continue

        output_folder = os.path.join(cleaned_data_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)

        data_by_timestamp = load_data(input_folder)
        common_timestamps = get_common_timestamps(data_by_timestamp)

        velocity_data_points = {
            ts: {
                'speed': data[0]['data'].get('speed')
                }
            for ts, data in data_by_timestamp["_system_emergency_control_cmd.json"].items()
            if ts in common_timestamps
            }


        timestamps_to_keep = process_velocity_data(velocity_data_points)
        common_timestamps = [ts for ts in common_timestamps if ts in timestamps_to_keep]

        processed_data = process_data(data_by_timestamp, common_timestamps)
        write_output(output_folder, processed_data)
        copy_route_file(input_folder, output_folder)

    print("Data cleaning and velocity filtering completed. Cleaned data stored in 'Cleaned_Dataset' folder.")
    for metric, values in metrics.items():
        print(f"Minimum {metric}: {values['min']}")
        print(f"Maximum {metric}: {values['max']}")

if __name__ == "__main__":
    main()
