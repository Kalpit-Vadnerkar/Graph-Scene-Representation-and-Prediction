import json
import os
from collections import defaultdict

files_to_clean = {
    "_perception_object_recognition_tracking_objects.json",
    "_vehicle_status_steering_status.json",
    "_perception_traffic_light_recognition_traffic_signals.json",
    "_tf.json",
    "_vehicle_status_velocity_status.json",
}

main_folder = input("Enter data folder name: ")
raw_data_folder = os.path.join(main_folder, "Raw_Dataset")
cleaned_data_folder = os.path.join(main_folder, "Cleaned_Dataset")
stopped_limit = 4

min_longitudinal_velocity = float('inf')
max_longitudinal_velocity = float('-inf')
min_lateral_velocity = float('inf')
max_lateral_velocity = float('-inf')

def process_velocity_data(data_points):
    zero_velocity_count = 0
    timestamps_to_keep = set()

    global min_longitudinal_velocity, max_longitudinal_velocity, min_lateral_velocity, max_lateral_velocity

    for timestamp, data in sorted(data_points.items()):
        longitudinal_velocity = data.get('longitudinal_velocity')
        lateral_velocity = data.get('lateral_velocity')

        # Update min and max longitudinal velocities
        if longitudinal_velocity is not None:
            min_longitudinal_velocity = min(min_longitudinal_velocity, longitudinal_velocity)
            max_longitudinal_velocity = max(max_longitudinal_velocity, longitudinal_velocity)

        # Update min and max lateral velocities
        if lateral_velocity is not None:
            min_lateral_velocity = min(min_lateral_velocity, lateral_velocity)
            max_lateral_velocity = max(max_lateral_velocity, lateral_velocity)

        if longitudinal_velocity is not None and abs(longitudinal_velocity) < 1e-6:  # Consider velocities very close to 0 as 0
            zero_velocity_count += 1
        else:
            zero_velocity_count = 0

        if zero_velocity_count <= stopped_limit:
            timestamps_to_keep.add(timestamp)

    return timestamps_to_keep

def extract_timestamp(data):
    if isinstance(data, dict):
        return data.get('timestamp_sec')
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get('timestamp_sec')
    return None

def extract_velocity(data):
    if isinstance(data, dict):
        return data.get('longitudinal_velocity'), data.get('lateral_velocity')
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get('longitudinal_velocity'), data[0].get('lateral_velocity')
    return None, None

# Create the Cleaned_Dataset folder
os.makedirs(cleaned_data_folder, exist_ok=True)

# Iterate over all subfolders in the Raw_Dataset folder
for subfolder in os.listdir(raw_data_folder):
    input_folder = os.path.join(raw_data_folder, subfolder)
    if not os.path.isdir(input_folder):
        continue

    # Create a folder to store the cleaned JSON files
    output_folder = os.path.join(cleaned_data_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    # First, collect all timestamps from all files
    all_timestamps = defaultdict(set)
    for filename in files_to_clean:
        if not os.path.isfile(os.path.join(input_folder, filename)):
            continue

        with open(os.path.join(input_folder, filename), 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    timestamp_sec = extract_timestamp(data['data'])
                    if timestamp_sec is not None:
                        all_timestamps[filename].add(timestamp_sec)
                except json.JSONDecodeError:
                    continue
                except KeyError as e:
                    print(f"KeyError in file {filename}: {e}")
                    print(f"Data structure: {data}")
                except Exception as e:
                    print(f"Unexpected error in file {filename}: {e}")
                    print(f"Data structure: {data}")

    # Find common timestamps across all files
    common_timestamps = set.intersection(*all_timestamps.values())

    # Now process velocity data
    velocity_file = "_vehicle_status_velocity_status.json"
    velocity_data_points = {}

    with open(os.path.join(input_folder, velocity_file), 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                timestamp_sec = extract_timestamp(data['data'])
                longitudinal_velocity, lateral_velocity = extract_velocity(data['data'])
                if timestamp_sec in common_timestamps and longitudinal_velocity is not None:
                    velocity_data_points[timestamp_sec] = {'longitudinal_velocity': longitudinal_velocity, 'lateral_velocity': lateral_velocity}
            except json.JSONDecodeError:
                continue
            except KeyError as e:
                print(f"KeyError in velocity file: {e}")
                print(f"Data structure: {data}")
            except Exception as e:
                print(f"Unexpected error in velocity file: {e}")
                print(f"Data structure: {data}")

    timestamps_to_keep = process_velocity_data(velocity_data_points)

    # Now process all files
    for filename in files_to_clean:
        if not os.path.isfile(os.path.join(input_folder, filename)):
            continue

        data_points = {}
        with open(os.path.join(input_folder, filename), 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    timestamp_sec = extract_timestamp(data['data'])
                    if timestamp_sec in common_timestamps and timestamp_sec in timestamps_to_keep:
                        data_points[timestamp_sec] = data
                except json.JSONDecodeError:
                    continue
                except KeyError as e:
                    print(f"KeyError in file {filename}: {e}")
                    print(f"Data structure: {data}")
                except Exception as e:
                    print(f"Unexpected error in file {filename}: {e}")
                    print(f"Data structure: {data}")

        # Write the cleaned data points to a new file
        with open(os.path.join(output_folder, filename), 'w') as file:
            for data_point in sorted(data_points.values(), key=lambda x: extract_timestamp(x['data'])):
                file.write(json.dumps(data_point) + '\n')

    # Copy the route file without modifications
    route_file = "route.json"
    if os.path.isfile(os.path.join(input_folder, route_file)):
        with open(os.path.join(input_folder, route_file), 'r') as infile, \
             open(os.path.join(output_folder, route_file), 'w') as outfile:
            outfile.write(infile.read())

print("Data cleaning and velocity filtering completed. Cleaned data stored in 'Cleaned_Dataset' folder.")
print(f"Minimum longitudinal velocity: {min_longitudinal_velocity}")
print(f"Maximum longitudinal velocity: {max_longitudinal_velocity}")
print(f"Minimum lateral velocity: {min_lateral_velocity}")
print(f"Maximum lateral velocity: {max_lateral_velocity}")
