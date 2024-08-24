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

main_folder = "Dataset"
raw_data_folder = os.path.join(main_folder, "Raw_Dataset")
cleaned_data_folder = os.path.join(main_folder, "Cleaned_Dataset")
stopped_limit = 4

def process_velocity_data(data_points):
    zero_velocity_count = 0
    timestamps_to_keep = set()
    
    for timestamp, data in sorted(data_points.items()):
        velocity = data['longitudinal_velocity']
        if abs(velocity) < 1e-6:  # Consider velocities very close to 0 as 0
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
        return data.get('longitudinal_velocity')
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get('longitudinal_velocity')
    return None

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
                velocity = extract_velocity(data['data'])
                if timestamp_sec in common_timestamps and velocity is not None:
                    velocity_data_points[timestamp_sec] = {'longitudinal_velocity': velocity}
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