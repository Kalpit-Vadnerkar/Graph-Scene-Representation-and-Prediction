import json
import os
import math

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

max_velocity = float('-inf')
min_velocity = float('inf')
max_steering_angle = float('-inf')
min_steering_angle = float('inf')

def process_velocity_data(data_points):
    global max_velocity, min_velocity
    zero_velocity_count = 0
    timestamps_to_keep = set()
    
    for timestamp, data in sorted(data_points.items()):
        velocity = data['data']['longitudinal_velocity']
        max_velocity = max(max_velocity, velocity)
        min_velocity = min(min_velocity, velocity)
        if abs(velocity) < 1e-6:  # Consider velocities very close to 0 as 0
            zero_velocity_count += 1
        else:
            zero_velocity_count = 0
        
        if zero_velocity_count <= stopped_limit:
            timestamps_to_keep.add(timestamp)
    
    return timestamps_to_keep

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

    # Process velocity data first
    velocity_file = "_vehicle_status_velocity_status.json"
    velocity_data_points = {}
    
    with open(os.path.join(input_folder, velocity_file), 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                timestamp_sec = data['data']['timestamp_sec']
                if "Transform not found" not in str(data['data']):
                    velocity_data_points[timestamp_sec] = data
            except json.JSONDecodeError:
                continue

    timestamps_to_keep = process_velocity_data(velocity_data_points)

    for filename in files_to_clean:
        # Check if the file exists in the input folder
        if not os.path.isfile(os.path.join(input_folder, filename)):
            continue

        # Open the file and read the lines
        with open(os.path.join(input_folder, filename), 'r') as file:
            lines = file.readlines()

        # Initialize a dictionary to store the data points
        data_points = {}

        # Iterate over the lines
        for line in lines:
            try:
                # Try to parse the line as JSON
                data = json.loads(line)

                # If the line contains "Transform not found", skip it
                if "Transform not found" in str(data['data']):
                    continue

                # Extract the timestamp_sec
                timestamp_sec = data['data']['timestamp_sec']

                # If the filename is the steering status file, update max_steering_angle and min_steering_angle
                if filename == "_vehicle_status_steering_status.json":
                    steering_angle = data['data']['steering_angle']
                    max_steering_angle = max(max_steering_angle, steering_angle)
                    min_steering_angle = min(min_steering_angle, steering_angle)

                # If the timestamp_sec is in the set of timestamps to keep, add the data point
                if timestamp_sec in timestamps_to_keep:
                    data_points[timestamp_sec] = data
            except json.JSONDecodeError:
                # If the line is not valid JSON, skip it
                continue

        # Write the cleaned data points to a new file
        with open(os.path.join(output_folder, filename), 'w') as file:
            for data_point in data_points.values():
                file.write(json.dumps(data_point) + '\n')

    # Copy the route file without modifications
    route_file = "route.json"
    if os.path.isfile(os.path.join(input_folder, route_file)):
        with open(os.path.join(input_folder, route_file), 'r') as infile, \
             open(os.path.join(output_folder, route_file), 'w') as outfile:
            outfile.write(infile.read())

print("Data cleaning and velocity filtering completed.")
print(f"Maximum velocity: {max_velocity:.2f} m/s")
print(f"Minimum velocity: {min_velocity:.2f} m/s")
print(f"Maximum steering angle: {max_steering_angle:.2f} radians ({math.degrees(max_steering_angle):.2f} degrees)")
print(f"Minimum steering angle: {min_steering_angle:.2f} radians ({math.degrees(min_steering_angle):.2f} degrees)")