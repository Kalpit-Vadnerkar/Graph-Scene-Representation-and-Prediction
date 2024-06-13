import json
import os


files_to_clean = {"_perception_object_recognition_tracking_objects.json",
                  "_vehicle_status_steering_status.json",
                  "_perception_traffic_light_recognition_traffic_signals.json",
                  "_tf.json",
                  "_vehicle_status_velocity_status.json",
}

input_folder = input("Provide input folder:")

# Create a folder to store the JSON files
output_folder = "Cleaned_" + input_folder
  
os.makedirs(output_folder, exist_ok=True)

for filename in files_to_clean:
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
        
            # If the timestamp_sec is not in the dictionary, add the data point
            if timestamp_sec not in data_points:
                data_points[timestamp_sec] = data
        except json.JSONDecodeError:
            # If the line is not valid JSON, skip it
            continue

    # Write the cleaned data points to a new file
    with open(os.path.join(output_folder, filename), 'w') as file:
        for data_point in data_points.values():
            file.write(json.dumps(data_point) + '\n')
