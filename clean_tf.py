import json

# Open the file and read the lines
with open('Sequence_1/_tf.json', 'r') as file:
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
with open('cleaned_tf_data.json', 'w') as file:
    for data_point in data_points.values():
        file.write(json.dumps(data_point) + '\n')
