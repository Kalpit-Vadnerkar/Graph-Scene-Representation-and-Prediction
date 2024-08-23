import os
from DataProcessor import DataProcessor


user_folder = input("Please provide folder name: ")
map_file = "lanelet2_map.osm"
input_folder = os.path.join(user_folder, "Cleaned_Dataset")
output_folder = os.path.join(user_folder, "Sequence_Dataset")

past_trajectory = 3 # Number of past timesteps
prediction_horizon = 3  # Number of future timesteps to predict

processor = DataProcessor(map_file, input_folder, output_folder, past_trajectory, prediction_horizon)
processor.process_all_sequences()