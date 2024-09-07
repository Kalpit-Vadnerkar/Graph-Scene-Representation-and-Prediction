import os
from Data_Curation.DataProcessor import DataProcessor
from Data_Curation.config import config
from tqdm import tqdm

user_folder = input("Please provide folder name: ")
config.set_folders(user_folder)

processor = DataProcessor(config)

# Get the total number of folders to process
total_folders = sum(1 for folder_name in os.listdir(config.INPUT_FOLDER) if os.path.isdir(os.path.join(config.INPUT_FOLDER, folder_name)))

# Create a progress bar
with tqdm(total=total_folders, desc="Processing folders") as pbar:
    processor.process_all_sequences(pbar)