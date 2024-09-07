import os
import pickle
import math
from tqdm import tqdm
from Data_Curation.MapProcessor import MapProcessor
from Data_Curation.SequenceProcessor import SequenceProcessor
from Data_Curation.SequenceAugmenter import SequenceAugmenter
from Data_Curation.GraphBuilder import GraphBuilder
from Data_Curation.DataReader import DataReader
from Data_Curation.config import config

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.map_processor = MapProcessor()  
        self.graph_builder = None
        self.sequence_processor = SequenceProcessor(config.PAST_TRAJECTORY, config.PREDICTION_HORIZON, config.REFERENCE_POINTS)
        self.all_data = []
    
    def process_run(self, folder_name, folder_path, inner_pbar):
        self.map_processor.load_route(folder_path)
        self.graph_builder = GraphBuilder(self.map_processor.map_data, self.map_processor.get_route(), 
                                          self.config.MIN_DIST_BETWEEN_NODE, self.config.CONNECTION_THRESHOLD, 
                                          self.config.MAX_NODES, self.config.MIN_NODES)
        
        data_reader = DataReader(folder_path)
        data = data_reader.read_scene_data()
        
        sequences = self.sequence_processor.create_sequences(data, self.graph_builder, inner_pbar)
        
        # Update inner progress bar for each sequence
        #for _ in sequences:
        #    inner_pbar.update(1)
        
        return sequences
    
    def augment_sequence(self, sequence):
        augmenter = SequenceAugmenter(sequence)
        return augmenter.augment()

    
    def data_augmentation(self, sequences, inner_pbar):
        for sequence in sequences:
            augmented_sequences = self.augment_sequence(sequence)
            self.all_data.extend(augmented_sequences)
            inner_pbar.update(len(augmented_sequences))
        
    def reset_run_data(self):
        self.all_data = []

    def save_sequences(self, sequences, folder_name):
        max_sequences_per_file = 1000
        num_files = math.ceil(len(sequences) / max_sequences_per_file)
        
        for i in range(num_files):
            start_idx = i * max_sequences_per_file
            end_idx = min((i + 1) * max_sequences_per_file, len(sequences))
            
            sequences_subset = sequences[start_idx:end_idx]
            
            if num_files == 1:
                output_file = os.path.join(self.config.OUTPUT_FOLDER, f"{folder_name}.pkl")
            else:
                output_file = os.path.join(self.config.OUTPUT_FOLDER, f"{folder_name}_part{i+1}.pkl")
            
            with open(output_file, 'wb') as f:
                pickle.dump(sequences_subset, f)
            
            print(f"Saved {len(sequences_subset)} sequences for {folder_name} in {output_file}")
        
        print(f"Total {len(sequences)} sequences saved for {folder_name} in {num_files} file(s)")


    def process_all_sequences(self, outer_pbar):
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
        folders = [f for f in os.listdir(self.config.INPUT_FOLDER) if os.path.isdir(os.path.join(self.config.INPUT_FOLDER, f))]
        
        for folder_name in folders:
            folder_path = os.path.join(self.config.INPUT_FOLDER, folder_name)
            print(f"Processing folder: {folder_name}")
            
            # First pass to count sequences
            data_reader = DataReader(folder_path)
            data = data_reader.read_scene_data()
            num_sequences = len(data) - self.config.PAST_TRAJECTORY - self.config.PREDICTION_HORIZON + 1
            total_augmented_sequences = num_sequences * (1 + config.NUM_ROTATIONS * (1 + len(config.MIRRORS)))
            
            # Create inner progress bar for each folder
            with tqdm(total=total_augmented_sequences, desc=f"  {folder_name}", leave=False) as inner_pbar:
                sequences = self.process_run(folder_name, folder_path, inner_pbar)
                self.data_augmentation(sequences, inner_pbar)
                self.save_sequences(self.all_data, folder_name)
                self.reset_run_data()
            
            outer_pbar.update(1)  # Update outer progress bar