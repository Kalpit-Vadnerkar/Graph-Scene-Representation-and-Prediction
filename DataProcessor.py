import os
import pickle
from tqdm import tqdm
from MapProcessor import MapProcessor
from SequenceProcessor import SequenceProcessor
from SequenceAugmenter import SequenceAugmenter
from GraphBuilder import GraphBuilder
from DataReader import DataReader
from config import config

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
        augmented_sequences = [sequence]  # Include the original sequence
        augmented_sequences.extend(augmenter.augment(rotations=self.config.ROTATIONS))
        augmented_sequences.extend(augmenter.augment(mirrors=self.config.MIRRORS))
        return augmented_sequences
    
    def data_augmentation(self, sequences, inner_pbar):
        for sequence in sequences:
            augmented_sequences = self.augment_sequence(sequence)
            for augmented_seq in augmented_sequences:
                self.all_data.append(augmented_seq)
            inner_pbar.update(len(augmented_sequences))
        
    def reset_run_data(self):
        self.all_data = []

    def save_sequences(self, sequences, folder_name):
        output_file = os.path.join(self.config.OUTPUT_FOLDER, f"{folder_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved {len(sequences)} sequences for {folder_name}")

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
            total_sequences = num_sequences * (1 + len(self.config.ROTATIONS) + len(self.config.MIRRORS))
            
            # Create inner progress bar for each folder
            with tqdm(total=num_sequences, desc=f"  {folder_name}", leave=False) as inner_pbar:
                sequences = self.process_run(folder_name, folder_path, inner_pbar)
                self.data_augmentation(sequences, inner_pbar)
                self.save_sequences(self.all_data, folder_name)
                self.reset_run_data()
            
            outer_pbar.update(1)  # Update outer progress bar