import os
import pickle
from MapProcessor import MapProcessor
from SequenceProcessor import SequenceProcessor
from SequenceAugmenter import SequenceAugmenter
from GraphBuilder import GraphBuilder
from DataReader import DataReader


class DataProcessor:
    def __init__(self, map_file, input_folder, output_folder, min_dist_between_node, connection_threshold, max_nodes, min_nodes, window_size, prediction_horizon):
        self.map_processor = MapProcessor(map_file)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.graph_builder = None
        self.rotations = [90, 180, 270]
        self.mirrors = ['x', 'y']
        self.reference_points = [
            ((81370.40, 49913.81), (3527.96, 1775.78)),
            ((81375.16, 49917.01), (3532.70, 1779.04)),
            ((81371.85, 49911.62), (3529.45, 1773.63)),
            ((81376.60, 49914.82), (3534.15, 1776.87)),
        ]
        self.sequence_processor = SequenceProcessor(window_size, prediction_horizon, self.reference_points)
        self.min_dist_between_node = min_dist_between_node
        self.connection_threshold = connection_threshold
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.all_data = []
    
    def process_run(self, folder_name, folder_path):
        print(f"Processing folder: {folder_name}")
        self.map_processor.load_route(folder_path)
        self.graph_builder = GraphBuilder(self.map_processor.map_data, self.map_processor.get_route(), 
                                          self.min_dist_between_node, self.connection_threshold, 
                                          self.max_nodes, self.min_nodes)
        
        data_reader = DataReader(folder_path)
        data = data_reader.read_scene_data()
        
        all_sequences = []

        return self.sequence_processor.create_sequences(data, self.graph_builder)
    
    def augment_sequence(self, sequence):
        augmenter = SequenceAugmenter(sequence)
        augmented_sequences = [sequence]  # Include the original sequence
        
        # Add rotated sequences
        augmented_sequences.extend(augmenter.augment(rotations=self.rotations))
        # Add mirrored sequences
        augmented_sequences.extend(augmenter.augment(mirrors=self.mirrors))
        
        return augmented_sequences
    
    def data_augmentation(self, sequences):
        for sequence in sequences:
            augmented_sequences = self.augment_sequence(sequence)
            for augmented_seq in augmented_sequences:
                self.all_data.append(augmented_seq)
        
    def reset_run_data(self):
        self.all_data = []

    def save_sequences(self, sequences, folder_name):
        output_file = os.path.join(self.output_folder, f"{folder_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved {len(sequences)} sequences for {folder_name}")

    def process_all_sequences(self):
        os.makedirs(self.output_folder, exist_ok=True)
        for folder_name in os.listdir(self.input_folder):
            folder_path = os.path.join(self.input_folder, folder_name)
            if os.path.isdir(folder_path):
                sequences = self.process_run(folder_name, folder_path)
                self.data_augmentation(sequences)
                self.save_sequences(self.all_data, folder_name)
                self.reset_run_data()
