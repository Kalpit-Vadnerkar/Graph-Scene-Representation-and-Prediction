from Data_Curator.SequenceProcessor import SequenceProcessor
from Data_Curator.GraphBuilder import GraphBuilder
from Data_Curator.DataReader import DataReader

from collections import defaultdict
import numpy as np

class DataStreamer:
    def __init__(self, config, map_processor):
        self.config = config
        self.map_processor = map_processor
        self.graph_builder = None
        self.sequence_processor = SequenceProcessor(
            config.PAST_TRAJECTORY, 
            config.PREDICTION_HORIZON,
            config.REFERENCE_POINTS
        )
        self.data_points_per_second = 10
        self.topic_buffers = defaultdict(list)
        self.current_timestamp = None
        self.points_in_current_second = 0
        
    def initialize_graph_builder(self, route):
        self.map_processor.route = route
        self.graph_builder = GraphBuilder(
            self.map_processor.map_data,
            self.map_processor.get_route(),
            self.config.MIN_DIST_BETWEEN_NODE,
            self.config.CONNECTION_THRESHOLD,
            self.config.MAX_NODES,
            self.config.MIN_NODES
        )
    
    def process_current_data(self, current_data):
        timestamp_sec = current_data['/tf'].get('timestamp_sec')
        if timestamp_sec is None:
            return
            
        if self.current_timestamp is None:
            self.current_timestamp = timestamp_sec
                
        if timestamp_sec > self.current_timestamp:
            if self.points_in_current_second < self.data_points_per_second:
                last_point = {topic: buffer[-1] for topic, buffer in self.topic_buffers.items()}
                points_to_add = self.data_points_per_second - self.points_in_current_second
                for _ in range(points_to_add):
                    for topic, point in last_point.items():
                        self.topic_buffers[topic].append(point)
            elif self.points_in_current_second > self.data_points_per_second:
                for topic in self.topic_buffers:
                    points = self.topic_buffers[topic][-self.points_in_current_second:]
                    indices = np.linspace(0, len(points) - 1, self.data_points_per_second, dtype=int)
                    self.topic_buffers[topic] = self.topic_buffers[topic][:-self.points_in_current_second] + [points[i] for i in indices]
            
            self.current_timestamp = timestamp_sec
            self.points_in_current_second = 0
                
        for topic, data in current_data.items():
            self.topic_buffers[topic].append(data)
        self.points_in_current_second += 1
        
        # Process if buffer is full
        if all(len(buffer) >= self.config.PAST_TRAJECTORY for buffer in self.topic_buffers.values()):
            processed_data = self._process_timestep_data(current_data)
            if processed_data:
                self.sequence_processor.process_timestep(processed_data)
            
            # Remove oldest point from each buffer
            for topic in self.topic_buffers:
                self.topic_buffers[topic].pop(0)