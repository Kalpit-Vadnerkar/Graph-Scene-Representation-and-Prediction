from Data_Curator.DataReader import DataReader

from collections import defaultdict
import numpy as np

class DataStreamer:
    def __init__(self, config):
        self.config = config
        self.data_points_per_second = 10
        self.topic_buffers = defaultdict(list)
        self.current_timestamp = None
        self.points_in_current_second = 0
        self.data_buffer = []

    def process_current_data(self, current_data):
        '''
        Input -> Cleaned ROS Message Stream
        Output -> Structured Timestamp Data used for State Estimation
        '''
        tf_data = current_data['/tf']
        timestamp_sec = tf_data[0].get('timestamp_sec') if isinstance(tf_data, list) else tf_data.get('timestamp_sec')
        
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
        
        processed_data = self._process_timestep_data(current_data)
        if processed_data:
            self.data_buffer.append(processed_data)
            
            if len(self.data_buffer) > self.config.PAST_TRAJECTORY:
                self.data_buffer.pop(0)
                
            return self.data_buffer, timestamp_sec
        return None, None
    
    def _process_timestep_data(self, data):
        reader = DataReader(None)
        return reader._process_timestamp_data(
            {'data': data.get('/tf', {})},
            {'data': data.get('/perception/object_recognition/tracking/objects', {})},
            {'data': data.get('/perception/traffic_light_recognition/traffic_signals', {})},
            {'data': data.get('/vehicle/status/velocity_status', {})},
            {'data': data.get('/system/emergency/control_cmd', {})}
        )
