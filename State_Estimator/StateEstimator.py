from Data_Curator.Point import Point
from State_Estimator.MapProcessor import MapProcessor
from State_Estimator.GraphBuilder import GraphBuilder
from State_Estimator.SequenceProcessor import SequenceProcessor

class StateEstimator:
    def __init__(self, config):
        self.config = config
        self.map_processor = MapProcessor()
        self.sequence_processor = SequenceProcessor(
            config.PAST_TRAJECTORY,
            config.PREDICTION_HORIZON,
            config.REFERENCE_POINTS
        )
        self.graph_builder = None
        self.route = None
        self._initialize()

    def _initialize(self):
        self.graph_builder = GraphBuilder(
            self.map_processor.map_data,
            self.route,
            self.config.MIN_DIST_BETWEEN_NODE,
            self.config.CONNECTION_THRESHOLD,
            self.config.MAX_NODES,
            self.config.MIN_NODES
        )

    def update_route(self, route):
        self.route = route
        self.map_processor.route = route
        self._initialize()

    def estimate_state(self, data_buffer, timestamp):
        if not self.route or len(data_buffer) < self.config.PAST_TRAJECTORY:
            return None

        initial_position = self.sequence_processor.extract_ego_data(data_buffer[0])['position']
        final_position = self.sequence_processor.extract_ego_data(data_buffer[-1])['position']
        
        G = self.graph_builder.create_expanded_graph(initial_position, final_position)
        G, x_min, x_max, y_min, y_max = self.sequence_processor.scale_graph(G)
        
        past_sequence = []
        for data in data_buffer[-self.config.PAST_TRAJECTORY:]:
            past_sequence.append(
                self.sequence_processor.process_timestep(
                    data, G, x_min, x_max, y_min, y_max, 
                    is_past=True  # Added parameter for SequenceProcessor compatibility
                )
            )
        
        return {
            'timestamp': timestamp,
            'past': past_sequence,
            'graph': G,
            'graph_bounds': [x_min, x_max, y_min, y_max]
        }