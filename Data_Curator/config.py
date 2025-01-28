import os
import torch

class Config:
    def __init__(self):
        # Map processing
        self.MAP_FILE = "map/lanelet2_map.osm"
        
        # Graph building
        self.MIN_DIST_BETWEEN_NODE = 5
        self.CONNECTION_THRESHOLD = 5
        self.MAX_NODES = 150
        self.MIN_NODES = 150
        self.NODE_FEATURES = 4
        
        # Sequence processing
        self.PAST_TRAJECTORY = 30
        self.PREDICTION_HORIZON = 30
        self.STRIDE = 1
        
        #  Data augmentation
        self.NUM_ROTATIONS = 0  # Number of rotations to perform
        self.MIRRORS = ['x', 'y']
        
        # Reference points for coordinate conversion
        self.REFERENCE_POINTS = [
            ((81370.40, 49913.81), (3527.96, 1775.78)),
            ((81375.16, 49917.01), (3532.70, 1779.04)),
            ((81371.85, 49911.62), (3529.45, 1773.63)),
            ((81376.60, 49914.82), (3534.15, 1776.87)),
        ]

        # Velocity and steering angle scaling factors
        self.MAX_VELOCITY_X = 12.0  # m/s, adjust as needed
        self.MIN_VELOCITY_X = -0.0  # m/s, adjust as needed
        self.MAX_VELOCITY_Y = 0.4  # m/s, adjust as needed
        self.MIN_VELOCITY_Y = -0.4  # m/s, adjust as needed
        self.MAX_STEERING = 0.5  # radians, adjust as needed
        self.MIN_STEERING = -0.5  # radians, adjust as needed
        self.MIN_ACCELERATION = -1.0 # m/s^2
        self.MAX_ACCELERATION = 1.0 # m/s^2

        # Prediction Model values
        self.MAX_GRAPH_NODES = 150
        self.MODEL_PATH = 'models/graph_attention_network.pth'   #GAT
        #self.model_path = 'models/lstm_trajectory_model.pth'        #LSTM
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_folders(self, user_folder):
        self.INPUT_FOLDER = os.path.join(user_folder, "Cleaned_Dataset")
        self.OUTPUT_FOLDER = os.path.join(user_folder, "Sequence_Dataset")

# Create a global instance of the config
config = Config()
