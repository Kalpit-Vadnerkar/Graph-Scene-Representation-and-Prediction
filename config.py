import os

class Config:
    def __init__(self):
        # Map processing
        self.MAP_FILE = "lanelet2_map.osm"
        
        # Graph building
        self.MIN_DIST_BETWEEN_NODE = 5
        self.CONNECTION_THRESHOLD = 5
        self.MAX_NODES = 500
        self.MIN_NODES = 200
        
        # Sequence processing
        self.PAST_TRAJECTORY = 3
        self.PREDICTION_HORIZON = 3
        
        # Data augmentation
        self.ROTATIONS = [90, 180, 270]
        self.MIRRORS = ['x', 'y']
        
        # Reference points for coordinate conversion
        self.REFERENCE_POINTS = [
            ((81370.40, 49913.81), (3527.96, 1775.78)),
            ((81375.16, 49917.01), (3532.70, 1779.04)),
            ((81371.85, 49911.62), (3529.45, 1773.63)),
            ((81376.60, 49914.82), (3534.15, 1776.87)),
        ]

        # Velocity and steering angle scaling factors
        self.MAX_VELOCITY = 12.0  # m/s, adjust as needed
        self.MIN_VELOCITY = -0.0  # m/s, adjust as needed
        self.MAX_STEERING = 0.4  # radians, adjust as needed
        self.MIN_STEERING = -0.4  # radians, adjust as needed
        
    def set_folders(self, user_folder):
        self.INPUT_FOLDER = os.path.join(user_folder, "Cleaned_Dataset")
        self.OUTPUT_FOLDER = os.path.join(user_folder, "Sequence_Dataset")

# Create a global instance of the config
config = Config()