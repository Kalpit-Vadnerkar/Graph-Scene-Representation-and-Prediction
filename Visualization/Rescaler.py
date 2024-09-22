import numpy as np

class GraphBoundsScaler:
    def __init__(self, graph_bounds):
        self.x_min, self.x_max, self.y_min, self.y_max = graph_bounds

    def restore_position(self, scaled_x, scaled_y):
        original_x = scaled_x * (self.x_max - self.x_min) + self.x_min
        original_y = scaled_y * (self.y_max - self.y_min) + self.y_min
        return np.array([original_x, original_y])

    def restore_mean(self, scaled_mean_x, scaled_mean_y):
        return self.restore_position(scaled_mean_x, scaled_mean_y)

    def restore_variance(self, scaled_variance_x, scaled_variance_y):
        original_variance_x = scaled_variance_x * (self.x_max - self.x_min)**2 
        original_variance_y = scaled_variance_y * (self.y_max - self.y_min)**2
        return np.array([original_variance_x, original_variance_y])