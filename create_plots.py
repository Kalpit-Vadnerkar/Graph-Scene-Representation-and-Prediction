import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from TrajectoryDataset import TrajectoryDataset
from DLModels import GraphTrajectoryLSTM
from visualizer import visualize_predictions, make_predictions


def load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device):
    model = GraphTrajectoryLSTM(input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def restore_mean(scaled_mean_x, scaled_mean_y, graph_bounds):
    x_min, x_max, y_min, y_max = graph_bounds
    original_mean_x = scaled_mean_x * (x_max - x_min) + x_min
    original_mean_y = scaled_mean_y * (y_max - y_min) + y_min
    return [original_mean_x, original_mean_y]

def restore_variance(scaled_variance_x, scaled_variance_y, graph_bounds):
    x_min, x_max, y_min, y_max = graph_bounds
    original_variance_x = scaled_variance_x * (x_max - x_min)**2 
    original_variance_y = scaled_variance_y * (y_max - y_min)**2
    return [original_variance_x, original_variance_y]

def restore_position(scaled_x, scaled_y, graph_bounds):
    x_min, x_max, y_min, y_max = graph_bounds
    original_x = scaled_x * (x_max - x_min) + x_min
    original_y = scaled_y * (y_max - y_min) + y_min
    return [original_x, original_y]

def plot_3d_distributions(predictions, past_positions, future_positions, all_graph_bounds):
    fig = plt.figure(figsize=(20, 20))
    colors = ['viridis', 'plasma', 'inferno']
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        
        # Extract data
        past_pos = np.array(past_positions[i])
        future_pos = np.array(future_positions[i])
        pred_pos = predictions[i]['position_mean']
        pred_var = predictions[i]['position_var']
        
        # Unscale the predicted mean and variance using the corresponding graph bounds
        graph_bounds = all_graph_bounds[i] 
        future_pos = np.array([restore_position(x, y, graph_bounds) for x, y in future_pos])
        pred_pos = np.array([restore_mean(x, y, graph_bounds) for x, y in pred_pos])
        pred_var = np.array([restore_variance(x, y, graph_bounds) for x, y in pred_var])
        
        # Plot past trajectory
        #ax.plot(past_pos[:, 0], past_pos[:, 1], np.zeros_like(past_pos[:, 0]), 'b-', label='Past')
        
        # Plot ground truth future
        ax.scatter(future_pos[:, 0], future_pos[:, 1], np.zeros_like(future_pos[:, 0]), 'g-', label='Ground Truth')
        
        # Plot predicted mean
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], np.zeros_like(pred_pos[:, 0]), 'r-', label='Predicted Mean')
        
        # Create grid for probability distribution
        padding = 50
        x = np.linspace(min(pred_pos[:, 0]) - padding, max(pred_pos[:, 0]) + padding, 100)
        y = np.linspace(min(pred_pos[:, 1]) - padding, max(pred_pos[:, 1]) + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        # Plot probability distribution for each predicted point
        for t in range(len(pred_pos)):
            mean = pred_pos[t]
            cov = np.diag(pred_var[t])
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)

            # Normalize Z
            #Z = Z * 1000
            #Z = Z / Z.max()  # Scale Z so that the maximum value is 1

            ax.plot_surface(X, Y, Z, cmap=colors[t], alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Sequence {i+1}')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("predictions/pdf.png")

def plot_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds):
    fig = plt.figure(figsize=(20, 20))
    colors = ['viridis', 'plasma', 'inferno'] 
    num_timesteps = 11

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(4, 3, timestep + 1, projection='3d')

        relevant_sequences = [i for i in range(9) if timestep in range(i, i + 3)]

        # Calculate the mean of future positions for this timestep (once, before the prediction loop)
        future_pos_for_timestep = []
        for seq_idx in relevant_sequences:
            future_pos = np.array(future_positions[seq_idx])
            graph_bounds = all_graph_bounds[seq_idx]
            future_pos = np.array([restore_position(x, y, graph_bounds) for x, y in future_pos])
            future_pos_for_timestep.append(future_pos[timestep - seq_idx])

        if future_pos_for_timestep:  # Check if we have future positions for this timestep
            mean_future_pos = np.mean(future_pos_for_timestep, axis=0)

        padding = 100
        x = np.linspace(mean_future_pos[0] - padding, mean_future_pos[0] + padding, 100)
        y = np.linspace(mean_future_pos[1] - padding, mean_future_pos[1] + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        for seq_idx in relevant_sequences:
            # Extract data for this sequence
            past_pos = np.array(past_positions[seq_idx])
            future_pos = np.array(future_positions[seq_idx])
            pred_pos = predictions[seq_idx]['position_mean']
            pred_var = predictions[seq_idx]['position_var']

            # Unscale using graph bounds
            graph_bounds = all_graph_bounds[seq_idx]
            future_pos = np.array([restore_position(x, y, graph_bounds) for x, y in future_pos])
            pred_pos = np.array([restore_mean(x, y, graph_bounds) for x, y in pred_pos])
            pred_var = np.array([restore_variance(x, y, graph_bounds) for x, y in pred_var])

            # Plot predicted mean and distribution for the relevant timestep within this sequence
            t = timestep - seq_idx
            if 0 <= t < len(pred_pos):
                mean = pred_pos[t]
                cov = np.diag(pred_var[t])

                # Adjust predicted mean based on the shift in ground truth
                if future_pos_for_timestep:
                    original_future_pos = np.array(future_positions[seq_idx][t])
                    original_future_pos = restore_position(original_future_pos[0], original_future_pos[1], graph_bounds)
                    mean_shift = mean_future_pos - original_future_pos
                    mean += mean_shift

                # Create grid for probability distribution
                #padding = 50
                #x = np.linspace(min(pred_pos[:, 0]) - padding, max(pred_pos[:, 0]) + padding, 100)
                #y = np.linspace(min(pred_pos[:, 1]) - padding, max(pred_pos[:, 1]) + padding, 100)
                #X, Y = np.meshgrid(x, y)

                Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)

                # Plot
                ax.plot_surface(X, Y, Z, cmap=colors[seq_idx % 3], alpha=0.3)
                ax.scatter(mean[0], mean[1], 0, c='r', marker='x',
                           label='Predicted Mean' if seq_idx == relevant_sequences[0] else "")

        if future_pos_for_timestep:
            ax.scatter(mean_future_pos[0],
                       mean_future_pos[1],
                       0,
                       c='g', marker='o', label='Mean Ground Truth')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.show()
    plt.savefig("predictions/pdf_by_timestep.png")




input_sizes = {
        'node_features': 4,
        'position': 2,
        'velocity': 2,
        'steering': 1,
        'object_in_path': 1,
        'traffic_light_detected': 1
    }
hidden_size = 64
num_layers = 2
input_seq_len = 3  # past trajectory length
output_seq_len = 3  # future prediction length

scaling_factor = 10

# Data loading
data_folder = "Test_Dataset/Sequence_Dataset"
dataset = TrajectoryDataset(data_folder, position_scaling_factor=10, velocity_scaling_factor=100, steering_scaling_factor=100)

model_path = 'models/graph_trajectory_model.pth'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device)

print(device)

# Make predictions
predictions, sampled_sequences = make_predictions(model, dataset, device, num_samples=9)

# Extract past and future positions
past_positions = []
future_positions = []
all_graph_bounds = []
for i in range(10,19):
    past, future, graph, graph_bounds = dataset[i]
    past_positions.append(past['position'].numpy())
    future_positions.append(future['position'].numpy())
    all_graph_bounds.append(graph_bounds)


#visualize_predictions(model, dataset, scaling_factor, device)

print("trajectory viz. done!")

# Plot 3D distributions grouped by time steps
#plot_3d_distributions(predictions, past_positions, future_positions, all_graph_bounds)

print("sequence distribution viz. done!")

plot_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds)

print("time step distribution viz. done!")











