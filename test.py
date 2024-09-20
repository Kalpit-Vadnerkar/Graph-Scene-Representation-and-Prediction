import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from TrajectoryDataset import TrajectoryDataset
from DLModels import GraphTrajectoryLSTM


def load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device):
    model = GraphTrajectoryLSTM(input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def make_predictions(model, dataset, device, num_samples=9):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for idx in range(num_samples):
            past, future, graph = dataset[idx+10]
            
            # Ensure consistent dimensions
            past = {k: v.unsqueeze(0).to(device) for k, v in past.items()}  # Add batch dimension
            graph = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}  # Add batch dimension
            
            predictions = model(past, graph)

            all_predictions.append({k: v.squeeze().cpu().numpy() for k, v in predictions.items()})

    return all_predictions

def plot_3d_distributions(predictions, past_positions, future_positions):
    fig = plt.figure(figsize=(20, 20))
    colors = ['viridis', 'plasma', 'inferno']
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        
        # Extract data
        past_pos = np.array(past_positions[i])
        future_pos = np.array(future_positions[i])
        pred_pos = predictions[i]['position_mean']
        pred_var = predictions[i]['position_var']
        
        # Plot past trajectory
        #ax.plot(past_pos[:, 0], past_pos[:, 1], np.zeros_like(past_pos[:, 0]), 'b-', label='Past')
        
        # Plot ground truth future
        ax.scatter(future_pos[:, 0], future_pos[:, 1], np.zeros_like(future_pos[:, 0]), 'g-', label='Ground Truth')
        
        # Plot predicted mean
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], np.zeros_like(pred_pos[:, 0]), 'r-', label='Predicted Mean')
        
        # Create grid for probability distribution
        padding = 1
        x = np.linspace(min(pred_pos[:, 0]) - padding, max(pred_pos[:, 0]) + padding, 100)
        y = np.linspace(min(pred_pos[:, 1]) - padding, max(pred_pos[:, 1]) + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        # Plot probability distribution for each predicted point
        for t in range(len(pred_pos)):
            mean = pred_pos[t]
            cov = np.diag(pred_var[t])
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
            ax.plot_surface(X, Y, Z, cmap=colors[t], alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Sequence {i+1}')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("pdf.png")

def plot_distributions_by_time_step(predictions, future_positions, num_time_steps=6):
    fig, axs = plt.subplots(2, 3, figsize=(20, 15), subplot_kw={'projection': '3d'})
    axs = axs.ravel()

    colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight']
    
    for t in range(num_time_steps):
        ax = axs[t]
        
        # Collect all predictions for this time step
        all_preds_x = []
        all_preds_y = []
        all_vars_x = []
        all_vars_y = []
        ground_truth = future_positions[t]
        
        for i in range(t, t+3):
            all_preds_x.append(predictions[i]['position_mean'][2][0])
            all_preds_x.append(predictions[i]['position_mean'][1][0])
            all_preds_x.append(predictions[i]['position_mean'][0][0])
            all_preds_y.append(predictions[i]['position_mean'][2][1])
            all_preds_y.append(predictions[i]['position_mean'][1][1])
            all_preds_y.append(predictions[i]['position_mean'][0][1])
            all_vars_x.append(predictions[i]['position_var'][2][0])
            all_vars_x.append(predictions[i]['position_var'][1][0])
            all_vars_x.append(predictions[i]['position_var'][0][0])
            all_vars_y.append(predictions[i]['position_var'][2][1])
            all_vars_y.append(predictions[i]['position_var'][1][1])
            all_vars_y.append(predictions[i]['position_var'][0][1])

        #for i, (pred, truth) in enumerate(zip(predictions, future_positions)):
        #    if t < len(pred['position_mean']):
        #        all_preds_x.append(pred['position_mean'][t][0])
        #        all_preds_y.append(pred['position_mean'][t][1])
        #        all_vars_x.append(pred['position_var'][t][0])
        #        all_vars_y.append(pred['position_var'][t][1])
        #        if i + t == 5:  # This ensures we get the correct ground truth for each time step
        #            ground_truth = truth[t]
        
        all_preds_x = np.array(all_preds_x)
        all_preds_y = np.array(all_preds_y)
        all_vars_x = np.array(all_vars_x)
        all_vars_y = np.array(all_vars_y)
        
        # Plot ground truth
        if ground_truth is not None:
            ax.scatter(ground_truth[0], ground_truth[1], 0, c='g', s=100, label='Ground Truth', marker='*')
        
        # Plot predicted means
        ax.scatter(all_preds_x, all_preds_y, np.zeros_like(all_preds_x), c='r', label='Predicted Mean')
        
        # Create grid for probability distribution
        padding = 1
        x = np.linspace(min(all_preds_x) - padding, max(all_preds_x) + padding, 100)
        y = np.linspace(min(all_preds_y) - padding, max(all_preds_y) + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        # Plot probability distribution for each prediction
        for pred_x, pred_y, var_x, var_y in zip(all_preds_x, all_preds_y, all_vars_x, all_vars_y):
            mean = [pred_x, pred_y]
            cov = np.diag([var_x, var_y])
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
            ax.plot_surface(X, Y, Z, cmap=colors[t], alpha=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.legend()
        ax.set_title(f'Time Step {t+4}')  # +4 because predictions start at t=4
    
    plt.tight_layout()
    plt.show()
    plt.savefig("time_step_distributions.png")


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

# Data loading
data_folder = "Test_Dataset/Sequence_Dataset"
dataset = TrajectoryDataset(data_folder, position_scaling_factor=10, velocity_scaling_factor=100, steering_scaling_factor=100)

model_path = 'graph_trajectory_model.pth'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len, device)

# Make predictions
predictions = make_predictions(model, dataset, device, num_samples=9)

# Extract past and future positions
past_positions = []
future_positions = []
for i in range(10,19):
    past, future, _ = dataset[i]
    past_positions.append(past['position'].numpy())
    future_positions.append(future['position'].numpy())

# Plot 3D distributions grouped by time steps
plot_3d_distributions(predictions, past_positions, future_positions)

plot_distributions_by_time_step(predictions, future_positions)
