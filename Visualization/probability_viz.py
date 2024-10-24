import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

def plot_probabilities(config, predictions, future_values, condition, variable_name, scale=None):
    """
    Plots the probability trend of ground truth values for a given variable.

    Args:
        config: Dictionary containing configuration parameters (e.g., 'output_seq_len').
        predictions: List of predictions, where each prediction is a dictionary
                     containing 'mean' and 'var' for the variable.
        future_values: Ground truth values for the variable.
        condition: String describing the condition (e.g., 'nominal', 'sensor1 faulty').
        variable_name: String representing the name of the variable (e.g., 'steering_angle', 'pos_x').
        range: Tuple specifying the valid range of the variable (e.g., (0, 10) for steering angle).
               If None, no range constraint is applied.
    """
    num_samples = len(predictions)
    sequence_length = config['output_seq_len']
    num_timesteps = int(num_samples + sequence_length - 1)
   
    probabilities = np.zeros((sequence_length, num_timesteps))

    X = np.linspace(0, 10, 100)

    for timestep in range(num_timesteps):
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + sequence_length)]
        future_val_for_timestep = []

        for seq_idx in relevant_sequences:
            future_val = np.array([v for v in future_values[seq_idx]])
            future_val_for_timestep.append(future_val[timestep - seq_idx])

        if future_val_for_timestep:
            ground_truth = np.mean(future_val_for_timestep, axis=0)

            for seq_idx in relevant_sequences:
                pred_mean = predictions[seq_idx][f'{variable_name}_mean']
                pred_var = predictions[seq_idx][f'{variable_name}_var']

                t = timestep - seq_idx
                if 0 <= t < len(pred_mean):
                    mean = pred_mean[t]
                    variance = pred_var[t]
                    actual_future_val = future_values[seq_idx][t]

                    mean_shift = ground_truth - actual_future_val
                    mean += mean_shift

                    if scale:
                        mean = np.clip(mean, scale[0], scale[1])

                    # Create the distribution
                    distribution = norm.pdf(X, mean, np.sqrt(variance))
                    
                    # Normalize distribution
                    distribution = distribution / np.sum(distribution)
                    
                    # Calculate the probability of the ground truth
                    index = np.argmin(np.abs(X - ground_truth))
                    probability = distribution[index]
                    probabilities[t, timestep] = probability

    plt.figure(figsize=(10, 6))
    #for i in range(num_timesteps):
    for i in range(55, 60):
        plt.plot(probabilities[:, i], label=f'Timestep {i+1}')
    plt.title(f'Probability Trend of Ground Truth {variable_name} - {condition}')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(f"Probability_Plots/{variable_name}/{condition}.png")


def plot_probabilities2(config, predictions, future_values, condition, variable_name, scale=None):
    num_samples = len(predictions)
    sequence_length = config['output_seq_len']
    num_timesteps = int(num_samples + sequence_length - 1)

    probabilities_x = np.zeros((sequence_length, num_timesteps))
    probabilities_y = np.zeros((sequence_length, num_timesteps))

    X = np.linspace(0, 10, 100)
    
    for timestep in range(num_timesteps):
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + sequence_length)]
        future_val_for_timestep = []

        for seq_idx in relevant_sequences:
            future_val = np.array(future_values[seq_idx])
            future_val_for_timestep.append(future_val[timestep - seq_idx])

        if future_val_for_timestep:
            ground_truth = np.mean(future_val_for_timestep, axis=0)

            for seq_idx in relevant_sequences:
                pred_mean = np.array(predictions[seq_idx][f'{variable_name}_mean'])
                pred_var = np.array(predictions[seq_idx][f'{variable_name}_var'])

                t = timestep - seq_idx
                if 0 <= t < len(pred_mean):
                    mean_x, mean_y = pred_mean[t]
                    variance_x, variance_y = pred_var[t]
                    actual_future_val = future_values[seq_idx][t]

                    mean_shift = ground_truth - actual_future_val
                    mean_x += mean_shift[0]
                    mean_y += mean_shift[1]

                    if scale:
                        mean_x = np.clip(mean_x, scale[0], scale[1])
                        mean_y = np.clip(mean_y, scale[0], scale[1])

                    # Create separate distributions for x and y
                    distribution_x = norm.pdf(X, mean_x, np.sqrt(variance_x))
                    distribution_y = norm.pdf(X, mean_y, np.sqrt(variance_y))

                    # Normalize distributions
                    distribution_x = distribution_x / np.sum(distribution_x)
                    distribution_y = distribution_y / np.sum(distribution_y)

                    # Calculate the probability of the ground truth
                    index_x = np.argmin(np.abs(X - ground_truth[0]))
                    index_y = np.argmin(np.abs(X - ground_truth[1]))
                    probability_x = distribution_x[index_x]
                    probability_y = distribution_y[index_y]
                    probabilities_x[t, timestep] = probability_x
                    probabilities_y[t, timestep] = probability_y

    # Plot the probabilities
    plt.figure(figsize=(10, 6))
    for i in range(30, num_timesteps - 30, 30):
    #for i in range(55, 60):
        plt.plot(probabilities_x[:, i], label=f'Timestep {i+1} (x)')
    plt.title(f'Probability Trend of Ground Truth {variable_name} (x) - {condition}')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(f"Probability_Plots/{variable_name}_X/{condition}.png")

    plt.figure(figsize=(10, 6))
    for i in range(30, num_timesteps - 30, 30):
    #for i in range(55, 60):
        plt.plot(probabilities_y[:, i], label=f'Timestep {i+1} (y)')
    plt.title(f'Probability Trend of Ground Truth {variable_name} (y) - {condition}')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(f"Probability_Plots/{variable_name}_Y/{condition}.png")