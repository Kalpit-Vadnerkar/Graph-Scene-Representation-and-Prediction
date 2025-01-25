import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from Visualization.Rescaler import GraphBoundsScaler

def plot_graph_and_trajectories(sequence, scaling_factor, predicted_future, ax):
    # Extract the graph
    G = sequence['graph']

    scaler = GraphBoundsScaler(sequence['graph_bounds'])
    
    # Plot graph nodes
    pos = {node: scaler.restore_position(data['x'] * scaling_factor, data['y'] * scaling_factor) for node, data in G.nodes(data=True)}

    # Plot regular map nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='black', node_size=1, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 0 and d['path_node'] == 0])

    # Plot traffic light nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='red', node_size=2, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['traffic_light_detection_node'] == 1])

    # Plot path nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='yellow', node_size=2, 
                           nodelist=[n for n, d in G.nodes(data=True) if d['path_node'] == 1])

    # Plot edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1)

    # Plot past trajectory
    past_positions = np.array([scaler.restore_position(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) for step in sequence['past']])
    ax.scatter(past_positions[:, 0], past_positions[:, 1], c='darkgreen', s=10, label='Past positions')

    # Plot actual future trajectory
    future_positions = np.array([scaler.restore_position(step['position'][0] * scaling_factor, step['position'][1] * scaling_factor) for step in sequence['future']])
    ax.scatter(future_positions[:, 0], future_positions[:, 1], c='blue', s=10, label='Actual future')

    # Plot predicted future trajectory with uncertainty
    pred_positions = np.array([scaler.restore_mean(x, y) for x, y in predicted_future['position_mean']])
    pred_variances = np.array([scaler.restore_variance(x, y) for x, y in predicted_future['position_var']])

    print(f"Position Variances: {predicted_future['position_var']}")
    #print(f"Velocity Variances: {predicted_future['velocity_var']}")
    #print(f"Steering Variances: {predicted_future['steering_var']}")
    #print(f"Acceleration Variances: {predicted_future['acceleration_var']}")

    ax.scatter(pred_positions[:, 0], pred_positions[:, 1], c='red', s=10, label='Predicted future mean')
    
    # Visualize uncertainty as distributions
    for i in range(len(pred_positions)):
        x, y = np.mgrid[pred_positions[i, 0] - 3*np.sqrt(pred_variances[i, 0]):pred_positions[i, 0] + 3*np.sqrt(pred_variances[i, 0]):0.1, 
                       pred_positions[i, 1] - 3*np.sqrt(pred_variances[i, 1]):pred_positions[i, 1] + 3*np.sqrt(pred_variances[i, 1]):0.1]
        pos = np.dstack((x, y))
    
        rv = multivariate_normal([pred_positions[i, 0], pred_positions[i, 1]], [[pred_variances[i, 0], 0], [0, pred_variances[i, 1]]])
        ax.contour(x, y, rv.pdf(pos), cmap="RdYlGn", alpha=0.1)

    # Visualize uncertainty as ellipses
    #for i in range(len(pred_positions)):
    #    ellipse = Ellipse(xy=pred_positions[i], 
    #                  width=6*np.sqrt(pred_variances[i, 0]), 
    #                  height=6*np.sqrt(pred_variances[i, 1]),
    #                  angle=0,  # Assuming no correlation between x and y uncertainties
    #                  facecolor='orange', alpha=0.3)
    #    ax.add_patch(ellipse)

    # Visualize uncertainty as heatmaps
    #for i in range(len(pred_positions)):
    #    x, y = np.mgrid[pred_positions[i, 0] - 3*np.sqrt(pred_variances[i, 0]):pred_positions[i, 0] + 3*np.sqrt(pred_variances[i, 0]):0.1, 
    #                    pred_positions[i, 1] - 3*np.sqrt(pred_variances[i, 1]):pred_positions[i, 1] + 3*np.sqrt(pred_variances[i, 1]):0.1]
    #    pos = np.dstack((x, y))

    #    rv = multivariate_normal([pred_positions[i, 0], pred_positions[i, 1]], [[pred_variances[i, 0], 0], [0, pred_variances[i, 1]]])
    #    ax.imshow(rv.pdf(pos), extent=[x.min(), x.max(), y.min(), y.max()], cmap="Reds", alpha=0.5, origin='lower')
    
    ax.legend()
    ax.set_aspect('equal')

def plot_3d_distributions(predictions, past_positions, future_positions, all_graph_bounds, condition):
    num_samples = len(predictions)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    colors = ['viridis', 'plasma', 'inferno']
    
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        scaler = GraphBoundsScaler(all_graph_bounds[i])
        
        future_pos = np.array([scaler.restore_position(x, y) for x, y in future_positions[i]])
        pred_pos = np.array([scaler.restore_mean(x, y) for x, y in predictions[i]['position_mean']])
        pred_var = np.array([scaler.restore_variance(x, y) for x, y in predictions[i]['position_var']])
        
        ax.scatter(future_pos[:, 0], future_pos[:, 1], np.zeros_like(future_pos[:, 0]), c='g', marker='o', label='Ground Truth')
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], np.zeros_like(pred_pos[:, 0]), c='r', marker='x', label='Predicted Mean')
        
        padding = 50
        x = np.linspace(min(pred_pos[:, 0]) - padding, max(pred_pos[:, 0]) + padding, 100)
        y = np.linspace(min(pred_pos[:, 1]) - padding, max(pred_pos[:, 1]) + padding, 100)
        X, Y = np.meshgrid(x, y)
        
        for t in range(len(pred_pos)):
            mean = pred_pos[t]
            cov = np.diag(pred_var[t])
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
            Z = Z * 1000
            ax.plot_surface(X, Y, Z, cmap=colors[t], alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.set_zlim(0, 1)
        ax.legend()
        ax.set_title(f'Sequence {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"predictions/sequence_{condition}.png")
    plt.close()

def plot_pos_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, condition):
    num_samples = len(predictions)
    num_timesteps = num_samples + 2
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))

    colors = ['viridis', 'plasma', 'inferno']

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(rows, cols, timestep + 1, projection='3d')
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + 20)]

        future_pos_for_timestep = []
        for seq_idx in relevant_sequences:
            scaler = GraphBoundsScaler(all_graph_bounds[seq_idx])
            future_pos = np.array([scaler.restore_position(x, y) for x, y in future_positions[seq_idx]])
            future_pos_for_timestep.append(future_pos[timestep - seq_idx])

        if future_pos_for_timestep:
            mean_future_pos = np.mean(future_pos_for_timestep, axis=0)
            padding = 10
            x = np.linspace(mean_future_pos[0] - padding, mean_future_pos[0] + padding, 100)
            y = np.linspace(mean_future_pos[1] - padding, mean_future_pos[1] + padding, 100)
            X, Y = np.meshgrid(x, y)
        
            for seq_idx in relevant_sequences:
                scaler = GraphBoundsScaler(all_graph_bounds[seq_idx])
                pred_pos = np.array([scaler.restore_mean(x, y) for x, y in predictions[seq_idx]['position_mean']])
                pred_var = np.array([scaler.restore_variance(x, y) for x, y in predictions[seq_idx]['position_var']])

                t = timestep - seq_idx
                if 0 <= t < len(pred_pos):
                    mean = pred_pos[t]
                    cov = np.diag(pred_var[t])

                    original_future_pos = scaler.restore_position(future_positions[seq_idx][t][0], future_positions[seq_idx][t][1])
                    mean_shift = mean_future_pos - original_future_pos
                    mean += mean_shift

                    Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
                    
                    Z = Z / np.sum(Z) 
                    Z = Z * 1000

                    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=(t+1)/20)
                    #ax.scatter(mean[0], mean[1], 0, c='r', marker='x',
                    #           label='Predicted Mean' if seq_idx == relevant_sequences[0] else "")

            ax.scatter(mean_future_pos[0], mean_future_pos[1], 0, c='g', marker='o', label='Mean Ground Truth')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.set_zlim(0, 1)
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.savefig(f"predictions/timestep_position_{condition}.png")
    plt.close()

def plot_vel_distributions_by_timestep(predictions, past_velocities, future_velocities, condition):
    num_samples = len(predictions)
    num_timesteps = num_samples + 2
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))

    colors = ['viridis', 'plasma', 'inferno']

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(rows, cols, timestep + 1, projection='3d')
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + 20)]

        future_vel_for_timestep = []
        for seq_idx in relevant_sequences:
            future_vel = np.array([[x, y] for x, y in future_velocities[seq_idx]])
            future_vel_for_timestep.append(future_vel[timestep - seq_idx])

        if future_vel_for_timestep:
            mean_future_vel = np.mean(future_vel_for_timestep, axis=0)
            
            padding = 1
            x = np.linspace(mean_future_vel[0] - padding, mean_future_vel[0] + padding, 100)
            y = np.linspace(mean_future_vel[1] - padding, mean_future_vel[1] + padding, 100)

            #x = np.linspace(0, 10, 100)
            #y = np.linspace(0, 10, 100)
            
            X, Y = np.meshgrid(x, y)
        
            for seq_idx in relevant_sequences:
                pred_mean = np.array([[x, y] for x, y in predictions[seq_idx]['velocity_mean']])
                pred_var = np.array([[x, y] for x, y in predictions[seq_idx]['velocity_var']])

                t = timestep - seq_idx
                if 0 <= t < len(pred_mean):
                    mean = pred_mean[t]
                    cov = np.diag(pred_var[t])

                    original_future_vel = future_velocities[seq_idx][t]
                    mean_shift = mean_future_vel - original_future_vel
                    mean += mean_shift

                    Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
                    Z = Z / np.sum(Z)
                    Z = Z * 100
                    ax.plot_surface(X, Y, Z, cmap=colors[seq_idx % 3], alpha=0.3)
                    ax.scatter(mean[0], mean[1], 0, c='r', marker='x',
                               label='Predicted Mean' if seq_idx == relevant_sequences[0] else "")

            ax.scatter(mean_future_vel[0], mean_future_vel[1], 0, c='g', marker='o', label='Mean Ground Truth')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability')
        ax.set_zlim(0, 0.6)
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.savefig(f"predictions/timestep_velocity_{condition}.png")
    plt.close()

def plot_steer_distributions_by_timestep(predictions, past_steering, future_steering, condition):
    num_samples = len(predictions)
    num_timesteps = num_samples + 29
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    #print(f'length of output: {len(future_steering)}')

    colors = []
    for i in range(20):
        hex_color = '#%02x0000' % (10 + 3 * i)  # Generates hex codes from #100000 to #680000
        colors.append(hex_color)

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(rows, cols, timestep + 1)
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + 30)]
        #print(f'Relevent sequences: {relevant_sequences}')
        future_ste_for_timestep = []
        for seq_idx in relevant_sequences:
            future_ste = np.array([s for s in future_steering[seq_idx]])
            
            # Rescaling steer values down to -0.5 to 0.5
            #future_ste = -0.5 + future_ste / 10
            
            future_ste_for_timestep.append(future_ste[timestep - seq_idx])
        

        if future_ste_for_timestep:
            mean_future_ste = np.mean(future_ste_for_timestep, axis=0)

            #X = np.linspace(-0.5, 0.5, 100)
            X = np.linspace(0, 10, 100)

            for seq_idx in relevant_sequences:
                pred_mean = predictions[seq_idx]['steering_mean']
                pred_var = predictions[seq_idx]['steering_var']

                # Rescaling steer values down to -0.5 to 0.5
                #pred_mean = -0.5 + pred_mean / 10
                #pred_var = pred_var * (0.1)**2

                t = timestep - seq_idx
                if 0 <= t < len(pred_mean):
                    mean = pred_mean[t]
                    std_dev = np.sqrt(pred_var[t])
                    
                    original_future_ste = future_steering[seq_idx][t]

                    # Rescaling
                    #original_future_ste = -0.5 + original_future_ste / 10

                    mean_shift = mean_future_ste - original_future_ste
                    mean += mean_shift

                    Y = norm.pdf(X, mean, std_dev)

                    # Normalize the PDF
                    if np.sum(Y):
                        Y = Y / np.sum(Y) 
                    else:
                        Y = None
                    #Y = Y / 10
                    if Y is not None:
                        #ax.plot(X, Y, c = 'red', alpha=(t+1)/30)
                        ax.plot(X, Y, c = 'red', alpha=(30-t)/30)
                    
                    #ax.vlines(mean, ymin=0, ymax=1, colors=colors[t], linestyles='--', label='Predicted Sequence ' + str(seq_idx+1))
                    
            ax.vlines(mean_future_ste, ymin=0, ymax=0.1, colors='blue', linestyles='-', label='Ground Truth')

        ax.set_xlabel('Steering')
        ax.set_ylabel('Probability')
        #ax.set_xlim(-0.5, 0.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.savefig(f"predictions/steer/{condition}.png")
    plt.close()

def plot_acceleration_distributions_by_timestep(predictions, past_acceleration, future_acceleration, condition):
    num_samples = len(predictions)
    num_timesteps = num_samples + 29
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig = plt.figure(figsize=(5*cols, 5*rows))

    colors = []
    for i in range(20):
        hex_color = '#%02x0000' % (10 + 3 * i)  # Generates hex codes from #100000 to #680000
        colors.append(hex_color)

    for timestep in range(num_timesteps):
        ax = fig.add_subplot(rows, cols, timestep + 1)
        relevant_sequences = [i for i in range(num_samples) if timestep in range(i, i + 30)]

        future_ste_for_timestep = []
        for seq_idx in relevant_sequences:
            future_ste = np.array([s for s in future_acceleration[seq_idx]])
            
            # Rescaling steer values down to -1 to 1
            #future_ste = (future_ste / 5) - 1
            
            future_ste_for_timestep.append(future_ste[timestep - seq_idx])
        

        if future_ste_for_timestep:
            mean_future_ste = np.mean(future_ste_for_timestep, axis=0)

            X = np.linspace(0, 10, 100)

            for seq_idx in relevant_sequences:
                pred_mean = predictions[seq_idx]['acceleration_mean']
                pred_var = predictions[seq_idx]['acceleration_var']

                #print(f'Acceleration Variance: {pred_var}')
                
                # Rescaling acceleration values down to -1 to 1
                #pred_mean = (pred_mean / 5) - 1
                #pred_var = pred_var * (0.5)**2

                t = timestep - seq_idx
                if 0 <= t < len(pred_mean):
                    mean = pred_mean[t]
                    std_dev = np.sqrt(pred_var[t])
                    
                    original_future_ste = future_acceleration[seq_idx][t]

                    # Rescaling
                    #original_future_ste = -1 + original_future_ste / 5

                    mean_shift = mean_future_ste - original_future_ste
                    mean += mean_shift

                    Y = norm.pdf(X, mean, std_dev)

                    # Normalize the PDF
                    Y = Y / np.sum(Y) 

                    #ax.plot(X, Y, c = 'red', alpha=(t+1)/30)
                    ax.plot(X, Y, c = 'red', alpha=(30-t)/30)

                    #ax.vlines(mean, ymin=0, ymax=1, colors=colors[t], linestyles='--', label='Predicted Sequence ' + str(seq_idx+1))
                    
            ax.vlines(mean_future_ste, ymin=0, ymax=0.1, colors='blue', linestyles='-', label='Ground Truth')

        ax.set_xlabel('acceleration')
        ax.set_ylabel('Probability')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f'Timestep {timestep + 1}')

    plt.tight_layout()
    plt.savefig(f"predictions/acceleration/{condition}.png")
    plt.close()


def visualize_predictions(dataset, scaling_factor, predictions, sampled_indices, condition):
    num_samples = len(predictions)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    #fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    #fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    #fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=True, sharey=True)

    fig = plt.figure(figsize=(5*cols, 5*rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    fig.suptitle(f"Trajectory Predictions - {condition}", fontsize=16)

    for i, (pred, idx) in enumerate(zip(predictions, sampled_indices)):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        #ax = axes[i // cols, i % cols] if num_samples > 1 else axes
        sequence = dataset.data[idx]
        plot_graph_and_trajectories(sequence, scaling_factor, pred, ax)
        ax.set_title(f"Sample {i+1}")
        print(f"Plotted Sample {i+1}")

    # Hide any unused subplots
    #for i in range(num_samples, rows * cols):
    #    axes.flatten()[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"predictions/trajectory_prediction_{condition}.png")
    plt.close()