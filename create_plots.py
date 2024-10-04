from model_config import CONFIG
from Prediction_Model.model_utils import load_model, make_predictions
from Prediction_Model.TrajectoryDataset import TrajectoryDataset 
from Visualization.visualizer import *

def main():
    for condition in CONFIG['conditions']:
        # Update data folder for current condition
        CONFIG['data_folder'] = f"Test_Dataset/Sequence_Dataset/{condition}"
        
        # Data loading
        dataset = TrajectoryDataset(
            CONFIG['data_folder'],
            position_scaling_factor=CONFIG['position_scaling_factor'],
            velocity_scaling_factor=100,
            steering_scaling_factor=100
        )

        # Load the model
        model = load_model(CONFIG)

        # Make predictions
        predictions, sampled_sequences = make_predictions(model, dataset, CONFIG)

        #CONFIG['num_samples'] = len(predictions)

        # Extract past and future positions
        past_positions = []
        future_positions = []
        past_velocities = []
        future_velocities = []
        past_steering = []
        future_steering = []
        all_graph_bounds = []
        for i in range(CONFIG['sample_start_index'], CONFIG['sample_start_index'] + CONFIG['num_samples']):
            past, future, graph, graph_bounds = dataset[i]
            
            past_positions.append(past['position'].numpy())
            future_positions.append(future['position'].numpy())
            
            past_velocities.append(past['velocity'].numpy())
            future_velocities.append(future['velocity'].numpy())
            
            past_steering.append(past['steering'].numpy())
            future_steering.append(future['steering'].numpy())
            
            all_graph_bounds.append(graph_bounds)

        # Visualizations
        visualize_predictions(dataset, CONFIG['position_scaling_factor'], predictions, sampled_sequences, condition)
        print(f"{condition} trajectory viz. done!")

        plot_3d_distributions(predictions, past_positions, future_positions, all_graph_bounds, condition)
        print(f"{condition} sequence distribution viz. done!")

        plot_pos_distributions_by_timestep(predictions, past_positions, future_positions, all_graph_bounds, condition)
        print(f"{condition} time step position distribution viz. done!")

        plot_vel_distributions_by_timestep(predictions, past_velocities, future_velocities, condition)
        print(f"{condition} time step velocity distribution viz. done!")

        plot_steer_distributions_by_timestep(predictions, past_steering, future_steering, condition)
        print(f"{condition} time step steering distribution viz. done!")

if __name__ == "__main__":
    main()