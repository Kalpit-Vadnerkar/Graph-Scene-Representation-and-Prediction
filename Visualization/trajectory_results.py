import numpy as np

from Visualization.Rescaler import GraphBoundsScaler

def position_result_metrics(dataset, scaling_factor, predictions, condition, threshold=20.0):
    """
    Calculate trajectory forecasting metrics:
    - minFDE (Final Displacement Error)
    - minADE (Average Displacement Error)
    - MR (Miss Rate)
    
    Args:
        dataset: Dataset containing ground truth trajectories
        scaling_factor: Scale factor for position coordinates
        predictions: Model predictions
        condition: Filtering condition
        threshold: Distance threshold for miss rate calculation (default: 2.0)
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    fde_values = []
    ade_values = []
    miss_count = 0
    total_count = 0
    
    for i, pred in enumerate(predictions):
        sequence = dataset.data[i]
            
        scaler = GraphBoundsScaler(sequence['graph_bounds'])
        
        # Get ground truth future positions
        future_positions = np.array([
            scaler.restore_position(
                step['position'][0] * scaling_factor,
                step['position'][1] * scaling_factor
            ) for step in sequence['future']
        ])
        
        # Get predicted positions
        pred_positions = np.array([
            scaler.restore_mean(x, y) 
            for x, y in pred['position_mean']
        ])
        
        # Calculate L2 distances between prediction and ground truth
        distances = np.sqrt(np.sum((pred_positions - future_positions) ** 2, axis=1))
        
        # Calculate FDE (Final Displacement Error)
        fde = distances[-1]  # Error at final timestep
        fde_values.append(fde)
        
        # Calculate ADE (Average Displacement Error)
        ade = np.mean(distances)  # Average error across all timesteps
        ade_values.append(ade)
        
        # Update miss rate calculation
        total_count += 1
        if fde > threshold:
            miss_count += 1
    
    # Compute final metrics
    metrics = {
        'minFDE': np.mean(fde_values) if fde_values else 0.0,
        'minADE': np.mean(ade_values) if ade_values else 0.0,
        'MR': (miss_count / total_count) if total_count > 0 else 0.0
    }
    
    # Print metrics
    print(f"Trajectory Forecasting Metrics:")
    print(f"minFDE: {metrics['minFDE']:.4f}")
    print(f"minADE: {metrics['minADE']:.4f}")
    print(f"Miss Rate: {metrics['MR']:.4f}")
    
    return metrics