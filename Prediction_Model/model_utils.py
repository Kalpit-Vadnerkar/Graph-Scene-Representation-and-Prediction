from torchinfo import summary
from typing import Dict, Any 
import torch
from Prediction_Model.DLModels import GraphTrajectoryLSTM, TrajectoryLSTM, GraphAttentionLSTM

def print_model_summary(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: The PyTorch model
        config: Configuration dictionary containing model parameters
    """
    # Calculate input shapes based on config
    batch_size = 1
    #batch_size = config['batch_size']
    seq_len = config['input_seq_len']
    
    # Create sample inputs that match the model's expected input format
    sample_inputs = (
        # First argument: x (temporal features dictionary)
        {
            'position': torch.zeros(batch_size, seq_len, config['feature_sizes']['position']),
            'velocity': torch.zeros(batch_size, seq_len, config['feature_sizes']['velocity']),
            'steering': torch.zeros(batch_size, seq_len, config['feature_sizes']['steering']),
            'acceleration': torch.zeros(batch_size, seq_len, config['feature_sizes']['acceleration']),
            'object_distance': torch.zeros(batch_size, seq_len, config['feature_sizes']['object_distance']),
            'traffic_light_detected': torch.zeros(batch_size, seq_len, config['feature_sizes']['traffic_light_detected'])
        },
        # Second argument: graph (graph features dictionary)
        {
            'node_features': torch.zeros(batch_size, config['graph_sizes']['number_of_nodes'], 
                                       config['graph_sizes']['node_features']),
            'adj_matrix': torch.zeros(batch_size, config['graph_sizes']['number_of_nodes'], 
                                    config['graph_sizes']['number_of_nodes'])
        }
    )

    print("\nModel Summary:")
    print("==============")
    
    # Generate detailed summary with torchinfo using sample inputs
    model_summary = summary(
        model,
        input_data=sample_inputs,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"],
        device=config['device']
    )
    
    # Print additional parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

def load_model(config):
    model = GraphAttentionLSTM(config)
    #model = TrajectoryLSTM(config)
    #model = GraphTrajectoryLSTM(config)
    
    model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    model.to(config['device'])
    model.eval()
    return model

def make_predictions(model, dataset, config):
    model.eval()
    all_predictions = []
    #sampled_sequences = [i + config['sample_start_index'] for i in range(config['num_samples'])]

    sampled_sequences = [i for i in range(len(dataset))]

    with torch.no_grad():
        for idx in sampled_sequences:
            past, future, graph, graph_bounds = dataset[idx]
            
            past = {k: v.unsqueeze(0).to(config['device']) for k, v in past.items()}
            graph = {k: v.unsqueeze(0).to(config['device']) for k, v in graph.items()}
            
            predictions = model(past, graph)
            all_predictions.append({k: v.squeeze().cpu().numpy() for k, v in predictions.items()})

    return all_predictions

def make_limited_predictions(model, dataset, config):
    model.eval()
    all_predictions = []
    sampled_sequences = [i + config['sample_start_index'] for i in range(config['num_samples'])]

    with torch.no_grad():
        for idx in sampled_sequences:
            past, future, graph, graph_bounds = dataset[idx]
            
            past = {k: v.unsqueeze(0).to(config['device']) for k, v in past.items()}
            graph = {k: v.unsqueeze(0).to(config['device']) for k, v in graph.items()}
            
            predictions = model(past, graph)
            #print(predictions)
            all_predictions.append({k: v.squeeze().cpu().numpy() for k, v in predictions.items()})

    return all_predictions, sampled_sequences