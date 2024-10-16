import torch
from Prediction_Model.DLModels import GraphTrajectoryLSTM

def load_model(config):
    model = GraphTrajectoryLSTM(config)
    #model = GraphTrajectoryLSTM(
    #    config['input_sizes'],
    #    config['hidden_size'],
    #    config['num_layers'],
    #    config['input_seq_len'],
    #    config['output_seq_len']
    #)
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

    return all_predictions, sampled_sequences