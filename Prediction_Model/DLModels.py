import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output

class GraphTrajectoryLSTM(nn.Module):
    def __init__(self, config):
        super(GraphTrajectoryLSTM, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.input_seq_len = config['input_seq_len']
        self.output_seq_len = config['output_seq_len']
        self.dropout_rate = config['dropout_rate']
        self.graph_sizes = config['graph_sizes']
        self.feature_sizes = config['feature_sizes']
        
        self.feature_types = list(config['feature_sizes'].keys())
        
        # Graph Convolutional layers
        self.gc1 = GraphConvolution(self.graph_sizes['node_features'], self.hidden_size)
        self.gc2 = GraphConvolution(self.hidden_size, self.hidden_size)
        
        # Dropout layer after Graph Convolution
        self.dropout_gc = nn.Dropout(self.dropout_rate)
        
        # Attention mechanism for graph features
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=4)

        # Hidden layer for temporal features
        self.total_feature_size = sum(self.feature_sizes.values())
        self.temporal_hidden = nn.Linear(self.total_feature_size, self.hidden_size)
        
        # Calculate the input size for LSTM layers
        #self.lstm_input_size = self.hidden_size + self.total_feature_size
        self.lstm_input_size = self.hidden_size * 2  # temporal features + graph features

        # LSTM layers
        self.position_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.velocity_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.steering_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.acceleration_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.object_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.traffic_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        
        self.dropout_lstm = nn.Dropout(self.dropout_rate)
        
        # Output layers
        self.position_output = nn.Linear(self.hidden_size, 4 * self.output_seq_len)
        self.velocity_output = nn.Linear(self.hidden_size, 4 * self.output_seq_len)
        self.steering_output = nn.Linear(self.hidden_size, 2 * self.output_seq_len)
        self.acceleration_output = nn.Linear(self.hidden_size, 2 * self.output_seq_len)
        self.object_distance_output = nn.Linear(self.hidden_size, self.output_seq_len)
        self.traffic_light_output = nn.Linear(self.hidden_size, self.output_seq_len)

    def forward(self, x, graph):
        node_features, adj_matrix = graph['node_features'], graph['adj_matrix']
        batch_size = x['position'].size(0)

        # Ensure all input tensors have 3 dimensions
        for key in x:
            if x[key].dim() == 2:
                x[key] = x[key].unsqueeze(-1)

        # Process graph features
        graph_features_list = []
        for i in range(batch_size):
            single_graph_features = node_features[i]
            single_adj_matrix = adj_matrix[i]
            
            single_graph_features = F.relu(self.gc1(single_graph_features, single_adj_matrix))
            single_graph_features = self.dropout_gc(single_graph_features)
            single_graph_features = F.relu(self.gc2(single_graph_features, single_adj_matrix))
            single_graph_features = self.dropout_gc(single_graph_features)
            
            graph_features_list.append(single_graph_features)

        graph_features = torch.stack(graph_features_list)
        
        graph_features = graph_features.permute(1, 0, 2)
        graph_features, _ = self.attention(graph_features, graph_features, graph_features)
        graph_features = graph_features.mean(dim=0)
        
        graph_features = graph_features.unsqueeze(1).repeat(1, self.input_seq_len, 1)

        # Process temporal features
        temporal_features = torch.cat([x[f] for f in self.feature_types], dim=2)
        
        temporal_features = self.temporal_hidden(temporal_features)

        # Concatenate temporal and graph features
        combined_features = torch.cat([temporal_features, graph_features], dim=-1)

        # Process combined features through separate LSTMs
        position_output, _ = self.position_lstm(combined_features)
        velocity_output, _ = self.velocity_lstm(combined_features)
        steering_output, _ = self.steering_lstm(combined_features)
        acceleration_output, _ = self.acceleration_lstm(combined_features)
        object_output, _ = self.object_lstm(combined_features)
        traffic_output, _ = self.traffic_lstm(combined_features)
        
        # Apply dropout
        position_output = self.dropout_lstm(position_output[:, -1])
        velocity_output = self.dropout_lstm(velocity_output[:, -1])
        steering_output = self.dropout_lstm(steering_output[:, -1])
        acceleration_output = self.dropout_lstm(acceleration_output[:, -1])
        object_output = self.dropout_lstm(object_output[:, -1])
        traffic_output = self.dropout_lstm(traffic_output[:, -1])

        # Process predictions
        position = self.position_output(position_output).view(batch_size, self.output_seq_len, 4)
        velocity = self.velocity_output(velocity_output).view(batch_size, self.output_seq_len, 4)
        steering = self.steering_output(steering_output).view(batch_size, self.output_seq_len, 2)
        acceleration = self.acceleration_output(acceleration_output).view(batch_size, self.output_seq_len, 2)
        object_distance = self.object_distance_output(object_output).view(batch_size, self.output_seq_len)
        traffic_light = self.traffic_light_output(traffic_output).view(batch_size, self.output_seq_len)

        # Split the outputs into mean and variance
        predictions = {
            'position_mean': position[..., :2],
            'position_var': F.softplus(position[..., 2:]),
            'velocity_mean': velocity[..., :2],
            'velocity_var': F.softplus(velocity[..., 2:]),
            'steering_mean': steering[..., 0],
            'steering_var': F.softplus(steering[..., 1]),
            'acceleration_mean': acceleration[..., 0],
            'acceleration_var': F.softplus(acceleration[..., 1]),
            'object_distance': torch.sigmoid(object_distance),
            'traffic_light_detected': torch.sigmoid(traffic_light)
        }

        return predictions