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
    def __init__(self, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len):
        super(GraphTrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # Graph Convolutional layers
        self.gc1 = GraphConvolution(4, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, hidden_size)
        
        # Attention mechanism for graph features
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # LSTM layers
        self.lstm_position = nn.LSTM(input_sizes['position'] + hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_velocity = nn.LSTM(input_sizes['velocity'] + hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_steering = nn.LSTM(input_sizes['steering'] + hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_object = nn.LSTM(input_sizes['object_in_path'] + hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_traffic = nn.LSTM(input_sizes['traffic_light_detected'] + hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layers for prediction
        self.fc_position = nn.Linear(hidden_size, 2 * output_seq_len)
        self.fc_velocity = nn.Linear(hidden_size, 2 * output_seq_len)
        self.fc_steering = nn.Linear(hidden_size, output_seq_len)
        self.fc_object = nn.Linear(hidden_size, output_seq_len)
        self.fc_traffic = nn.Linear(hidden_size, output_seq_len)
        
    def forward(self, x, graph):
        # Process graph features
        node_features, adj_matrix = graph['node_features'], graph['adj_matrix']
        batch_size = x['position'].size(0)

        # Process each graph in the batch separately
        graph_features_list = []
        for i in range(batch_size):
            # Extract features and adjacency matrix for a single graph
            single_graph_features = node_features[i]
            single_adj_matrix = adj_matrix[i]
            
            # Apply Graph Convolutional layers
            single_graph_features = F.relu(self.gc1(single_graph_features, single_adj_matrix))
            single_graph_features = F.relu(self.gc2(single_graph_features, single_adj_matrix))
            
            graph_features_list.append(single_graph_features)

        # Stack the processed graph features
        graph_features = torch.stack(graph_features_list)

        # Reshape graph features for attention mechanism
        graph_features = graph_features.permute(1, 0, 2)

        # Apply attention mechanism
        graph_features, _ = self.attention(graph_features, graph_features, graph_features)

        # Average pooling over nodes
        graph_features = graph_features.mean(dim=0)

        # Repeat graph features for each timestep
        graph_features = graph_features.unsqueeze(1).repeat(1, x['position'].size(1), 1)
        
        # Concatenate graph features with input features
        position_input = torch.cat((x['position'], graph_features), dim=-1)
        velocity_input = torch.cat((x['velocity'], graph_features), dim=-1)
        steering_input = torch.cat((x['steering'], graph_features), dim=-1)
        object_input = torch.cat((x['object_in_path'], graph_features), dim=-1)
        traffic_input = torch.cat((x['traffic_light_detected'], graph_features), dim=-1)
        
        # Process with LSTM layers
        position_out, _ = self.lstm_position(position_input)
        velocity_out, _ = self.lstm_velocity(velocity_input)
        steering_out, _ = self.lstm_steering(steering_input)
        object_out, _ = self.lstm_object(object_input)
        traffic_out, _ = self.lstm_traffic(traffic_input)
        
        # Predict future trajectory
        position_pred = self.fc_position(position_out[:, -1]).view(-1, self.output_seq_len, 2)
        velocity_pred = self.fc_velocity(velocity_out[:, -1]).view(-1, self.output_seq_len, 2)
        steering_pred = self.fc_steering(steering_out[:, -1]).unsqueeze(-1)
        object_pred = torch.sigmoid(self.fc_object(object_out[:, -1])).unsqueeze(-1)
        traffic_pred = torch.sigmoid(self.fc_traffic(traffic_out[:, -1])).unsqueeze(-1)
        
        return {
            'position': position_pred,
            'velocity': velocity_pred,
            'steering': steering_pred,
            'object_in_path': object_pred,
            'traffic_light_detected': traffic_pred
        }



class TrajectoryLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, input_seq_len, output_seq_len):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # Separate LSTMs for x and y components of position and velocity
        self.lstm_position_x = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.lstm_position_y = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.lstm_velocity_x = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.lstm_velocity_y = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.lstm_steering = nn.LSTM(input_sizes['steering'], hidden_size, num_layers, batch_first=True)
        self.lstm_object = nn.LSTM(input_sizes['object_in_path'], hidden_size, num_layers, batch_first=True)
        self.lstm_traffic = nn.LSTM(input_sizes['traffic_light_detected'], hidden_size, num_layers, batch_first=True)
        
        # Separate fully connected layers for x and y components
        self.fc_position_x = nn.Linear(hidden_size, output_seq_len)
        self.fc_position_y = nn.Linear(hidden_size, output_seq_len)
        self.fc_velocity_x = nn.Linear(hidden_size, output_seq_len)
        self.fc_velocity_y = nn.Linear(hidden_size, output_seq_len)
        self.fc_steering = nn.Linear(hidden_size, output_seq_len)
        self.fc_object = nn.Linear(hidden_size, output_seq_len)
        self.fc_traffic = nn.Linear(hidden_size, output_seq_len)
        
    def forward(self, x):
        # Split position and velocity into x and y components
        position_x = x['position'][:, :, 0].unsqueeze(-1)
        position_y = x['position'][:, :, 1].unsqueeze(-1)
        velocity_x = x['velocity'][:, :, 0].unsqueeze(-1)
        velocity_y = x['velocity'][:, :, 1].unsqueeze(-1)
        
        # Process each component separately
        position_x_out, _ = self.lstm_position_x(position_x)
        position_y_out, _ = self.lstm_position_y(position_y)
        velocity_x_out, _ = self.lstm_velocity_x(velocity_x)
        velocity_y_out, _ = self.lstm_velocity_y(velocity_y)
        steering_out, _ = self.lstm_steering(x['steering'])
        object_out, _ = self.lstm_object(x['object_in_path'])
        traffic_out, _ = self.lstm_traffic(x['traffic_light_detected'])
        
        # Use only the last hidden state for prediction
        position_x_pred = self.fc_position_x(position_x_out[:, -1]).unsqueeze(-1)
        position_y_pred = self.fc_position_y(position_y_out[:, -1]).unsqueeze(-1)
        velocity_x_pred = self.fc_velocity_x(velocity_x_out[:, -1]).unsqueeze(-1)
        velocity_y_pred = self.fc_velocity_y(velocity_y_out[:, -1]).unsqueeze(-1)
        steering_pred = self.fc_steering(steering_out[:, -1]).unsqueeze(-1)
        object_pred = torch.sigmoid(self.fc_object(object_out[:, -1])).unsqueeze(-1)
        traffic_pred = torch.sigmoid(self.fc_traffic(traffic_out[:, -1])).unsqueeze(-1)
        
        # Combine x and y components
        position_pred = torch.cat((position_x_pred, position_y_pred), dim=-1)
        velocity_pred = torch.cat((velocity_x_pred, velocity_y_pred), dim=-1)
        
        return {
            'position': position_pred,
            'velocity': velocity_pred,
            'steering': steering_pred,
            'object_in_path': object_pred,
            'traffic_light_detected': traffic_pred
        }