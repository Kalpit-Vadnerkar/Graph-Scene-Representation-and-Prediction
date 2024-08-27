import torch
import torch.nn as nn

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