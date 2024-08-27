import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_position = nn.LSTM(input_sizes['position'], hidden_size, num_layers, batch_first=True)
        self.lstm_velocity = nn.LSTM(input_sizes['velocity'], hidden_size, num_layers, batch_first=True)
        self.lstm_steering = nn.LSTM(input_sizes['steering'], hidden_size, num_layers, batch_first=True)
        self.lstm_object = nn.LSTM(input_sizes['object_in_path'], hidden_size, num_layers, batch_first=True)
        self.lstm_traffic = nn.LSTM(input_sizes['traffic_light_detected'], hidden_size, num_layers, batch_first=True)
        
        self.fc_position = nn.Linear(hidden_size, 2)
        self.fc_velocity = nn.Linear(hidden_size, 2)
        self.fc_steering = nn.Linear(hidden_size, 1)
        self.fc_object = nn.Linear(hidden_size, 1)
        self.fc_traffic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        position_out, _ = self.lstm_position(x['position'])
        velocity_out, _ = self.lstm_velocity(x['velocity'])
        steering_out, _ = self.lstm_steering(x['steering'])
        object_out, _ = self.lstm_object(x['object_in_path'])
        traffic_out, _ = self.lstm_traffic(x['traffic_light_detected'])
        
        position_pred = self.fc_position(position_out[:, -1, :])
        velocity_pred = self.fc_velocity(velocity_out[:, -1, :])
        steering_pred = self.fc_steering(steering_out[:, -1, :])
        object_pred = torch.sigmoid(self.fc_object(object_out[:, -1, :]))
        traffic_pred = torch.sigmoid(self.fc_traffic(traffic_out[:, -1, :]))
        
        return {
            'position': position_pred,
            'velocity': velocity_pred,
            'steering': steering_pred,
            'object_in_path': object_pred,
            'traffic_light_detected': traffic_pred
        }