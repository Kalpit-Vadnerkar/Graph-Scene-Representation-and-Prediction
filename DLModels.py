import torch

class TrajectoryLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_continuous = torch.nn.Linear(hidden_size, 5)  # position (2), velocity (2), steering (1)
        self.fc_binary = torch.nn.Linear(hidden_size, 2)  # object_in_path, traffic_light_detected
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        continuous_pred = self.fc_continuous(out)
        binary_pred = torch.sigmoid(self.fc_binary(out))
        
        return continuous_pred, binary_pred