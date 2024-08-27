import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, pred, target):
        loss = 0
        for key in pred.keys():
            if key in ['object_in_path', 'traffic_light_detected']:
                loss += self.bce_loss(pred[key], target[key])
            else:
                loss += self.mse_loss(pred[key], target[key])
        return loss