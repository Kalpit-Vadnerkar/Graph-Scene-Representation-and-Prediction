import torch
import torch.nn as nn
import math

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')
        
    def gaussian_nll_loss(self, mean, log_std, target):
        std = torch.exp(log_std)
        var = std.pow(2)
        nll = 0.5 * (log_std + ((target - mean).pow(2) / var) + math.log(2 * math.pi))
        return nll.sum()
    
    def forward(self, pred, target):
        loss = 0
        for key in pred.keys():
            if key in ['object_in_path', 'traffic_light_detected']:
                loss += self.bce_loss(pred[key], target[key])
            else:
                mean, log_std = pred[key]
                loss += self.gaussian_nll_loss(mean, log_std, target[key])
        return loss