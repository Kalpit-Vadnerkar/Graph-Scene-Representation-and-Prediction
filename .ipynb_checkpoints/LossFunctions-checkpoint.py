import torch
import torch.nn as nn
import math

class CombinedLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.eps = eps
        
    def gaussian_nll_loss(self, mean, log_var, target):
        var = torch.exp(log_var) + self.eps
        loss = 0.5 * (log_var + ((target - mean)**2 / var))
        return loss.sum()
    
    def forward(self, pred, target):
        losses = {}
        batch_size = target['position'].shape[0]
        
        for key in pred.keys():
            if key in ['object_in_path', 'traffic_light_detected']:
                losses[key] = self.bce_loss(pred[key], target[key]) / batch_size
            else:
                mean, log_var = pred[key]
                losses[key] = self.gaussian_nll_loss(mean, log_var, target[key]) / batch_size
        
        total_loss = sum(losses.values())
        
        return total_loss, losses