import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, min_var=1e-2, epsilon=1e-2):
        super(CombinedLoss, self).__init__()
        self.min_var = min_var
        self.epsilon = epsilon

    def forward(self, pred, target):
        losses = {}
        total_loss = 0

        for key in ['position', 'velocity', 'steering', 'acceleration', 'object_distance']:
            mean_key = f'{key}_mean'
            var_key = f'{key}_var'
            if mean_key in pred and var_key in pred:
                variance = torch.clamp(pred[var_key], min=self.min_var)
                
                gnll_loss = 0.5 * (torch.log(variance + self.epsilon) +
                                   (target[key] - pred[mean_key])**2 / (variance + self.epsilon))
                
                # Sum over all dimensions except batch
                gnll_loss = gnll_loss.sum(dim=tuple(range(1, gnll_loss.dim()))).mean()
                
                regularization = torch.mean(1 / (variance + self.epsilon))
                gnll_loss += 0.01 * regularization
                
                losses[f'{key}_loss'] = gnll_loss.item()
                total_loss += gnll_loss
            elif key in pred:
                mse_loss = torch.mean((pred[key] - target[key])**2)
                losses[f'{key}_loss'] = mse_loss.item()
                total_loss += mse_loss

        for key in ['traffic_light_detected']:
            if key in pred:
                bce_loss = F.binary_cross_entropy(pred[key], target[key], reduction='mean')
                losses[f'{key}_loss'] = bce_loss.item()
                total_loss += bce_loss

        losses['total_loss'] = total_loss.item()
        return losses, total_loss