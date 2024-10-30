import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, min_var=1e-2, epsilon=0, reg_weights=None):
        super(CombinedLoss, self).__init__()
        self.min_var = min_var
        self.epsilon = epsilon
        # Default weights: equal importance to all variables
        self.reg_weights = reg_weights if reg_weights is not None else {
            'position': 1.0, 
            'velocity': 1.0, 
            'steering': 1.0, 
            'acceleration': 1.0, 
            'object_distance': 1.0
        }

    def forward(self, pred, target):
        losses = {}
        total_loss = 0

        for key in ['position', 'velocity', 'steering', 'acceleration', 'object_distance']:
            mean_key = f'{key}_mean'
            var_key = f'{key}_var'
            if mean_key in pred and var_key in pred:
                variance = pred[var_key]
                #variance = torch.clamp(pred[var_key], min=self.min_var)

                gnll_loss = 0.5 * (torch.log(variance + self.epsilon) + 
                                   (target[key] - pred[mean_key])**2 / (variance + self.epsilon))
                gnll_loss = gnll_loss.sum(dim=tuple(range(1, gnll_loss.dim()))).mean()

                # Weighted regularization
                regularization = torch.mean(1 / (variance + self.epsilon))
                gnll_loss += 0.01 * self.reg_weights[key] * regularization 

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