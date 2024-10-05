import torch
import torch.nn as nn
import logging

class CombinedLoss(nn.Module):
    def __init__(self, min_var=1e-2, epsilon=1e-2):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.min_var = min_var
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        loss = 0
        for key in ['position', 'velocity', 'steering', 'acceleration', 'object_distance']:
            mean_key = f'{key}_mean'
            var_key = f'{key}_var'
            if mean_key in pred and var_key in pred:
                # Clip variance to minimum value
                variance = torch.clamp(pred[var_key], min=self.min_var)
                
                # Calculate GNLL loss
                gnll_loss = 0.5 * torch.mean(torch.log(variance + self.epsilon) + 
                                             (target[key] - pred[mean_key])**2 / (variance + self.epsilon))
                
                # Add regularization term
                regularization = torch.mean(1 / (variance + self.epsilon))
                gnll_loss += 0.01 * regularization
                
                loss += gnll_loss
                
                #if gnll_loss < 0:
                    #logging.warning(f"Negative GNLL loss encountered for {key}: {gnll_loss.item()}")
            elif key in pred:
                loss += self.mse_loss(pred[key], target[key])
        
        for key in ['traffic_light_detected']:
            if key in pred:
                loss += self.bce_loss(pred[key], target[key])
        
        #if loss < 0:
            #logging.warning(f"Total loss is negative: {loss.item()}")
        
        return loss