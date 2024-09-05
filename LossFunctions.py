import torch
import torch.nn as nn
import math

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.log_2pi = torch.log(torch.tensor(2 * math.pi))
        
    def gaussian_nll_loss(self, mean, log_var, target):
        var = torch.exp(log_var.clamp(min=-20, max=20))
        return 0.5 * (log_var + ((target - mean)**2 / var) + self.log_2pi).sum(dim=-1).mean()
    
    def single_gaussian_nll_loss(self, mean, log_var, target):
        var = torch.exp(log_var.clamp(min=-20, max=20))
        return (0.5 * (log_var + ((target - mean)**2 / var) + self.log_2pi)).mean()
    
    def forward(self, pred, target):
        losses = {}
        batch_size = next(iter(target.values())).shape[0]
        
        for key in pred.keys():
            if key in ['object_in_path', 'traffic_light_detected']:
                losses[key] = self.bce_loss(pred[key].clamp(min=1e-7, max=1-1e-7), target[key]) / batch_size
            elif key == 'steering':
                mean, log_var = pred[key]
                losses[key] = self.single_gaussian_nll_loss(mean, log_var, target[key])
            else:
                mean, log_var = pred[key]
                losses[key] = self.gaussian_nll_loss(mean, log_var, target[key])
        
        total_loss = sum(losses.values())
        
        # Print individual losses
        #for key, value in losses.items():
            #print(f"{key} loss: {value.item():.4f}")
        #print(f"Total loss: {total_loss.item():.4f}")
        
        return total_loss