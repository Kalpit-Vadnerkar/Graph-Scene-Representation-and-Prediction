import torch

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()
        
    def forward(self, continuous_pred, binary_pred, continuous_target, binary_target):
        mse = self.mse_loss(continuous_pred, continuous_target)
        bce = self.bce_loss(binary_pred, binary_target)
        return mse + 0*bce