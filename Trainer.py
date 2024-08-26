import torch
from tqdm import tqdm
from LossFunctions import CombinedLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        
        for past, future in tqdm(self.train_loader, desc="Training"):
            past, future = past.to(self.device), future.to(self.device)
            
            self.optimizer.zero_grad()
            continuous_pred, binary_pred = self.model(past)
            
            continuous_target = future[:, :, :5]
            binary_target = future[:, :, 5:]
            
            loss = self.criterion(continuous_pred, binary_pred, continuous_target, binary_target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, future in self.val_loader:
                past, future = past.to(self.device), future.to(self.device)
                continuous_pred, binary_pred = self.model(past)
                
                continuous_target = future[:, :, :5]
                binary_target = future[:, :, 5:]
                
                loss = self.criterion(continuous_pred, binary_pred, continuous_target, binary_target)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.model