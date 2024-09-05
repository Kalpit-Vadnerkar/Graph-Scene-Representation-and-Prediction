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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.device = device

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        
        for past, future, graph in tqdm(self.train_loader, desc="Training"):
            past = {k: v.to(self.device) for k, v in past.items()}
            future = {k: v.to(self.device) for k, v in future.items()}
            graph = {k: v.to(self.device) for k, v in graph.items()}
            
            self.optimizer.zero_grad()
            predictions = self.model(past, graph)

            # Print shapes for debugging
            #print("Prediction shapes:")
            #for key, value in predictions.items():
            #    print(f"{key}: {value.shape}")
            #print("Target shapes:")
            #for key, value in future.items():
            #    print(f"{key}: {value.shape}")
            
            loss = self.criterion(predictions, future)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        # Update learning rate
        self.scheduler.step(train_loss)
        
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, future, graph in self.val_loader:
                past = {k: v.to(self.device) for k, v in past.items()}
                future = {k: v.to(self.device) for k, v in future.items()}
                graph = {k: v.to(self.device) for k, v in graph.items()}
                
                predictions = self.model(past, graph)
                
                loss = self.criterion(predictions, future)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.model