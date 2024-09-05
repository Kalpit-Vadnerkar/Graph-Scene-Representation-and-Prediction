import torch
from tqdm import tqdm
from LossFunctions import CombinedLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        self.device = device

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        individual_losses = {k: 0.0 for k in ['position', 'velocity', 'steering', 'object_in_path', 'traffic_light_detected']}
        
        for past, future, graph in tqdm(self.train_loader, desc="Training"):
            past = {k: v.to(self.device) for k, v in past.items()}
            future = {k: v.to(self.device) for k, v in future.items()}
            graph = {k: v.to(self.device) for k, v in graph.items()}
            
            self.optimizer.zero_grad()
            predictions = self.model(past, graph)
            
            loss, losses = self.criterion(predictions, future)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            for k, v in losses.items():
                individual_losses[k] += v.item()
        
        return train_loss / len(self.train_loader), {k: v / len(self.train_loader) for k, v in individual_losses.items()}

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        individual_losses = {k: 0.0 for k in ['position', 'velocity', 'steering', 'object_in_path', 'traffic_light_detected']}
        with torch.no_grad():
            for past, future, graph in self.val_loader:
                past = {k: v.to(self.device) for k, v in past.items()}
                future = {k: v.to(self.device) for k, v in future.items()}
                graph = {k: v.to(self.device) for k, v in graph.items()}
                
                predictions = self.model(past, graph)
                
                loss, losses = self.criterion(predictions, future)
                val_loss += loss.item()
                for k, v in losses.items():
                    individual_losses[k] += v.item()
        
        return val_loss / len(self.val_loader), {k: v / len(self.val_loader) for k, v in individual_losses.items()}

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_individual_losses = self.train_epoch()
            val_loss, val_individual_losses = self.validate()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Train Individual Losses:")
            for k, v in train_individual_losses.items():
                print(f"  {k}: {v:.4f}")
            print("Val Individual Losses:")
            for k, v in val_individual_losses.items():
                print(f"  {k}: {v:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
        
        return self.model