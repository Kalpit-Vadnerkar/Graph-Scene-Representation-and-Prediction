import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from Prediction_Model.LossFunctions import CombinedLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = CombinedLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        epoch_losses = {}
        
        for past, future, graph in tqdm(self.train_loader, desc="Training"):
            past = {k: v.to(self.device) for k, v in past.items()}
            future = {k: v.to(self.device) for k, v in future.items()}
            graph = {k: v.to(self.device) for k, v in graph.items()}
            
            self.optimizer.zero_grad()
            predictions = self.model(past, graph)
            
            losses, loss = self.criterion(predictions, future)
            loss.backward()
            self.optimizer.step()
            
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value
        
        # Average the losses over the epoch
        epoch_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    def validate(self):
        self.model.eval()
        val_losses = {}
        with torch.no_grad():
            for past, future, graph in self.val_loader:
                past = {k: v.to(self.device) for k, v in past.items()}
                future = {k: v.to(self.device) for k, v in future.items()}
                graph = {k: v.to(self.device) for k, v in graph.items()}
                
                predictions = self.model(past, graph)
                
                losses, _ = self.criterion(predictions, future)
                
                for key, value in losses.items():
                    val_losses[key] = val_losses.get(key, 0) + value
        
        # Average the losses over the validation set
        val_losses = {k: v / len(self.val_loader) for k, v in val_losses.items()}
        
        return val_losses

    def train(self, num_epochs, seed=None):
        for epoch in range(num_epochs):
            train_losses = self.train_epoch()
            val_losses = self.validate()
            
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_losses['total_loss']}")
            print(f"validation Loss: {val_losses['total_loss']}")
            #print("Train Losses:")
            #for key, value in train_losses.items():
            #    print(f"  {key}: {value:.4f}")
            #print("Validation Losses:")
            #for key, value in val_losses.items():
            #    print(f"  {key}: {value:.4f}")
            
            self.scheduler.step(val_losses['total_loss'])
        
        self.plot_losses(seed)
        return self.model

    def plot_losses(self, seed):
        os.makedirs(f'Model_Training_Results/model_seed_{seed}', exist_ok=True)
        
        # Convert train and validation losses to DataFrames
        train_losses_df = pd.DataFrame(self.train_losses)
        val_losses_df   = pd.DataFrame(self.val_losses)
        
        # Save each as CSV
        train_losses_df.to_csv(f'Model_Training_Results/model_seed_{seed}/train_losses.csv', index=False)
        val_losses_df.to_csv(f'Model_Training_Results/model_seed_{seed}/val_losses.csv', index=False)
    
    # Plot total loss
        # Plot total loss
        plt.figure(figsize=(10, 6))
        plt.plot([losses['total_loss'] for losses in self.train_losses], label='Train')
        plt.plot([losses['total_loss'] for losses in self.val_losses], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'Model_Training_Results/model_seed_{seed}/total_loss.png')
        plt.close()

        # Plot individual losses
        loss_keys = [key for key in self.train_losses[0].keys() if key != 'total_loss']
        for key in loss_keys:
            plt.figure(figsize=(10, 6))
            plt.plot([losses[key] for losses in self.train_losses], label='Train')
            plt.plot([losses[key] for losses in self.val_losses], label='Validation')
            plt.title(f'{key} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'Model_Training_Results/model_seed_{seed}/{key}_loss.png')
            plt.close()