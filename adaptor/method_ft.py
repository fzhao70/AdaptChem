import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TraditionalFineTuner:
    def __init__(self, 
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-5,
                ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.uncertainty_threshold = 1e-10
        self.max_uncertainty_weight = 2.0
        self.num_mc_samples = 5

    def custom_uncertainty_loss(self,
                              model: nn.Module,
                              inputs: torch.Tensor,
                              targets: torch.Tensor,
                              uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function that considers input uncertainty through Monte Carlo sampling
        """
        batch_predictions = []
        
        # Multiple forward passes with noisy inputs
        for _ in range(self.num_mc_samples):
            # Add noise based on uncertainty
            noisy_inputs = inputs + torch.randn_like(inputs) * uncertainties
            predictions = model(noisy_inputs)
            batch_predictions.append(predictions)
        
        # Average predictions across MC samples
        mean_predictions = torch.stack(batch_predictions).mean(dim=0)
        
        # Calculate uncertainty-weighted MSE loss
        uncertainty_weights = 1.0 / (1.0 + torch.mean(uncertainties, dim=1))
        mse_loss = (mean_predictions - targets) ** 2
        
        # Apply uncertainty weights to loss
        weighted_loss = mse_loss * uncertainty_weights.unsqueeze(1)
        
        return torch.mean(weighted_loss)
    
    def fine_tune(self,
                model: nn.Module,
                observed_inputs: torch.Tensor,
                observed_outputs: torch.Tensor,
                input_uncertainty: torch.Tensor,
                num_epochs: int = 100,
                batch_size: int = 32,
                patience: int = None,
                user_optimizer: optim.Optimizer = optim.Adam
                ) -> nn.Module:
        """Fine-tune the model using traditional optimization"""
        
        dataset = torch.utils.data.TensorDataset(
            observed_inputs, observed_outputs, input_uncertainty
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        # Use Adam as default, if you need to squeeze out the last drop of performance, use SGD
        optimizer = user_optimizer(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if patience is not None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=patience,
            )
        
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for inputs, targets, uncertainties in dataloader:
                optimizer.zero_grad()
                
                # Calculate loss using Monte Carlo sampling
                loss = self.custom_uncertainty_loss(
                    model, inputs, targets, uncertainties
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches

            # Update learning rate scheduler if it exists
            if patience is not None:
                scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Avg Loss: {avg_loss:.6f}, "
                      f"Learning Rate: {current_lr:.2e}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model