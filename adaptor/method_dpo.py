import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import numpy as np

class PreferenceDataset(Dataset):
    """Dataset for preference learning with chemical mechanism data"""
    def __init__(self, 
                inputs: torch.Tensor,
                outputs: torch.Tensor,
                uncertainties: torch.Tensor,
                preferences: torch.Tensor):
        """
        Args:
            inputs: Input features [N, input_dim]
            outputs: Target outputs [N, output_dim]
            uncertainties: Input uncertainties [N, input_dim]
            preferences: Binary preference labels [N/2, 2] where each row indicates
                        which of two consecutive samples is preferred (1) over the other (0)
        """
        assert len(inputs) % 2 == 0, "Need even number of samples for pairwise preferences"
        assert len(preferences) == len(inputs) // 2, "Preferences should be pairs"
        
        self.inputs = inputs
        self.outputs = outputs
        self.uncertainties = uncertainties
        self.preferences = preferences
        
    def __len__(self) -> int:
        return len(self.preferences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get pair of samples
        idx_1 = idx * 2
        idx_2 = idx * 2 + 1
        
        return {
            'input_1': self.inputs[idx_1],
            'input_2': self.inputs[idx_2],
            'output_1': self.outputs[idx_1],
            'output_2': self.outputs[idx_2],
            'uncertainty_1': self.uncertainties[idx_1],
            'uncertainty_2': self.uncertainties[idx_2],
            'preference': self.preferences[idx]
        }

class DPOLoss(nn.Module):
    """Direct Preference Optimization loss for chemistry mechanism"""
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self, 
                pred_1: torch.Tensor,
                pred_2: torch.Tensor,
                target_1: torch.Tensor,
                target_2: torch.Tensor,
                preference: torch.Tensor,
                uncertainty_1: torch.Tensor,
                uncertainty_2: torch.Tensor) -> torch.Tensor:
        # Calculate accuracy scores for each prediction
        score_1 = -torch.mean(((pred_1 - target_1) ** 2) / uncertainty_1.mean(dim=1, keepdim=True))
        score_2 = -torch.mean(((pred_2 - target_2) ** 2) / uncertainty_2.mean(dim=1, keepdim=True))
        
        # Calculate logits
        logits = (score_1 - score_2) / self.beta

        # Calculate DPO loss
        loss = F.binary_cross_entropy_with_logits(logits, preference.float())
        
        return loss

class ChemistryDPOTuner:
    """Fine-tuner using Direct Preference Optimization"""
    def __init__(self,
                learning_rate: float = 1e-4,
                beta: float = 0.1,
                batch_size: int = 32):
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_size = batch_size
        
    def create_preference_data(self,
                            inputs: torch.Tensor,
                            outputs: torch.Tensor,
                            uncertainties: torch.Tensor,
                            num_pairs: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Create preference pairs based on prediction accuracy and uncertainties
        """
        if num_pairs is None:
            num_pairs = len(inputs) // 2
            
        # Randomly sample pairs
        indices = torch.randperm(len(inputs))
        paired_indices = indices[:num_pairs * 2].reshape(-1, 2)
        
        # Create paired data
        paired_inputs = inputs[paired_indices]
        paired_outputs = outputs[paired_indices]
        paired_uncertainties = uncertainties[paired_indices]
        
        # Calculate preferences based on data quality (lower uncertainty = better quality)
        # In the absence of a reference model, we use uncertainty as a proxy for data quality
        with torch.no_grad():
            # Calculate average uncertainty per sample (lower is better)
            quality_scores = uncertainties.mean(dim=1)
            # Prefer samples with lower uncertainty
            preferences = (
                quality_scores[paired_indices[:, 0]] < quality_scores[paired_indices[:, 1]]
            ).float()
        
        return (
            paired_inputs.reshape(-1, inputs.shape[1]),
            paired_outputs.reshape(-1, outputs.shape[1]),
            paired_uncertainties.reshape(-1, uncertainties.shape[1]),
            preferences
        )
    
    def fine_tune(self,
                model: nn.Module,
                inputs: torch.Tensor,
                outputs: torch.Tensor,
                uncertainties: torch.Tensor,
                num_epochs: int = 100,
                num_pairs: Optional[int] = None) -> nn.Module:
        """
        Fine-tune model using DPO
        """
        # Create preference dataset
        paired_inputs, paired_outputs, paired_uncertainties, preferences = \
            self.create_preference_data(inputs, outputs, uncertainties, num_pairs)
        
        dataset = PreferenceDataset(
            paired_inputs, paired_outputs, paired_uncertainties, preferences
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = DPOLoss(beta=self.beta)
        
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Get predictions for both samples in pairs
                pred_1 = model(batch['input_1'])
                pred_2 = model(batch['input_2'])
                
                # Calculate DPO loss
                loss = criterion(
                    pred_1, pred_2,
                    batch['output_1'], batch['output_2'],
                    batch['preference'],
                    batch['uncertainty_1'], batch['uncertainty_2']
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                        f"Avg Loss: {avg_loss:.6f}")
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model

if __name__ == '__main__':
    # Example usage
    def fine_tune_with_dpo(model: nn.Module,
                          inputs: torch.Tensor,
                          outputs: torch.Tensor,
                          uncertainties: torch.Tensor,
                          **kwargs) -> nn.Module:
        """
        Convenience function for DPO fine-tuning
        """
        tuner = ChemistryDPOTuner(**kwargs)
        return tuner.fine_tune(model, inputs, outputs, uncertainties)
    
    from .base_model import DemoModel
    pertrained_model = DemoModel(40, 20, 20)
    
    obs = torch.randn(100, 40)
    unc = torch.randn(100, 40)
    tgt = torch.randn(100, 20)
    
    fine_tuned_model = fine_tune_with_dpo(
    model=pertrained_model,
    inputs=obs,
    outputs=tgt,
    uncertainties=unc,
    )