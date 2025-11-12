import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Optional, Union, Tuple

from .base_model import DemoModel
from .method_ft import TraditionalFineTuner
from .method_rl import RLFineTuner
from .method_dpo import ChemistryDPOTuner

class ChemistryMechanismAdapter:
    """Main class combining all fine-tuning approaches"""
    def __init__(self,
                 method: str = "traditional",
                 **kwargs):
        """
        Initialize solver with specified method
        
        Args:
            method: str
            **kwargs: Additional arguments for the chosen method
        """
        self.method = method.lower()
        if self.method == "finetune":
            self.tuner = TraditionalFineTuner(**kwargs)
        elif self.method == "rl":
            self.tuner = RLFineTuner(**kwargs)
        elif self.method == "dpo":
            self.tuner = ChemistryDPOTuner(**kwargs)
        else:
            raise ValueError("Method has not been implemented")
    
    def fine_tune(self,
                pretrained_model: nn.Module,
                observed_inputs: torch.Tensor,
                observed_outputs: torch.Tensor,
                input_uncertainty: torch.Tensor,
                **kwargs) -> nn.Module:
        """
        Fine-tune the pretrained model using the selected method
        
        Args:
            pretrained_model: Initial PyTorch model
            observed_data: Tensor of observed values
            uncertainty_data: Tensor of uncertainty values
            **kwargs: Additional arguments for specific tuning method
        
        Returns:
            Fine-tuned PyTorch model
        """
        # Validate inputs
        if not isinstance(pretrained_model, nn.Module):
            raise TypeError("Pretrained_model must be a PyTorch module")
        
        if observed_inputs.shape != input_uncertainty.shape:
            raise ValueError("Input and Uncertainty must have same shape")
        
        # Perform fine-tuning
        finetuned_model = self.tuner.fine_tune(
            pretrained_model,
            observed_inputs,
            observed_outputs,
            input_uncertainty,
            **kwargs
        )
        
        return finetuned_model

if __name__ == '__main__':
    pertrained_model = DemoModel(40, 20, 20)
    
    obs = torch.randn(100, 40)
    unc = torch.randn(100, 40)
    tgt = torch.randn(100, 20)
    """
    solver = ChemistryMechanismAdapter(
        method="finetune",
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    fine_tuned_model = solver.fine_tune(
        pretrained_model=pertrained_model,
        observed_inputs=obs,
        observed_outputs=tgt,
        input_uncertainty=unc,
        num_epochs=1000,
        patience = 100,
    )

    # OR for reinforcement learning
    solver = ChemistryMechanismAdapter(
        method="rl",
        total_timesteps=10000,
        learning_rate=1e-4
    )
    
    fine_tuned_model = solver.fine_tune(
        pretrained_model=pertrained_model,
        observed_inputs=obs,
        observed_outputs=tgt,
        input_uncertainty=unc,
    )
    """
    solver = ChemistryMechanismAdapter(
        method="dpo"
    )
    
    fine_tuned_model = solver.fine_tune(
        pretrained_model=pertrained_model,
        observed_inputs=obs,
        observed_outputs=tgt,
        input_uncertainty=unc,
    )