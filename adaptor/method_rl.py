import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Optional, Tuple, Union

class ChemistryEnv(gym.Env):
    """Custom Environment for chemistry mechanism fine-tuning using RL"""
    def __init__(self, 
                 pretrained_model: nn.Module, 
                 observed_inputs: torch.Tensor,    # Full input data including meteo factors
                 observed_outputs: torch.Tensor,   # Target outputs (smaller dimension)
                 input_uncertainty: torch.Tensor,  # Uncertainty for observed inputs
                 max_steps: int = 100):
        super().__init__()
        
        assert observed_inputs.shape[0] == observed_outputs.shape[0], "Batch sizes must match"
        assert observed_inputs.shape == input_uncertainty.shape, "Input and uncertainty shapes must match"
        
        self.pretrained_model = pretrained_model
        self.observed_inputs = observed_inputs
        self.observed_outputs = observed_outputs
        self.input_uncertainty = input_uncertainty
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces
        num_params = sum(p.numel() for p in pretrained_model.parameters())
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(num_params,), 
            dtype=np.float32
        )
        
        # Observation space combines model predictions and target outputs
        output_dim = observed_outputs.shape[1]  # Smaller dimension
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(output_dim * 2,),  # Predictions + targets
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Get initial prediction using full input
        with torch.no_grad():
            initial_pred = self.pretrained_model(
                self.observed_inputs[0].unsqueeze(0)
            ).squeeze(0)
            
        # State combines prediction and target (both in output dimension)
        state = torch.cat([
            initial_pred,
            self.observed_outputs[0]
        ]).numpy()
        
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Apply action (parameter updates) to model
        self._apply_action(action)
        
        # Get current input and its uncertainty
        current_input = self.observed_inputs[self.current_step]
        current_uncertainty = self.input_uncertainty[self.current_step]
        
        # Forward pass with noisy input based on uncertainty
        with torch.no_grad():
            noisy_input = current_input + torch.randn_like(current_input) * current_uncertainty
            pred = self.pretrained_model(noisy_input.unsqueeze(0)).squeeze(0)
        
        # Calculate reward considering uncertainty weight
        target = self.observed_outputs[self.current_step]
        # Higher uncertainty = lower weight in reward
        uncertainty_weight = 1.0 / (1.0 + torch.mean(current_uncertainty))
        reward = -torch.mean((pred - target) ** 2).item() * uncertainty_weight * 0.01
        
        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Next state combines prediction and target
        next_state = torch.cat([pred, target]).numpy()
        
        return next_state, reward, done, False, {}
    
    def _apply_action(self, action: np.ndarray) -> None:
        idx = 0
        for param in self.pretrained_model.parameters():
            num_params = param.numel()
            delta = torch.tensor(
                action[idx:idx + num_params],
                dtype=param.dtype
            ).reshape(param.shape)
            param.data += delta * 0.001  # Small update scale
            idx += num_params

class RLFineTuner:
    """Reinforcement learning fine-tuning approach using SAC"""
    def __init__(self,
                total_timesteps: int = 10000,
                learning_rate: float = 1e-4):
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate

    def fine_tune(self,
                pretrained_model: nn.Module,
                observed_inputs: torch.Tensor,
                observed_outputs: torch.Tensor,
                input_uncertainty: torch.Tensor) -> nn.Module:
        """Fine-tune the model using SAC"""
        env = ChemistryEnv(pretrained_model,
                        observed_inputs,
                        observed_outputs,
                        input_uncertainty
                        )
        
        sac_agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            verbose=1
        )
        
        sac_agent.learn(total_timesteps=self.total_timesteps)
        
        return pretrained_model
