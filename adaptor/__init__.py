"""
Adaptor Module - Fine-tune atmospheric chemistry models using various methods.

This module provides tools for adapting pre-trained neural network models
to new atmospheric chemistry mechanisms or observations using:
- Traditional fine-tuning with uncertainty quantification
- Reinforcement learning (RL) with Soft Actor-Critic
- Direct Preference Optimization (DPO)
"""

from .adaptor import ChemistryMechanismAdapter
from .method_ft import TraditionalFineTuner
from .method_rl import RLFineTuner, ChemistryEnv
from .method_dpo import ChemistryDPOTuner
from .base_model import DemoModel

__all__ = [
    'ChemistryMechanismAdapter',
    'TraditionalFineTuner',
    'RLFineTuner',
    'ChemistryEnv',
    'ChemistryDPOTuner',
    'DemoModel',
]
