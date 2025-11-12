"""
Expander Module - Extend neural network input/output dimensions.

This module provides tools for expanding existing neural network models
to handle additional input features or output predictions while preserving
previously learned weights.
"""

from .expander import extend_input_output
from .base_model import DemoModel

__all__ = [
    'extend_input_output',
    'DemoModel',
]
