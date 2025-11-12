import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from .base_model import DemoModel

def extend_input_output(model, input_extension, output_extension):
    """
    Extend the input and output dimensions of a model by creating new layers
    with extended dimensions while preserving learned weights.

    Args:
        model: PyTorch model (must be nn.Sequential or contain Linear layers)
        input_extension: Number of dimensions to add to the input
        output_extension: Number of dimensions to add to the output

    Returns:
        model: Extended model with new input/output dimensions

    Note:
        New dimensions are initialized to zero to preserve the model's
        existing behavior on original dimensions (zero-shot extension).
    """
    if input_extension < 0 or output_extension < 0:
        raise ValueError("Extension dimensions must be non-negative")

    # Get the first and last linear layers
    first_layer = None
    last_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if first_layer is None:
                first_layer = module
            last_layer = module

    if first_layer is None or last_layer is None:
        raise ValueError("Model must contain at least one Linear layer")

    # Extend input dimension by creating a new weight matrix
    if input_extension > 0:
        old_weight = first_layer.weight.data
        old_bias = first_layer.bias.data if first_layer.bias is not None else None
        new_in_features = first_layer.in_features + input_extension
        new_out_features = first_layer.out_features

        # Create extended weight matrix (preserve old weights, initialize new to zero)
        new_weight = torch.zeros(new_out_features, new_in_features, dtype=old_weight.dtype)
        new_weight[:, :old_weight.size(1)] = old_weight

        # Update layer with new parameters
        first_layer.weight = nn.Parameter(new_weight)
        first_layer.in_features = new_in_features

    # Extend output dimension by creating a new weight and bias
    if output_extension > 0:
        old_weight = last_layer.weight.data
        old_bias = last_layer.bias.data if last_layer.bias is not None else None
        new_out_features = last_layer.out_features + output_extension
        new_in_features = last_layer.in_features

        # Create extended weight matrix (preserve old weights, initialize new to zero)
        new_weight = torch.zeros(new_out_features, new_in_features, dtype=old_weight.dtype)
        new_weight[:old_weight.size(0), :] = old_weight

        # Create extended bias (preserve old bias, initialize new to zero)
        if old_bias is not None:
            new_bias = torch.zeros(new_out_features, dtype=old_bias.dtype)
            new_bias[:old_bias.size(0)] = old_bias
            last_layer.bias = nn.Parameter(new_bias)

        # Update layer with new parameters
        last_layer.weight = nn.Parameter(new_weight)
        last_layer.out_features = new_out_features

    return model


# Usage example
#model = nn.Sequential(
#    nn.Linear(2, 20),
#    nn.ReLU(),
#    nn.Linear(20, 5),
#    nn.ReLU(),
#    nn.Linear(5, 5),
#    nn.ReLU(),
#    nn.Linear(5, 2)
#)
#
#obs = torch.randn(4, 2)
#print(model(obs))
#
## Extend input by 2 dimensions and output by 3 dimensions
#modified_model = extend_input_output(model, 2, 2)
#
#obs = torch.cat((obs, torch.randn(4, 2)), axis = 1)
#print(modified_model(obs))
#
#print(modified_model)