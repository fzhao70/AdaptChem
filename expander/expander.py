import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from base_model import DemoModel

def extend_input_output(model, input_extension, output_extension):
    """
    Extend the input and output dimensions of a model
    
    Args:
        model: PyTorch model
        input_extension: Number of dimensions to add to the input
        output_extension: Number of dimensions to add to the output
        
    Returns:
        model: Extended model
    """
    # Get the first and last linear layers
    first_layer = None
    last_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if first_layer is None:
                first_layer = module
            last_layer = module
    
    # Extend input dimension
    old_weight = first_layer.weight
    old_bias = first_layer.bias
    new_in_features = first_layer.in_features + input_extension
    new_out_features = first_layer.out_features
    
    first_layer.in_features = new_in_features
    first_layer.weight = nn.Parameter(torch.zeros(new_out_features, new_in_features))
    first_layer.weight.data[:, :old_weight.size(1)] = old_weight
    
    # Extend output dimension
    old_weight = last_layer.weight
    old_bias = last_layer.bias
    new_out_features = last_layer.out_features + output_extension
    
    last_layer.out_features = new_out_features
    last_layer.weight = nn.Parameter(torch.zeros(new_out_features, last_layer.in_features))
    last_layer.weight.data[:old_weight.size(0), :] = old_weight
    last_layer.bias = nn.Parameter(torch.zeros(new_out_features))
    last_layer.bias.data[:old_bias.size(0)] = old_bias
    
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