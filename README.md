# AdaptChem

**Adaptive Atmospheric Chemistry Mechanism Toolkit and Framework with Hybrid Machine Learning**

AdaptChem is a Python framework for adapting pre-trained neural network models to new atmospheric chemistry mechanisms and observations. It provides three complementary approaches for model adaptation: traditional fine-tuning with uncertainty quantification, reinforcement learning, and direct preference optimization.

## Features

- **Multiple Adaptation Methods**: Choose from traditional fine-tuning, reinforcement learning (RL), or direct preference optimization (DPO)
- **Uncertainty Quantification**: Built-in support for input uncertainty through Monte Carlo sampling
- **Network Extension**: Dynamically extend model input/output dimensions while preserving learned weights
- **C/Fortran Integration**: Export adapted models as TorchScript for use in legacy atmospheric chemistry codes

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/fzhao70/AdaptChem.git
```

### From Source

```bash
git clone https://github.com/fzhao70/AdaptChem.git
cd AdaptChem
pip install -e .
```

### Dependencies

- Python >= 3.6
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Gymnasium
- Stable-Baselines3

## Structure of the Framework

This repository follows a modular structure:

### **Adaptor** (`adaptor/`)
Fine-tune existing neural network models using new mechanism data or observations. Three methods available:
- **Traditional Fine-Tuning** (`method_ft.py`): Supervised learning with uncertainty-weighted loss
- **Reinforcement Learning** (`method_rl.py`): RL-based adaptation using Soft Actor-Critic (SAC)
- **Direct Preference Optimization** (`method_dpo.py`): Preference-based learning for model adaptation

### **Expander** (`expander/`)
Extend the input/output dimensions of existing models to accommodate additional features or predictions while preserving previously learned weights.

### **Wrapper** (`wrapper/`)
Export PyTorch models as TorchScript and compile C++ interfaces for integration with C or Fortran atmospheric chemistry codes.

## Quick Start

### 1. Traditional Fine-Tuning with Uncertainty

```python
import torch
from adaptor import ChemistryMechanismAdapter, DemoModel

# Create or load a pre-trained model
pretrained_model = DemoModel(input_dim=40, hidden_dim=20, output_dim=20)

# Prepare your data
observed_inputs = torch.randn(100, 40)      # Input features
observed_outputs = torch.randn(100, 20)     # Target outputs
input_uncertainty = torch.randn(100, 40)    # Measurement uncertainty

# Initialize adapter with traditional fine-tuning
adapter = ChemistryMechanismAdapter(
    method="finetune",
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Fine-tune the model
fine_tuned_model = adapter.fine_tune(
    pretrained_model=pretrained_model,
    observed_inputs=observed_inputs,
    observed_outputs=observed_outputs,
    input_uncertainty=input_uncertainty,
    num_epochs=1000,
    patience=100  # For learning rate scheduling
)
```

### 2. Reinforcement Learning Adaptation

```python
from adaptor import ChemistryMechanismAdapter

# Initialize adapter with RL method
adapter = ChemistryMechanismAdapter(
    method="rl",
    total_timesteps=10000,
    learning_rate=1e-4
)

# Fine-tune using RL
fine_tuned_model = adapter.fine_tune(
    pretrained_model=pretrained_model,
    observed_inputs=observed_inputs,
    observed_outputs=observed_outputs,
    input_uncertainty=input_uncertainty
)
```

### 3. Direct Preference Optimization

```python
from adaptor import ChemistryMechanismAdapter

# Initialize adapter with DPO method
adapter = ChemistryMechanismAdapter(
    method="dpo",
    learning_rate=1e-4,
    beta=0.1  # Temperature parameter for DPO
)

# Fine-tune using DPO
fine_tuned_model = adapter.fine_tune(
    pretrained_model=pretrained_model,
    observed_inputs=observed_inputs,
    observed_outputs=observed_outputs,
    input_uncertainty=input_uncertainty,
    num_epochs=100
)
```

### 4. Extending Model Dimensions

```python
from expander import extend_input_output
import torch.nn as nn

# Create a model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Extend input by 3 dimensions and output by 2 dimensions
extended_model = extend_input_output(
    model=model,
    input_extension=3,   # Add 3 input features
    output_extension=2   # Add 2 output predictions
)

# Now the model accepts 13 inputs and produces 7 outputs
# Original weights are preserved, new dimensions initialized to zero
```

### 5. Exporting for C/Fortran Integration

```python
from wrapper import save_model, create_dynamic_library

# Save model as TorchScript
save_model(fine_tuned_model, "my_model.pt")

# Compile C++ wrapper library (requires LibTorch)
create_dynamic_library()

# Now you can load and use the model in C/Fortran code
# See wrapper/wrapper.cpp for the C interface
```

## Advanced Usage

### Custom Uncertainty Loss

The traditional fine-tuner uses Monte Carlo sampling to account for input uncertainty:

```python
from adaptor import TraditionalFineTuner

tuner = TraditionalFineTuner(
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Customize number of MC samples (default: 5)
tuner.num_mc_samples = 10

# Fine-tune with custom settings
fine_tuned = tuner.fine_tune(
    model=pretrained_model,
    observed_inputs=inputs,
    observed_outputs=outputs,
    input_uncertainty=uncertainty,
    num_epochs=500,
    batch_size=64
)
```

### Custom Optimizer

```python
import torch.optim as optim

adapter = ChemistryMechanismAdapter(method="finetune")

fine_tuned_model = adapter.fine_tune(
    pretrained_model=model,
    observed_inputs=inputs,
    observed_outputs=outputs,
    input_uncertainty=uncertainty,
    user_optimizer=optim.SGD  # Use SGD instead of Adam
)
```

### RL Environment Customization

```python
from adaptor import ChemistryEnv, RLFineTuner
from stable_baselines3 import SAC

# Create custom environment
env = ChemistryEnv(
    pretrained_model=model,
    observed_inputs=inputs,
    observed_outputs=outputs,
    input_uncertainty=uncertainty,
    max_steps=100  # Maximum steps per episode
)

# Create RL tuner with custom settings
tuner = RLFineTuner(
    total_timesteps=50000,
    learning_rate=3e-4
)

fine_tuned = tuner.fine_tune(model, inputs, outputs, uncertainty)
```

## API Reference

### ChemistryMechanismAdapter

Main class for model adaptation.

**Methods:**
- `__init__(method, **kwargs)`: Initialize with method ('finetune', 'rl', or 'dpo')
- `fine_tune(pretrained_model, observed_inputs, observed_outputs, input_uncertainty, **kwargs)`: Adapt the model

### extend_input_output

Extend model dimensions while preserving learned weights.

**Parameters:**
- `model`: PyTorch model to extend
- `input_extension`: Number of input dimensions to add
- `output_extension`: Number of output dimensions to add

**Returns:** Extended model

### save_model

Export model as TorchScript.

**Parameters:**
- `model`: PyTorch model
- `filename`: Output file path (*.pt)

### create_dynamic_library

Compile C++ wrapper for Fortran/C integration.

**Returns:** Path to compiled shared library

## Examples

See the `examples/` directory for complete working examples:
- Basic adaptation workflow
- Comparing different adaptation methods
- Extending models for new chemical species
- Integration with atmospheric chemistry models

## Citation

If you use AdaptChem in your research, please cite:

```bibtex
@software{adaptchem2024,
  author = {Zhao, Fanghe and Wang, Yuhang},
  title = {AdaptChem: Adaptive Atmospheric Chemistry Mechanism Framework},
  year = {2024},
  url = {https://github.com/fzhao70/AdaptChem}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Acknowledgements

We would like to acknowledge high-performance computing support from the Derecho system (doi:10.5065/qx9a-pg09) provided by the NSF National Center for Atmospheric Research (NCAR), sponsored by the National Science Foundation under project number UGIT0038.

## Contact

- **Fanghe Zhao** - fzhao97@gmail.com
- **Yuhang Wang** - School of Earth & Atmospheric Sciences, Georgia Institute of Technology

## Project Status

This project is actively maintained and under development. Feature requests and bug reports are welcome via GitHub issues.
