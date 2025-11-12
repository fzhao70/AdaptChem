# AdaptChem Examples

This directory contains example scripts demonstrating the capabilities of the AdaptChem framework.

## Available Examples

### 1. `basic_example.py` - Complete Adaptation Workflow

Demonstrates all three adaptation methods provided by AdaptChem:
- Traditional fine-tuning with uncertainty quantification
- Reinforcement learning using Soft Actor-Critic (SAC)
- Direct Preference Optimization (DPO)

**What it does:**
- Generates synthetic atmospheric chemistry data
- Adapts a pre-trained model using each method
- Compares performance across methods
- Generates visualization of results

**Run it:**
```bash
cd examples
python basic_example.py
```

**Expected output:**
- Console output showing training progress for each method
- `adaptation_comparison.png` - Bar chart comparing MSE across methods
- Performance comparison table

**Estimated runtime:** 2-5 minutes (depending on hardware)

---

### 2. `extension_example.py` - Model Dimension Extension

Demonstrates how to extend model input/output dimensions for new chemical species.

**What it does:**
- Creates a model for 10 species producing 5 outputs
- Extends it to handle 13 species and produce 7 outputs
- Verifies that original behavior is preserved
- Shows how new dimensions respond to input

**Use case:** When you need to add new chemical species to your mechanism without retraining from scratch.

**Run it:**
```bash
cd examples
python extension_example.py
```

**Expected output:**
- Console output showing model shapes before/after extension
- Verification that original predictions are preserved
- Demonstration of new dimension behavior

**Estimated runtime:** < 10 seconds

---

## Running the Examples

### Prerequisites

Make sure AdaptChem is installed:

```bash
# From the repository root
pip install -e .
```

Or if you're running from within the repository without installation:

```bash
# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:/home/user/AdaptChem"
```

### Running All Examples

```bash
# From the examples directory
python basic_example.py
python extension_example.py
```

## Understanding the Output

### basic_example.py Output

```
==================================================================
AdaptChem Example: Adapting Atmospheric Chemistry Models
==================================================================

Generating 200 synthetic samples...
Training data: torch.Size([200, 40])
Test data: torch.Size([50, 40])

Pre-trained model MSE: 0.123456

------------------------------------------------------------------
Method 1: Traditional Fine-Tuning with Uncertainty
------------------------------------------------------------------
Fine-tuning model (this may take a minute)...
Epoch 10/100, Avg Loss: 0.098765, Learning Rate: 1.00e-03
Epoch 20/100, Avg Loss: 0.087654, Learning Rate: 1.00e-03
...
Fine-tuned model MSE: 0.067890
Improvement: 45.00%

[Similar output for RL and DPO methods]

==================================================================
RESULTS SUMMARY
==================================================================

Method                          MSE           Improvement
------------------------------------------------------------------
Pretrained                   0.123456       (baseline)
Fine-Tuning                  0.067890       +45.00%
RL (SAC)                     0.078901       +36.12%
DPO                          0.089012       +27.89%
```

### extension_example.py Output

```
==================================================================
AdaptChem Extension Example: Adding New Chemical Species
==================================================================

Creating initial model...
  - Input dimension: 10 (representing 10 chemical species)
  - Output dimension: 5 (representing 5 reaction rates)

Original input shape: torch.Size([4, 10])
Original output shape: torch.Size([4, 5])

------------------------------------------------------------------
Extending model dimensions...
------------------------------------------------------------------
  - Adding 3 new input dimensions (new species)
  - Adding 2 new output dimensions (new rates)

Extended input shape: torch.Size([4, 13])
Extended output shape: torch.Size([4, 7])

Maximum difference in original outputs: 0.000001
✓ Original behavior successfully preserved!
```

## Customizing the Examples

### Modify Data Generation

In `basic_example.py`, adjust the synthetic data parameters:

```python
# Change dimensions
INPUT_DIM = 50    # More species
OUTPUT_DIM = 30   # More reactions

# Change sample size
train_inputs, train_outputs, train_uncertainties = create_synthetic_data(
    n_samples=500,  # More training data
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM
)
```

### Modify Training Parameters

```python
# Longer training for fine-tuning
adapter_ft = ChemistryMechanismAdapter(
    method="finetune",
    learning_rate=5e-4,    # Smaller learning rate
    weight_decay=1e-4      # More regularization
)

fine_tuned_model = adapter_ft.fine_tune(
    ...,
    num_epochs=500,        # More epochs
    batch_size=64,         # Larger batches
    patience=20            # More patience for scheduler
)
```

### Modify RL Parameters

```python
# More RL training
adapter_rl = ChemistryMechanismAdapter(
    method="rl",
    total_timesteps=50000,  # More training steps
    learning_rate=1e-4
)
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'adaptor'`:

```bash
# Make sure you're running from the examples directory
cd /path/to/AdaptChem/examples

# And that AdaptChem is installed or PYTHONPATH is set
pip install -e ..
# OR
export PYTHONPATH="${PYTHONPATH}:/path/to/AdaptChem"
```

### Memory Issues

If you run out of memory:
- Reduce `n_samples` in data generation
- Reduce `batch_size` in training
- Reduce `total_timesteps` for RL
- Reduce model size (`HIDDEN_DIM`)

### Slow Performance

If examples run too slowly:
- Reduce `num_epochs` for fine-tuning
- Reduce `total_timesteps` for RL
- Reduce `n_samples` for smaller datasets
- Use GPU if available (models will automatically use CUDA if available)

## Next Steps

After running these examples:

1. **Adapt to your data:** Replace synthetic data with your actual atmospheric chemistry observations
2. **Experiment with methods:** Try different adaptation methods to see which works best for your use case
3. **Tune hyperparameters:** Adjust learning rates, batch sizes, and other parameters for optimal performance
4. **Export models:** Use the wrapper module to export your adapted models for production use

## Additional Resources

- [Main README](../README.md) - Full framework documentation
- [API Reference](../README.md#api-reference) - Detailed API documentation
- [Paper](../paper.md) - Scientific background and motivation

## Questions?

If you have questions or encounter issues:
- Open an issue on GitHub: https://github.com/fzhao70/AdaptChem/issues
- Contact: fzhao97@gmail.com
