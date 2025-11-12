"""
Basic Example: Adapting a Chemistry Model using AdaptChem

This example demonstrates the three main adaptation methods provided by AdaptChem:
1. Traditional fine-tuning with uncertainty quantification
2. Reinforcement learning using Soft Actor-Critic
3. Direct Preference Optimization

Author: Fanghe Zhao
Date: December 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from adaptor import ChemistryMechanismAdapter, DemoModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def create_synthetic_data(n_samples=200, input_dim=40, output_dim=20):
    """
    Create synthetic atmospheric chemistry data with uncertainty

    Args:
        n_samples: Number of data samples
        input_dim: Input feature dimension (e.g., concentrations, temperature, pressure)
        output_dim: Output dimension (e.g., reaction rates)

    Returns:
        inputs, outputs, uncertainties
    """
    print(f"Generating {n_samples} synthetic samples...")

    # Generate input features (e.g., chemical concentrations, meteorological variables)
    inputs = torch.randn(n_samples, input_dim) * 0.5 + 1.0

    # Generate measurement uncertainties (varying by feature)
    # Some measurements are more uncertain than others
    base_uncertainty = torch.rand(input_dim) * 0.2 + 0.05
    uncertainties = base_uncertainty.unsqueeze(0).repeat(n_samples, 1)
    uncertainties += torch.randn(n_samples, input_dim) * 0.02  # Add some variation
    uncertainties = torch.abs(uncertainties)  # Ensure positive

    # Generate target outputs (simplified chemical reaction rates)
    # In reality, these would come from detailed chemical simulations or observations
    W = torch.randn(input_dim, output_dim) * 0.1
    outputs = torch.matmul(inputs, W) + torch.randn(n_samples, output_dim) * 0.1

    return inputs, outputs, uncertainties


def evaluate_model(model, test_inputs, test_outputs):
    """
    Evaluate model performance on test data

    Args:
        model: PyTorch model to evaluate
        test_inputs: Test input data
        test_outputs: Test output data

    Returns:
        Mean squared error
    """
    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs)
        mse = torch.mean((predictions - test_outputs) ** 2).item()
    return mse


def main():
    """Main example demonstrating all three adaptation methods"""

    print("=" * 70)
    print("AdaptChem Example: Adapting Atmospheric Chemistry Models")
    print("=" * 70)
    print()

    # Configuration
    INPUT_DIM = 40   # Number of input features
    HIDDEN_DIM = 64  # Hidden layer size
    OUTPUT_DIM = 20  # Number of outputs (e.g., reaction rates)

    # Generate synthetic training and test data
    train_inputs, train_outputs, train_uncertainties = create_synthetic_data(
        n_samples=200, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )

    test_inputs, test_outputs, test_uncertainties = create_synthetic_data(
        n_samples=50, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )

    print(f"Training data: {train_inputs.shape}")
    print(f"Test data: {test_inputs.shape}")
    print()

    # Create a pre-trained model (in practice, this would be loaded from file)
    print("Creating pre-trained model...")
    pretrained_model = DemoModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # Evaluate pre-trained model
    initial_mse = evaluate_model(pretrained_model, test_inputs, test_outputs)
    print(f"Pre-trained model MSE: {initial_mse:.6f}")
    print()

    # Store results for comparison
    results = {
        'Pretrained': initial_mse
    }

    # =========================================================================
    # Method 1: Traditional Fine-Tuning with Uncertainty Quantification
    # =========================================================================
    print("-" * 70)
    print("Method 1: Traditional Fine-Tuning with Uncertainty")
    print("-" * 70)

    # Create a fresh copy of the model for this method
    model_ft = DemoModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model_ft.load_state_dict(pretrained_model.state_dict())

    # Initialize adapter
    adapter_ft = ChemistryMechanismAdapter(
        method="finetune",
        learning_rate=1e-3,
        weight_decay=1e-5
    )

    print("Fine-tuning model (this may take a minute)...")
    fine_tuned_model = adapter_ft.fine_tune(
        pretrained_model=model_ft,
        observed_inputs=train_inputs,
        observed_outputs=train_outputs,
        input_uncertainty=train_uncertainties,
        num_epochs=100,
        batch_size=32,
        patience=10  # For learning rate scheduling
    )

    ft_mse = evaluate_model(fine_tuned_model, test_inputs, test_outputs)
    results['Fine-Tuning'] = ft_mse
    print(f"Fine-tuned model MSE: {ft_mse:.6f}")
    print(f"Improvement: {((initial_mse - ft_mse) / initial_mse * 100):.2f}%")
    print()

    # =========================================================================
    # Method 2: Reinforcement Learning
    # =========================================================================
    print("-" * 70)
    print("Method 2: Reinforcement Learning (Soft Actor-Critic)")
    print("-" * 70)

    # Create a fresh copy of the model
    model_rl = DemoModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model_rl.load_state_dict(pretrained_model.state_dict())

    # Initialize RL adapter
    adapter_rl = ChemistryMechanismAdapter(
        method="rl",
        total_timesteps=5000,  # Reduced for faster demo
        learning_rate=3e-4
    )

    print("Training with RL (this may take a few minutes)...")
    rl_model = adapter_rl.fine_tune(
        pretrained_model=model_rl,
        observed_inputs=train_inputs,
        observed_outputs=train_outputs,
        input_uncertainty=train_uncertainties
    )

    rl_mse = evaluate_model(rl_model, test_inputs, test_outputs)
    results['RL (SAC)'] = rl_mse
    print(f"RL-adapted model MSE: {rl_mse:.6f}")
    print(f"Improvement: {((initial_mse - rl_mse) / initial_mse * 100):.2f}%")
    print()

    # =========================================================================
    # Method 3: Direct Preference Optimization
    # =========================================================================
    print("-" * 70)
    print("Method 3: Direct Preference Optimization")
    print("-" * 70)

    # Create a fresh copy of the model
    model_dpo = DemoModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model_dpo.load_state_dict(pretrained_model.state_dict())

    # Initialize DPO adapter
    adapter_dpo = ChemistryMechanismAdapter(
        method="dpo",
        learning_rate=1e-4,
        beta=0.1
    )

    print("Training with DPO...")
    dpo_model = adapter_dpo.fine_tune(
        pretrained_model=model_dpo,
        observed_inputs=train_inputs,
        observed_outputs=train_outputs,
        input_uncertainty=train_uncertainties,
        num_epochs=50
    )

    dpo_mse = evaluate_model(dpo_model, test_inputs, test_outputs)
    results['DPO'] = dpo_mse
    print(f"DPO-adapted model MSE: {dpo_mse:.6f}")
    print(f"Improvement: {((initial_mse - dpo_mse) / initial_mse * 100):.2f}%")
    print()

    # =========================================================================
    # Summary and Comparison
    # =========================================================================
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nMethod                          MSE           Improvement")
    print("-" * 70)
    for method, mse in results.items():
        if method == 'Pretrained':
            print(f"{method:25s}  {mse:.6f}       (baseline)")
        else:
            improvement = (initial_mse - mse) / initial_mse * 100
            print(f"{method:25s}  {mse:.6f}       {improvement:+.2f}%")

    # Visualize results
    print("\nGenerating comparison plot...")
    methods = list(results.keys())
    mse_values = list(results.values())

    plt.figure(figsize=(10, 6))
    colors = ['gray', 'blue', 'green', 'orange']
    bars = plt.bar(methods, mse_values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.title('Comparison of Adaptation Methods', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_file = 'adaptation_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
