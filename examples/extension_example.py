"""
Extension Example: Extending Model Dimensions for New Chemical Species

This example demonstrates how to use the expander module to add new
input/output dimensions to an existing trained model without retraining
from scratch. This is useful when adding new chemical species to an
existing atmospheric chemistry model.

Author: Fanghe Zhao
Date: December 2024
"""

import torch
import torch.nn as nn
from expander import extend_input_output


def create_simple_model():
    """Create a simple sequential model"""
    model = nn.Sequential(
        nn.Linear(10, 30),
        nn.ReLU(),
        nn.Linear(30, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    return model


def main():
    """Main example demonstrating model dimension extension"""

    print("=" * 70)
    print("AdaptChem Extension Example: Adding New Chemical Species")
    print("=" * 70)
    print()

    # =========================================================================
    # Scenario: We have a trained model for 10 species producing 5 outputs
    # We need to add 3 new species (inputs) and predict 2 new rates (outputs)
    # =========================================================================

    print("Creating initial model...")
    print("  - Input dimension: 10 (representing 10 chemical species)")
    print("  - Output dimension: 5 (representing 5 reaction rates)")
    print()

    # Create and "train" initial model (we'll just initialize it here)
    model = create_simple_model()

    # Show model architecture
    print("Initial model architecture:")
    print(model)
    print()

    # Test with original dimensions
    test_input_original = torch.randn(4, 10)  # Batch of 4 samples, 10 features
    print(f"Original input shape: {test_input_original.shape}")

    output_original = model(test_input_original)
    print(f"Original output shape: {output_original.shape}")
    print(f"Sample output values: {output_original[0].detach().numpy()}")
    print()

    # =========================================================================
    # Extend the model for new chemical species
    # =========================================================================
    print("-" * 70)
    print("Extending model dimensions...")
    print("-" * 70)

    INPUT_EXTENSION = 3   # Adding 3 new species
    OUTPUT_EXTENSION = 2  # Adding 2 new reaction rates

    print(f"  - Adding {INPUT_EXTENSION} new input dimensions (new species)")
    print(f"  - Adding {OUTPUT_EXTENSION} new output dimensions (new rates)")
    print()

    # Extend the model
    extended_model = extend_input_output(
        model=model,
        input_extension=INPUT_EXTENSION,
        output_extension=OUTPUT_EXTENSION
    )

    print("Extended model architecture:")
    print(extended_model)
    print()

    # =========================================================================
    # Test the extended model
    # =========================================================================
    print("-" * 70)
    print("Testing extended model...")
    print("-" * 70)
    print()

    # Create test input with new dimensions
    # First 10 dimensions: original species (use same values as before)
    # Last 3 dimensions: new species (initialize to zero for testing)
    test_input_extended = torch.cat([
        test_input_original,
        torch.zeros(4, INPUT_EXTENSION)
    ], dim=1)

    print(f"Extended input shape: {test_input_extended.shape}")

    # Get predictions from extended model
    output_extended = extended_model(test_input_extended)
    print(f"Extended output shape: {output_extended.shape}")
    print()

    # =========================================================================
    # Verify that original behavior is preserved
    # =========================================================================
    print("-" * 70)
    print("Verifying preservation of original behavior...")
    print("-" * 70)
    print()

    # The first 5 outputs should be similar to original (when new inputs are zero)
    # because new dimensions were initialized to zero
    original_outputs = output_extended[:, :5]
    new_outputs = output_extended[:, 5:]

    print("Original outputs from extended model:")
    print(f"  Shape: {original_outputs.shape}")
    print(f"  Sample: {original_outputs[0].detach().numpy()}")
    print()

    print("New outputs from extended model:")
    print(f"  Shape: {new_outputs.shape}")
    print(f"  Sample: {new_outputs[0].detach().numpy()}")
    print()

    # Check if original predictions are approximately preserved
    # (They should be very close when new inputs are zero)
    difference = torch.abs(original_outputs - output_original).max().item()
    print(f"Maximum difference in original outputs: {difference:.6f}")

    if difference < 1e-5:
        print("✓ Original behavior successfully preserved!")
    else:
        print("⚠ Warning: Some deviation in original behavior detected")
        print("  (This is expected if model has non-linear activation at output)")
    print()

    # =========================================================================
    # Demonstrate with non-zero new inputs
    # =========================================================================
    print("-" * 70)
    print("Testing with non-zero new species concentrations...")
    print("-" * 70)
    print()

    # Now test with actual values for new species
    test_input_with_new_species = torch.cat([
        test_input_original,
        torch.randn(4, INPUT_EXTENSION) * 0.5  # New species with some values
    ], dim=1)

    output_with_new_species = extended_model(test_input_with_new_species)

    print(f"Input shape with new species: {test_input_with_new_species.shape}")
    print(f"Output shape: {output_with_new_species.shape}")
    print()
    print("Sample output with new species:")
    print(f"  Original rates: {output_with_new_species[0, :5].detach().numpy()}")
    print(f"  New rates: {output_with_new_species[0, 5:].detach().numpy()}")
    print()

    # =========================================================================
    # Next steps
    # =========================================================================
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("After extending the model, you would typically:")
    print()
    print("1. Fine-tune the extended model using new observations")
    print("   that include the new species and reaction rates:")
    print()
    print("   from adaptor import ChemistryMechanismAdapter")
    print("   adapter = ChemistryMechanismAdapter(method='finetune')")
    print("   fine_tuned = adapter.fine_tune(")
    print("       pretrained_model=extended_model,")
    print("       observed_inputs=new_data_inputs,")
    print("       observed_outputs=new_data_outputs,")
    print("       input_uncertainty=uncertainties")
    print("   )")
    print()
    print("2. Export the adapted model for use in your atmospheric code:")
    print()
    print("   from wrapper import save_model")
    print("   save_model(fine_tuned, 'extended_chemistry_model.pt')")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
