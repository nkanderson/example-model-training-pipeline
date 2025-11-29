import torch
from pathlib import Path

# Load the saved weights
# Get the script directory and construct path relative to it
script_dir = Path(__file__).parent
weights_path = script_dir / "../train/weights/snn_xor_weights.pth"
weights_path = weights_path.resolve()  # Convert to absolute path

if not weights_path.exists():
    print(f"Weights file not found at {weights_path}")
    print("Please run train_xor.py first to generate the weights.")
else:
    state_dict = torch.load(weights_path)

    print("=" * 60)
    print("SNN XOR Network Trained Weights")
    print("=" * 60)

    print("\n--- Hidden Layer (fc1) ---")
    print("Weights shape:", state_dict["fc1.weight"].shape, "(2 neurons × 2 inputs)")
    print("\nWeight matrix:")
    print(state_dict["fc1.weight"])
    print("\nBiases:")
    print(state_dict["fc1.bias"])

    print("\n--- Output Layer (fc2) ---")
    print(
        "Weights shape:", state_dict["fc2.weight"].shape, "(1 neuron × 2 hidden inputs)"
    )
    print("\nWeight matrix:")
    print(state_dict["fc2.weight"])
    print("\nBiases:")
    print(state_dict["fc2.bias"])

    print("\n" + "=" * 60)
    print("Weight Statistics")
    print("=" * 60)

    all_weights = torch.cat(
        [
            state_dict["fc1.weight"].flatten(),
            state_dict["fc1.bias"].flatten(),
            state_dict["fc2.weight"].flatten(),
            state_dict["fc2.bias"].flatten(),
        ]
    )

    print(f"\nTotal parameters: {len(all_weights)}")
    print(f"Positive weights: {(all_weights > 0).sum().item()}")
    print(f"Negative weights: {(all_weights < 0).sum().item()}")
    print(f"Zero weights: {(all_weights == 0).sum().item()}")
    print(f"\nMin weight: {all_weights.min().item():.4f}")
    print(f"Max weight: {all_weights.max().item():.4f}")
    print(f"Mean weight: {all_weights.mean().item():.4f}")
