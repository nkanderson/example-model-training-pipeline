"""
Test the quantized weights by loading them back into the Python SNN model.
This validates that quantization error is acceptable before debugging hardware implementation.
"""

import torch
import sys
from pathlib import Path

# Add train directory to path to import Net class
train_dir = Path(__file__).parent.parent / "train"
sys.path.insert(0, str(train_dir))

from train_xor import Net, beta, num_steps, device

# Quantization parameters (must match quantize_weights.py)
SCALE_FACTOR = 100


def hex_to_signed_int8(hex_str):
    """Convert hex string to signed 8-bit integer using two's complement."""
    value = int(hex_str, 16)
    # If MSB is set, it's negative in two's complement
    if value & 0x80:
        value = value - 0x100
    return value


def load_quantized_weights(weights_dir):
    """Load quantized weights from .mem files and convert back to float32."""
    weights_dir = Path(weights_dir)

    # Read and parse each .mem file
    def read_mem_file(filename):
        filepath = weights_dir / filename
        with open(filepath, "r") as f:
            hex_values = [line.strip() for line in f if line.strip()]
        # Convert hex to signed int8, then to float32
        int_values = [hex_to_signed_int8(h) for h in hex_values]
        float_values = [v / SCALE_FACTOR for v in int_values]
        return torch.tensor(float_values, dtype=torch.float32)

    # Load all weight files
    hidden_w0 = read_mem_file("hidden_neuron_0.mem")  # fc1.weight[0, :]
    hidden_w1 = read_mem_file("hidden_neuron_1.mem")  # fc1.weight[1, :]
    hidden_bias = read_mem_file("hidden_bias.mem")  # fc1.bias
    output_w = read_mem_file("output_neuron.mem")  # fc2.weight[0, :]
    output_bias = read_mem_file("output_bias.mem")  # fc2.bias

    # Construct state_dict matching the Net architecture
    state_dict = {
        "fc1.weight": torch.stack([hidden_w0, hidden_w1]),  # Shape: [2, 2]
        "fc1.bias": hidden_bias,  # Shape: [2]
        "fc2.weight": output_w.unsqueeze(0),  # Shape: [1, 2]
        "fc2.bias": output_bias,  # Shape: [1]
    }

    return state_dict


def test_quantized_model(weights_dir):
    """Test the SNN with quantized weights on XOR dataset."""

    # Load quantized weights
    print("Loading quantized weights from .mem files...")
    quantized_state_dict = load_quantized_weights(weights_dir)

    # Print loaded weights for verification
    print("\nQuantized weights loaded:")
    for name, tensor in quantized_state_dict.items():
        print(f"  {name}: {tensor.cpu().numpy()}")

    # Create network and load quantized weights
    net = Net().to(device)
    # Use strict=False to only load fc1/fc2 weights, not LIF neuron parameters
    net.load_state_dict(quantized_state_dict, strict=False)
    net.eval()

    # XOR test data (same as training)
    data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    # Run inference
    print(
        f"\nRunning inference with quantized weights (beta={beta}, timesteps={num_steps})..."
    )
    with torch.no_grad():
        spk_rec, mem_rec = net(data)
        spike_count = spk_rec.sum(dim=0)

    # Display results
    print("\n" + "=" * 60)
    print("QUANTIZED WEIGHTS TEST RESULTS")
    print("=" * 60)
    print("Input [x0, x1] -> Spike Count -> Expected")
    print("-" * 60)
    for i in range(len(data)):
        input_str = f"[{data[i][0].item():.1f}, {data[i][1].item():.1f}]"
        spike_str = f"{spike_count[i].item():.2f}"
        target_str = f"{targets[i].item():.1f}"

        # Determine if result is correct (spike count close to target)
        error = abs(spike_count[i].item() - targets[i].item())
        status = "✓" if error < 0.5 else "✗"

        print(f"{input_str:12} -> {spike_str:6} -> {target_str:8} {status}")
    print("=" * 60)

    # Summary
    errors = torch.abs(spike_count - targets)
    avg_error = errors.mean().item()
    print(f"\nAverage error: {avg_error:.3f}")

    # Check if XOR pattern is correct (high/low discrimination)
    false_cases_avg = (spike_count[0] + spike_count[3]) / 2  # [0,0] and [1,1]
    true_cases_avg = (spike_count[1] + spike_count[2]) / 2  # [0,1] and [1,0]

    print(f"False cases ([0,0], [1,1]) average spikes: {false_cases_avg.item():.2f}")
    print(f"True cases ([0,1], [1,0]) average spikes: {true_cases_avg.item():.2f}")

    if true_cases_avg > false_cases_avg + 0.5:
        print("\n✓ XOR pattern is CORRECT with quantized weights!")
    else:
        print("\n✗ XOR pattern is INCORRECT - quantization may have broken the model")

    return spike_count


if __name__ == "__main__":
    # Path to weights directory
    weights_dir = Path(__file__).parent.parent / "sv" / "weights"

    if not weights_dir.exists():
        print(f"Error: Weights directory not found at {weights_dir}")
        print("Please run quantize_weights.py first to generate .mem files")
        sys.exit(1)

    test_quantized_model(weights_dir)
