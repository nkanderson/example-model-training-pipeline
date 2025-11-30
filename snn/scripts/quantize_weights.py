"""
Quantize SNN weights from PyTorch float32 to signed 8-bit integers.
Converts trained weights to hardware-compatible format for SystemVerilog simulation.
"""

import torch
import argparse
from pathlib import Path


def float_to_int8(value, scale_factor=100.0):
    """
    Convert floating point value to signed 8-bit integer.

    Args:
        value: Float value to convert
        scale_factor: Multiplier before quantization (default 100 for ~0.01 resolution)
                      Allows representation of values in range [-1.28, 1.27]

    Returns:
        Signed 8-bit integer in range [-128, 127]
    """
    # Scale and round
    scaled = round(value * scale_factor)

    # Clamp to 8-bit signed range
    clamped = max(-128, min(127, scaled))

    # Convert to unsigned (2's complement) representation for hex output
    if clamped < 0:
        unsigned = (256 + clamped) & 0xFF
    else:
        unsigned = clamped & 0xFF

    return unsigned


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SNN weights to signed 8-bit format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="../train/weights/snn_xor_weights.pth",
        help="Path to PyTorch weights file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../sv/weights",
        help="Output directory for .mem files",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale factor for quantization (default: 100)",
    )

    args = parser.parse_args()

    # Get script directory and resolve paths
    script_dir = Path(__file__).parent
    weights_path = (script_dir / args.weights).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    # Load weights
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Extract and quantize weights
    fc1_weight = state_dict["fc1.weight"]  # Shape: [2, 2] - [neurons, inputs]
    fc1_bias = state_dict["fc1.bias"]  # Shape: [2]
    fc2_weight = state_dict["fc2.weight"]  # Shape: [1, 2] - [neurons, inputs]
    fc2_bias = state_dict["fc2.bias"]  # Shape: [1]

    print("\n" + "=" * 60)
    print("Quantizing weights...")
    print("=" * 60)

    # Hidden neuron 0 weights (2 inputs)
    with open(output_dir / "hidden_neuron_0.mem", "w") as f:
        for i in range(2):
            val = fc1_weight[0, i].item()
            quant = float_to_int8(val, args.scale)
            f.write(f"{quant:02x}\n")
            print(
                f"hidden_neuron_0 weight[{i}]: {val:8.4f} -> 0x{quant:02x} ({quant:3d})"
            )

    # Hidden neuron 1 weights (2 inputs)
    with open(output_dir / "hidden_neuron_1.mem", "w") as f:
        for i in range(2):
            val = fc1_weight[1, i].item()
            quant = float_to_int8(val, args.scale)
            f.write(f"{quant:02x}\n")
            print(
                f"hidden_neuron_1 weight[{i}]: {val:8.4f} -> 0x{quant:02x} ({quant:3d})"
            )

    # Hidden biases (2 neurons)
    with open(output_dir / "hidden_bias.mem", "w") as f:
        for i in range(2):
            val = fc1_bias[i].item()
            quant = float_to_int8(val, args.scale)
            f.write(f"{quant:02x}\n")
            print(f"hidden_bias[{i}]:         {val:8.4f} -> 0x{quant:02x} ({quant:3d})")

    # Output neuron weights (2 hidden inputs)
    with open(output_dir / "output_neuron.mem", "w") as f:
        for i in range(2):
            val = fc2_weight[0, i].item()
            quant = float_to_int8(val, args.scale)
            f.write(f"{quant:02x}\n")
            print(
                f"output_neuron weight[{i}]: {val:8.4f} -> 0x{quant:02x} ({quant:3d})"
            )

    # Output bias (1 neuron)
    with open(output_dir / "output_bias.mem", "w") as f:
        val = fc2_bias[0].item()
        quant = float_to_int8(val, args.scale)
        f.write(f"{quant:02x}\n")
        print(f"output_bias:              {val:8.4f} -> 0x{quant:02x} ({quant:3d})")

    print("\n" + "=" * 60)
    print(f"Quantization complete! Files written to {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - hidden_neuron_0.mem (2 weights)")
    print("  - hidden_neuron_1.mem (2 weights)")
    print("  - hidden_bias.mem (2 biases)")
    print("  - output_neuron.mem (2 weights)")
    print("  - output_bias.mem (1 bias)")


if __name__ == "__main__":
    main()
