"""
Quantize floating-point neural network weights to fixed-point representation.

This script converts floating-point weights to signed fixed-point values
for hardware implementation. Each line of weights from the input file
is written to a separate .mem file for loading into hardware neurons.
"""

import argparse
import os
import sys
from pathlib import Path


def float_to_fixed(value, integer_bits, fractional_bits):
    """
    Convert a floating-point value to a signed fixed-point integer.

    Uses Q(integer_bits).(fractional_bits) format where:
    - Total bits = 1 (sign) + integer_bits + fractional_bits
    - For example, Q3.12 with integer_bits=3, fractional_bits=12:
      - 1 sign bit
      - 3 bits for integer part (range: -8 to +7 for the integer portion)
      - 12 bits for fractional part (precision: 1/4096 ≈ 0.000244)
      - Total: 16 bits

    Args:
        value: Floating-point value to convert
        integer_bits: Number of bits for the integer part (excluding sign bit)
        fractional_bits: Number of bits for the fractional part

    Returns:
        Integer representing the fixed-point value
    """
    # Scale the floating-point value by 2^fractional_bits
    scaled_value = value * (2**fractional_bits)

    # Round to nearest integer
    fixed_value = round(scaled_value)

    # Calculate total bit width (sign + integer + fractional)
    bit_width = 1 + integer_bits + fractional_bits

    # Determine the min and max values for the given bit width (signed)
    max_value = (2 ** (bit_width - 1)) - 1
    min_value = -(2 ** (bit_width - 1))

    # Clamp the value to the valid range
    if fixed_value > max_value:
        print(
            f"Warning: Value {value} clamped from {fixed_value} to {max_value}",
            file=sys.stderr,
        )
        fixed_value = max_value
    elif fixed_value < min_value:
        print(
            f"Warning: Value {value} clamped from {fixed_value} to {min_value}",
            file=sys.stderr,
        )
        fixed_value = min_value

    # Convert to unsigned two's complement representation for hardware
    if fixed_value < 0:
        fixed_value = (2**bit_width) + fixed_value

    return fixed_value


def quantize_weights(input_path, output_dir, integer_bits, fractional_bits):
    """
    Read floating-point weights and write fixed-point weights to separate files.

    Args:
        input_path: Path to input file containing floating-point weights
        output_dir: Directory to write output .mem files
        integer_bits: Number of bits for integer part (excluding sign)
        fractional_bits: Number of bits for fractional part
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read input file
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Process each line
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Parse floating-point values
        float_values = [float(x) for x in line.split()]

        # Convert to fixed-point
        fixed_values = [
            float_to_fixed(v, integer_bits, fractional_bits) for v in float_values
        ]

        # Write to output file (one value per line in hexadecimal)
        output_path = os.path.join(output_dir, f"neuron_{line_idx}.mem")
        with open(output_path, "w") as f:
            for fixed_val in fixed_values:
                f.write(f"{fixed_val:04x}\n")

        print(f"Wrote {len(fixed_values)} weights to {output_path}")


def main():
    # Determine script directory
    script_dir = Path(__file__).parent.resolve()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Quantize floating-point weights to fixed-point for hardware implementation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(script_dir / "../train/weights/mlp_2_2.txt"),
        help="Path to input floating-point weights file (default: ../train/weights/mlp_2_2.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "../sv/weights"),
        help="Directory for output .mem files (default: ../sv/weights)",
    )
    parser.add_argument(
        "--integer-bits",
        type=int,
        default=3,
        help="Number of bits for integer part, excluding sign bit (default: 3 for Q3.12)",
    )
    parser.add_argument(
        "--fractional-bits",
        type=int,
        default=12,
        help="Number of bits for fractional part (default: 12 for Q3.12)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    # Validate bit counts
    if args.integer_bits < 1 or args.integer_bits > 32:
        print(
            f"Error: Integer bits must be between 1 and 32, got {args.integer_bits}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.fractional_bits < 0 or args.fractional_bits > 32:
        print(
            f"Error: Fractional bits must be between 0 and 32, got {args.fractional_bits}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_bits = 1 + args.integer_bits + args.fractional_bits
    integer_range_max = (2**args.integer_bits) - 1
    integer_range_min = -(2**args.integer_bits)
    precision = 1.0 / (2**args.fractional_bits)

    print(f"Quantizing weights from: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Fixed-point format: Q{args.integer_bits}.{args.fractional_bits}")
    print("  Sign bits: 1")
    print(f"  Integer bits: {args.integer_bits}")
    print(f"  Fractional bits: {args.fractional_bits}")
    print(f"  Total bits: {total_bits}")
    print(f"  Integer range: [{integer_range_min}, {integer_range_max}]")
    print(f"  Precision: {precision:.6f}")
    print()

    # Perform quantization
    quantize_weights(
        args.input, args.output_dir, args.integer_bits, args.fractional_bits
    )

    print("\nQuantization complete!")


if __name__ == "__main__":
    main()
