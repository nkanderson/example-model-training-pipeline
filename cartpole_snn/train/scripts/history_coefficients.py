"""
Compare Grunwald-Letnikov binomial coefficients with bit-shift approximations.

This script analyzes whether bit-shift operations (which are computationally
efficient in hardware) can approximate the GL coefficients used in fractional-
order derivatives.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path is set
from utils import compute_gl_coefficients  # noqa: E402


def get_gl_coefficients(alpha: float, history_length: int) -> list[float]:
    """
    Get GL binomial coefficients for a given fractional order alpha.

    Args:
        alpha: Fractional order (0 < alpha <= 1 typically)
        history_length: Number of coefficients to compute

    Returns:
        List of GL coefficient values [g_0, g_1, ..., g_{H-1}]
    """
    coeffs = compute_gl_coefficients(alpha, history_length)
    return coeffs.tolist()


def get_bitshift_sequence(history_length: int) -> list[float]:
    """
    Generate sequence of values by repeatedly right-shifting (dividing by 2).

    This produces the sequence: [1, 0.5, 0.25, 0.125, 0.0625, ...]
    which corresponds to [2^0, 2^-1, 2^-2, 2^-3, 2^-4, ...]

    In hardware, these can be computed efficiently with right bit shifts.

    Args:
        history_length: Number of values to generate

    Returns:
        List of bit-shift values [1, 1>>1, 1>>2, 1>>3, ...]
    """
    values = []
    for k in range(history_length):
        # Each right shift by k bits is equivalent to dividing by 2^k
        values.append(1.0 / (2**k))
    return values


def get_slow_decay_bitshift_sequence(history_length: int) -> list[float]:
    """
    Generate slow-decay bit-shift sequence with special case for first coefficient.

    This produces: [1, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125, ...]
    which corresponds to: [2^0, 2^-1, 2^-1, 2^-2, 2^-2, 2^-3, 2^-3, ...]

    The first coefficient (2^0) does not repeat. Starting from the second position,
    each power of 2 appears twice before shifting to the next power.

    The slower decay may better approximate GL coefficients for certain alpha values.

    Args:
        history_length: Number of values to generate

    Returns:
        List of slow-decay bit-shift values where powers of 2 repeat (except first)
    """
    values = []
    for k in range(history_length):
        if k == 0:
            # First coefficient: 2^0 = 1.0, no repeat
            shift_amount = 0
        else:
            # For k >= 1: shift by (k+1)//2, so k=1,2 -> shift 1; k=3,4 -> shift 2, etc.
            shift_amount = (k + 1) // 2
        values.append(1.0 / (2**shift_amount))
    return values


def get_custom_bitshift_sequence(
    history_length: int, decay_rate: int = 3
) -> list[float]:
    """
    Generate custom bit-shift sequence with specific repetition pattern.

    Pattern:
    - 2^0 once (k=0)
    - 2^-1 once (k=1)
    - skip 2^-2
    - 2^-3 once (k=2)
    - 2^-4 once (k=3)
    - 2^-5 decay_rate times (e.g. k=4,5,6 for decay_rate = 3)
    - 2^-6 decay_rate times (e.g. k=7,8,9 for decay_rate = 3)
    - 2^-7 decay_rate times (e.g. k=10,11,12 for decay _rate = 3)
    - and so on...

    This produces: [1.0, 0.5, 0.125, 0.0625, 0.03125, 0.03125, 0.03125, ...]

    Args:
        history_length: Number of values to generate

    Returns:
        List of custom bit-shift values following the specified pattern
    """
    values = []
    shift_sequence = [0, 1, 3, 4]  # Initial shifts: 2^0, 2^-1, 2^-3, 2^-4

    # Build the full sequence by adding shifts >= 5 three times each
    shift = 5
    while len(shift_sequence) < history_length:
        shift_sequence.extend([shift] * decay_rate)
        shift += 1

    # Convert shifts to actual values
    for k in range(history_length):
        shift_amount = shift_sequence[k]
        values.append(1.0 / (2**shift_amount))

    return values


# TODO: Add plot showing divergence between GL coefficients and bit-shift values
# over time, as history_length increases to values as high as 256-512.

# TODO: Add method for finding closest bit-shift approximation to a given GL coefficient.
# This may be used to identify a particular bit shift sequence for approximating GL coefficients
# with a given alpha value. It may also be worthwhile testing an implementation with a simple
# incrementing bit shift, as compared to a more complex custom sequence.


def main():
    """Compare GL coefficients with bit-shift approximations."""
    # Example usage
    alpha = 0.5
    history_length = 32

    print(f"Comparing GL coefficients (alpha={alpha}) with bit-shift approximations")
    print(f"History length: {history_length}\n")

    gl_coeffs = get_gl_coefficients(alpha, history_length)
    bitshift_vals = get_bitshift_sequence(history_length)
    slow_decay_vals = get_slow_decay_bitshift_sequence(history_length)
    # TODO: Compare different decay rates for the custom sequence
    custom_vals = get_custom_bitshift_sequence(history_length, 4)

    # Regular bit-shift comparison
    print("=" * 85)
    print("REGULAR BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-2, ...)")
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Bit-Shift':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        bitshift_val = bitshift_vals[k]
        diff = gl_mag - bitshift_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {bitshift_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )

    # Slow-decay bit-shift comparison
    print("\n" + "=" * 85)
    print(
        "SLOW-DECAY BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-1, 2^-2, 2^-2, 2^-3, 2^-3, ...)"
    )
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Slow-Decay':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        slow_decay_val = slow_decay_vals[k]
        diff = gl_mag - slow_decay_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {slow_decay_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )

    # Custom bit-shift comparison
    print("\n" + "=" * 85)
    print("CUSTOM BIT-SHIFT COMPARISON (2^0, 2^-1, 2^-3, 2^-4, 2^-5×3, 2^-6×3, ...)")
    print("=" * 85)
    print(
        f"{'Index':<6} {'|GL Coeff|':<20} {'Custom':<20} {'Difference':<20} "
        f"{'Rel Error %':<15}"
    )
    print("-" * 85)

    for k in range(history_length):
        gl_mag = abs(gl_coeffs[k])
        custom_val = custom_vals[k]
        diff = gl_mag - custom_val
        # Calculate relative error as percentage
        rel_error = (diff / gl_mag * 100) if gl_mag != 0 else 0
        print(
            f"{k:<6} {gl_mag:<20.10f} {custom_val:<20.10f} {diff:<20.10f} "
            f"{rel_error:<15.2f}"
        )


if __name__ == "__main__":
    main()
