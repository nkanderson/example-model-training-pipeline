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
    history_length = 16

    print(f"Comparing GL coefficients (alpha={alpha}) with bit-shift sequence")
    print(f"History length: {history_length}\n")

    gl_coeffs = get_gl_coefficients(alpha, history_length)
    bitshift_vals = get_bitshift_sequence(history_length)

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


if __name__ == "__main__":
    main()
