"""
Fractional-Order Leaky Integrate-and-Fire (FLIF) Neuron

This module implements a fractional-order LIF neuron using the Grunwald-Letnikov (GL)
approximation for fractional derivatives. The neuron maintains a history buffer
for past membrane potential values.

Inherits from snnTorch Leaky neuron, overriding only the state function.
"""

import torch
import snntorch as snn
from collections import OrderedDict
from utils import compute_gl_coefficients


# Module-level cache for GL coefficients
# Key: (alpha, history_length) -> Value: torch.Tensor of coefficients (CPU, float64)
_GL_COEFF_CACHE = OrderedDict()
_CACHE_MAX_SIZE = 100  # Limit cache size to prevent memory issues


def get_gl_coefficients(alpha: float, history_length: int) -> torch.Tensor:
    """
    Get GL coefficients with caching.

    Wraps compute_gl_coefficients from utils with caching to avoid recomputation
    for the same (alpha, history_length) pairs.

    Args:
        alpha: Fractional order
        history_length: Number of coefficients

    Returns:
        Cached or newly computed GL coefficients
    """
    # Round alpha to avoid floating-point key issues
    alpha_rounded = round(alpha, 12)
    cache_key = (alpha_rounded, history_length)

    # Check cache
    if cache_key in _GL_COEFF_CACHE:
        # Move to end (LRU)
        _GL_COEFF_CACHE.move_to_end(cache_key)
        return _GL_COEFF_CACHE[cache_key]

    # Compute coefficients
    coeffs = compute_gl_coefficients(alpha, history_length)

    # Store in cache
    _GL_COEFF_CACHE[cache_key] = coeffs

    # Limit cache size (LRU eviction)
    if len(_GL_COEFF_CACHE) > _CACHE_MAX_SIZE:
        _GL_COEFF_CACHE.popitem(last=False)  # Remove oldest

    return coeffs


class FractionalLIF(snn.Leaky):
    """
    Fractional-Order Leaky Integrate-and-Fire (FLIF) neuron.

    Extends snnTorch Leaky neuron with fractional-order dynamics using the
    Grunwald-Letnikov approximation. Maintains a history buffer of past
    membrane potentials for fractional derivative computation.

    Inherits all snnTorch Leaky functionality (init_hidden, reset_hidden, etc.)
    and overrides only the state update to use fractional dynamics.

    Args:
        beta: Membrane potential decay rate for compatibility. Not used in fractional dynamics,
              but passed to parent. Use lam instead for fractional leak.
        alpha: Fractional order for derivative (0 < alpha <= 1). Default: 0.5
        lam: Leakage parameter in fractional equation (>= 0). Default: 0.111 (matches beta=0.9)
        history_length: Number of past values for GL approximation. Default: 256
        dt: Discrete timestep for GL approximation. Default: 1.0
        threshold: Spike threshold. Default: 1.0
        spike_grad: Optional spike gradient surrogate function.
        init_hidden: If True, instantiates state variables as instance variables. Default: False
        output: If True with init_hidden=True, states are returned. Default: False
        **kwargs: Additional arguments passed to snnTorch Leaky parent class

    Forward API (inherited from snnTorch):
        forward(input, mem) -> (spike, mem) or spike (if init_hidden=True and output=False)
    """

    def __init__(
        self,
        beta: float = 0.9,  # For compatibility, not used in fractional dynamics
        alpha: float = 0.5,
        lam: float = 0.111,  # Default matches beta=0.9: lam = (1-0.9)/0.9
        history_length: int = 256,
        dt: float = 1.0,
        threshold: float = 1.0,
        spike_grad=None,
        init_hidden: bool = False,
        output: bool = False,
        **kwargs,
    ):
        # Initialize parent snnTorch Leaky neuron
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            output=output,
            **kwargs,
        )

        # Validate fractional parameters
        assert 0 < alpha <= 1.0, "Fractional order alpha should be in (0, 1.0]"
        assert lam >= 0, "Leakage parameter lam must be non-negative"
        assert history_length > 0, "History length must be positive"

        # Store fractional-specific parameters
        self.alpha = alpha
        self.lam = lam
        self.history_length = history_length
        self.dt = dt

        # Precompute GL coefficients (will be moved to device on first forward)
        self._gl_coeffs = get_gl_coefficients(alpha, history_length)
        self._coeffs_device = None  # Track which device coeffs are on

        # Initialize history buffer if init_hidden=True
        if self.init_hidden:
            hist = torch.zeros(0)
            self.register_buffer("hist", hist, persistent=False)

    def _get_coeffs(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get GL coefficients on the correct device and dtype."""
        # Convert and cache device-specific coefficients
        if self._coeffs_device != device:
            self._gl_coeffs = self._gl_coeffs.to(device=device, dtype=dtype)
            self._coeffs_device = device
        return self._gl_coeffs.to(dtype=dtype)

    def _base_state_function(self, input_):
        """
        Override parent's state function to use fractional-order dynamics.

        Instead of: mem = beta * mem + input
        We use: mem = (input - C * Σ g_k*mem[t-k]) / (C + λ)

        This is called by parent's forward() after handling reset logic.
        """
        device = input_.device
        dtype = input_.dtype

        # Get GL coefficients
        coeffs = self._get_coeffs(device, dtype)

        # Initialize or reshape history buffer if needed
        if not hasattr(self, "hist") or self.hist.shape[0] == 0:
            hist_shape = (self.history_length,) + input_.shape
            self.hist = torch.zeros(hist_shape, device=device, dtype=dtype)
        elif self.hist.shape[1:] != input_.shape:
            # Reshape if batch size changed
            hist_shape = (self.history_length,) + input_.shape
            self.hist = torch.zeros(hist_shape, device=device, dtype=dtype)

        # Compute fractional membrane update
        # V[n] = (I[n] - C * Σ_{k=1}^{H-1} g_k * V[n-k]) / (C + λ)
        C = 1.0 / (self.dt**self.alpha)

        # Extract past coefficients (skip g_0=1 which multiplies V[n])
        coeffs_past = coeffs[1:]

        # History convolution using simple multiply-accumulate
        # coeffs_past shape: (history_length-1,)
        # hist shape: (history_length, batch, features)
        # We want: Σ coeffs_past[k] * hist[k] for k=0..history_length-2
        history_valid = self.hist[: self.history_length - 1]  # (H-1, batch, features)

        # Reshape coeffs for broadcasting: (H-1, 1, 1) -> broadcasts to (H-1, batch, features)
        coeffs_reshaped = coeffs_past.view(-1, 1, 1)

        # Element-wise multiply and sum over time dimension (much faster than einsum)
        history_sum = (coeffs_reshaped * history_valid).sum(dim=0)  # (batch, features)

        # Compute new membrane potential
        numerator = input_ - C * history_sum
        denominator = C + self.lam
        mem_new = numerator / denominator

        # Update history buffer: shift in-place (faster than cat)
        # Roll the history buffer and insert current mem at position 0
        self.hist = torch.roll(self.hist, shifts=1, dims=0)
        self.hist[0] = self.mem

        return mem_new

    @classmethod
    def reset_hidden(cls):
        """
        Reset hidden states for all FractionalLIF instances.

        Overrides parent's reset_hidden to also clear the history buffer,
        which is specific to fractional-order neurons.

        Called automatically by snnTorch's instance tracking system.
        """
        # First call parent's reset_hidden to clear mem (and any other parent state)
        super(FractionalLIF, cls).reset_hidden()

        # Then clear history buffers for all FractionalLIF instances
        for instance in cls.instances:
            if isinstance(instance, FractionalLIF) and hasattr(instance, "hist"):
                instance.hist = torch.zeros_like(
                    instance.hist, device=instance.hist.device
                )

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        parent_repr = super().extra_repr()
        fractional_repr = (
            f"alpha={self.alpha}, lam={self.lam}, "
            f"history_length={self.history_length}, dt={self.dt}"
        )
        return f"{parent_repr}, {fractional_repr}"
