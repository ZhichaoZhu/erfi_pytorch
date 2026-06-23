"""GPU-accelerated imaginary error function for real PyTorch tensors."""

from __future__ import annotations

import torch

from ._dispatch import dispatch_erfi, triton_available

__all__ = ["erfi", "triton_available"]
__version__ = "0.1.0"


def erfi(x: torch.Tensor) -> torch.Tensor:
    """Compute the imaginary error function for a real tensor.

    The first release is intentionally forward-only. Inputs must be float32
    or float64 tensors without gradient tracking.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"erfi expects a torch.Tensor, got {type(x).__name__}")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "erfi supports only torch.float32 and torch.float64 tensors, "
            f"got {x.dtype}"
        )
    if x.requires_grad:
        raise RuntimeError(
            "erfi_pytorch.erfi is forward-only and does not support "
            "requires_grad=True"
        )
    return dispatch_erfi(x)
