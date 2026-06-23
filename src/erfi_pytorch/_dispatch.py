from __future__ import annotations

import sys

import torch

from ._torch_impl import erfi_torch


TRITON_MIN_ELEMENTS = 65_536

_TRITON_AVAILABLE = False
_triton_erfi = None

if sys.platform != "win32":
    try:
        from ._triton import erfi_triton as _triton_erfi

        _TRITON_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


def triton_available() -> bool:
    """Return whether the optional Triton backend imported successfully."""
    return _TRITON_AVAILABLE


def _should_use_triton(x: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and x.numel() >= TRITON_MIN_ELEMENTS
    )


def dispatch_erfi(x: torch.Tensor) -> torch.Tensor:
    if _should_use_triton(x):
        assert _triton_erfi is not None
        return _triton_erfi(x)
    return erfi_torch(x)

