from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ._coefficients import POLYNOMIAL_WIDTH, W_IM_COEFFICIENTS
from ._coefficients_float64 import (
    POLYNOMIAL_WIDTH_FLOAT64,
    W_IM_COEFFICIENTS_FLOAT64,
)

if TYPE_CHECKING:
    from torch import Tensor


_SMALL_X_MAX = 3.0 / 97.0
_COEFFICIENT_CACHE: dict[tuple[str, int | None, torch.dtype], Tensor] = {}


def _coefficient_tensor(x: Tensor) -> Tensor:
    source = (
        W_IM_COEFFICIENTS_FLOAT64
        if x.dtype == torch.float64
        else W_IM_COEFFICIENTS
    )
    if torch.compiler.is_compiling():
        return torch.tensor(source, dtype=x.dtype, device=x.device)

    key = (x.device.type, x.device.index, x.dtype)
    coefficients = _COEFFICIENT_CACHE.get(key)
    if coefficients is None:
        coefficients = torch.tensor(
            source,
            dtype=x.dtype,
            device=x.device,
        )
        _COEFFICIENT_CACHE[key] = coefficients
    return coefficients


def _w_im_positive(x: Tensor, coefficients: Tensor) -> Tensor:
    """Evaluate w_im(x) for finite, nonnegative x in the useful erfi range."""
    x2 = x * x
    taylor = x * (
        1.1283791670955125739
        - x2
        * (
            0.75225277806367504925
            - x2
            * (
                0.30090111122547001970
                - x2
                * (
                    0.085971746064420005629
                    - x2 * 0.016931216931216931217
                )
            )
        )
    )

    y100 = 100.0 / (1.0 + x)
    index = torch.clamp(y100.to(torch.int64), 0, 96)
    t = 2.0 * y100 - (2.0 * index.to(x.dtype) + 1.0)
    selected = coefficients[index]

    width = (
        POLYNOMIAL_WIDTH_FLOAT64
        if x.dtype == torch.float64
        else POLYNOMIAL_WIDTH
    )
    polynomial = selected[..., width - 1]
    for column in range(width - 2, -1, -1):
        polynomial = selected[..., column] + polynomial * t

    return torch.where(x <= _SMALL_X_MAX, taylor, polynomial)


def erfi_torch(x: Tensor) -> Tensor:
    """Pure PyTorch implementation used as the portable reference backend."""
    finfo = torch.finfo(x.dtype)
    log_max = math.log(finfo.max)
    evaluation_limit = math.sqrt(log_max) + 1.0

    absolute = torch.abs(x)
    nan_mask = torch.isnan(x)

    # Clamp before squaring and table lookup. Values beyond this point are
    # guaranteed to overflow erfi for the target dtype.
    safe_absolute = torch.clamp(
        torch.where(nan_mask, torch.zeros_like(absolute), absolute),
        max=evaluation_limit,
    )
    coefficients = _coefficient_tensor(x)
    w_im = _w_im_positive(safe_absolute, coefficients)
    square = safe_absolute * safe_absolute
    log_magnitude = square + torch.log(w_im)
    overflow = (absolute > evaluation_limit) | (log_magnitude > log_max)

    # The direct formula is fastest away from overflow. Near the limit,
    # reconstruct relative to finfo.max so exp(square) never overflows before
    # multiplication by the small w_im factor.
    direct = torch.exp(square) * w_im
    delta = torch.clamp(log_magnitude - log_max, max=0.0)
    scaled = finfo.max * torch.exp(delta)
    magnitude = torch.where(square < log_max - 2.0, direct, scaled)
    magnitude = torch.where(
        overflow,
        torch.full_like(magnitude, float("inf")),
        magnitude,
    )

    result = torch.copysign(magnitude, x)
    return torch.where(nan_mask, x, result)
