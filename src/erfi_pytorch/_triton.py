from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._torch_impl import _coefficient_tensor


@triton.jit
def _erfi_kernel(
    x_pointer,
    coefficient_pointer,
    output_pointer,
    element_count,
    LOG_MAX: tl.constexpr,
    MAX_VALUE: tl.constexpr,
    EVALUATION_LIMIT: tl.constexpr,
    POLYNOMIAL_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    x = tl.load(x_pointer + offsets, mask=mask)

    nan_mask = x != x
    absolute = tl.abs(x)
    safe_absolute = tl.where(nan_mask, 0.0, tl.minimum(absolute, EVALUATION_LIMIT))
    square = safe_absolute * safe_absolute

    taylor = safe_absolute * (
        1.1283791670955125739
        - square
        * (
            0.75225277806367504925
            - square
            * (
                0.30090111122547001970
                - square
                * (
                    0.085971746064420005629
                    - square * 0.016931216931216931217
                )
            )
        )
    )

    y100 = 100.0 / (1.0 + safe_absolute)
    index = tl.minimum(tl.maximum(y100.to(tl.int32), 0), 96)
    t = 2.0 * y100 - (2.0 * index + 1.0)
    coefficient_offset = index * POLYNOMIAL_WIDTH

    polynomial = tl.load(
        coefficient_pointer + coefficient_offset + POLYNOMIAL_WIDTH - 1
    )
    for reverse_column in tl.static_range(0, POLYNOMIAL_WIDTH - 1):
        column = POLYNOMIAL_WIDTH - 2 - reverse_column
        coefficient = tl.load(coefficient_pointer + coefficient_offset + column)
        polynomial = coefficient + polynomial * t

    w_im = tl.where(safe_absolute <= (3.0 / 97.0), taylor, polynomial)
    log_magnitude = square + tl.log(w_im)
    overflow = (absolute > EVALUATION_LIMIT) | (log_magnitude > LOG_MAX)

    direct = tl.exp(square) * w_im
    delta = tl.minimum(log_magnitude - LOG_MAX, 0.0)
    scaled = MAX_VALUE * tl.exp(delta)
    magnitude = tl.where(square < LOG_MAX - 2.0, direct, scaled)
    magnitude = tl.where(overflow, float("inf"), magnitude)

    signed = tl.where(x < 0.0, -magnitude, magnitude)
    signed = tl.where(x == 0.0, x, signed)
    result = tl.where(nan_mask, x, signed)
    tl.store(output_pointer + offsets, result, mask=mask)


@triton_op("erfi_pytorch::erfi_triton", mutates_args={})
def _erfi_triton_op(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    finfo = torch.finfo(x.dtype)
    log_max = math.log(finfo.max)
    evaluation_limit = math.sqrt(log_max) + 1.0
    polynomial_width = coefficients.shape[1]
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_erfi_kernel)[grid](
        x,
        coefficients,
        output,
        x.numel(),
        LOG_MAX=log_max,
        MAX_VALUE=finfo.max,
        EVALUATION_LIMIT=evaluation_limit,
        POLYNOMIAL_WIDTH=polynomial_width,
        BLOCK_SIZE=256,
    )
    return output


def erfi_triton(x: torch.Tensor) -> torch.Tensor:
    coefficients = _coefficient_tensor(x)
    return _erfi_triton_op(x, coefficients)
