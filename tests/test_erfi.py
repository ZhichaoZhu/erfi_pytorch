from __future__ import annotations

import math

import mpmath
import pytest
import torch
from scipy import special

from erfi_pytorch import erfi, triton_available
from erfi_pytorch._dispatch import TRITON_MIN_ELEMENTS
from erfi_pytorch._torch_impl import erfi_torch


mpmath.mp.dps = 80


def _reference(values: torch.Tensor) -> torch.Tensor:
    result = []
    for value in values.detach().cpu().reshape(-1).tolist():
        if math.isnan(value):
            result.append(float("nan"))
        elif math.isinf(value):
            result.append(math.copysign(float("inf"), value))
        else:
            result.append(float(mpmath.erfi(value)))
    return torch.tensor(result, dtype=values.dtype).reshape(values.shape)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-6, 5e-7),
        (torch.float64, 5e-13, 5e-15),
    ],
)
def test_reference_accuracy(dtype: torch.dtype, rtol: float, atol: float) -> None:
    central = torch.linspace(-8.0, 8.0, 401, dtype=torch.float64)
    tiny = torch.tensor(
        [-1e-12, -1e-8, -3.0 / 97.0, -0.0, 0.0, 3.0 / 97.0, 1e-8, 1e-12],
        dtype=torch.float64,
    )
    values = torch.cat((central, tiny)).to(dtype)
    actual = erfi(values)
    expected = _reference(values)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        rtol=rtol,
        atol=atol,
    )
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_boundaries(dtype: torch.dtype) -> None:
    boundaries = []
    for interval in range(3, 97):
        boundary = 100.0 / (interval + 1.0) - 1.0
        boundaries.extend(
            [
                math.nextafter(boundary, -math.inf),
                boundary,
                math.nextafter(boundary, math.inf),
            ]
        )
    values = torch.tensor(boundaries, dtype=dtype)
    actual = erfi(values)
    expected = _reference(values)
    finite = torch.isfinite(expected)
    tolerances = (5e-6, 5e-7) if dtype == torch.float32 else (5e-13, 5e-15)
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        rtol=tolerances[0],
        atol=tolerances[1],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_values_and_symmetry(dtype: torch.dtype) -> None:
    values = torch.tensor(
        [-float("inf"), -10.0, -1.0, -0.0, 0.0, 1.0, 10.0, float("inf"), float("nan")],
        dtype=dtype,
    )
    result = erfi(values)

    assert torch.isneginf(result[0])
    assert torch.isposinf(result[-2])
    assert torch.isnan(result[-1])
    assert torch.signbit(result[3])
    assert not torch.signbit(result[4])
    torch.testing.assert_close(result[1:4], -torch.flip(result[4:7], dims=(0,)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_maximizes_finite_range(dtype: torch.dtype) -> None:
    finfo = torch.finfo(dtype)
    log_max = math.log(finfo.max)

    # Search with high precision for the positive erfi overflow boundary.
    low = mpmath.mpf("1")
    high = mpmath.mpf("30")
    for _ in range(160):
        middle = (low + high) / 2
        if mpmath.log(mpmath.erfi(middle)) <= log_max:
            low = middle
        else:
            high = middle

    below = torch.tensor(float(low), dtype=dtype)
    above = torch.tensor(float(high), dtype=dtype)
    below_result = erfi(below)
    above_result = erfi(above)
    assert torch.isfinite(below_result)
    assert torch.isinf(above_result) or above == below


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        erfi(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float32 and torch.float64"):
        erfi(torch.tensor([1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="forward-only"):
        erfi(torch.tensor([1.0], requires_grad=True))


def test_noncontiguous_tensor() -> None:
    source = torch.linspace(-3, 3, 24, dtype=torch.float64).reshape(4, 6)
    values = source[:, ::2]
    assert not values.is_contiguous()
    torch.testing.assert_close(erfi(values), _reference(values), rtol=5e-13, atol=5e-15)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 8e-6, 5e-7),
        (torch.float64, 5e-13, 5e-15),
    ],
)
def test_scipy_special_erfi_precision(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    values = torch.linspace(-10, 10, 8193, dtype=dtype)
    scipy_values = special.erfi(values.to(torch.float64).numpy())
    expected = torch.from_numpy(scipy_values).to(dtype)
    actual = erfi(values)
    finite = torch.isfinite(expected)

    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        rtol=rtol,
        atol=atol,
    )
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-6, 5e-7),
        (torch.float64, 5e-13, 5e-15),
    ],
)
def test_cuda_pytorch_fallback(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    values = torch.linspace(-8, 8, 4097, dtype=dtype, device="cuda")
    actual = erfi(values).cpu()
    expected = _reference(values)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        rtol=rtol,
        atol=atol,
    )
    assert actual.shape == values.shape
    assert actual.dtype == dtype


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_torch_compile_fullgraph() -> None:
    compiled = torch.compile(erfi, backend="eager", fullgraph=True)
    values = torch.linspace(-4, 4, 257, dtype=torch.float64)
    torch.testing.assert_close(compiled(values), erfi(values), rtol=5e-13, atol=5e-15)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA compiler backend unavailable",
)
def test_torch_compile_inductor_cuda() -> None:
    compiled = torch.compile(erfi, fullgraph=True)
    values = torch.linspace(-4, 4, 65_536, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(compiled(values), erfi(values), rtol=5e-6, atol=5e-7)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA Triton backend unavailable",
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-6, 5e-7),
        (torch.float64, 5e-13, 5e-15),
    ],
)
def test_triton_matches_torch(dtype: torch.dtype, rtol: float, atol: float) -> None:
    values = torch.linspace(
        -10,
        10,
        TRITON_MIN_ELEMENTS,
        dtype=dtype,
        device="cuda",
    )
    actual = erfi(values)
    expected = erfi_torch(values)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
