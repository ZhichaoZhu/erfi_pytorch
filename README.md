# erfi-pytorch

`erfi-pytorch` provides a forward-only imaginary error function for real
PyTorch tensors:

```python
import torch
from erfi_pytorch import erfi

x = torch.linspace(-4, 4, 1_000_000, device="cuda")
y = erfi(x)
```

The package supports `torch.float32` and `torch.float64`, preserves tensor
shape/dtype/device, and works with `torch.compile(fullgraph=True)`.

## Backends

- **Pure PyTorch:** portable CPU and CUDA implementation.
- **Triton:** fused path for contiguous NVIDIA CUDA tensors with at least
  65,536 elements, when Triton is available.

Windows and systems without Triton automatically use the pure PyTorch path.
No CUDA toolkit or native compiler is required.

## Installation

```bash
pip install .
```

For development and reference tests:

```bash
pip install -e ".[test]"
```

On Linux, install the optional Triton dependency if it is not already
provided by your PyTorch installation:

```bash
pip install -e ".[test,triton]"
```

## Numerical method

For real `x`, the implementation uses

```text
erfi(x) = exp(x^2) Im(w(x)),
```

where `w` is the Faddeeva function. `Im(w(x))` is evaluated with a Taylor
polynomial near zero and a 100-interval table of low-degree polynomial
approximations elsewhere. Near floating-point overflow, the final magnitude
is reconstructed in the log domain so representable results are not lost to
premature overflow in `exp(x^2)`.

The polynomial coefficients originate from Steven G. Johnson's
MIT-licensed Faddeeva implementation. The original license notice is retained
in [`third_party/faddeeva`](third_party/faddeeva/).

The detailed implementation notes are in
[`docs/faddeeva.md`](docs/faddeeva.md).

## License

This project is released under the MIT License. The vendored Faddeeva sources
and material derived from them retain the original Copyright (c) 2012
Massachusetts Institute of Technology attribution and MIT license notice.

## Limitations

- Inputs must be real `torch.float32` or `torch.float64` tensors.
- This release is forward-only. `requires_grad=True` raises an error.
- Triton acceleration currently targets NVIDIA CUDA.

## Benchmark

```bash
python benchmarks/benchmark_erfi.py --dtype float32
```

The benchmark covers powers of two from `2^10` through `2^24` and reports
eager PyTorch, compiled PyTorch, eager dispatch, and compiled dispatch.
Before timing, it compares the operator against `scipy.special.erfi` and
reports maximum absolute error, maximum and mean relative error, and infinity
mismatches. Use `--precision-elements` to change the comparison sample count
or `--skip-precision` to run timing only.
