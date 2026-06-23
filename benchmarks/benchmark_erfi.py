from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from scipy import special

from erfi_pytorch import erfi
from erfi_pytorch._torch_impl import erfi_torch


def _time_cuda(function, values: torch.Tensor, repeats: int = 50) -> float:
    for _ in range(10):
        function(values)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(values)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6


def _make_values(
    count: int,
    dtype: torch.dtype,
    distribution: str,
    device: str,
) -> torch.Tensor:
    if distribution == "central":
        bounds = (-4.0, 4.0)
    elif distribution == "wide":
        bounds = (-10.0, 10.0)
    else:
        edge = 9.4 if dtype == torch.float32 else 26.6
        bounds = (edge - 0.5, edge + 0.5)
    return torch.linspace(*bounds, count, dtype=dtype, device=device)


def _precision_against_scipy(
    dtype: torch.dtype,
    distribution: str,
    count: int,
) -> None:
    values = _make_values(count, dtype, distribution, "cuda")
    actual = erfi(values).cpu()

    # Evaluate SciPy in float64, then round to the operator's target dtype.
    scipy_input = values.cpu().to(torch.float64).numpy()
    reference64 = special.erfi(scipy_input)
    reference = torch.from_numpy(np.asarray(reference64)).to(dtype)

    finite = torch.isfinite(reference)
    absolute_error = torch.abs(actual[finite] - reference[finite])
    relative_error = absolute_error / torch.abs(reference[finite]).clamp_min(
        torch.finfo(dtype).tiny
    )
    infinity_mismatches = torch.count_nonzero(
        torch.isinf(actual) != torch.isinf(reference)
    ).item()

    print("# precision against scipy.special.erfi")
    print(f"# samples={count}")
    print(f"# finite_samples={finite.sum().item()}")
    print(f"# max_abs_error={absolute_error.max().item():.9e}")
    print(f"# max_rel_error={relative_error.max().item():.9e}")
    print(f"# mean_rel_error={relative_error.mean().item():.9e}")
    print(f"# infinity_mismatches={infinity_mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--distribution",
        choices=("central", "wide", "overflow"),
        default="central",
    )
    parser.add_argument(
        "--precision-elements",
        type=int,
        default=65_536,
        help="number of samples used for the SciPy precision comparison",
    )
    parser.add_argument(
        "--skip-precision",
        action="store_true",
        help="skip the scipy.special.erfi precision comparison",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = getattr(torch, args.dtype)
    if not args.skip_precision:
        _precision_against_scipy(
            dtype,
            args.distribution,
            args.precision_elements,
        )

    compiled_torch = torch.compile(erfi_torch, fullgraph=True)
    compiled_dispatch = torch.compile(erfi, fullgraph=True)
    compile_enabled = True

    probe = torch.zeros(1024, dtype=dtype, device="cuda")
    try:
        compiled_torch(probe)
        compiled_dispatch(probe)
        torch.cuda.synchronize()
    except Exception as error:
        compile_enabled = False
        print(f"# compiled modes unavailable: {type(error).__name__}: {error}")

    print("elements,eager_torch_us,compiled_torch_us,eager_dispatch_us,compiled_dispatch_us")
    for exponent in range(10, 25):
        count = 1 << exponent
        values = _make_values(count, dtype, args.distribution, "cuda")

        timings = [_time_cuda(erfi_torch, values)]
        timings.append(
            _time_cuda(compiled_torch, values) if compile_enabled else float("nan")
        )
        timings.append(_time_cuda(erfi, values))
        timings.append(
            _time_cuda(compiled_dispatch, values)
            if compile_enabled
            else float("nan")
        )
        print(f"{count}," + ",".join(f"{timing:.3f}" for timing in timings))


if __name__ == "__main__":
    main()
