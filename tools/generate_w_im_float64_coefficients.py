"""Generate higher-accuracy w_im polynomials for the float64 erfi path."""

from __future__ import annotations

from pathlib import Path

import mpmath

from extract_w_im_coefficients import extract


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "erfi_pytorch" / "_coefficients_float64.py"
POLYNOMIAL_COUNT = 15


def _w_im_from_local_t(interval: int, t: mpmath.mpf) -> mpmath.mpf:
    y100 = (t + 2 * interval + 1) / 2
    x = 100 / y100 - 1
    return mpmath.exp(-(x * x)) * mpmath.erfi(x)


def main() -> None:
    mpmath.mp.dps = 100
    original = extract()
    rows: list[list[str]] = []

    for interval in range(97):
        if interval < 3:
            row = original[interval] + ["0.0"] * (
                POLYNOMIAL_COUNT - len(original[interval])
            )
        else:
            descending = mpmath.chebyfit(
                lambda t, interval=interval: _w_im_from_local_t(interval, t),
                [-1, 1],
                POLYNOMIAL_COUNT,
            )
            row = [mpmath.nstr(value, 40) for value in reversed(descending)]
        rows.append(row)

    lines = [
        '"""High-accuracy float64 polynomial coefficients for real w_im.',
        "",
        "Generated with 100-digit mpmath arithmetic by",
        "tools/generate_w_im_float64_coefficients.py.",
        '"""',
        "",
        f"POLYNOMIAL_WIDTH_FLOAT64 = {POLYNOMIAL_COUNT}",
        "W_IM_COEFFICIENTS_FLOAT64 = (",
    ]
    for row in rows:
        lines.append("    (" + ", ".join(row) + "),")
    lines.extend([")", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} x {POLYNOMIAL_COUNT} coefficients to {OUTPUT}")


if __name__ == "__main__":
    main()
