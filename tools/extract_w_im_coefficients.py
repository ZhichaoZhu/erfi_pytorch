"""Extract the real w_im polynomial table from the vendored Faddeeva.cc.

This is a development helper. The generated module is committed so package
users do not need the C++ source at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "faddeeva" / "Faddeeva.cc"
OUTPUT = ROOT / "src" / "erfi_pytorch" / "_coefficients.py"
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?")


def extract() -> list[list[str]]:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("static double w_im_y100")
    end = text.index("case 97: case 98:", start)
    table_text = text[start:end]

    rows: list[list[str]] = []
    for case in range(97):
        match = re.search(
            rf"case {case}: \{{.*?return (.*?);",
            table_text,
            flags=re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"could not find w_im case {case}")
        expression = " ".join(match.group(1).split())
        coefficients = NUMBER.findall(expression)
        rows.append(coefficients)

    return rows


def render(rows: list[list[str]]) -> str:
    width = max(map(len, rows))
    padded = [row + ["0.0"] * (width - len(row)) for row in rows]
    lines = [
        '"""Polynomial coefficients for the real Faddeeva w_im function.',
        "",
        "Generated from third_party/faddeeva/Faddeeva.cc by",
        "tools/extract_w_im_coefficients.py.",
        "The coefficients originate from Steven G. Johnson's MIT-licensed",
        "Faddeeva implementation; see licenses/FADDEEVA-MIT.txt.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"POLYNOMIAL_WIDTH = {width}",
        "W_IM_COEFFICIENTS = (",
    ]
    for row in padded:
        lines.append("    (" + ", ".join(row) + "),")
    lines.extend([")", ""])
    return "\n".join(lines)


def main() -> None:
    rows = extract()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(rows), encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} x {max(map(len, rows))} coefficients to {OUTPUT}")


if __name__ == "__main__":
    main()
