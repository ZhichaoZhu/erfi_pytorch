# Faddeeva.cc and Faddeeva.hh

## Overview

This code implements a family of related special functions for real and
complex double-precision arguments:

- the Faddeeva function `w(z)`
- the scaled complementary error function `erfcx(z)`
- the error function `erf(z)`
- the imaginary error function `erfi(z)`
- the complementary error function `erfc(z)`
- Dawson's integral

The central idea is to implement the Faddeeva function accurately over the
complex plane, then derive most of the other functions from mathematical
identities. Special real-argument implementations and local Taylor expansions
are used where they are faster or avoid numerical cancellation.

The header places the C++ API in namespace `Faddeeva` and supplies overloads
for `double` and `std::complex<double>`. Complex functions accept an optional
relative-error target:

```cpp
std::complex<double> Faddeeva::w(std::complex<double> z,
                                 double relerr = 0);
```

A non-positive `relerr` means approximately machine precision. Internally,
the value is clamped to the range from `DBL_EPSILON` to `0.1`.

## Mathematical Relationships

The primary function is

```text
w(z) = exp(-z^2) erfc(-i z).
```

The other important identities are

```text
erfcx(z) = exp(z^2) erfc(z) = w(i z)
erfi(z)  = -i erf(i z)
D(z)     = sqrt(pi)/2 exp(-z^2) erfi(z)
```

where `D(z)` denotes Dawson's integral.

For real `x`, the Faddeeva function has the especially useful form

```text
w(x) = exp(-x^2) + i * 2/sqrt(pi) * D(x).
```

Therefore,

```text
w_im(x) = Im(w(x)) = 2/sqrt(pi) * D(x)
erfi(x) = exp(x^2) * w_im(x).
```

This last identity is the basis of the optimized real `erfi` implementation.

## Structure of the Source

### Portability layer

The opening macros allow the same source to compile as either C++ using
`std::complex<double>` or C using C99 complex numbers. They also normalize
construction of complex values, infinity, NaN, `isnan`, `isinf`, and
`copysign` across older compilers.

In C++, macros such as

```cpp
#define FADDEEVA(name) Faddeeva::name
#define C(a,b) std::complex<double>(a,b)
```

turn the shared implementation into the namespace API declared by
`Faddeeva.hh`.

### The `w(z)` numerical engine

`w(z)` first handles the coordinate axes:

- For purely imaginary `z = i y`, `w(i y) = erfcx(y)`.
- For real `z = x`, it returns
  `exp(-x*x) + i*w_im(x)`.

For general complex arguments, it chooses between two main algorithms.

1. **Large arguments:** a continued-fraction expansion is used because it is
   fast and asymptotically accurate. For extremely large values this reduces
   to one or two terms, such as
   `w(z) ~= i / (sqrt(pi) z)`.
2. **Smaller arguments:** a convergent summation based on Algorithm 916 is
   used. Precomputed exponential coefficients accelerate the normal
   machine-precision path.

The branch boundary is not just a simple `|z|` test. The code avoids the
continued fraction near parts of the real axis where the real component of
`w(z)` would have poor relative accuracy.

For arguments in the lower half-plane, it uses the symmetry

```text
w(z) = 2 exp(-z^2) - w(-z).
```

Several expressions are algebraically rearranged to avoid intermediate
overflow. For example, the real part of `-z^2` is computed as

```cpp
(y - x) * (x + y)
```

instead of directly evaluating `y*y - x*x`.

### Real helper functions

`erfcx(double)` and `w_im(double)` use similar three-region strategies:

- continued fractions for large magnitude arguments;
- piecewise Chebyshev polynomial approximations for the middle range;
- Taylor expansions near zero.

The Chebyshev lookup tables dominate the size of `Faddeeva.cc`. They are
precomputed polynomial coefficients, not separate conceptual algorithms.

## Imaginary Error Function

The imaginary error function is defined by

```text
erfi(z) = -i erf(i z)
        = 2/sqrt(pi) integral_0^z exp(t^2) dt.
```

Unlike `erf(x)`, which approaches `+1` or `-1` on the real axis, `erfi(x)`
grows approximately like

```text
erfi(x) ~ exp(x^2) / (sqrt(pi) x).
```

That rapid growth is why overflow handling matters.

### Complex implementation

The complex implementation is:

```cpp
cmplx FADDEEVA(erfi)(cmplx z, double relerr)
{
  cmplx e = FADDEEVA(erf)(C(-cimag(z),creal(z)), relerr);
  return C(cimag(e), -creal(e));
}
```

If `z = x + i y`, then

```text
i z = -y + i x.
```

That explains the input rotation:

```cpp
C(-imag(z), real(z))
```

If the result is `e = a + i b`, multiplying it by `-i` gives

```text
-i(a + i b) = b - i a.
```

That explains the output rotation:

```cpp
C(imag(e), -real(e))
```

Thus the complex `erfi` routine contains no independent approximation.
It delegates to the complex `erf` implementation and performs two exact
coordinate rotations. This also means that `relerr` is passed directly to
the underlying Faddeeva calculation.

### Why complex `erf` is the hard part

The delegated `erf(z)` routine derives its normal path from `w(z)`, but it
does not blindly evaluate one formula everywhere. It has several stability
branches:

- real and imaginary axes are handled directly;
- positive and negative real parts use different symmetry formulas;
- a Taylor series is used near `z = 0`;
- a second expansion is used near the imaginary axis when the direct
  expression would subtract nearly equal values.

The imaginary-axis identity used there is

```text
erf(i y) = i erfi(y)
         = i exp(y^2) w_im(y).
```

This is the `taylor_erfi` branch in `erf(z)`. Despite its label, it is a local
expansion of `erf(x+i y)` around the imaginary axis, using the real `erfi(y)`
value as its base term.

### Real implementation

The optimized real overload is:

```cpp
double FADDEEVA_RE(erfi)(double x)
{
  return x*x > 720 ? (x > 0 ? Inf : -Inf)
    : exp(x*x) * FADDEEVA(w_im)(x);
}
```

To derive this formula, begin with the definition of the Faddeeva function
for real `x`:

```text
w(x) = exp(-x^2) erfc(-i x).
```

Using

```text
erfc(-i x) = 1 - erf(-i x)
           = 1 + i erfi(x),
```

we obtain

```text
w(x) = exp(-x^2) [1 + i erfi(x)].
```

Taking its imaginary part gives

```text
Im(w(x)) = exp(-x^2) erfi(x),
```

and hence

```text
erfi(x) = exp(x^2) Im(w(x))
        = exp(x^2) w_im(x).
```

It is also useful to express this through Dawson's integral:

```text
w_im(x) = 2 D(x) / sqrt(pi)
erfi(x) = 2 exp(x^2) D(x) / sqrt(pi).
```

This specialized route is faster than constructing a complex value, calling
complex `erf`, and rotating the answer.

The `x*x > 720` check is a coarse explicit overflow guard. The largest finite
exponential has an exponent near 709.78, and `erfi` itself also eventually
exceeds the double range, so some smaller inputs can naturally overflow
during the multiplication. Above the guard, the code directly returns the
mathematically correct signed infinity. This is especially important for
infinite or enormous inputs, where evaluating the identity mechanically
could produce the indeterminate form `Inf * 0` because `w_im(x)` tends to
zero as `|x|` grows.

The result is odd because `w_im` is odd:

```text
erfi(-x) = -erfi(x).
```

### How `w_im(x)` is computed

`w_im(x)` is a scaled Dawson integral:

```text
w_im(x) = 2 D(x) / sqrt(pi).
```

The actual numerical work for real `erfi` therefore occurs inside `w_im`.
It divides the real axis into three numerical regions.

1. **Small `|x|`**, approximately `|x| <= 0.0309`:

   ```text
   w_im(x) = 2/sqrt(pi)
             * (x - 2x^3/3 + 4x^5/15 - 8x^7/105 + 16x^9/945).
   ```

   The source evaluates this in Horner form:

   ```cpp
   double x2 = x*x;
   return x * (1.1283791670955125739
               - x2 * (0.75225277806367504925
                       - x2 * (0.30090111122547001970
                               - x2 * (0.085971746064420005629
                                       - x2 * 0.016931216931216931217))));
   ```

   Horner form uses fewer multiplications and generally accumulates less
   rounding error. The explicit leading factor `x` naturally preserves odd
   symmetry. Near zero,

   ```text
   w_im(x) ~= 2x/sqrt(pi)
   erfi(x) ~= 2x/sqrt(pi),
   ```

   because `exp(x^2) ~= 1`.

2. **Moderate `|x|`**, up to 45:

   The code maps positive `x` using

   ```text
   y = 1 / (1 + x)
   y100 = 100 y.
   ```

   The call is:

   ```cpp
   return w_im_y100(100/(1+x), x);
   ```

   The integer part of `y100` selects one of 100 intervals in a `switch`.
   Each case maps its local interval approximately to `[-1,1]`, for example:

   ```cpp
   double t = 2*y100 - 1;
   ```

   It then evaluates a low-degree, Chebyshev-derived polynomial in `t`, again
   in Horner form. The large coefficient table in `Faddeeva.cc` is therefore
   a lookup table of fitted polynomials. At runtime the process is simply:

   ```text
   transform x -> select interval -> evaluate polynomial.
   ```

   For a moderate negative input, the code evaluates the positive argument
   and negates it:

   ```cpp
   return -w_im_y100(100/(1-x), -x);
   ```

   This explicitly applies `w_im(-x) = -w_im(x)`.

3. **Large `|x|`**, `45 < |x| <= 5e7`:

   A five-term continued-fraction approximation is simplified into a rational
   expression:

   ```cpp
   return ispi * ((x*x) * (x*x-4.5) + 2)
        / (x * ((x*x) * (x*x-5) + 3.75));
   ```

   Here `ispi = 1/sqrt(pi)`. This is an algebraically simplified form of

   ```text
                  1/sqrt(pi)
   ------------------------------------------------
   x - (1/2)/(x - 1/(x - (3/2)/(x - 2/x))).
   ```

   For `|x| > 5e7`, only the leading asymptotic term is needed:

   ```text
   w_im(x) ~= 1 / (sqrt(pi) x).
   ```

Combining this asymptotic form with the real `erfi` formula gives the expected
large-argument behavior:

```text
erfi(x) ~= exp(x^2) / (sqrt(pi) x).
```

The complete real execution path is therefore:

```text
erfi(x)
  |
  +-- x^2 > 720? --> return signed infinity
  |
  +-- compute w_im(x)
  |     +-- tiny x: Taylor polynomial
  |     +-- moderate x: piecewise Chebyshev polynomials
  |     +-- large x: continued-fraction rational approximation
  |     +-- enormous x: 1/(sqrt(pi) x)
  |
  +-- return exp(x^2) * w_im(x)
```

## Other Exported Functions

- `erfcx(z)` is a direct rotation into `w`: `erfcx(z) = w(i z)`.
- `erf(z)` uses `w` plus symmetry and Taylor expansions to avoid
  cancellation.
- `erfc(z)` uses direct formulas rather than always computing `1-erf(z)`,
  because that subtraction would lose precision when `erfc(z)` is tiny.
- `Dawson(z)` uses `w`, axis-specific formulas, continued fractions, and
  Taylor expansions. For real input it is simply
  `sqrt(pi)/2 * w_im(x)`.

## Numerical Design Lessons

The implementation is less about finding one universal formula and more
about choosing an equivalent formula that is well-conditioned in each
region:

- use asymptotic continued fractions when arguments are large;
- use convergent sums or fitted polynomials in the middle range;
- use Taylor expansions near zeros and cancellation points;
- exploit oddness, conjugation, rotations, and reflection identities;
- special-case axes, infinities, NaNs, signed zero, overflow, and underflow;
- avoid forming huge and tiny intermediate quantities whose final product
  would be representable.

For `erfi` specifically, the important implementation chain is:

```text
complex erfi(z)
    -> rotate z to i*z
    -> stabilized complex erf(i*z)
    -> rotate result by -i

real erfi(x)
    -> specialized w_im(x)
    -> multiply by exp(x^2)
    -> return signed infinity before overflow becomes indeterminate
```

The complex path maximizes reuse of the robust `erf` implementation, while
the real path maximizes speed and preserves accuracy through the
Faddeeva/Dawson relationship.
