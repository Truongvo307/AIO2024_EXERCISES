# Description: This module contains functions that approximate the sin, cos, sinh, and cosh functions using the Taylor series expansion.
def approx_sin(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    sin_approx = 0
    for n in range(n):
        term = ((-1)**n) * (x**(2*n+1)) / factorial(2*n+1)
        sin_approx += term
    print(sin_approx)


def approx_cos(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    cos_approx = 0
    for n in range(n):
        term = ((-1)**n) * (x**(2*n)) / factorial(2*n)
        cos_approx += term
    print(cos_approx)


def approx_sinh(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    sinh_approx = 0
    for n in range(n):
        term = (x**(2*n+1)) / factorial(2*n+1)
        sinh_approx += term
    print(sinh_approx)


def approx_cosh(x, n):
    if not (0 <= x <= 2 * np.pi):
        print("x must be a radian value")
        return
    if (n <= 0):
        print("n must be greater than 0")
        return
    cosh_approx = 0
    for n in range(n):
        term = (x**(2*n)) / factorial(2*n)
        cosh_approx += term
    print(cosh_approx)
