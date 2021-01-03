# -*- coding: utf-8 -*-
import sympy as sp
import numpy as np
import sys
sys.path.append("../src/")
from lagrange import get_FunctionsExp

if __name__ == "__main__":
    t = sp.symbols("t")

    x = sp.Function("x")(t)
    y = sp.Function("y")(t)
    q = sp.Function("q")(t)
    U = [x, y, q]

    x_ = sp.diff(x, t)
    y_ = sp.diff(y, t)
    q_ = sp.diff(q, t)
    U_ = [x_, y_, q_]

    # Problem Data:
    I0 = 1.3e-2  # kg*m^2
    m2 = 0.44  # kg
    m3 = 0.470  # kg
    a = 0.045  # m
    l = 0.145  # m
    grav = 0  # m/s^2
    # Force applied in the system
    # is a function of t, U and U_
    P = (0, 0, 0)
    # P(t, U, U_)

    g = (x**2 + y**2 - a**2,
         y**2 + q**2 - l**2)

    t0 = 0
    U0 = (a, 0, l)
    U0_ = (0, 1, 0)

    m1 = I0 / a**2
    M11 = m1 + m2 + m3
    M22 = m1 + m2 / 3
    M33 = m2 / 3 + m3
    M13 = m2 / 2 + m3

    Ep = 0
    Ec = 0

    # M11, M22, M33, M13 = sp.symbols("M11 M22 M33 M13")
    Ec += M11 * x_**2 / 2
    Ec += M22 * y_**2 / 2
    Ec += M33 * q_**2 / 2
    Ec += M13 * x_ * q_

    Ep += m2 * y * grav / 2

    Mexp, Fexp = get_FunctionsExp(Ec, Ep, U, g, P)
    print("Mexp = ")
    print(Mexp)
    print("Fexp = ")
    print(Fexp)
    Mexp = sp.lambdify([U], Mexp, "numpy")
    Fexp = sp.lambdify([t, U, U_], Fexp, "numpy")

    Mexp0 = Mexp(U0)
    Fexp0 = Fexp(t0, U0, U0_)
    Mexp0 = np.array(Mexp0)
    Fexp0 = np.array(Fexp0)
    Uexp0 = np.linalg.solve(Mexp0, Fexp0)
    U0__ = Uexp0[:len(U)]

    print("Mexp0 = " + str(type(Mexp0)) + " - shape = " + str(Mexp0.shape))
    print(Mexp0)
    print("Fexp0 = " + str(type(Fexp0)) + " - shape = " + str(Fexp0.shape))
    print(Fexp0)
    print("Uexp0 = " + str(type(Uexp0)) + " - shape = " + str(Uexp0.shape))
    print(Uexp0)
    print("U0__ = " + str(type(U0__)) + " - shape = " + str(U0__.shape))
    print(U0__)
