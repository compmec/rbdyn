# -*- coding: utf-8 -*-
import sympy as sp
import numpy as np
import sys
sys.path.append("../src/")
from lagrange import get_FunctionsExp

if __name__ == "__main__":
    t = sp.symbols("t")

    theta = sp.Function("theta")(t)
    beta = sp.Function("beta")(t)
    p = sp.Function("p")(t)
    U = [theta, beta, p]

    theta_ = sp.diff(theta, t)
    beta_ = sp.diff(beta, t)
    p_ = sp.diff(p, t)
    U_ = [theta_, beta_, p_]

    # Problem Data
    I0 = 1.3e-2  # kg*m^2
    m2 = 0.44  # kg
    m3 = 0.470  # kg
    a = 0.045  # m
    l = 0.145  # m
    grav = 0  # m/s^2
    P = (0, 0, 0)

    g = [a * sp.sin(theta) + l * sp.sin(beta),
         p - a * sp.cos(theta) - l * sp.cos(beta)]

    t0 = 0
    U0 = [0, 0, 0]  # Initial positionating of each variable
    U0_ = [1, 0, 0]  # Initial speed of each variable

    M11 = I0 + m2 * a**2
    M22 = m2 * l**2 / 3
    M33 = m3
    M12 = m2 * a * l / 2

    Ec = 0
    Ep = 0

    Ec += M11 * theta_**2 / 2
    Ec += M22 * beta_**2 / 2
    Ec += M33 * p**2 / 2
    Ec += M12 * theta_ * beta_ * sp.cos(theta - beta)

    Ep += m2 * a * sp.sin(theta) / 2

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
