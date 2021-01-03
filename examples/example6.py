# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
from lagrange import get_FunctionsExp, get_Rexp
from solveRungeKutta import RungeKutta


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
    Ttotal = 10
    dt = 0.001
    divisions = int((Ttotal / dt)) + 1

    # a, l = sp.symbols("a l")

    g = (x**2 + y**2 - a**2,
         y**2 + q**2 - l**2)

    t0 = 0

    m1 = I0 / a**2
    M11 = m1 + m2 + m3
    M22 = m1 + m2 / 3
    M33 = m2 / 3 + m3
    M13 = m2 / 2 + m3

    Ep = 0
    Ec = 0

    M12 = 0
    M23 = 0
    # M12, M13, M23 = sp.symbols("M12 M13 M23")
    # M11, M22, M33 = sp.symbols("M11 M22 M33")
    Ec += M11 * x_**2 / 2
    Ec += M22 * y_**2 / 2
    Ec += M33 * q_**2 / 2
    Ec += M13 * x_ * q_
    Ec += M12 * x_ * y_
    Ec += M23 * y_ * q_

    # Ep += m2 * y * grav / 2

    start = time.process_time()
    # Mexp, Fexp = get_FunctionsExp(Ec, Ep, U, g, P)
    Mexp, Fexp = get_FunctionsExp(Ec, Ep, U, g, P)
    time1 = time.process_time()
    print("Got Mexp and Fexp")
    print("    Time = " + str(time1 - start) + " s")
    Rexp = get_Rexp(Mexp, Fexp)
    Rexp = sp.lambdify([t, U, U_], Rexp, "numpy")
    time2 = time.process_time()
    print("Got Rexp")
    print("    Time = " + str(time2 - time1) + " s")

    if 1:  # Initial Conditions
        theta0 = -np.pi / 2
        theta0_ = 2 * np.pi
        x0 = a * np.cos(theta0)
        y0 = a * np.sin(theta0)
        beta0 = np.arcsin(-y0 / l)
        q0 = l * np.cos(beta0)
        x0_ = -a * np.sin(theta0) * theta0_
        y0_ = a * np.cos(theta0) * theta0_
        beta0_ = -a * theta0_ * np.cos(theta0) / (l * np.cos(beta0))
        q0_ = -l * np.sin(beta0) * beta0_

        U0 = (x0, y0, q0)
        U0_ = (x0_, y0_, q0_)
    time3 = time.process_time()
    print("Set initial conditions")
    print("    Time = " + str(time3 - time2) + " s")

    t, u, u_, u__ = RungeKutta(U0, U0_, Ttotal, divisions, Rexp)
    end = time.process_time()
    print("Runge Kutta solved")
    print("    Time = " + str(end - time3) + " s")

    Xvalues = u[:, 0]
    Yvalues = u[:, 1]
    Qvalues = u[:, 2]

    plt.figure()
    plt.plot(t, Xvalues, color="b", label=r"$x$")
    plt.plot(t, Yvalues, color="r", label=r"$y$")
    plt.plot(t, Qvalues, color="g", label=r"$q$")
    plt.title("Values of solved system")
    plt.legend()

    g_ = []
    for i in range(len(g)):
        g_.append(sp.derive_by_array(g[i], U))
    g = sp.lambdify([U], g, "numpy")
    g_ = sp.lambdify([U], g_, "numpy")

    g1 = np.zeros(divisions + 1)
    g2 = np.zeros(divisions + 1)
    for i in range(divisions + 1):
        gi = g(u[i])
        g1[i] = gi[0]
        g2[i] = gi[1]
    plt.figure()
    plt.plot(t, g1, color="b", label=r"$g_1$")
    plt.plot(t, g2, color="r", label=r"$g_2$")
    plt.legend()
    plt.title("Equality constraints functions")

    g1_ = np.zeros(divisions + 1)
    g2_ = np.zeros(divisions + 1)
    for i in range(divisions + 1):
        gi_ = g_(u[i])
        gi_ = np.dot(gi_, u_[i])
        g1_[i] = gi_[0]
        g2_[i] = gi_[1]
    plt.figure()
    plt.plot(t, g1_, color="b", label=r"$g_{1}'$")
    plt.plot(t, g2_, color="r", label=r"$g_{2}'$")
    plt.legend()
    plt.title("Derivative of equality constraints")
    plt.show()
