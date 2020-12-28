
import sympy as sp
import numpy as np
import time
import matplotlib.pyplot as plt
from lagrange import get_FunctionsExp, get_Rexp
from solveRungeKutta import RungeKutta


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

    g = [a * sp.sin(theta) + l * sp.sin(beta),
         p - a * sp.cos(theta) - l * sp.cos(beta)]

    t0 = 0
    P = (0, 0, 0)
    Ttotal = 10
    dt = 0.001
    divisions = int((Ttotal / dt)) + 1

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

    Ep += m2 * grav * a * sp.sin(theta) / 2

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
        p0 = x0 + q0
        x0_ = -a * np.sin(theta0) * theta0_
        y0_ = a * np.cos(theta0) * theta0_
        beta0_ = -a * theta0_ * np.cos(theta0) / (l * np.cos(beta0))
        q0_ = -l * np.sin(beta0) * beta0_
        p0_ = x0_ + q0_

        U0 = (theta0, beta0, p0)
        U0_ = (theta0_, beta0_, p0_)
    time3 = time.process_time()
    print("Set initial conditions")
    print("    Time = " + str(time3 - time2) + " s")

    t, u, u_, u__ = RungeKutta(U0, U0_, Ttotal, divisions, Rexp)
    end = time.process_time()
    print("Runge Kutta solved")
    print("    Time = " + str(end - time3) + " s")

    THETAvalues = u[:, 0]
    BETAvalues = u[:, 1]
    Pvalues = u[:, 2]

    plt.figure()
    plt.plot(t, THETAvalues, color="b", label=r"$\theta$")
    plt.plot(t, BETAvalues, color="r", label=r"$\beta$")
    plt.plot(t, Pvalues, color="g", label=r"$p$")
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
    plt.show()
