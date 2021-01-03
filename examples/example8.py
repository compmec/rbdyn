
import sympy as sp
import numpy as np
import sys
sys.path.append("../src/")
from lagrange import get_M, get_K
from vibrations import linearize, get_vibrationmodules

"""
Exemple of vibrations
"""


if __name__ == "__main__":
    t = sp.symbols("t")

    a, l0, MB, m = sp.symbols("a l0 MB m", positive=True, real=True)
    grav = sp.symbols("g")  # gravity
    k = sp.symbols("k")  # spring constant

    b = sp.Function("b")(t)
    theta1 = sp.Function("theta1")(t)
    theta2 = sp.Function("theta2")(t)
    theta3 = sp.Function("theta3")(t)
    U = [b, theta1, theta2, theta3]

    g = []

    b_ = sp.diff(b, t)
    theta1_ = sp.diff(theta1, t)
    theta2_ = sp.diff(theta2, t)
    theta3_ = sp.diff(theta3, t)
    U_ = [b_, theta1_, theta2_, theta3_]

    Ec = 0
    Ep = 0

    Ec += (1 / 2) * b_**2 * (MB + 3 * m)
    Ec += (1 / 2) * theta1_**2 * m * a**2
    Ec += (1 / 2) * theta2_**2 * m * a**2
    Ec += (1 / 2) * theta3_**2 * m * a**2
    Ec += m * a * b_ * theta1_
    Ec += m * a * b_ * theta2_
    Ec += m * a * b_ * theta3_

    s1 = sp.sin(theta1)
    c1 = sp.cos(theta1)
    s2 = sp.sin(theta2)
    c2 = sp.cos(theta2)
    s3 = sp.sin(theta3)
    c3 = sp.cos(theta3)

    dx12 = sp.sqrt(l0**2 + 2 * l0 * a * (s2 - s1)
                   + a ** 2 * ((s2 - s1)**2 + (c2 - c1)**2)) - l0
    dx23 = sp.sqrt(l0**2 + 2 * l0 * a * (s3 - s2)
                   + a ** 2 * ((s3 - s2)**2 + (c3 - c2)**2)) - l0
    Ep += (1 / 2) * k * b**2
    Ep += (1 / 2) * k * dx12**2
    Ep += (1 / 2) * k * dx23**2

    Ep += m * grav * a * (3 - (c1 + c2 + c3))

    Et = Ec + Ep

    M = get_M(Ec, U_)
    K = get_K(Ep, U)

    U0 = [0, 0, 0, 0]

    M = linearize(M, U, U0)
    Ec = np.dot(U_, np.dot(M, U_)) / 2

    K = linearize(K, U, U0)
    Ep = np.dot(U, np.dot(K, U)) / 2

    print("M = ")
    print(M)
    print("K = ")
    print(K)

    eigen = get_vibrationmodules(K, M)
    print("eigen = ")
    print(eigen)
