
import sympy as sp
import numpy as np
import time

# def f(t, w, Mexp, Fexp):
#     n = len(w) // 2
#     U0 = w[:n]
#     U0_ = w[n:]
#     Mexp0 = Mexp(U0)
#     Mexp0 = np.array(Mexp0)
#     Fexp0 = Fexp(t, U0, U0_)
#     Fexp0 = np.array(Fexp0)
#     Rexp0 = np.linalg.solve(Mexp0, Fexp0)
#     U0__ = Rexp0[:n]
#     w1 = np.zeros(2 * n)
#     w1[:n] = U0_
#     w1[n:] = U0__
#     return w1


def f(t, w, Rexp):
    n = len(w) // 2
    U0 = w[:n]
    U0_ = w[n:]
    Rexp0 = Rexp(t, U0, U0_)
    U0__ = Rexp0[:n]
    w1 = np.zeros(2 * n)
    w1[:n] = U0_
    w1[n:] = U0__
    return w1


def RungeKutta(U0, U0_, Ttotal, divisions, Rexp):
    # U0 = np.array(U0, dtype=np.float64)
    # U0_ = np.array(U0, dtype=np.float64)

    h = Ttotal / divisions  # The step size
    n = len(U0)

    t = np.linspace(0, Ttotal, divisions + 1)
    u = np.zeros((divisions + 1, n))
    u_ = np.zeros((divisions + 1, n))
    u__ = np.zeros((divisions + 1, n))

    # print("____________")
    # print(type(U0))
    # print(type(U0_))

    w0 = np.concatenate((U0, U0_))
    w0_ = f(t, w0, Rexp)
    begin = time.process_time()
    u[0] = U0
    u_[0] = U0_
    u__[0] = w0_[n:]
    for i in range(divisions):
        k1 = h * w0_
        k2 = h * f(t[i] + h / 2, w0 + h * k1 / 2, Rexp)
        k3 = h * f(t[i] + h / 2, w0 + h * k2 / 2, Rexp)
        k4 = h * f(t[i] + h, w0 + h * k3, Rexp)
        dw = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # print("type = " + str(type(dw)))
        # print("type = " + str(type(w0)))
        # print("dw = " + str(dw))
        # print("w0 = " + str(w0))
        w0 += dw
        w0_ = f(t[i + 1], w0, Rexp)
        if i == divisions // 999:
            end = time.process_time()
            DeltaT = (end - begin) * 1000 / divisions
            TotalT = divisions * DeltaT
            print("Estimated time to solve Runge Kutta: " + str(TotalT) + " s")
        u[i + 1] = w0[:n]
        u_[i + 1] = w0[n:]
        u__[i + 1] = w0_[n:]

    return t, u, u_, u__
