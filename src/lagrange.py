
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def get_M(Ec, U_):
    n = len(U_)

    M = []
    for i in range(n):
        M.append([])
        for j in range(n):
            M[i].append(0)

    for i in range(n):
        dEcdui_ = sp.diff(Ec, U_[i])
        for j in range(i, n):
            mij = sp.diff(dEcdui_, U_[j])
            M[i][j] = mij
            M[j][i] = mij

    return M


def get_K(Ep, U):
    return get_M(Ep, U)


def transform_MtoN(M, U):
    n = len(U)

    N = []
    for i in range(n):
        N.append([])
        for j in range(n):
            N[i].append([])
            for k in range(n):
                N[i][j].append(0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                P1 = sp.diff(M[k][i], U[j])
                P2 = sp.diff(M[i][j], U[k])
                N[k][i][j] = P1 - (P2 / 2)

    return N


def transform_EptoG(Ep, U):
    n = len(U)
    # if 1:
    if isinstance(Ep, (int, float)):
        G = sp.Array([0 for i in range(len(U))])
    else:
        G = sp.derive_by_array(Ep, U)
        G = G.subs([U], np.zeros(n))
    return G


def get_FunctionsExp(Ec, Ep, U, g, P):
    t = sp.symbols("t")
    Ec = sp.sympify(Ec)
    Ep = sp.sympify(Ep)
    U = sp.Array(U)
    U_ = U.diff(t)
    M = get_M(Ec, U_)
    N = transform_MtoN(M, U)
    K = get_K(Ep, U)
    G = transform_EptoG(Ep, U)

    g = sp.Array(g)
    g_ = sp.derive_by_array(g, U)
    g__ = sp.derive_by_array(g_, U)

    Mexp = get_Mexp(M, g_, U)
    Fexp = get_Fexp(N, K, G, g, g_, g__, U, U_, P)
    Mexp = sp.Matrix(Mexp)

    return Mexp, Fexp


def get_Rexp(Mexp, Fexp):
    # Lexp, Dexp = LDLFactorization(Mexp)
    t = sp.symbols("t")
    detMexp = 1 #  Mexp.det()

    Mexp_inv = Mexp.inv()
    Mexp_inv = Mexp_inv * detMexp
    Mexp_inv = sp.expand(Mexp_inv)
    # Mexp_inv = sp.simplify(Mexp_inv)
    Rexp = sp.tensorproduct(Mexp_inv, Fexp)
    Rexp = sp.tensorcontraction(Rexp, (1, 2))

    Rexp /= detMexp
    return Rexp


def get_Mexp(M, g_, U):
    n, a = g_.shape

    Mexp = []
    for i in range(a + n):
        Mexp.append([])
        for j in range(a + n):
            Mexp[i].append(0)

    for i in range(n):
        for j in range(n):
            Mexp[i][j] = M[i][j]
    for i in range(a):
        for j in range(n):
            Mexp[i + n][j] = g_[j][i]
            Mexp[j][i + n] = g_[j][i]
    Mexp = sp.Array(Mexp)
    return Mexp


def get_Fexp(N, K, G, g, g_, g__, U, U_, P):

    alpha = 5
    # alpha = sp.symbols("alpha")

    n, a = g_.shape
    N = sp.Array(N)
    K = sp.Array(K)
    G = sp.Array(G)
    # g = sp.Array(g)

    # FUU_ = sp.lambdify([U], FUU_, "numpy")
    # FU_ = sp.lambdify([U], FU_, "numpy")
    # FU = sp.lambdify([U], FU, "numpy")
    # FC = sp.lambdify([U], FC, "numpy")

    U = sp.Array(U)
    U_ = sp.Array(U_)
    UU_ = sp.tensorproduct(U_, U_)

    FUU_ = []
    FU_ = []
    FU = []
    FC = []
    for i in range(n + a):
        FC.append(0)
        FU.append([])
        FU_.append([])
        FUU_.append([])
        for j in range(n):
            FU[i].append(0)
            FU_[i].append(0)
            FUU_[i].append([])
            for k in range(n):
                FUU_[i][j].append(0)

    for i in range(n):
        FC[i] = G[i]
        FC[i] += P[i]
        for j in range(n):
            FU[i][j] = K[i][j]
            for k in range(n):
                FUU_[i][j][k] = -N[i][j][k]

    for i in range(a):
        FC[i + n] = -alpha**2 * g[i]
        for j in range(n):
            FU_[i + n][j] = -2 * alpha * g_[j][i]
            for k in range(n):
                FUU_[i + n][j][k] = -g__[j][k][i]

    FUU_ = sp.Array(FUU_)
    FU_ = sp.Array(FU_)
    FU = sp.Array(FU)
    FC = sp.Array(FC)

    # print("######## Shapes ##########")
    # print("FUU_ = " + str(FUU_.shape))
    # print(" FU_ = " + str(FU_.shape))
    # print("  FU = " + str(FU.shape))
    # print("  FC = " + str(FC.shape))
    # print("  U_ = " + str(U_.shape))
    # print("   U = " + str(U.shape))

    Pt1 = sp.tensorproduct(FUU_, UU_)
    # print("Pt1_shape = " + str(Pt1.shape))
    Pt1 = sp.tensorcontraction(Pt1, (1, 3))
    # print("Pt1_shape = " + str(Pt1.shape))
    Pt1 = sp.tensorcontraction(Pt1, (1, 2))
    # print("Pt1_shape = " + str(Pt1.shape))

    Pt2 = sp.tensorproduct(FU_, U_)
    # print("Pt2_shape = " + str(Pt2.shape))
    Pt2 = sp.tensorcontraction(Pt2, (1, 2))
    # print("Pt2_shape = " + str(Pt2.shape))

    Pt3 = sp.tensorproduct(FU, U)
    # print("Pt3_shape = " + str(Pt3.shape))
    Pt3 = sp.tensorcontraction(Pt3, (1, 2))
    # print("Pt3_shape = " + str(Pt3.shape))

    Pt4 = FC

    Pt1 = sp.simplify(Pt1)
    Pt2 = sp.simplify(Pt2)
    Pt3 = sp.simplify(Pt3)
    Pt4 = sp.simplify(Pt4)
    soma = Pt1 + Pt2 + Pt3 + Pt4
    return sp.simplify(soma)


def LDLFactorization(A):
    n, n = A.shape

    L = []
    D = []
    for i in range(n):
        D.append(0)
        L.append([])
        for j in range(n):
            L[i].append(0)
        L[i][i] = 1

    D[0] = A[0, 0]
    for i in range(n):
        for j in range(i):
            soma = 0
            for k in range(j):
                soma += L[i][k] * L[j][k] * D[k]
            L[i][j] = (A[i][j] - soma) / D[j]
            L[i][j] = sp.expand(L[i][j])
            L[i][j] = sp.simplify(L[i][j])
        soma = 0
        for j in range(i):
            soma += L[i][j]**2 * D[j]
        D[i] = A[i, i] - soma
        D[i] = sp.expand(D[i])
        D[i] = sp.simplify(D[i])
    return L, D
