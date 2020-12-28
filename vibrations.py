
import sympy as sp


def linearize(M, U, U0):
    n = len(U)
    if isinstance(M, sp.Expr):
        for i in range(n):
            # dM = sp.diff(M, U[i])
            # ddM = sp.diff(dM, U[i])
            M = M.subs(U[i], U0[i])  # + dM.subs(U[i], U0[i]) * \
            # (U[i] - U0[i]) # + ddM.subs(U[i], U0[i]) * (U[i] - U0[i])**2 / 2
    elif type(M) == list:
        for i in range(n):
            M[i] = linearize(M[i], U, U0)
    elif type(M) == int or type(M) == float:
        return 0
    else:
        raise Exception("Not expected get here. Type = " + str(type(M)))
    return M


def get_vibrationmodules(K, M):
    """
    We suppose that K and M doesn't depend of the state vector U
    """
    # caracteristic polynominal:
    w2 = sp.symbols("w2")
    K = sp.Matrix(K)
    M = sp.Matrix(M)
    Minv = M.inv()
    Mat = Minv * K
    # polynominal = Mat.charpoly(w2)
    # print("polynominal = " + str(polynominal))
    # co = polynominal.all_coeffs()
    # print("co = " + str(co))
    eigen = Mat.eigenvals()
    return eigen
