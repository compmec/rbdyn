import numpy as np
import sympy as sp
from compmec.rbdyn.__validation__ import Validation_Energy
from compmec.rbdyn.__classes__ import EnergyClass
from compmec.rbdyn.variable import Variable


def KineticEnergy(mass, velocity):
    """
    Receives mass as an scalar, velocity as a 3D vector
    And returns a Energy instance, with Kinetic Energy calculated
    """
    listvariables = []
    for j in range(3):
        for var in sp.sympify(velocity[j]).atoms(sp.Function):
            if var not in listvariables:
                listvariables.append(var)
    X = Variable.sort(listvariables)
    V = np.zeros((len(X), 3), dtype="object")
    for i, xi in enumerate(X):
        dvar = xi.dt
        for j in range(3):
            V[i, j] = sp.diff(velocity[j], dvar)
    M = mass * np.dot(V, V.T)
    return Energy(X=X, M=M)


class Energy(EnergyClass):
    """
    This class will store the energy
    There are some types of energy: Kinetic and Potential
    All the energy can be expressed like

        E = [dX].T * [M] * [dX]
             + [dX].T * [V] * [X]
             + [X].T * [K] * [X]
             + [A] * [dX]
             + [B] * [X]
             + C

    The linear terms are stored in vectors [A] and [B]
    The quadratic terms are stored in vectors [M], [V] and [K]
    A constant term is stored in C

    If there are any non-linear terms, it will be inside the 
        [M], [V], [K], [A], [B] and C
    """

    def __init__(self, *args, **kwargs):
        self._X = None
        self._dX = None
        self._M = None
        self._K = None
        self._V = None
        self._A = None
        self._B = None
        self._C = 0
        if len(args) == 1 and len(kwargs) == 0:
            # We expect an expression
            expression = args[0]
            self.__init__transformExpression(expression)
        if len(kwargs) > 0:
            if not "X" in kwargs:
                raise ValueError("You must pass X when you use kwargs")
            self.X = kwargs["X"]
            if "M" in kwargs:
                self.M = kwargs["M"]
            if "V" in kwargs:
                self.V = kwargs["V"]
            if "K" in kwargs:
                self.K = kwargs["K"]
            if "A" in kwargs:
                self.A = kwargs["A"]
            if "B" in kwargs:
                self.B = kwargs["B"]
            if "C" in kwargs:
                self.C = kwargs["C"]

    @property
    def X(self):
        if self._X is None:
            raise AttributeError("You must initialize X")
        return self._X

    @property
    def dX(self):
        if self._dX is None:
            raise AttributeError("You must initialize X")
        return self._dX

    @X.setter
    def X(self, value):
        if not isinstance(value, tuple):
            try:
                value = tuple(value)
            except Exception as e:
                error = "X must be a tuple, not '%s'"
                raise TypeError(error % str(type(value)))
        for i, var in enumerate(value):

            if not isinstance(var, sp.core.function.Basic):
                error = "Each one inside X must be a Variable instance, not %s"
                raise TypeError(error % type(var))
        self._X = []
        self._dX = []
        for i, var in enumerate(value):
            self._X.append(var)
            self._dX.append(var.dt)
        self._X = tuple(self._X)
        self._dX = tuple(self._dX)

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        Validation_Energy.M(value, len(self.X))
        self._M = value

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        Validation_Energy.K(value, len(self.X))
        self._K = value

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        Validation_Energy.V(value, len(self.X))
        self._V = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        Validation_Energy.A(value, len(self.X))
        self._A = value

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        Validation_Energy.B(value, len(self.X))
        self._B = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        Validation_Energy.C(value)
        self._C = value

    def __init__transformExpression(self, expression):
        listvariables = sp.sympify(expression).atoms(sp.Function)
        self.X = Variable.sort(listvariables)
        n = len(listvariables)
        if n:
            self.__computeM(expression)
            self.__computeK(expression)
            self.__computeV(expression)
            self.__computeA(expression)
            self.__computeB(expression)
        self.__computeC(expression)

    def __computeM(self, expression):
        n = len(self.X)
        M = np.zeros((n, n), dtype="object")  # If we have kinetic energy
        for i in range(n):
            dE = sp.diff(expression, self.dX[i])
            M[i, i] = sp.diff(dE, self.dX[i])
            for j in range(i + 1, n):
                M[i, j] = sp.diff(dE, self.dX[j]) / 2
                M[j, i] = M[i, j]
        if not np.all(M == 0):
            self._M = M

    def __computeK(self, expression):
        n = len(self.X)
        K = np.zeros((n, n), dtype="object")  # If we have kinetic energy
        for i in range(n):
            dE = sp.diff(expression, self.X[i])
            K[i, i] = sp.diff(dE, self.X[i])
            for j in range(i + 1, n):
                K[i, j] = sp.diff(dE, self.X[j]) / 2
                K[j, i] = K[i, j]
        if not np.all(K == 0):
            self._K = K

    def __computeV(self, expression):
        n = len(self.X)
        V = np.zeros((n, n), dtype="object")  # If we have kinetic energy
        for i, dxi in enumerate(self.dX):
            dE = sp.diff(expression, dxi)
            for j, xj in enumerate(self.X):
                V[i, j] = sp.diff(dE, xj)
        if not np.all(V == 0):
            self._V = V

    def __computeA(self, expression):
        expression -= self.expr_dXMdX()
        expression -= self.expr_dXVX()
        n = len(self.X)
        A = np.zeros((n), dtype="object")  # If we have kinetic energy
        for i in range(n):
            A[i] = sp.diff(expression, self.dX[i])
        if not np.all(A == 0):
            self._A = A

    def __computeB(self, expression):
        expression -= self.expr_dXVX()
        expression -= self.expr_XKX()
        n = len(self.X)
        B = np.zeros((n), dtype="object")  # If we have kinetic energy
        for i in range(n):
            B[i] = sp.diff(expression, self.dX[i])
        if not np.all(B == 0):
            self._B = B

    def __computeC(self, expression):
        expression -= self.expr_dXMdX()
        expression -= self.expr_dXVX()
        expression -= self.expr_XKX()
        expression -= self.expr_AdX()
        expression -= self.expr_BX()
        expression = sp.expand(expression)
        expression = sp.simplify(expression)
        self._C = expression

    def __str__(self):
        return str(self.expr())

    def __repr__(self):
        return str(self.expr())

    def expr(self):
        result = sp.sympify(0)
        result += self.expr_C()
        if self._X is None:
            return result
        if len(self._X) == 0:
            return result
        result += self.expr_AdX()
        result += self.expr_BX()
        result += self.expr_dXMdX()
        result += self.expr_dXVX()
        result += self.expr_XKX()

        return result

    def expr_dXMdX(self):
        if self.M is not None:
            return np.dot(self.dX, np.dot(self.M, self.dX)) / 2
        return 0

    def expr_XKX(self):
        if self.K is not None:
            return np.dot(self.X, np.dot(self.K, self.X)) / 2
        return 0

    def expr_dXVX(self):
        if self.V is not None:
            return np.dot(self.dX, np.dot(self.V, self.X))
        return 0

    def expr_AdX(self):
        if self.A is not None:
            return np.dot(self.A, self.dX)
        return 0

    def expr_BX(self):
        if self.B is not None:
            return np.dot(self.B, self.X)
        return 0

    def expr_C(self):
        return self.C

    def __eq__(self, value):
        if isinstance(value, Energy):
            if not np.all(self.M == value.M):
                return False
            if not np.all(self.V == value.V):
                return False
            if not np.all(self.K == value.K):
                return False
            if not np.all(self.A == value.A):
                return False
            if not np.all(self.B == value.B):
                return False
            if not np.all(self.C == value.C):
                return False
            return True
        else:
            try:
                value = sp.sympify(value)
            except Exception:
                error = "Cannot compare '%s'-type with energy"
                raise Exception(error % type(value))
            expr = self.expr()
            diff = expr - value
            diff = sp.expand(diff)
            diff = sp.simplify(diff)
            diff = sp.nsimplify(diff, tolerance=1e-10, rational=True)
            return diff == 0

    def __ne__(self, value):
        return not self.__eq__(value)
