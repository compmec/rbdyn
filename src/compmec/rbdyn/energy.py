import numpy as np
import sympy as sp
from compmec.rbdyn.__validation__ import Validation_Energy, Validation_EnergyMatrix
from compmec.rbdyn.__classes__ import EnergyBaseClass
from compmec.rbdyn.variable import Variable, VariableList


def MatrixSymbolicFunction1(name, n):
    M = np.zeros(n, dtype="object")  
    for i in range(n):
        M[i] = sp.Function(f"{name}{i}")
    return sp.MutableDenseNDimArray(M)

def MatrixSymbolicFunction2(name, n, m):
    M = np.zeros((n, n), dtype="object")
    for i in range(n):
        for j in range(m):
            M[i, j] = sp.Function(f"{name}{i}{j}")
    return sp.MutableDenseNDimArray(M)


def KineticEnergy(mass, velocity):
    """
    Receives mass as an scalar, velocity as a 3D vector
    And returns a Energy instance, with Kinetic Energy calculated
    """
    return Energy.Kinetic(mass, velocity)

class EnergyMatrix(EnergyBaseClass):
    validaiton = Validation_EnergyMatrix
    
    @property
    def X(self):
        return self._X

    @property
    def dX(self):
        if self.X is None:
            return None
        return sp.Array([x.dt for x in self.X])

    @property
    def n(self):
        if self.X is None:
            return 0
        return len(self.X)
    
    
    @property
    def M(self):
        return self._M

    @property
    def V(self):
        return self._V
    
    @property
    def K(self):
        return self._K    
    
    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._B    
    
    @property
    def C(self):
        return self._C

    @X.setter
    def X(self, value):
        self.validation.Xsetter(self, value)
        if value is None:
            self._X = None
        elif len(value) == 0:
            self._X = None
        else:
            value = VariableList(value)
            self._X = sp.Array(value)
    
    @M.setter
    def M(self, value):
        self.validation.Msetter(self, value)
        if value is None:
            if self.X is None:
                self._M = None
            else:
                self._M = sp.Array(np.zeros((self.n, self.n), dtype="int32"))
        else:
            self._M = sp.Array(value)
            self._M = sp.nsimplify(self._M, tolerance=1e-6, rational=True)

    
    @V.setter
    def V(self, value):
        self.validation.Vsetter(self, value)
        if value is None:
            if self.X is None:
                self._V = None
            else:
                self._V = sp.Array(np.zeros((self.n, self.n), dtype="int32"))
        else:
            self._V = sp.Array(value)
            self._V = sp.nsimplify(self._V, tolerance=1e-6, rational=True)

    @K.setter
    def K(self, value):
        self.validation.Ksetter(self, value)
        if value is None:
            if self.X is None:
                self._K = None
            else:
                self._K = sp.Array(np.zeros((self.n, self.n), dtype="int32"))
        else:
            self._K = sp.Array(value)
            self._K = sp.nsimplify(self._K, tolerance=1e-6, rational=True)

    @A.setter
    def A(self, value):
        self.validation.Asetter(self, value)
        if value is None:
            if self.X is None:
                self._A = None
            else:
                self._A = sp.Array(np.zeros(self.n, dtype="int32"))
        else:
            self._A = sp.Array(value)
            self._A = sp.nsimplify(self._A, tolerance=1e-6, rational=True)

    @B.setter
    def B(self, value):
        self.validation.Bsetter(self, value)
        if value is None:
            if self.X is None:
                self._B = None
            else:
                self._B = sp.Array(np.zeros(self.n, dtype="int32"))
        else:
            self._B = sp.Array(value)
            self._B = sp.nsimplify(self._B, tolerance=1e-6, rational=True)

    
    @C.setter
    def C(self, value):
        self.validation.Csetter(self, value)
        if value is None:
            self._C = 0
        else:
            self._C = sp.nsimplify(value, tolerance=1e-6, rational=True)
            self._C = sp.sympify(self._C)


class Energy(EnergyMatrix, EnergyBaseClass):
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
    validation = Validation_Energy

    def __init__(self, expression):
        self.validation.init(self, expression)
        self.__init(expression)

    def __init(self, expression):
        expression = sp.expand(sp.sympify(expression))
        self.X = VariableList.fromexpression(expression)
        self.M = None
        self.V = None
        self.K = None
        self.A = None
        self.B = None
        self.C = None
        self.__init__transformExpression(expression)
        

        

    @classmethod
    def frommatrix(cls, *, X, M=None, V=None, K=None, A=None, B=None, C=None):
        cls.validation.frommatrix(cls, X, M, V, K, A, B, C)
        return cls.__frommatrix(X, M, V, K, A, B, C)

    @classmethod
    def __frommatrix(cls, X, M, V, K, A, B, C):
        self = object.__new__(cls)
        self.X = X
        self.M = M
        self.V = V
        self.K = K
        self.A = A
        self.B = B
        self.C = C
        return self

    @classmethod
    def Kinetic(cls, mass, velocity):
        cls.validation.Kinetic(cls, mass, velocity)
        return cls.__Kinetic(mass, velocity)

    @classmethod
    def __Kinetic(cls, mass, velocity):
        listvars = []

        velocity = sp.Array(velocity)
        X = VariableList.fromexpression(velocity)
        if len(X) == 0:
            return cls.__frommatrix(X=None, M=None, V=None, K=None, A=None, B=None, C=None)
        
        dX = sp.Array([x.dt for x in X])
        v = sp.diff(velocity, dX)
        v = np.array(v)
        M = mass * np.dot(v, v.T)
        M = sp.Array(M)
        
        return cls.__frommatrix(X=X, M=M,V=None, K=None, A=None,B=None,C=None)

        

    def __init__transformExpression(self, expression):
        if self.n:
            self.__compute_expression(expression)
        else:
            self.C = expression

    def __compute_expression(self, expression):
        A = sp.MutableDenseNDimArray(np.zeros(self.n))
        B = sp.MutableDenseNDimArray(np.zeros(self.n))
        M = sp.MutableDenseNDimArray(np.zeros((self.n, self.n)))
        V = sp.MutableDenseNDimArray(np.zeros((self.n, self.n)))
        K = sp.MutableDenseNDimArray(np.zeros((self.n, self.n)))
        C = sp.Function("C")(*tuple(self.X))

        all_variables = [C]
        equations = []
        for i in range(self.n):
            A[(i,)] = sp.Function(f"A{i}")(*tuple(self.X))
            B[(i,)] = sp.Function(f"B{i}")(*tuple(self.X))
            all_variables.append(A[(i,)])
            all_variables.append(B[(i,)])
            for j in range(self.n):
                M[i, j] = sp.Function(f"M{i}{j}")(*tuple(self.X))
                V[i, j] = sp.Function(f"V{i}{j}")(*tuple(self.X))
                K[i, j] = sp.Function(f"K{i}{j}")(*tuple(self.X))
                all_variables.append(M[i, j])
                all_variables.append(V[i, j])
                all_variables.append(K[i, j])
        for i in range(self.n):
            for j in range(i+1, self.n):
                equations.append(M[i, j] - M[j, i])
                equations.append(K[i, j] - K[j, i])

        Esup = C
        for i in range(self.n):
            Esup += A[(i,)]*self.dX[i]
            Esup += B[(i,)]*self.X[i]
            for j in range(self.n):
                Esup += M[i, j]*self.dX[i]*self.dX[j]
                Esup += V[i, j]*self.dX[i]*self.X[j]
                Esup += K[i, j]*self.X[i]*self.X[j]
        poly = sp.Poly(Esup-expression, list(self.X)+list(self.dX))
        equations += poly.coeffs()
        solution = sp.solve(equations, all_variables)
        if solution == []:
            raise ValueError("Could not find solution to get the expression")
        C = C.subs(solution.items())
        for var, val in solution.items():
            # C = C.subs(var, val)
            for i in range(self.n):
                A[(i,)] = A[(i,)].subs(var, val)
                B[(i,)] = B[(i,)].subs(var, val)
                for j in range(self.n):
                    M[i, j] = M[i, j].subs(var, val)
                    V[i, j] = V[i, j].subs(var, val)
                    K[i, j] = K[i, j].subs(var, val)

        self.M = sp.Array(2*M)
        self.V = sp.Array(V)
        self.K = sp.Array(2*K)
        self.A = sp.Array(A)
        self.B = sp.Array(B)
        self.C = C




    def __str__(self):
        return str(self.expr())

    def __repr__(self):
        return str(self.expr())

    def expr(self):
        result = sp.sympify(0)
        result += self.expr_C()
        if self.X is None:
            return result
        result += self.expr_AdX()
        result += self.expr_BX()
        result += self.expr_dXMdX()
        result += self.expr_dXVX()
        result += self.expr_XKX()
        result = sp.expand(result)
        result = sp.simplify(result)

        return result

    def expr_dXMdX(self):
        if self.n == 0:
            return 0
        mult = np.dot(self.dX, np.dot(self.M, self.dX))
        if mult is None:
            return 0
        return sp.expand(sp.simplify(mult)) / 2
        
    def expr_XKX(self):
        mult = np.dot(self.X, np.dot(self.K, self.X))
        if mult is None:
            return 0
        return sp.expand(sp.simplify(mult)) / 2

    def expr_dXVX(self):
        mult = np.dot(self.dX, np.dot(self.V, self.X))
        if mult is None:
            return 0
        return sp.expand(sp.simplify(mult))
        
    def expr_AdX(self):
        mult = np.dot(self.A, self.dX)
        if mult is None:
            return 0        
        return sp.expand(sp.simplify(mult))

    def expr_BX(self):
        mult = np.dot(self.B, self.X)
        if mult is None:
            return 0
        return sp.expand(sp.simplify(mult))
        
    def expr_C(self):
        return self.C

    def __add__(self, value):
        if not isinstance(value, Energy):
            value = Energy(value)
        
        newX = VariableList(self.X) + VariableList(value.X)
        
        indexs1 = []
        indexs2 = []
        for i, nx in enumerate(newX):
            if nx in self.X:
                indexs1.append(i)
            if nx in value.X:
                indexs2.append(i)

        newn = len(newX)
        M = np.zeros((newn, newn), dtype="object")
        V = np.zeros((newn, newn), dtype="object")
        K = np.zeros((newn, newn), dtype="object")
        A = np.zeros((newn), dtype="object")
        B = np.zeros((newn), dtype="object")
        C = self.C + value.C
        for indexs, energ in [(indexs1, self), (indexs2, value)]:
            for ind_i, new_i in enumerate(indexs):
                A[new_i] += energ.A[ind_i]
                B[new_i] += energ.B[ind_i]
                for ind_j, new_j in enumerate(indexs1):
                    M[new_i, new_j] += energ.M[ind_i, ind_j]
                    V[new_i, new_j] += energ.V[ind_i, ind_j]
                    K[new_i, new_j] += energ.K[ind_i, ind_j]
        M = sp.Array(M)
        V = sp.Array(V)
        K = sp.Array(K)
        A = sp.Array(A)
        B = sp.Array(B)
        C = sp.expand(C)
        C = sp.simplify(C)

        return Energy.frommatrix(X=newX, M=M, V=V, K=K, A=A, B=B, C=C)


    def __sub__(self, value):
        if not isinstance(value, Energy):
            value = Energy(value)
        return self + (-value)


    def __neg__(self):
        return Energy.frommatrix(X=self.X,
                      M=np.copy(-self.M),
                      V=np.copy(-self.V),
                      K=np.copy(-self.K),
                      A=np.copy(-self.A),
                      B=np.copy(-self.B),
                      C=-self.C)


    def __eq__(self, value):
        if value == 0:
            if self.n:
                zerosM = sp.zeros(self.n)
                zerosA = sp.Matrix(np.zeros(self.n))
                if sp.Matrix(self.M) != zerosM:
                    return False
                if sp.Matrix(self.V) != zerosM:
                    return False
                if sp.Matrix(self.K) != zerosM:
                    return False
                if sp.Matrix(self.A) != zerosA:
                    return False
                if sp.Matrix(self.B) != zerosA:
                    return False
            if self.C != 0:
                return False
            return True



        if not isinstance(value, Energy):
            value = Energy(value)
        if self.n != value.n:
            return False
        if self.n == 0:
            return self.C == value.C
        if self.X != value.X:
            return False

        diff = self-value
        return diff == 0

    def __ne__(self, value):
        return not self.__eq__(value)


def isSympyEqual(M1, M2):
    if isinstance(M1, np.ndarray) and isinstance(M2, np.ndarray):
        if M1.shape != M2.shape:
            return False
        
        for i in range(M1.shape[0]):
            if not isSympyEqual(M1[0], M2[0]):
                return False
        return True

    else:
        diff = M1 - M2
        diff = sp.expand(diff)
        diff = sp.simplify(diff)
        return diff == 0