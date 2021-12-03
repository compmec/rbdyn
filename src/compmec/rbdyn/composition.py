import numpy as np
import sympy as sp
from numpy import linalg as la
from compmec.rbdyn.__validation__ import Validation_Compute, IS


def cos(x):
    if isinstance(x, sp.core.basic.Basic):
        c = sp.cos(x)
        c = sp.expand_trig(c)
        c = sp.simplify(c)
        c = sp.trigsimp(c)
        return c
    else:
        return np.cos(x)

def sin(x):
    if isinstance(x, sp.core.basic.Basic):
        s = sp.sin(x)
        s = sp.expand_trig(s)
        s = sp.simplify(s)
        s = sp.trigsimp(s)
        return s
    else:
        return np.sin(x)

def arccos(x):
    if isinstance(x, sp.core.basic.Basic):
        return sp.simplify(sp.acos(x))
    else:
        return np.arccos(x)

def sqrt(x):
    if isinstance(x, sp.core.basic.Basic):
        return sp.simplify(sp.sqrt(x))
    else:
        return np.sqrt(x)    

def norm(x):
    if IS.NumericValue(x):
        return la.norm(x)
    if SymbolicValue(x):
        RF = sp.physics.vector.ReferenceFrame("M")
        vector = x[0] * RF.x + x[1]*RF.y + x[2]*RF.z
        no = sp.sqrt(sp.physics.vector.dot(vector, vector))
        no = sp.simplify(no)
        return no

def unitizevector(x):
    if IS.NumericValue(x):
        return x/la.norm(x)
    else:
        if x[0] == 0 and x[1] == 0:
            return np.array((0, 0, 1))
        elif x[0] == 0 and x[2] == 0:
            return np.array((0, 1, 0))
        elif x[1] == 0 and x[2] == 0:
            return np.array((1, 0, 0))
        else:
            return x/norm(x)



class Compute:
    @staticmethod
    def Ux2u(Ux):
        Validation_Compute.Ux2u(Ux)
        return Compute._Ux2u(Ux)

    @staticmethod
    def u2Ux(u):
        Validation_Compute.u2Ux(u)
        return Compute._u2Ux(u)

    @staticmethod
    def R(angle, u):
        Validation_Compute.R(angle, u)
        return Compute._R(angle, u)

    @staticmethod
    def r2R(r):
        Validation_Compute.r2R(r)
        return Compute._r2R(r)

    @staticmethod
    def R2r(R):
        Validation_Compute.R2r(R)
        return Compute._R2r(R)

    @staticmethod
    def w2W(w):
        Validation_Compute.w2W(w)
        return Compute.u2Ux(w)

    @staticmethod
    def W2w(W):
        Validation_Compute.W2w(W)
        return Compute.Ux2u(W)

    @staticmethod
    def q2Q(q):
        Validation_Compute.q2Q(q)
        return Compute.u2Ux(q)

    @staticmethod
    def Q2q(Q):
        Validation_Compute.Q2q(Q)
        return Compute.Ux2u(Q)

    @staticmethod
    def PAT(CM):
        # PAT is the Parallel Axis Theorem
        Validation_Compute.CM(CM)
        return Compute._PAT(CM)

    @staticmethod
    def _Ux2u(Ux):
        Ux = np.array(Ux)
        u = np.array([Ux[2, 1] - Ux[1, 2],
                      Ux[0, 2] - Ux[2, 0],
                      Ux[1, 0] - Ux[0, 1]])
        u = u / 2
        return u

    @staticmethod
    def _u2Ux(u):
        Ux = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        return Ux

    @staticmethod
    def _R(angle, u):
        c = cos(angle)
        s = sin(angle)
            
        if isinstance(u, str):
            pass
        elif np.all(u == (1, 0, 0)):
            u = "x"
        elif np.all(u == (0, 1, 0)):
            u = "y"
        elif np.all(u == (0, 0, 1)):
            u = "z"

        if isinstance(u, str):
            if u == "x":
                R = np.array([[1, 0, 0],
                              [0, c, -s],
                              [0, s, c]])
            elif u == "y":
                R = np.array([[c, 0, s],
                              [0, 1, 0],
                              [-s, 0, c]])
            elif u == "z":
                R = np.array([[c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]])
        else:
            I = np.eye(3)
            # I = np.array([[1, 0, 0],
            #               [0, 1, 0],
            #               [0, 0, 1]])
            U = np.tensordot(u, u, axes=0)
            # U = np.array([[ux * ux, ux * uy, ux * uz],
            #               [ux * uy, uy * uy, uy * uz],
            #               [ux * uz, uy * uz, uz * uz]])
            Ux = Compute.u2Ux(u)
            # Ux = np.array([[0, -uz, uy],
            #                [uz, 0, -ux],
            #                [-uy, ux, 0]])
            R = (1 - c) * U + c * I + s * Ux
        return R

    @staticmethod
    def _r2R(r):
        angle = norm(r)
        if angle != 0:
            u = r / angle
            return Compute.R(angle, u)
        else:
            return np.eye(3)

    @staticmethod
    def _R2r(R):
        tr = np.trace(R)
        if tr == 3:
            return np.zeros(3)
        angle = arccos((tr - 1) / 2)
        v = Compute._Ux2u(R)
        u = unitizevector(v)
        return angle * u
