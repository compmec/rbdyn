import numpy as np
from numpy import linalg as la
from compmec.rbdyn.__classes__ import KinematicBaseClass
from compmec.rbdyn.__validation__ import Validation_Kinematic, Validation_ObjectKinematic
from compmec.rbdyn.composition import Compute


class Kinematic(KinematicBaseClass):

    validation = Validation_Kinematic

    def __init__(self, init=True):
        self.validation.init(self, init)
        if init:
            self.p = np.zeros(3)  # linear position
            self.v = np.zeros(3)  # linear velocity
            self.a = np.zeros(3)  # linear aceleration
            self.R = np.eye(3)  # angular position
            self.w = np.zeros(3)  # angular speed
            self.q = np.zeros(3)  # angular acceleration
        else:
            self._p = None
            self._v = None
            self._a = None
            self._R = None
            self._w = None
            self._q = None



    @property
    def p(self):
        return self._p

    @property
    def v(self):
        return self._v
    
    @property
    def a(self):
        return self._a

    @property
    def r(self):
        if self.R is None:
            return None
        return Compute.R2r(self.R)
    
    @property
    def w(self):
        return self._w

    @property
    def q(self):
        return self._q

    @property
    def R(self):
        return self._R

    @property
    def W(self):
        if self.w is None:
            return None
        return Compute.w2W(self.w)

    @property
    def Q(self):
        if self.q is None:
            return None
        return Compute.q2Q(self.q)


    @p.setter
    def p(self, value):
        self.validation.psetter(self, value)
        self._p = value


    @v.setter
    def v(self, value):
        self.validation.vsetter(self, value)
        self._v = value


    @a.setter
    def a(self, value):
        self.validation.asetter(self, value)
        self._a = value


    @r.setter
    def r(self, value):
        self.validation.rsetter(self, value)
        if len(value) == 2:
            angle, direction = value
            self.R = Compute.R(angle, direction)
        else:
            self.R = Compute.r2R(value)

    @w.setter
    def w(self, value):
        self.validation.wsetter(self, value)
        self._w = value

    @q.setter
    def q(self, value):
        self.validation.qsetter(self, value)
        self._q = value

    @R.setter
    def R(self, value):
        self.validation.Rsetter(self, value)
        self._R = value

    @W.setter
    def W(self, value):
        self.validation.Wsetter(self, value)
        self.w = Compute.W2w(value)

    @Q.setter
    def Q(self, value):
        self.validation.Qsetter(self, value)
        self.q = Compute.Q2q(value)


class ObjectKinematic(Kinematic):

    validation = Validation_ObjectKinematic

    def __init__(self, init=True):
        self.validation.init(self, init)
        Kinematic.__init__(self, init)
        if init:
            self.CM = np.zeros(3)  # Center of mass
            self.II = np.zeros((3, 3))  # Inertia
        else:
            self._CM = None
            self._II = None

    @property
    def CM(self):
        return self._CM

    @property
    def II(self):
        return self._II

    @CM.setter
    def CM(self, value):
        self.validation.CMsetter(self, value)
        self._CM = value


    @II.setter
    def II(self, value):
        self.validation.IIsetter(self, value)
        self._II = value

    def get(self, element):
        self.validation.get(self, element)
        if element == "p":
            return self.p
        elif element == "v":
            return self.v
        elif element == "a":
            return self.a
        elif element == "r":
            return self.r
        elif element == "w":
            return self.w
        elif element == "q":
            return self.q
        elif element == "R":
            return self.R
        elif element == "W":
            return self.W
        elif element == "Q":
            return self.Q
        elif element == "CM":
            return self.CM
        elif element == "II":
            return self.II
