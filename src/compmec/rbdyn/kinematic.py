import numpy as np
from numpy import linalg as la
from compmec.rbdyn.__classes__ import KinematicClass
from compmec.rbdyn.__validation__ import Validation_Kinematic, Validation_ObjectKinematic
from compmec.rbdyn.composition import Compute


class Kinematic(KinematicClass):

    def __init__(self, init=True):
        Validation_Kinematic.init(init)
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

    @p.setter
    def p(self, value):
        Validation_Kinematic.p(value)
        self._p = value

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        Validation_Kinematic.v(value)
        self._v = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        Validation_Kinematic.a(value)
        self._a = value

    @property
    def r(self):
        if self.R is None:
            return None
        return Compute.R2r(self.R)

    @r.setter
    def r(self, value):
        Validation_Kinematic.r(value)
        if len(value) == 2:
            angle, direction = value
            self.R = Compute.R(angle, direction)
        else:
            self.R = Compute.r2R(value)

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        Validation_Kinematic.w(value)
        self._w = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        Validation_Kinematic.q(value)
        self._q = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        Validation_Kinematic.R(value)
        self._R = value

    @property
    def W(self):
        if self.w is None:
            return None
        return Compute.w2W(self.w)

    @W.setter
    def W(self, value):
        Validation_Kinematic.W(value)
        self.w = Compute.W2w(value)

    @property
    def Q(self):
        if self.q is None:
            return None
        return Compute.q2Q(self.q)

    @Q.setter
    def Q(self, value):
        Validation_Kinematic.Q(value)
        self.q = Compute.Q2q(value)


class ObjectKinematic(Kinematic):

    def __init__(self, init=True):
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

    @CM.setter
    def CM(self, value):
        Validation_ObjectKinematic.CM(value)
        self._CM = value

    @property
    def II(self):
        return self._II

    @II.setter
    def II(self, value):
        Validation_ObjectKinematic.II(value)
        self._II = value

    def get(self, element):
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
