import numpy as np
import sympy as sp
from compmec.rbdyn.kinematic import Kinematic
from compmec.rbdyn.__classes__ import FrameReferenceBaseClass, timesymb
from compmec.rbdyn.__validation__ import Validation_FrameReference, Validation_FrameComposition, IS

def TakeOutSmallNumbers(value, tolerance=1e-15):
    if isinstance(value, (int, float)):
        if np.abs(value) < tolerance:
            return 0
        return value
    elif isinstance(value, sp.core.basic.Basic):
        return sp.nsimplify(value, tolerance=tolerance, rational=True)
    elif isinstance(value, (list, tuple)):
        value = np.array(value)
    elif isinstance(value, np.ndarray):
        pass
    else:
        raise TypeError("To take out small numbers, value must be a number/array. Received %s" % type(value))

    for i in range(value.shape[0]):
        value[i] = TakeOutSmallNumbers(value[i])
    return value

    


class FrameReference(FrameReferenceBaseClass):
    validation = Validation_FrameReference
    instances = []

    def __new__(cls, base=None, translation=None, rotation=None):
        cls.validation.init(cls, base, translation, rotation)

        if len(FrameReference.instances) == 0:
            return cls.__new(None, None, None)
        if base is None:
            return FrameReference.instances[0]
        
        return cls.__new(base, translation, rotation)

    @classmethod
    def __new(cls, base, translation, rotation):
        self = object.__new__(cls)
        FrameReference.instances.append(self)
        self.__init(base, translation, rotation)
        return self


    def __init(self, base, translation, rotation):
        self.baseframe = base
        self._id = len(FrameReference.instances) - 1
        self._kine = Kinematic(init=False)
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = np.zeros(3)
        self._kine.p = translation
        self._kine.v = np.zeros(3, dtype="object")
        self._kine.a = np.zeros(3, dtype="object")
        self._kine.r = rotation
        self._kine.w = np.zeros(3, dtype="object")
        self._kine.q = np.zeros(3, dtype="object")
        for i in range(3):
            self._kine.v[i] = sp.diff(self._kine.p[i], timesymb)
            self._kine.a[i] = sp.diff(self._kine.v[i], timesymb)
            self._kine.w[i] = sp.diff(self._kine.r[i], timesymb)
            self._kine.q[i] = sp.diff(self._kine.w[i], timesymb)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return "R%d" % self.id

    @property
    def kine(self):
        return self._kine

    def __compute_w(self):
        raise NotImplementedError("__compute_w is not yet implemented")

    def __compute_q(self):
        raise NotImplementedError("__compute_w is not yet implemented")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.string()

    def string(self):
        msg = "FrameReference name: " + self.name + "\n"
        basename = "None" if (self.baseframe is None) else self.baseframe.name
        msg += "FrameReference base: %s\n" % basename
        msg += "Kinematic data:"
        msg += "    Linear: " + "\n"
        msg += "        p: %s\n" % str(self.kine.p)
        msg += "        v: %s\n" % str(self.kine.v)
        msg += "        a: %s\n" % str(self.kine.a)
        msg += "    Angular: " + "\n"
        msg += "        r: %s\n" % str(self.kine.r)
        msg += "        w: %s\n" % str(self.kine.w)
        msg += "        q: %s\n" % str(self.kine.q)
        return msg

    def __del__(self):
        if self in FrameReference.instances:
            FrameReference.instances.remove(self)


class FrameComposition:


    validation = Validation_FrameComposition
    @classmethod
    def p1(cls, kine01, kine0):
        cls.validation.p1(kine01, kine0)
        return cls._p1(kine01, kine0)

    @classmethod
    def v1(cls, kine01, kine0):
        cls.validation.v1(kine01, kine0)
        return cls._v1(kine01, kine0)

    @classmethod
    def a1(cls, kine01, kine0):
        cls.validation.a1(kine01, kine0)
        return cls._a1(kine01, kine0)

    @classmethod
    def R1(cls, kine01, kine0):
        cls.validation.R1(kine01, kine0)
        return cls._R1(kine01, kine0)

    @classmethod
    def w1(cls, kine01, kine0):
        cls.validation.w1(kine01, kine0)
        return cls._w1(kine01, kine0)

    @classmethod
    def q1(cls, kine01, kine0):
        cls.validation.q1(kine01, kine0)
        return cls._q1(kine01, kine0)

    @classmethod
    def p0(cls, kine01, kine1):
        cls.validation.p0(kine01, kine1)
        return cls._p0(kine01, kine1)

    @classmethod
    def v0(cls, kine01, kine1):
        cls.validation.v0(kine01, kine1)
        return cls._v0(kine01, kine1)

    @classmethod
    def a0(cll, kine01, kine1):
        cls.validation.a0(kine01, kine1)
        return cls._a0(kine01, kine1)

    @classmethod
    def R0(cls, kine01, kine1):
        cls.validation.R0(kine01, kine1)
        return cls._R0(kine01, kine1)

    @classmethod
    def w0(kine01, kine1):
        cls.validation.w0(kine01, kine1)
        return cls._w0(kine01, kine1)

    @classmethod
    def q0(kine01, kine1):
        cls.validation.q0(kine01, kine1)
        return cls._q0(kine01, kine1)

    @classmethod
    def PAT(CM):
        cls.validation.PAT(CM)
        return cls._PAT(CM)

    @classmethod
    def _p1(cls, kine01, kine0):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (kine1.R) sees the point P at position kine1.p
            The position of (kine1.R) in relation to (R0) is given by
            the vectors kine01.p and rotation matrix R01
        """
        dp = kine1.p - kine01.p

        Rt = np.transpose(kine01.R)
        return TakeOutSmallNumbers(np.dot(Rt, dp))

    @classmethod
    def _v1(cls, kine01, kine0):
        dp = kine0.p - kine01.p
        dv = kine0.v - kine01.v
        Wdp = np.dot(kine01.W, dp)

        Rt = np.transpose(kine01.R)
        return TakeOutSmallNumbers(np.dot(Rt, dv - Wdp))

    @classmethod
    def _a1(cls, kine01, kine0):
        Rp0 = np.dot(kine01.R, kine0.p)
        Rv0 = np.dot(kine01.R, kine0.v)
        Ra0 = np.dot(kine01.R, kine0.a)
        Wp0 = np.dot(kine01.W, Rp0)
        Wv0 = np.dot(kine01.W, Rv0)
        Qp0 = np.dot(kine01.Q, Rp0)
        WWp0 = np.dot(kine01.W, Wp0)

        return TakeOutSmallNumbers(kine01.a + Qp0 + WWp0 + 2 * Wv0 + Ra0)

    @classmethod
    def _R1(cls, kine01, kine0):
        newR1 = np.dot(kine01.R, kine0.R)
        return TakeOutSmallNumbers(newR1)

    @classmethod
    def _w1(cls, kine01, kine0):
        Rw0 = np.dot(kine01.R, kine0.w)
        return TakeOutSmallNumbers(kine01.w + Rw0)

    @classmethod
    def _q1(cls, kine01, kine0):
        Rq0 = np.dot(kine01.R, kine0.q)
        q1 = kine01.q + Rq0
        return TakeOutSmallNumbers(q1)

    @classmethod
    def _p0(cls, kine01, kine1):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (kine1.R) sees the point P at position kine1.p
            The position of (kine1.R) in relation to (R0) is given by
            the vectors kine01.p and rotation matrix kine01.R
        """

        Rp1 = np.dot(kine01.R, kine1.p)
        return TakeOutSmallNumbers(kine01.p + Rp1)

    @classmethod
    def _v0(cls, kine01, kine1):
        Rp1 = np.dot(kine01.R, kine1.p)
        Rv1 = np.dot(kine01.R, kine1.v)
        return TakeOutSmallNumbers(kine01.v + np.dot(kine01.W, Rp1) + Rv1)

    @classmethod
    def _a0(cls, kine01, kine1):
        dp = kine1.p - kine01.p
        dv = kine1.v - kine01.v
        da = kine1.a - kine01.a
        Wdv = np.dot(kine01.W, dv)
        WWdp = np.dot(kine01.W, np.dot(kine01.W, dp))
        Qdp = np.dot(kine01.Q, dp)

        Rt = np.transpose(kine01.R)
        return TakeOutSmallNumbers(np.dot(Rt, da - Qdp + WWdp - 2 * Wdv))

    @classmethod
    def _R0(cls, kine01, kine1):
        Rt = np.transpose(kine01.R)
        newR0 = np.dot(Rt, kine1.R)
        for i in range(3):
            for j in range(3):
                newR0[i, j] = sp.nsimplify(newR0[i, j])
        return TakeOutSmallNumbers(newR0)

    @classmethod
    def _w0(cls, kine01, kine1):
        dw = w0 - w01

        Rt = np.transpose(kine01.R)
        return TakeOutSmallNumbers(np.dot(Rt, dw))

    @classmethod
    def _q0(cls, kine01, kine1):
        dq = kine1.q - kine01.q

        Rt = np.transpose(kine01.R)
        return TakeOutSmallNumbers(np.dot(Rt, dq))

    @classmethod
    def _PAT(cls, CM):
        return np.eye(3) * np.dot(CM, CM) - np.tensordot(CM, CM, axes=0)
