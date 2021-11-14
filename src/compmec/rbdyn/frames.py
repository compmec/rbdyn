import numpy as np
import sympy as sp
from compmec.rbdyn.kinematic import Kinematic
from compmec.rbdyn import time
from compmec.rbdyn.__classes__ import FrameReferenceClass
from compmec.rbdyn.__validation__ import Validation_FrameReference, Validation_FrameComposition


class FrameReference(FrameReferenceClass):

    def __new__(cls, base=None, translation=None, rotation=None):
        Validation_FrameReference.init(base, translation, rotation)

        if len(FrameReference.instances) == 0:
            self = object.__new__(cls)
            FrameReference.instances.append(self)
            self.__init(None, None, None)

        if base is None:
            return FrameReference.instances[0]
        else:
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
            self._kine.v[i] = sp.diff(self._kine.p[i], time)
            self._kine.a[i] = sp.diff(self._kine.v[i], time)
            self._kine.w[i] = sp.diff(self._kine.r[i], time)
            self._kine.q[i] = sp.diff(self._kine.w[i], time)

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

    @staticmethod
    def p1(kine01, kine0):
        Validation_FrameComposition.p1(kine01, kine0)
        return FrameComposition._p1(kine01, kine0)

    @staticmethod
    def v1(kine01, kine0):
        Validation_FrameComposition.v1(kine01, kine0)
        return FrameComposition._v1(kine01, kine0)

    @staticmethod
    def a1(kine01, kine0):
        Validation_FrameComposition.a1(kine01, kine0)
        return FrameComposition._a1(kine01, kine0)

    @staticmethod
    def R1(kine01, kine0):
        Validation_FrameComposition.R1(kine01, kine0)
        return FrameComposition._R1(kine01, kine0)

    @staticmethod
    def w1(kine01, kine0):
        Validation_FrameComposition.w1(kine01, kine0)
        return FrameComposition._w1(kine01, kine0)

    @staticmethod
    def q1(kine01, kine0):
        Validation_FrameComposition.q1(kine01, kine0)
        return FrameComposition._q1(kine01, kine0)

    @staticmethod
    def p0(kine01, kine1):
        Validation_FrameComposition.p0(kine01, kine1)
        return FrameComposition._p0(kine01, kine1)

    @staticmethod
    def v0(kine01, kine1):
        Validation_FrameComposition.v0(kine01, kine1)
        return FrameComposition._v0(kine01, kine1)

    @staticmethod
    def a0(kine01, kine1):
        Validation_FrameComposition.a0(kine01, kine1)
        return FrameComposition._a0(kine01, kine1)

    @staticmethod
    def R0(kine01, kine1):
        Validation_FrameComposition.R0(kine01, kine1)
        return FrameComposition._R0(kine01, kine1)

    @staticmethod
    def w0(kine01, kine1):
        Validation_FrameComposition.w0(kine01, kine1)
        return FrameComposition._w0(kine01, kine1)

    @staticmethod
    def q0(kine01, kine1):
        Validation_FrameComposition.q0(kine01, kine1)
        return FrameComposition._q0(kine01, kine1)

    @staticmethod
    def PAT(CM):
        Validation_FrameComposition.PAT(CM)
        return FrameComposition._PAT(CM)

    @staticmethod
    def _p1(kine01, kine0):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (kine1.R) sees the point P at position kine1.p
            The position of (kine1.R) in relation to (R0) is given by
            the vectors kine01.p and rotation matrix R01
        """
        dp = kine1.p - kine01.p

        Rt = np.transpose(kine01.R)
        return np.dot(Rt, dp)

    @staticmethod
    def _v1(kine01, kine0):
        dp = kine0.p - kine01.p
        dv = kine0.v - kine01.v
        Wdp = np.dot(kine01.W, dp)

        Rt = np.transpose(kine01.R)
        return np.dot(Rt, dv - Wdp)

    @staticmethod
    def _a1(kine01, kine0):
        Rp0 = np.dot(kine01.R, kine0.p)
        Rv0 = np.dot(kine01.R, kine0.v)
        Ra0 = np.dot(kine01.R, kine0.a)
        Wp0 = np.dot(kine01.W, Rp0)
        Wv0 = np.dot(kine01.W, Rv0)
        Qp0 = np.dot(kine01.Q, Rp0)
        WWp0 = np.dot(kine01.W, Wp0)

        return kine01.a + Qp0 + WWp0 + 2 * Wv0 + Ra0

    @staticmethod
    def _R1(kine01, kine0):
        return np.dot(kine01.R, kine0.R)

    @staticmethod
    def _w1(kine01, kine0):
        Rw0 = np.dot(kine01.R, kine0.w)
        return kine01.w + Rw0

    @staticmethod
    def _q1(kine01, kine0):
        Rq0 = np.dot(kine01.R, kine0.q)
        q1 = kine01.q + Rq0
        return q1

    @staticmethod
    def _p0(kine01, kine1):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (kine1.R) sees the point P at position kine1.p
            The position of (kine1.R) in relation to (R0) is given by
            the vectors kine01.p and rotation matrix kine01.R
        """

        Rp1 = np.dot(kine01.R, kine1.p)
        return kine01.p + Rp1

    @staticmethod
    def _v0(kine01, kine1):
        Rp1 = np.dot(kine01.R, kine1.p)
        Rv1 = np.dot(kine01.R, kine1.v)
        return kine01.v + np.dot(kine01.W, Rp1) + Rv1

    @staticmethod
    def _a0(kine01, kine1):
        dp = kine1.p - kine01.p
        dv = kine1.v - kine01.v
        da = kine1.a - kine01.a
        Wdv = np.dot(kine01.W, dv)
        WWdp = np.dot(kine01.W, np.dot(kine01.W, dp))
        Qdp = np.dot(kine01.Q, dp)

        Rt = np.transpose(kine01.R)
        return np.dot(Rt, da - Qdp + WWdp - 2 * Wdv)

    @staticmethod
    def _R0(kine01, kine1):
        Rt = np.transpose(kine01.R)
        return np.dot(Rt, kine1.R)

    @staticmethod
    def _w0(kine01, kine1):
        dw = w0 - w01

        Rt = np.transpose(kine01.R)
        return np.dot(Rt, dw)

    @staticmethod
    def _q0(kine01, kine1):
        dq = kine1.q - kine01.q

        Rt = np.transpose(kine01.R)
        return np.dot(Rt, dq)

    @staticmethod
    def _PAT(CM):
        return np.eye(3) * np.dot(CM, CM) - np.tensordot(CM, CM, axes=0)
