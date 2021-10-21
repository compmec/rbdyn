import numpy as np
from numpy import linalg as la


def Verify3DVector(value):
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, tuple):
        pass
    elif isinstance(value, list):
        pass
    else:
        raise TypeError("The 3D Vector must be a numpy.ndarray/list/tuple")

    value = np.array(value)
    if value.ndim != 1:
        raise ValueError("The ndim of 3D vector must be equal to 1!")
    if len(value) != 3:
        raise ValueError("The 3D vector must have 3 elements!")


def Verify3DTensor(value):
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, tuple):
        pass
    elif isinstance(value, list):
        pass
    else:
        raise TypeError("The 3D Vector must be a numpy.ndarray/list/tuple")

    value = np.array(value)
    if value.ndim != 2:
        raise ValueError("The ndim of 3D tensor must be equal to 2!")
    if value.shape != (3, 3):
        raise ValueError("The 3D tensor must have 3x3 elements!")


def VerifyAntiSymmetric(value):
    value = np.array(value)
    if value.ndim != 2:
        raise ValueError("The given tensor must have ndim = 2")
    zerostest = np.abs(value + np.transpose(value))
    if np.any(zerostest > 1e-10):
        raise ValueError("The given tensor is not AntiSymmetric")


def VerifyUnit3DVector(value):
    Verify3DVector(value)
    if np.abs(la.norm(value) - 1) > 1e-10:
        raise ValueError("The given vector must have module 1")


def VerifyValidRotation(value):
    R = np.array(value)
    RT = np.transpose(R)
    M = np.dot(RT, R)
    print("M = ")
    print(M)
    diff = np.abs(M - np.eye(3))
    print("M = ")
    print(M)
    if np.any(diff > 1e-10):
        raise ValueError("The given R is not a rotation matrix!")


class Compute:

    @staticmethod
    def Ux2u(Ux):
        Verify3DTensor(Ux)
        VerifyAntiSymmetric(Ux)
        return Compute._Ux2u(Ux)

    @staticmethod
    def u2Ux(u):
        Verify3DVector(u)
        return Compute._u2Ux(u)

    @staticmethod
    def R(angle, u):
        VerifyUnit3DVector(u)
        return Compute._R(angle, u)

    @staticmethod
    def r2R(r):
        Verify3DVector(r)
        return Compute._r2R(r)

    @staticmethod
    def R2r(R):
        Verify3DTensor(R)
        return Compute._R2r(R)

    @staticmethod
    def _Ux2u(Ux):
        Ux = np.array(Ux)
        u = np.array([Ux[2, 1] - Ux[1, 2],
                      Ux[0, 2] - Ux[2, 0],
                      Ux[1, 0] - Ux[0, 1]])
        u /= 2
        return u

    @staticmethod
    def _u2Ux(u):
        Ux = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        return Ux

    @staticmethod
    def _R(angle, u):
        c = np.cos(angle)
        s = np.sin(angle)
        if np.all(u == (1, 0, 0)):
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s, c]])
        elif np.all(u == (0, 1, 0)):
            R = np.array([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]])
        elif np.all(u == (0, 0, 1)):
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
        angle = la.norm(r)
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
        angle = np.arccos((tr - 1) / 2)
        v = Compute._Ux2u(R)
        norm_v = la.norm(v)
        u = v / norm_v
        return angle * u


class Kinematic:

    def __init__(self, initializezero=True):
        if initializezero:
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
        Verify3DVector(value)
        self._p = value

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        Verify3DVector(value)
        self._v = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        Verify3DVector(value)
        self._a = value

    @property
    def r(self):
        if self.R is None:
            return None
        return Compute.R2r(self.R)

    @r.setter
    def r(self, value):
        raise NotImplementedError("r is not yet valid")
        Verify3DVector(value)
        self._r = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        Verify3DVector(value)
        self._w = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        Verify3DVector(value)
        self._q = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        Verify3DTensor(value)
        VerifyValidRotation(value)
        self._R = value

    @property
    def W(self):
        if self.w is None:
            return None
        return Compute.u2Ux(self.w)

    @W.setter
    def W(self, value):
        Verify3DTensor(value)
        self.w = Compute.Ux2u(value)

    @property
    def Q(self):
        if self.q is None:
            return None
        return Compute.u2Ux(self.q)

    @Q.setter
    def Q(self, value):
        Verify3DTensor(value)
        self.q = Compute.Ux2u(value)


class ComputeComposition:

    @staticmethod
    def direct_linear_position(t01, R01, p0):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (R1) sees the point P at position p1
            The position of (R1) in relation to (R0) is given by
            the vectors t01 and rotation matrix R01
        """
        Rp0 = np.dot(R01, p0)
        p1 = t01 + Rp1
        return p1

    @staticmethod
    def direct_linear_speed(v01, R01, W01, p0, v0):
        Rp0 = np.dot(R01, p0)
        Rv0 = np.dot(R01, v0)

        v1 = v01 + np.dot(W01, Rp0) + Rv0
        return v1

    @staticmethod
    def direct_linear_acceleration(a01, R01, W01, Q01, p0, v0, a0):
        Rp0 = np.dot(R01, p0)
        Rv0 = np.dot(R01, v0)
        Ra0 = np.dot(R01, a0)
        Wp0 = np.dot(W01, Rp0)
        Wv0 = np.dot(W01, Rv0)
        Qp0 = np.dot(Q01, Rp0)
        WWp0 = np.dot(W01, Wp0)

        a1 = a01 + Qp0 + WWp0 + 2 * Wv0 + Ra0
        return a1

    @staticmethod
    def direct_rotation_position(r0, R01):
        R0 = Compute.r2R(r0)
        R1 = np.dot(R01, R0)
        r1 = Compute.R2r(R1)
        return r1

    @staticmethod
    def direct_rotation_speed(w01, R01, w0):
        Rw0 = np.dot(R01, w0)
        w1 = w01 + Rw0
        return w1

    @staticmethod
    def direct_rotation_acceleration(q01, R01, q0):
        Rq0 = np.dot(R01, q0)
        q1 = q01 + Rq0
        return q1

    @staticmethod
    def inverse_linear_position(t01, R01, p1):
        """
        Computes the position of a point P in the reference R0.
        The FrameReference1 (R1) sees the point P at position p1
            The position of (R1) in relation to (R0) is given by
            the vectors t01 and rotation matrix R01
        """
        dp = p1 - t01

        Rt = np.transpose(R01)
        p0 = np.dot(Rt, dp)
        return p0

    @staticmethod
    def inverse_linear_speed(t01, v01, R01, W01, p1, v1):
        dp = p1 - t01
        dv = v1 - v01
        Wdp = np.dot(W01, dp)

        Rt = np.transpose(R01)
        v0 = np.dot(Rt, dv - Wdp)
        return v0

    @staticmethod
    def inverse_linear_acceleration(t01, v01, a01, R01, W01, Q01, p1, v1, a1):
        dp = p1 - t01
        dv = v1 - v01
        da = a1 - a01
        Wdv = np.dot(W01, dv)
        WWdp = np.dot(W01, np.dot(W01, dp))
        Qdp = np.dot(Q01, dp)

        Rt = np.transpose(R01)
        a0 = np.dot(Rt, da - Qdp + WWdp - 2 * Wdv)
        return a0

    @staticmethod
    def inverse_rotation_position(R01, r1):
        """
        Not implemented yet
        """
        R1 = Compute.r2R(r1)
        Rt = np.transpose(R01)
        R0 = np.dot(Rt, R1)
        r0 = Compute.R2r(R0)
        return r0

    @staticmethod
    def inverse_rotation_speed(w01, R01, w0):
        dw = w0 - w01

        Rt = np.transpose(R01)
        q0 = np.dot(Rt, dw)
        return w0

    @staticmethod
    def inverse_rotation_acceleration(q01, R01, q0):
        dq = q0 - q01

        Rt = np.transpose(R01)
        q0 = np.dot(Rt, dq)
        return q0

    @staticmethod
    def direct_element(frame, kine, element):
        if element == "p":
            t01 = frame.p
            R01 = frame.R
            p0 = kine.p
            return ComputeComposition.direct_linear_position(t01, R01, p0)
        if element == "v":
            v01 = frame.p
            R01 = frame.R
            W01 = frame.W
            p0 = kine.p
            v0 = kine.v
            return ComputeComposition.direct_linear_speed(v01, R01, W01, p0, v0)
        if element == "a":
            t01 = frame.p
            R01 = frame.R
            p0 = kine.p
            v0 = kine.v
            a0 = kine.a
            return ComputeComposition.direct_linear_acceleration(a01, R01, W01, Q01, p0, v0, a0)
        if element == "r":
            raise NotImplementedError("rotation position not yet implemented")
        if element == "R":
            R01 = frame.R
            R0 = kine.R
            return np.dot(R01, R0)

        if element == "W":
            W01 = frame.W
            R01 = frame.R
            p0 = kine.p
            W1 = ComputeComposition.direct_rotation_speed(W01, R01, w0)
            return W1
        if element == "Q":
            t01 = frame.p
            R01 = frame.R
            p0 = kine.p
            return ComputeComposition.direct_rotation_acceleration(q01, R01, q0)
        if element == "w":
            W1 = ComputeComposition.direct_element(frame, kine, "W")
            return Ux2u(W1)
        if element == "q":
            Q1 = ComputeComposition.direct_element(frame, kine, "Q")
            return Ux2u(Q1)
        raise Exception("direct_element: Not expected got here")

    # @staticmethod
    # def center_mass(frame0to1, kine1):
    #     if not isinstance(frame0to1, Kinematic):
    #         raise TypeError("frame0to1 must be Kinematic instance")
    #     if not isinstance(kine1, Kinematic):
    #         raise TypeError("kine1 must be Kinematic instance")
    #     return KinematicComposition.position(frame0to1, kine1)

    # @staticmethod
    # def matrix_inertia(frame0to1, kine1):
    #     if not isinstance(frame0to1, Kinematic):
    #         raise TypeError("frame0to1 must be Kinematic instance")
    #     if not isinstance(kine1, Kinematic):
    #         raise TypeError("kine1 must be Kinematic instance")
    #     # Basically we have that
    #     # II0 = R01 @ II1 @ R01^T
    #     R01 = frame0to1.R

    #     II1 = kine1.II

    #     II0 = np.dot(II1, np.transpose(R01))
    #     II0 = np.array(II0)
    #     II0 = II0.reshape((3, 3))
    #     for i in range(3):
    #         for j in range(3):
    #             II0[i, j] = sp.simplify(II0[i, j])
    #     II0 = np.dot(R01, II0)
    #     II0 = np.array(II0)
    #     II0 = II0.reshape((3, 3))
    #     for i in range(3):
    #         for j in range(3):
    #             II0[i, j] = sp.simplify(II0[i, j])
    #     return II0


# class ObjectKinematic(Kinematic):
#     def __init__(self, initializezero=True):
#         super().__init__(initializezero)
#         if initializezero:
#             self._CM = np.zeros(3)
#             self._II = np.zeros((3, 3))
#         else:
#             self._CM = None
#             self._II = None

#     @property
#     def CM(self):
#         return self._CM

#     @CM.setter
#     def CM(self, value):
#         Verify3DVector(value)
#         self._CM = value

#     @property
#     def II(self):
#         return self._II

#     @II.setter
#     def II(self, value):
#         Verify3DTensor(value)
#         self._II = value

#     def get(self, element):
#         if not isinstance(element, str):
#             raise TypeError("element must be a string")
#         if element == "CM":
#             return self.CM
#         elif element == "II":
#             return self.II
#         else:
#             super(Kinematic).get
