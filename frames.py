import numpy as np
import numpy.linalg as la
import sympy as sp


class TreatInput:

    @staticmethod
    def translation(translation):
        if translation is None:
            data_t = np.array((0, 0, 0))
        else:
            data_t = np.array(translation)
            data_t = sp.simplify(data_t)
            data_t = np.array(data_t)
        return data_t

    @staticmethod
    def rotation(rotation):
        """
        rotation can be None
        If rotation is not None, so
            rotation = (theta, axis)
            theta is the rotation angle
            axis is the vector of rotation
            Exemple:
                rotation = (theta, (0, 0, 1))
                rotation = (theta, "z")
                rotation = (theta, "x")
                rotation = (theta, (1, 1, 1))
        """
        if rotation is None:
            data_angle = 0
            data_u = np.zeros(3)
        else:
            angle, axis = rotation
            data_angle = sp.simplify(angle)
            data_u = TreatInput.axis_into_u(axis)
        return data_angle, data_u

    @staticmethod
    def axis_into_u(axis):
        if axis is None:
            raise Exception("If there is not rotation, put None")
        elif axis == (0, 0, 0):
            raise Exception(
                "The vector of rotation must be different of 0! If there is not rotation, put None")
        elif axis == (1, 0, 0) or axis == "x":
            u = np.zeros(3)
            u[0] = 1
        elif axis == (0, 1, 0) or axis == "y":
            u = np.zeros(3)
            u[1] = 1
        elif axis == (0, 0, 1) or axis == "z":
            u = np.zeros(3)
            u[2] = 1
        else:
            try:
                if type(axis) is not tuple:
                    raise Exception("Axis must be a tuple")
                axis = np.array(axis)
                axis_mod = np.linalg.norm(axis)
                u = axis / axis_mod  # u must be a unit vector
            except:
                raise Exception("Axis must be valid. Axis = " + str(axis))
        return u


class Calculate:
    @staticmethod
    def Ux_to_u(Ux):
        u = np.array([Ux[2, 1] - Ux[1, 2],
                      Ux[0, 2] - Ux[0, 2],
                      Ux[1, 0] - Ux[0, 1]])
        u /= 2
        return u

    @staticmethod
    def u_to_Ux(u):
        Ux = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        return Ux

    @staticmethod
    def linear_speed(data_t, time):
        data_v = sp.diff(data_t, time)
        data_v = sp.simplify(data_v)
        data_v = np.array(data_v)
        return data_v

    @staticmethod
    def linear_acceleration(data_v, time):
        data_a = sp.diff(data_v, time)
        data_a = np.array(data_a)
        data_a = sp.simplify(data_a)
        data_a = np.array(data_a)
        return data_a

    def rotation_speed(R, time):
        data_W = Calculate.W(R, time)
        # data_w = Calculate.w(W)
        return data_W

    def rotation_acceleration(W, time):
        data_Q = Calculate.Q(W, time)
        # data_q = Calculate.q(Q)
        return data_Q

    @staticmethod
    def R(angle, u):
        c = sp.cos(angle)
        s = sp.sin(angle)
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
            Ux = Calculate.u_to_Ux(u)
            # Ux = np.array([[0, -uz, uy],
            #                [uz, 0, -ux],
            #                [-uy, ux, 0]])
            R = (1 - c) * U + c * I + s * Ux
        R = sp.simplify(R)
        R = np.array(R)
        R = R.reshape((3, 3))
        return R

    @staticmethod
    def W(R, time):
        R = sp.Matrix(R)
        R_inv = sp.transpose(R)
        dRdt = sp.diff(R, time)
        dRdt = np.array(dRdt)
        dRdt = dRdt.reshape((3, 3))
        W = np.dot(R_inv, dRdt)

        for i in range(3):
            for j in range(3):
                W[i, j] = sp.simplify(W[i, j])
        return W

    @staticmethod
    def Q(W, time):
        Q = sp.diff(W, time)
        Q = np.array(Q)
        Q = Q.reshape((3, 3))

        for i in range(3):
            for j in range(3):
                Q[i, j] = sp.simplify(Q[i, j])
        return Q

    @staticmethod
    def w(W):
        return Calculate.Ux_to_u(W)

    @staticmethod
    def q(Q):
        return Calculate.Ux_to_u(Q)


class FrameReference:
    instances = []

    def __init__(self, base=None, translation=None, rotation=None, name=None):

        FrameReference.valid_base(base, translation, rotation)
        translation = TreatInput.translation(translation)
        angle, u = TreatInput.rotation(rotation)

        if base is not None:
            self.base = base
            # self.base_id = base.get_id()
        else:
            self.base = None
            # self.base_id = base
        self.time = sp.symbols("t", real=True)
        self.id = len(FrameReference.instances)
        if name is None:
            name = str(self.id)
        self.name = name

        self.data = {}

        self.data["t"] = translation
        self.data["R"] = Calculate.R(angle, u)

        self.next = []
        if self.base is not None:
            self.base.next.append(self)
        FrameReference.instances.append(self)

    def get_id(self):
        return self.id

    @staticmethod
    def get_frame(i):
        if i > len(FrameReference.instances):
            raise Exception("The base solicited is not valid")
        return FrameReference.instances[i]

    @staticmethod
    def valid_base(base, translation, rotation):
        if base is None and len(FrameReference.instances) != 0:
            raise Exception(
                "The base must be one of the FrameReference already existent")
        elif base is None:
            if translation is not None:
                raise Exception(
                    "For the frist FrameReference, translation must be None")
            if rotation is not None:
                raise Exception(
                    "For the frist FrameReference, rotation must be None")
        elif base not in FrameReference.instances:
            raise Exception("Base is not known")

    def get_translation(self):
        return self.data["t"]

    def get_linear_speed(self):
        if "v" not in self.data:
            self.data["v"] = Calculate.linear_speed(self.data["t"], self.time)
        return self.data["v"]

    def get_linear_acceleration(self):
        if "a" not in self.data:
            self.data["a"] = Calculate.linear_speed(self.data["v"], self.time)
        return self.data["a"]

    def get_rotation(self):
        return self.data["R"]

    def get_rotation_speed(self):
        if "W" not in self.data:
            self.data["W"] = Calculate.rotation_speed(
                self.data["R"], self.time)
        return self.data["W"]

    def get_rotation_acceleration(self):
        if "Q" not in self.data:
            self.data["Q"] = Calculate.rotation_acceleration(
                self.data["W"], self.time)
        return self.data["Q"]

    def get_vector_rotation_speed(self):
        return Calculate.w(self.data["W"])

    def get_vector_rotation_acceleration(self):
        return Calculate.q(self.data["Q"])

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.string()

    def string(self):
        msg = "FrameReference Number: " + self.name + "\n"
        if self.base is None:
            msg += "FrameReference base: " + "None" + "\n"
        else:
            msg += "FrameReference base: " + self.base.name + "\n"
        msg += "Translation: " + "\n"
        if "t" in self.data:
            data_t = self.get_translation()
            msg += "t = " + str(data_t) + "\n"
        if "v" in self.data:
            data_v = self.get_linear_speed()
            msg += "v = " + str(data_v) + "\n"
        if "a" in self.data:
            data_a = self.get_linear_acceleration()
            msg += "a = " + str(data_a) + "\n"
        msg += "Rotation: " + "\n"
        if "r" in self.data:
            data_R = self.get_rotation()
            data_r = Calculate.Ux_to_u(data_R)
            msg += "r = " + str(data_r) + "\n"
        if "w" in self.data:
            data_W = self.get_rotation_speed()
            data_w = Calculate.Ux_to_u(data_W)
            msg += "w = " + str(data_w) + "\n"
        if "q" in self.data:
            data_Q = self.get_rotation_acceleration()
            data_q = Calculate.Ux_to_u(data_Q)
            msg += "q = " + str(data_q) + "\n"
        return msg


class FrameComposition:
    @staticmethod
    def calcule_position(frame, p1):
        t_01 = frame.get_translation()
        R_01 = frame.get_rotation()
        R_p1 = R_01 @ p1
        p0 = t_01 + R_p1
        return p0

    @staticmethod
    def calcule_linear_speed(frame, p1, v1):
        v_01 = frame.get_linear_speed()
        R_01 = frame.get_matrix_rotation()
        W_01 = frame.get_matrix_rotation_speed()

        R_p1 = np.dot(R_01, p1)
        R_v1 = np.dot(R_01, v1)

        v0 = v_01 + np.dot(W_01, R_p1) + R_v1
        return v0

    @staticmethod
    def calcule_linear_acceleration(frame, p1, v1, a1):
        a_01 = frame.get_linear_speed()
        R_01 = frame.get_matrix_rotation()
        W_01 = frame.get_matrix_rotation_speed()
        Q_01 = frame.get_matrix_rotation_acceleration()

        R_p1 = R_01 @ p1
        R_v1 = R_01 @ v1
        R_a1 = R_01 @ a1
        W_p1 = W_01 @ R_p1
        W_v1 = W_01 @ R_v1
        Q_p1 = Q_01 @ R_p1
        WW_p1 = W_01 @ W_p1

        a0 = a_01 + Q_p1 + WW_p1 + 2 * W_v1 + R_a1
        return a0

    @staticmethod
    def calcule_rotation(frame, r1):
        '''
        Not implemented yet
        '''
        r0 = np.array([0, 0, 0])
        return r0

    @staticmethod
    def calcule_rotation_speed(frame, w1):
        w_01 = frame.get_vector_rotation_speed()
        R_01 = frame.get_matrix_rotation()

        R_w1 = R_01 @ w1

        w0 = w_01 + R_w1
        return w0

    @staticmethod
    def calcule_rotation_acceleration(frame, q1):
        q_01 = frame.get_vector_rotation_acceleration()
        R_01 = frame.get_matrix_rotation()

        R_q1 = R_01 @ q1

        q0 = q_01 + R_q1
        return q0

    @staticmethod
    def calcule_matrix_inertia(frame, II1):
        # Basically we have that
        # II0 = R_01 @ II1 @ R_01^T
        R_01 = frame.get_matrix_rotation()

        II0 = np.dot(II1, np.transpose(R_01))
        II0 = np.array(II0)
        II0 = II0.reshape((3, 3))
        for i in range(3):
            for j in range(3):
                II0[i, j] = sp.simplify(II0[i, j])
        II0 = np.dot(R_01, II0)
        II0 = np.array(II0)
        II0 = II0.reshape((3, 3))
        for i in range(3):
            for j in range(3):
                II0[i, j] = sp.simplify(II0[i, j])
        return Inertia0
