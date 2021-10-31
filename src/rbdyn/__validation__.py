import numpy as np
import rbdyn.__classes__ as classes


def Verify3DVector(value):
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, tuple):
        pass
    elif isinstance(value, list):
        pass
    else:
        error = "The 3D Vector must be a numpy.ndarray/list/tuple not %s" % str(
            type(value))
        raise TypeError(error)

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


def VerifySymmetric(value):
    value = np.array(value)
    if value.ndim != 2:
        raise ValueError("The given tensor must have ndim = 2")
    zerostest = np.abs(value - np.transpose(value))
    if np.any(zerostest > 1e-10):
        raise ValueError("The given tensor is not Symmetric")


def VerifyAntiSymmetric(value):
    value = np.array(value)
    if value.ndim != 2:
        raise ValueError("The given tensor must have ndim = 2")
    zerostest = np.abs(value + np.transpose(value))
    if np.any(zerostest > 1e-10):
        raise ValueError("The given tensor is not AntiSymmetric")


def VerifyUnit3DVector(value):
    Verify3DVector(value)
    if np.abs(np.linalg.norm(value) - 1) > 1e-10:
        raise ValueError("The given vector must have module 1")


def VerifyValidRotation(value):
    R = np.array(value)
    RT = np.transpose(R)
    M = np.dot(RT, R)
    diff = np.abs(M - np.eye(3))
    if np.any(diff > 1e-10):
        raise ValueError("The given R is not a rotation matrix!")


def VerifyAngle(angle):
    pass


def VerifyBoolean(init):
    if not isinstance(init, bool):
        raise TypeError("The argument of Kinematic must be boolean")


def VerifyString(name):
    if not isinstance(name, str):
        raise TypeError("name must be a string")


def VerifyKinematicClass(kine):
    if not isinstance(kine, classes.KinematicClass):
        raise TypeError("The argument must be a Kinematic instance")


class Validation_Kinematic:
    @staticmethod
    def init(init):
        VerifyBoolean(init)

    @staticmethod
    def p(value):
        Verify3DVector(value)

    @staticmethod
    def v(value):
        Verify3DVector(value)

    @staticmethod
    def a(value):
        Verify3DVector(value)

    @staticmethod
    def r(value):
        if value is None:
            raise TypeError("rotation must be not None")
        Validation_Rotation.init(value)

    @staticmethod
    def w(value):
        Verify3DVector(value)

    @staticmethod
    def q(value):
        Verify3DVector(value)

    @staticmethod
    def R(value):
        Verify3DTensor(value)
        VerifyValidRotation(value)

    @staticmethod
    def W(value):
        Verify3DTensor(value)
        VerifyAntiSymmetric(value)

    @staticmethod
    def Q(value):
        Verify3DTensor(value)
        VerifyAntiSymmetric(value)


class Validation_ObjectKinematic:
    @staticmethod
    def CM(value):
        Verify3DVector(value)

    @staticmethod
    def II(value):
        Verify3DTensor(value)


class Validation_Compute:
    @staticmethod
    def Ux2u(Ux):
        Verify3DTensor(Ux)
        VerifyAntiSymmetric(Ux)

    @staticmethod
    def u2Ux(u):
        Verify3DVector(u)

    @staticmethod
    def R(angle, u):
        VerifyAngle(angle)
        Validation_Rotation.angle(angle)
        Validation_Rotation.direction(u)

    @staticmethod
    def r2R(r):
        Verify3DVector(r)

    @staticmethod
    def R2r(R):
        Verify3DTensor(R)

    @staticmethod
    def w2W(w):
        Verify3DVector(w)

    @staticmethod
    def W2w(W):
        Verify3DTensor(W)
        VerifyAntiSymmetric(W)

    @staticmethod
    def q2Q(q):
        Verify3DVector(q)

    @staticmethod
    def Q2q(Q):
        Verify3DTensor(Q)
        VerifyAntiSymmetric(Q)


class Validation_Translation:

    @staticmethod
    def init(translation):
        if translation is None:
            pass
        else:
            Verify3DVector(translation)


class Validation_Rotation:
    @staticmethod
    def init(rotation):
        if rotation is None:
            return
        elif isinstance(rotation, tuple):
            pass
        elif isinstance(rotation, list):
            pass
        elif isinstance(rotation, np.ndarray):
            pass
        else:
            error = "Rotation must be a tuple/list/numpy ndarray"
            raise TypeError(error)

        if len(rotation) == 2:
            angle, direction = rotation
            Validation_Rotation.angle(angle)
            Validation_Rotation.direction(direction)
        else:
            Verify3DVector(rotation)

    @staticmethod
    def angle(angle):
        pass

    @staticmethod
    def direction(direction):
        if isinstance(direction, str):
            if direction in ("x", "y", "z"):
                return
            raise ValueError("When direction is string, it must be x/y/z")
        VerifyUnit3DVector(direction)

class Validation_FrameReference:

    @staticmethod
    def init(base, translation, rotation):
        if base is None:
            if translation is not None:
                raise ValueError("translation must be None when base is None")
            if rotation is not None:
                raise ValueError("rotation must be None when base is None")
        Validation_FrameReference.base(base)
        Validation_FrameReference.translation(translation)
        Validation_FrameReference.rotation(rotation)

    @staticmethod
    def base(base):
        if base is None:
            pass
        elif isinstance(base, classes.FrameReferenceClass):
            pass
        else:
            raise TypeError("The base must be a FrameReference instance")

    @staticmethod
    def translation(translation):
        Validation_Translation.init(translation)

    @staticmethod
    def rotation(rotation):
        Validation_Rotation.init(rotation)


class Validation_FrameComposition:

    @staticmethod
    def p1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def v1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def a1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def R1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def w1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def q1(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def p0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def v0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def a0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def R0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def w0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def q0(kine01, kine1):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine1)

    @staticmethod
    def PAT(CM):
        Verify3DVector(CM)


class Validation_Object:
    KINEMATIC_ELEMENTS = ("p", "v", "a", "r", "w", "q",
                          "R", "W", "Q", "CM", "II")

    @staticmethod
    def init(baseframe, name):
        Validation_Object.baseframe(baseframe)
        Validation_Object.name(name)

    @staticmethod
    def baseframe(baseframe):
        Validation_Object.verifyFrame(baseframe)

    @staticmethod
    def name(name):
        if name is not None:
            VerifyString(name)

    @staticmethod
    def verifyFrame(frame):
        if not isinstance(frame, classes.FrameReferenceClass):
            error = "The given frame must be a FrameReference instance. Received %s" % str(
                type(frame))
            raise TypeError(error)

    @staticmethod
    def verifyElement(element):
        if not isinstance(element, str):
            raise TypeError("The element must be a string!")
        if element not in Validation_Object.KINEMATIC_ELEMENTS:
            error = "The element must be in %s" % str(
                Validation_Object.KINEMATIC_ELEMENTS)
            raise ValueError(error)

    @staticmethod
    def mass(value):
        if isinstance(value, str):
            raise TypeError("Mass must not be a string")

    @staticmethod
    def CM(value):
        Validation_ObjectKinematic.CM(value)

    @staticmethod
    def II(value):
        Validation_ObjectKinematic.II(value)

    @staticmethod
    def getCM(frame):
        Validation_Object.verifyFrame(frame)

    @staticmethod
    def getII(frame):
        Validation_Object.verifyFrame(frame)

    @staticmethod
    def get(frame, element):
        Validation_Object.verifyFrame(frame)
        Validation_Object.verifyElement(element)
