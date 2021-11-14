import numpy as np
import compmec.rbdyn.__classes__ as classes


def VerifyScalar(value):
    if isinstance(value, (int, float)):
        pass
    if isinstance(value, (tuple, list)):
        raise TypeError("The argument is not an scalar")

    if np.array(value).ndim != 0:
        raise ValueError("The argument is not an scalar")


def VerifyConvertToNumpyArray(value):
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, tuple):
        pass
    elif isinstance(value, list):
        pass
    else:
        raise TypeError("The argument must be a numpy.ndarray/list/tuple")
    try:
        value = np.array(value)
    except Exception as e:
        error = "We cannot convert the argument to numpy array: cause %s"
        raise ValueError(error % str(e))


def VerifyIsVector(value):
    VerifyConvertToNumpyArray(value)
    value = np.array(value)
    if value.ndim != 1:
        raise ValueError("The parameter must have ndim = 1")


def VerifyIsMatrix(value):
    VerifyConvertToNumpyArray(value)
    value = np.array(value)
    if value.ndim != 2:
        raise ValueError("The given argument must be a matrix")


def Verify3DVector(value):
    VerifyIsVector(value)
    if len(value) != 3:
        raise ValueError("The 3D vector must have 3 elements!")


def VerifySquareMatrix(matrix):
    VerifyIsMatrix(matrix)
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        error = "The argument must be a square matrix. Shape = %s"
        raise ValueError(error % str(matrix.shape))


def VerifySymmetricMatrix(value):
    VerifySquareMatrix(value)
    value = np.array(value)
    zerostest = np.abs(value - np.transpose(value))
    if np.any(zerostest > 1e-10):
        raise ValueError("The given tensor is not SymmetricMatrix")


def VerifyAntiSymmetricMatrix(value):
    VerifySquareMatrix(value)
    value = np.array(value)
    zerostest = np.abs(value + np.transpose(value))
    if np.any(zerostest > 1e-10):
        raise ValueError("The given tensor is not AntiSymmetricMatrix")


def VerifySquareMatrixSize3(value):
    VerifySquareMatrix(value)
    value = np.array(value)
    if value.shape[0] != 3:
        raise ValueError("The 3D tensor must have 3x3 elements!")


def Verify3DTensor(value):
    VerifySymmetricMatrix(value)
    VerifySquareMatrixSize3(value)


def VerifyUnit3DVector(value):
    Verify3DVector(value)
    if np.abs(np.linalg.norm(value) - 1) > 1e-10:
        raise ValueError("The given vector must have module 1")


def VerifyValidRotation(value):
    VerifySquareMatrix(value)
    VerifySquareMatrixSize3(value)
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


def VerifyFrameReference(frame):
    if not isinstance(frame, classes.FrameReferenceClass):
        raise TypeError("The argument must be a FrameReference instance")


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
        VerifySquareMatrixSize3(value)
        VerifyValidRotation(value)

    @staticmethod
    def W(value):
        VerifySquareMatrixSize3(value)
        VerifyAntiSymmetricMatrix(value)

    @staticmethod
    def Q(value):
        VerifySquareMatrixSize3(value)
        VerifyAntiSymmetricMatrix(value)

    @staticmethod
    def element(element):
        VALIDS = ("p", "v", "a", "r", "w", "q", "R", "W", "Q")
        if not isinstance(element, str):
            raise TypeError("The element must be a string")
        if element not in VALIDS:
            raise ValueError("The element must be in %s" % str(VALIDS))

    @staticmethod
    def get(frame, element):
        VerifyFrameReference(frame)
        Validation_Kinematic.element(element)


class Validation_ObjectKinematic:
    @staticmethod
    def CM(value):
        Verify3DVector(value)

    @staticmethod
    def II(value):
        Verify3DTensor(value)

    @staticmethod
    def get(frame, element):
        VerifyFrameReference(frame)
        Validation_ObjectKinematic.element(element)

    @staticmethod
    def element(element):
        VALIDS = ("p", "v", "a", "r", "w", "q", "R", "W", "Q", "CM", "II")
        if not isinstance(element, str):
            raise TypeError("The element must be a string")
        if element not in VALIDS:
            raise ValueError("The element must be in %s" % str(VALIDS))


class Validation_Compute:
    @staticmethod
    def Ux2u(Ux):
        VerifySquareMatrixSize3(Ux)
        VerifyAntiSymmetricMatrix(Ux)

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
        VerifySquareMatrixSize3(R)

    @staticmethod
    def w2W(w):
        Verify3DVector(w)

    @staticmethod
    def W2w(W):
        VerifySquareMatrixSize3(W)
        VerifyAntiSymmetricMatrix(W)

    @staticmethod
    def q2Q(q):
        Verify3DVector(q)

    @staticmethod
    def Q2q(Q):
        VerifySquareMatrixSize3(Q)
        VerifyAntiSymmetricMatrix(Q)


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
    def init(baseframe, translation, rotation):
        if baseframe is None:
            if translation is not None:
                raise ValueError(
                    "translation must be None when baseframe is None")
            if rotation is not None:
                raise ValueError(
                    "rotation must be None when baseframe is None")
        Validation_FrameReference.baseframe(baseframe)
        Validation_FrameReference.translation(translation)
        Validation_FrameReference.rotation(rotation)

    @staticmethod
    def baseframe(baseframe):
        if baseframe is not None:
            VerifyFrameReference(baseframe)

    @staticmethod
    def translation(translation):
        Validation_Translation.init(translation)

    @staticmethod
    def rotation(rotation):
        Validation_Rotation.init(rotation)


class Validation_FrameComposition:

    @staticmethod
    def p1(kine01, kine0):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def v1(kine01, kine0):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def a1(kine01, kine0):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def R1(kine01, kine0):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def w1(kine01, kine0):
        VerifyKinematicClass(kine01)
        VerifyKinematicClass(kine0)

    @staticmethod
    def q1(kine01, kine0):
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
        VerifyFrameReference(baseframe)

    @staticmethod
    def name(name):
        if name is not None:
            VerifyString(name)

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
        VerifyFrameReference(frame)

    @staticmethod
    def getII(frame):
        VerifyFrameReference(frame)

    @staticmethod
    def get(frame, element):
        VerifyFrameReference(frame)
        Validation_Object.verifyElement(element)


class Validation_Variable:
    @staticmethod
    def init(name):
        VerifyString(name)
        if " " in name:
            raise ValueError("The variable's name must not contain space")


class Validation_Energy:
    @staticmethod
    def init(*args, **kwargs):
        pass

    @staticmethod
    def shape1(array, n):
        if array.shape[0] != n:
            error = "The shape of parameter must be (%d) cause len(X) = %d"
            raise AttributeError(error % (n, n))

    @staticmethod
    def shape2(array, n):
        if array.shape[0] != n:
            error = "The shape of parameter must be (%d, %d) cause len(X) = %d"
            raise AttributeError(error % (n, n, n))

    @staticmethod
    def M(M, n):
        VerifySymmetricMatrix(M)
        Validation_Energy.shape2(M, n)

    @staticmethod
    def V(V, n):
        VerifySquareMatrix(V)
        Validation_Energy.shape2(V, n)

    @staticmethod
    def K(K, n):
        VerifySymmetricMatrix(K)
        Validation_Energy.shape2(K, n)

    @staticmethod
    def A(A, n):
        VerifyIsVector(A)
        Validation_Energy.shape1(A, n)

    @staticmethod
    def B(B, n):
        VerifyIsVector(B)
        Validation_Energy.shape1(B, n)

    @staticmethod
    def C(C):
        VerifyScalar(C)
