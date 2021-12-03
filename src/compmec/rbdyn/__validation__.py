import numpy as np
import sympy as sp
import compmec.rbdyn.__classes__ as classes


class IS:

    @staticmethod
    def List(value):
        return isinstance(value, list)

    @staticmethod
    def Tuple(value):
        return isinstance(value, tuple)

    @staticmethod
    def SympyBasic(value):
        return isinstance(value, sp.core.basic.Basic)

    @staticmethod
    def SympyExpression(value):
        if not IS.SympyBasic(value):
            return False
        try:
            value = sp.sympify(value)
            return True
        except Exception as e:
            return False

    @staticmethod
    def SympyArray(value):
        if not IS.SympyBasic:
            return False
        if isinstance(value, sp.tensor.array.dense_ndim_array.ImmutableDenseNDimArray):
            return True
        return False

    @staticmethod
    def NumpyArray(value):
        if isinstance(value, np.ndarray):
            return True
        return False

    @staticmethod
    def Array(value):
        if IS.SympyArray(value):
            return True
        if IS.NumpyArray(value):
            return True
        if IS.List(value):
            return True
        if IS.Tuple(value):
            return True
        return False

    @staticmethod
    def SympyArray1D(value):
        if not IS.SympyArray(value):
            return False
        if value.rank() != 1:
            return False
        return True

    @staticmethod
    def SympyArray2D(value):
        if not IS.SympyArray(value):
            return False
        if value.rank() != 2:
            return False
        return True

    @staticmethod
    def SympyMatrix(value):
        if not IS.SympyBasic(value):
            return False
        return isinstance(value, sp.matrices.dense.MutableDenseMatrix)

    @staticmethod
    def NumpyArray1D(value):
        if not IS.NumpyArray(value):
            return False
        if value.ndim != 1:
            return False
        return True

    @staticmethod
    def NumpyArray2D(value):
        if not IS.NumpyArray(value):
            return False
        if value.ndim != 2:
            return False
        return True


    @staticmethod
    def Array1D(value):
        if IS.NumpyArray1D(value):
            return True
        if IS.SympyArray1D(value):
            return True
        return np.array(value).ndim == 1

    @staticmethod
    def Array2D(value):
        if IS.SympyArray2D(value):
            return True
        if IS.NumpyArray2D(value):
            return True
        return np.array(value).ndim == 2


    @staticmethod
    def Matrix(value):
        if IS.SympyMatrix(value):
            return True
        return IS.Array2D(value)

    @staticmethod
    def SquareMatrix(value):
        if not IS.Matrix(value):
            return False
        if IS.NumpyArray(value):
            shape = value.shape
            if shape[0] == shape[1]:
                return True
            return False
        if IS.SympyBasic(value):
            if IS.SympyMatrix(value):
                shape = sp.shape(value)
            elif IS.SympyArray(value):
                shape = sp.shape(value)
            else:
                return False
            if shape[0] == shape[1]:
                return True
            return False
        if IS.Tuple(value) or IS.List(value):
            value = np.array(value)
            return IS.SquareMatrix(value)
        return False


    @staticmethod
    def SymmetricMatrix(value):
        """
        This means that the matrix is the same as its transpose
        """
        if not IS.SquareMatrix(value):
            return False
        if IS.NumpyArray(value):
            return np.all(np.abs(value - np.transpose(value)) < 1e-13)
        if IS.SympyBasic(value):
            if IS.SympyMatrix(value):
                return matrix == matrix.T
            if IS.SympyArray(value):
                matrix = sp.Matrix(value)
                return (matrix == matrix.T)
        if IS.Tuple(value) or IS.List(value):
            value = np.array(value)
            return IS.SymmetricMatrix(value)
        return False

    @staticmethod
    def AntiSymmetricMatrix(value):
        """
        That means all the values in the diagonal are zero
        And M[i, j] = -M[j, i] 
        """
        if not IS.SquareMatrix(value):
            return False
        if IS.NumpyArray(value):
            if np.any(np.diag(value) != 0):
                return False
            matrix = np.copy(value)
            return np.all(np.abs(matrix + np.transpose(matrix)) < 1e-13)
        if IS.SympyBasic(value):
            if IS.SympyMatrix(value):
                pass
            elif IS.SympyArray(value):
                value = sp.Matrix(value)
            else:
                return False
            n, = sp.shape(value)[0]
            return (value + value.T) == sp.zeros(n, n)
        if IS.Tuple(value) or IS.List(value):
            value = np.array(value)
            return IS.AntiSymmetricMatrix(value)
        return False





def isNumericValue(value):
    if isinstance(value, int):
        return True
    elif isinstance(value, float):
        return True
    elif isinstance(value, str):
        return False
    elif isinstance(value, np.ndarray):
        pass
    elif isinstance(value, (list, tuple)):
        try:
            value = np.array(value)
        except Exception:
            return False
    else:
        return False

    if np.issubdtype(value.dtype, np.number):
        return True
    else:
        return False

def isSympyExpression(expr):
    try:
        convexpr = sp.sympify(expr)
        return True
    except Exception as e:
        return False

def isSymbolicValue(value):
    if isinstance(value, (list, tuple)):
        value = np.array(value)
    elif isinstance(value, np.ndarray):
        pass
    elif isSympyExpression(value):
        return True
    else:
        return False

    for i in range(value.shape[0]):
        v = value[0]
        if isSymbolicValue(v):
            return True
    return False


class Verify:

    @staticmethod
    def Scalar(value):
        if isinstance(value, (int, float)):
            pass
        if isinstance(value, (tuple, list, np.ndarray)):
            error = "The argument is not an scalar, but a %s"
            raise TypeError(error % str(type(value)))
        
        if isinstance(value, sp.tensor.ImmutableDenseNDimArray):
            raise TypeError("The argument must be scalar, not sympy.ImmutableDenseNDimArray")

    @staticmethod
    def ConvertToNumpyArray(value):
        if isinstance(value, np.ndarray):
            pass
        elif isinstance(value, tuple):
            pass
        elif isinstance(value, list):
            pass
        else:
            error = "The argument must be a numpy.ndarray/list/tuple. Received %s"
            raise TypeError(error % type(value))
        try:
            value = np.array(value)
        except Exception as e:
            error = "We cannot convert the argument to numpy array: cause %s"
            raise ValueError(error % str(e))

    @staticmethod
    def IsVector(value):
        if IS.List(value):
            pass
        elif IS.Tuple(value):
            pass
        elif IS.NumpyArray(value):
            pass
        elif IS.SympyArray(value):
            pass
        else:
            error = "Received vector must be list/tuple/np.ndarray/sp.Array, not %s"
            raise TypeError(error % str(type(value)))

        if IS.Array1D(value):
            pass
        else:
            raise ValueError("Vector must have ndim = 1")

    @staticmethod
    def IsMatrix(value):
        Verify.ConvertToNumpyArray(value)
        value = np.array(value)
        if value.ndim != 2:
            raise ValueError("The given argument must be a matrix")


    @staticmethod
    def Vector3D(value):
        Verify.IsVector(value)
        if len(value) != 3:
            raise ValueError("The 3D vector must have 3 elements!")

    @staticmethod
    def SquareMatrix(matrix):
        Verify.IsMatrix(matrix)
        matrix = np.array(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            error = "The argument must be a square matrix. Shape = %s"
            raise ValueError(error % str(matrix.shape))



    @staticmethod
    def SymmetricMatrix(value):
        if not IS.Matrix(value):
            if IS.NumpyArray(value):
                error = "Received a numpy array, but not a matrix"
                raise ValueError(error)
            elif IS.SympyArray(value):
                error = "Received a sympy array, but not a matrix"
                raise ValueError(value)
            else:
                error = f"Received value is not a matrix! type={type(value)}"
                raise TypeError(error)
        if not IS.SquareMatrix(value):
            raise ValueError("Received a matrix, but it's not square")
        if not IS.SymmetricMatrix(value):
            raise ValueError("Received a square matrix, but it's not symmetric")



    @staticmethod
    def AntiSymmetricMatrix(value):
        if not IS.Matrix(value):
            if IS.NumpyArray(value):
                error = "Received a numpy array, but not a matrix"
                raise ValueError(error)
            elif IS.SympyArray(value):
                error = "Received a sympy array, but not a matrix"
                raise ValueError(value)
            else:
                error = f"Received value is not a matrix! type={type(value)}"
                raise TypeError(error)
        if not IS.SquareMatrix(value):
            raise ValueError("Received a matrix, but it's not square")
        if not IS.AntiSymmetricMatrix(value):
            raise ValueError("Received a square matrix, but it's not anti-symmetric")


    @staticmethod
    def SquareMatrixSize3(value):
        Verify.SquareMatrix(value)
        value = np.array(value)
        if value.shape[0] != 3:
            raise ValueError("The 3D tensor must have 3x3 elements!")

    @staticmethod
    def Tensor3D(value):
        Verify.SymmetricMatrix(value)
        Verify.SquareMatrixSize3(value)


    @staticmethod
    def UnitVector3D(value):
        Verify.Vector3D(value)
        if np.abs(np.linalg.norm(value) - 1) > 1e-10:
            raise ValueError("The given vector must have module 1")

    @staticmethod
    def ValidRotation(value):
        Verify.SquareMatrix(value)
        Verify.SquareMatrixSize3(value)
        R = np.array(value)
        RT = np.transpose(R)
        M = np.dot(RT, R)
        diff = M - np.eye(3)
        if isNumericValue(R):
            pass
        elif isSymbolicValue(R):
            for i in range(3):
                for j in range(3):
                    diff[i, j] = sp.expand(diff[i, j])
                    diff[i, j] = sp.simplify(diff[i, j])
        if np.any(np.abs(diff) > 1e-10):
            raise ValueError("The given R is not a rotation matrix!")

    @staticmethod
    def Angle(angle):
        if isNumericValue(angle):
            pass
        elif isSymbolicValue(angle):
            pass
        else:
            raise TypeError("Angle must be a number or a variable")

    @staticmethod
    def Boolean(init):
        if not isinstance(init, bool):
            raise TypeError("The argument of Kinematic must be boolean")


    @staticmethod
    def String(name):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, not {type(name)}")


    @staticmethod
    def KinematicBaseClass(kine):
        if not isinstance(kine, classes.KinematicBaseClass):
            raise TypeError("The argument must be a Kinematic instance")

    @staticmethod
    def FrameReference(frame):
        if not isinstance(frame, classes.FrameReferenceBaseClass):
            raise TypeError("The argument must be a FrameReference instance")

    @staticmethod
    def Energy(energy):
        if not isinstance(energy, classes.EnergyBaseClass):
            raise TypeError("The energy must be a Energy instance")

    @staticmethod
    def Variable(var):
        if not isinstance(type(var), sp.core.function.UndefinedFunction):
            errormsg = "The variable must be a Variable instance. Received %s"
            raise TypeError(errormsg % type(var))

    @staticmethod
    def SympyExpression(expr):
        if not isSympyExpression(expr):
            error = "The argument must be a sympy expression, not %s"
            raise TypeError(error % type(expr))



class ValidationBase:
    pass

class Validation_EnergyMatrix(ValidationBase):
    pass



class Validation_Kinematic(ValidationBase):
    @staticmethod
    def init(self, init):
        Verify.Boolean(init)

    @staticmethod
    def psetter(self, value):
        Verify.Vector3D(value)

    @staticmethod
    def vsetter(self, value):
        Verify.Vector3D(value)

    @staticmethod
    def asetter(self, value):
        Verify.Vector3D(value)

    @staticmethod
    def rsetter(self, value):
        if value is None:
            raise TypeError("rotation must be not None")
        Validation_Rotation.init(value)

    @staticmethod
    def wsetter(self, value):
        Verify.Vector3D(value)

    @staticmethod
    def qsetter(self, value):
        Verify.Vector3D(value)

    @staticmethod
    def Rsetter(self, value):
        Verify.SquareMatrixSize3(value)
        Verify.ValidRotation(value)

    @staticmethod
    def Wsetter(self, value):
        Verify.SquareMatrixSize3(value)
        Verify.AntiSymmetricMatrix(value)

    @staticmethod
    def Qsetter(self, value):
        Verify.SquareMatrixSize3(value)
        Verify.AntiSymmetricMatrix(value)

    @staticmethod
    def element(element):
        VALIDS = ("p", "v", "a", "r", "w", "q", "R", "W", "Q")
        if not isinstance(element, str):
            raise TypeError("The element must be a string")
        if element not in VALIDS:
            raise ValueError("The element must be in %s" % str(VALIDS))

    @staticmethod
    def get(frame, element):
        Verify.FrameReference(frame)
        Validation_Kinematic.element(element)


class Validation_ObjectKinematic(Validation_Kinematic):

    @staticmethod
    def CMsetter(inst, value):
        Verify.Vector3D(value)

    @staticmethod
    def IIsetter(inst, value):
        Verify.Tensor3D(value)

    @staticmethod
    def get(inst, element):
        # Verify.FrameReference(frame)
        Validation_ObjectKinematic.element(inst, element)

    @staticmethod
    def element(self, element):
        VALIDS = ("p", "v", "a", "r", "w", "q", "R", "W", "Q", "CM", "II")
        if not isinstance(element, str):
            raise TypeError("The element must be a string")
        if element not in VALIDS:
            raise ValueError("The element must be in %s" % str(VALIDS))


class Validation_Compute:
    @staticmethod
    def Ux2u(Ux):
        Verify.SquareMatrixSize3(Ux)
        Verify.AntiSymmetricMatrix(Ux)

    @staticmethod
    def u2Ux(u):
        Verify.Vector3D(u)

    @staticmethod
    def R(angle, u):
        Verify.Angle(angle)
        Validation_Rotation.angle(angle)
        Validation_Rotation.direction(u)

    @staticmethod
    def r2R(r):
        Verify.Vector3D(r)

    @staticmethod
    def R2r(R):
        Verify.SquareMatrixSize3(R)

    @staticmethod
    def w2W(w):
        Verify.Vector3D(w)

    @staticmethod
    def W2w(W):
        Verify.SquareMatrixSize3(W)
        Verify.AntiSymmetricMatrix(W)

    @staticmethod
    def q2Q(q):
        Verify.Vector3D(q)

    @staticmethod
    def Q2q(Q):
        Verify.SquareMatrixSize3(Q)
        Verify.AntiSymmetricMatrix(Q)


class Validation_Translation:

    @staticmethod
    def init(translation):
        if translation is None:
            pass
        else:
            Verify.Vector3D(translation)


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
            Verify.Vector3D(rotation)

    @staticmethod
    def angle(angle):
        Verify.Angle(angle)

    @staticmethod
    def direction(direction):
        if isinstance(direction, str):
            if direction in ("x", "y", "z"):
                return
            raise ValueError("When direction is string, it must be x/y/z")
        Verify.UnitVector3D(direction)

class Validation_FrameReference:

    @staticmethod
    def init(self, baseframe, translation, rotation):
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
            Verify.FrameReference(baseframe)

    @staticmethod
    def translation(translation):
        Validation_Translation.init(translation)

    @staticmethod
    def rotation(rotation):
        Validation_Rotation.init(rotation)


class Validation_FrameComposition:

    @staticmethod
    def p1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def v1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def a1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def R1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def w1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def q1(kine01, kine0):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine0)

    @staticmethod
    def p0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def v0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def a0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def R0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def w0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def q0(kine01, kine1):
        Verify.KinematicBaseClass(kine01)
        Verify.KinematicBaseClass(kine1)

    @staticmethod
    def PAT(CM):
        Verify.Vector3D(CM)


class Validation_Object:
    KINEMATIC_ELEMENTS = ("p", "v", "a", "r", "w", "q",
                          "R", "W", "Q", "CM", "II")

    @staticmethod
    def init(baseframe, name):
        Validation_Object.baseframe(baseframe)
        Validation_Object.name(name)

    @staticmethod
    def baseframe(baseframe):
        Verify.FrameReference(baseframe)

    @staticmethod
    def name(name):
        if name is not None:
            Verify.String(name)

    @staticmethod
    def verifyElement(element):
        if not isinstance(element, str):
            raise TypeError("The element must be a string!")
        if element not in Validation_Object.KINEMATIC_ELEMENTS:
            error = "The element must be in %s" % str(
                Validation_Object.KINEMATIC_ELEMENTS)
            raise ValueError(error)

    @staticmethod
    def masssetter(self, value):
        if isinstance(value, str):
            raise TypeError("Mass must not be a string")

    @staticmethod
    def CMsetter(self, value):
        Validation_ObjectKinematic.CMsetter(self, value)

    @staticmethod
    def IIsetter(self, value):
        Validation_ObjectKinematic.IIsetter(self, value)

    @staticmethod
    def getCM(frame):
        Verify.FrameReference(frame)

    @staticmethod
    def getII(frame):
        Verify.FrameReference(frame)

    @staticmethod
    def get(frame, element):
        Verify.FrameReference(frame)
        Validation_Object.verifyElement(element)


class Validation_Variable:

    @staticmethod
    def new(cls, name):
        Verify.String(name)
        if " " in name:
            raise ValueError("The variable's name must not contain space")

    @staticmethod
    def index(cls, value):
        pass
        # if isinstance(value, str):
        #     pass
        # elif value in :
        #     pass
        # else:
        #     raise TypeError("Index must receive 'name' or 'variable'")
        

    @staticmethod
    def init(cls, name):
        Verify.String(name)
        if " " in name:
            raise ValueError("The variable's name must not contain space")

class Validation_VariableList(ValidationBase):
    
    @staticmethod
    def init(self, iterable):
        pass

    @staticmethod
    def fromexpression(cls, expression):
        pass

    @staticmethod
    def add(self, item):
        pass

    @staticmethod
    def iadd(self, item):
        pass



class Validation_EnergyMatrix(ValidationBase):



    @staticmethod
    def Msetter(self, value):
        if value is None:
            return
        Verify.SymmetricMatrix(value)

    @staticmethod
    def Vsetter(self, value):
        if value is None:
            return

    @staticmethod
    def Ksetter(self, value):
        if value is None:
            return
        Verify.SymmetricMatrix(value)

    @staticmethod
    def Asetter(self, value):
        if value is None:
            return

    @staticmethod
    def Bsetter(self, value):
        if value is None:
            return
        pass

    @staticmethod
    def Csetter(self, value):
        if value is None:
            return
        pass

    @staticmethod
    def Xsetter(self, value):
        pass


class Validation_Energy(Validation_EnergyMatrix):

    @staticmethod
    def init(self, expression):
        Verify.SympyExpression(expression)

    @staticmethod
    def frommatrix(cls, X, M=None, V=None, K=None, A=None, B=None, C=None):
        Verify.IsVector(X)


    @staticmethod
    def Kinetic(cls, mass, velocity):
        pass






class Validation_Simulation:
    @staticmethod
    def run(initialconditions, timesteps, totalenergy, constraintfunctions, force):
        Validation_Simulation.initialconditions(initialconditions)
        Validation_Simulation.timesteps(timesteps)
        Validation_Simulation.totalenergy(totalenergy)
        Validation_Simulation.constraintfunctions(constraintfunctions)
        Validation_Simulation.force(force)
        Validation_Simulation.allvariableshasinitialconditions(initialconditions, totalenergy)



    @staticmethod
    def initialconditions(initialconditions):
        if not isinstance(initialconditions, dict):
            raise TypeError("The initial conditions must be a dictionary")
        for key, value in initialconditions.items():
            Verify.Variable(key)
            Verify.IsVector(value)
            if len(value) != 2:
                error = "The size of initial conditions for each variable must be 2"
                raise ValueError(error)

    @staticmethod
    def timesteps(timesteps):
        Verify.IsVector(timesteps)
        if not isNumericValue(timesteps):
            raise ValueError("The timesteps given must be an array with numeric values")


    @staticmethod
    def totalenergy(totalenergy):
        Verify.Energy(totalenergy)

    @staticmethod
    def constraintfunctions(constraintfunctions):
        Verify.IsVector(constraintfunctions)

    @staticmethod
    def force(force):
        if force is None:
            pass
        elif callable(force):
            pass
        else:
            raise TypeError("The force function must be callable")


    @staticmethod
    def allvariableshasinitialconditions(initialconditions, totalenergy):
        X = totalenergy.X
        for var in X:
            if var not in initialconditions:
                errormsg = "The initial conditions for the variable %s is missing"
                raise ValueError(errormsg % str(var))


class Validation_LagrangianMatrix(ValidationBase):
    @staticmethod
    def Xsetter(self, value):
        if value is None:
            return

    @staticmethod
    def Msetter(self, value):
        if value is None:
            return
    
    @staticmethod
    def Csetter(self, value):
        if value is None:
            return
    
    @staticmethod
    def Ksetter(self, value):
        if value is None:
            return
    
    @staticmethod
    def Zsetter(self, value):
        if value is None:
            return
   
    @staticmethod
    def Mvvsetter(self, value):
        if value is None:
            return
    
    @staticmethod
    def Mvpsetter(self, value):
        if value is None:
            return
    
    @staticmethod
    def Mppsetter(self, value):
        if value is None:
            return
    


class Validation_Lagrangian(Validation_LagrangianMatrix):
    @staticmethod
    def init(self, E):
        Verify.Energy(E)


    @staticmethod
    def get(self):
        pass
        # if (X is None) and (dX is None) and (ddX is None):
        #     return
        # if (X is not None) and (dX is not None) and (ddX is not None):
        #     pass
        # else:
        #     error = "(X, dX, ddX) must be (None, None, None) or (not None, not None, not None)"
        #     raise ValueError(error)

        # Verify.IsVector(X)
        # Verify.IsVector(dX)
        # Verify.IsVector(ddX)
        # X = np.array(X)
        # dX = np.array(dX)
        # ddX = np.array(ddX)
        # if X.shape != (n, ):
        #     raise ValueError("The shape of X must be [%d]" % n)
        # if dX.shape != (n, ):
        #     raise ValueError("The shape of dX must be [%d]" % n)
        # if ddX.shape != (n, ):
        #     raise ValueError("The shape of ddX must be [%d]" % n)




class Validation_Solver:

    @staticmethod
    def init(inst, energy, IC, G, F, timesteps):
        Validation_Solver.energy(energy)
        Validation_Solver.IC(IC)
        Validation_Solver.G(G)
        Validation_Solver.F(F)
        Validation_Solver.timesteps(timesteps)
        
    @staticmethod
    def energy(energy):
        if isinstance(energy, classes.EnergyBaseClass):
            pass
        else:
            Verify.SympyExpression(energy)


    @staticmethod
    def IC(IC):
        if not isinstance(IC, dict):
            raise TypeError("The Initial Conditions must be a dict")

    @staticmethod
    def G(G):
        if G is None:
            return
        Verify.IsVector(G)
        for gi in G:
            Verify.SympyExpression(gi)

    @staticmethod
    def F(F):
        Verify.IsVector(F)

    @staticmethod
    def timesteps(timesteps):
        Verify.IsVector(timesteps)

    @staticmethod
    def settert0(inst, t0):
        pass

    @staticmethod
    def setterX0(inst, X0):
        pass

    @staticmethod
    def setterdX0(inst, dX0):
        pass


class Validation_Euler:
    @staticmethod
    def init(inst, energy, IC, G, F, timesteps):
        Validation_Solver.init(inst, energy, IC, G, F, timesteps)



class Validation_Force:
    @staticmethod
    def init(inst, F, X):
        Verify.IsVector(F)
        Verify.IsVector(X)
        n = len(F)
        if len(X) != n:
            errormsg = "The size of X must be the same size of F"
            raise ValueError(errormsg)

    @staticmethod
    def call(inst, t, X, dX):
        if inst.isTDependent and t is None:
            error = "The parameter t must be not None. F is time dependent"
            raise ValueError(error)
        if inst.isXDependent and X is None:
            error = "The parameter X must be not None. F is X dependent"
            raise ValueError(error)
        if inst.isdXDependent and dX is None:
            error = "The parameter dX must be not None. F is dX dependent"
            raise ValueError(error)

