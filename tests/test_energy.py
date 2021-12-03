import pytest
import numpy as np
from compmec.rbdyn.variable import Variable
from compmec.rbdyn.energy import KineticEnergy, Energy


def randomSymmetricMatrix(n):
    M = np.random.rand(n, n)
    return M + np.transpose(M)


# @pytest.mark.dependency(
#     depends=["tests/test_variables.py::test_allgood"])
@pytest.mark.dependency()
def test_begin():
    pass



@pytest.mark.dependency(depends=["test_begin"])
@pytest.mark.timeout(2)
def test_Build():
    Energy(0)


@pytest.mark.dependency(depends=["test_Build"])
@pytest.mark.timeout(2)
def test_Constant():
    Ntests = 1  # 10
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a)
        assert E == a


@pytest.mark.dependency(depends=["test_Constant"])
@pytest.mark.timeout(2)
def test_LinearPosition():
    Ntests = 1  # 10
    x = Variable("x")
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x)
        assert E == a * x


@pytest.mark.dependency(depends=["test_Constant"])
@pytest.mark.timeout(2)
def test_LinearSpeed():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * dx)
        assert E == a * dx


@pytest.mark.dependency(depends=["test_LinearPosition"])
@pytest.mark.timeout(2)
def test_QuadPosition():
    Ntests = 1  # 10
    x = Variable("x")
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x**2)
        assert E == a * x**2


@pytest.mark.dependency(depends=["test_LinearPosition", "test_LinearSpeed"])
@pytest.mark.timeout(2)
def test_QuadPositionSpeed():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x * dx)
        assert E == a * x * dx


@pytest.mark.dependency(depends=["test_LinearSpeed"])
@pytest.mark.timeout(2)
def test_QuadSpeed():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * dx**2)
        assert E == a * dx**2


@pytest.mark.dependency(depends=["test_LinearPosition", "test_LinearSpeed"])
@pytest.mark.timeout(2)
def test_Linear():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a, b, c = np.random.rand(3)
        totalenergy = c
        totalenergy += b * x
        totalenergy += a * dx
        E = Energy(totalenergy)
        assert E == totalenergy


@pytest.mark.dependency(depends=["test_QuadPosition", "test_QuadPositionSpeed", "test_QuadSpeed"])
@pytest.mark.timeout(2)
def test_Quad():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a, b, c = np.random.rand(3)
        totalenergy = c
        totalenergy += b * x
        totalenergy += a * dx
        E = Energy(totalenergy)
        assert E == totalenergy


@pytest.mark.dependency(depends=["test_Linear", "test_Quad"])
@pytest.mark.timeout(20)
def test_AllQuantity():
    Ntests = 1  # 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        m, v, k, a, b, c = np.random.rand(6)
        totalenergy = c
        totalenergy += a * dx
        totalenergy += b * x
        totalenergy += k * x**2 / 2
        totalenergy += v * x * dx
        totalenergy += m * dx**2 / 2
        E = Energy(totalenergy)
        assert E == totalenergy

@pytest.mark.dependency(depends=["test_AllQuantity"])
@pytest.mark.timeout(2)
def test_CompareWithMatrixDim2_Standard():
    x = Variable("x")
    y = Variable("y")
    dx = x.dt
    dy = y.dt
    X = (x, y)
    
    M = np.array([[14, 2],
                  [2, 12]])
    V = np.array([[4, 6],
                  [3, 5]])
    K = np.array([[-6, -3],
                  [-3, -10]])

    A = np.array([1, -7])
    B = np.array([-5, 9])
    C = 80

    ener = 7*(dx**2) + 2*dx*dy + 6*(dy**2) + \
           4*dx*x + 6*dx*y + 3*dy*x + 5*dy*y + \
           (-3)*x**2 + (-3)*x*y + (-5)*y**2 + \
           1*dx + (-7)*dy + (-5)*x + 9*y + 80


    assert Energy(ener) == ener


@pytest.mark.dependency(depends=["test_CompareWithMatrixDim2_Standard"])
@pytest.mark.timeout(60)
def test_CompareWithMatrixDim2_Random():
    Ntests = 1  # 10
    x = Variable("x")
    y = Variable("y")
    X = np.array([x, y])
    dX = np.array([x.dt, y.dt])
    for i in range(Ntests):
        M = randomSymmetricMatrix(2)
        V = np.random.rand(2, 2)
        K = randomSymmetricMatrix(2)
        A = np.random.rand(2)
        B = np.random.rand(2)
        C = np.random.rand()
        totalenergy = C
        totalenergy += np.dot(dX, np.dot(M, dX)) / 2
        totalenergy += np.dot(dX, np.dot(V, X))
        totalenergy += np.dot(X, np.dot(K, X)) / 2
        totalenergy += np.dot(A, dX)
        totalenergy += np.dot(B, X)
        E = Energy(totalenergy)
        assert E == totalenergy


@pytest.mark.timeout(120)
@pytest.mark.dependency(depends=["test_CompareWithMatrixDim2_Random"])
def test_CompareWithMatrixDim5_Random():
    Ntests = 1  # 2
    dim = 5
    X = []
    dX = []
    for i in range(dim):
        xi = Variable("x%d" % i)
        X.append(xi)
        dX.append(xi.dt)
    X = np.array(X)
    dX = np.array(dX)
    for i in range(Ntests):
        M = randomSymmetricMatrix(dim)
        V = np.random.rand(dim, dim)
        K = randomSymmetricMatrix(dim)
        A = np.random.rand(dim)
        B = np.random.rand(dim)
        C = np.random.rand()
        totalenergy = C
        totalenergy += np.dot(dX, np.dot(M, dX)) / 2
        totalenergy += np.dot(dX, np.dot(V, X))
        totalenergy += np.dot(X, np.dot(K, X)) / 2
        totalenergy += np.dot(A, dX)
        totalenergy += np.dot(B, X)
        print("z")
        E = Energy(totalenergy)
        print("k")
        assert E == totalenergy

@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_CompareWithMatrixDim2_Standard"])
def test_SetMatrixDim2_Standard():
    x = Variable("x")
    y = Variable("y")
    dx = x.dt
    dy = y.dt
    X = (x, y)
    
    M = np.array([[14, 2],
                  [2, 12]])
    V = np.array([[4, 6],
                  [3, 5]])
    K = np.array([[-6, -3],
                  [-3, -10]])

    A = np.array([1, -7])
    B = np.array([-5, 9])
    C = 80

    ener = 7*(dx**2) + 2*dx*dy + 6*(dy**2) + \
           4*dx*x + 6*dx*y + 3*dy*x + 5*dy*y + \
           (-3)*x**2 + (-3)*x*y + (-5)*y**2 + \
           1*dx + (-7)*dy + (-5)*x + 9*y + 80
    E = Energy.frommatrix(X=X, M=M, V=V, K=K, A=A, B=B, C=C)
    ener = Energy(ener)
    assert E == ener
    

@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_SetMatrixDim2_Standard"])
def test_SetMatrixDim2_Random():
    Ntests = 1  # 2
    x = Variable("x")
    y = Variable("y")
    X = np.array([x, y])
    dX = np.array([x.dt, y.dt])
    for i in range(Ntests):
        M = randomSymmetricMatrix(2)
        V = np.random.rand(2, 2)
        K = randomSymmetricMatrix(2)
        A = np.random.rand(2)
        B = np.random.rand(2)
        C = np.random.rand()
        totalenergy = C
        totalenergy += np.dot(dX, np.dot(M, dX)) / 2
        totalenergy += np.dot(dX, np.dot(V, X))
        totalenergy += np.dot(X, np.dot(K, X)) / 2
        totalenergy += np.dot(A, dX)
        totalenergy += np.dot(B, X)
        E = Energy.frommatrix(X=X, M=M, V=V, K=K, A=A, B=B, C=C)
        assert E == totalenergy


@pytest.mark.timeout(120)
@pytest.mark.dependency(depends=["test_SetMatrixDim2_Random"])
def test_SetMatrixDim5_Random():
    Ntests = 1  # 2
    dim = 5
    X = []
    dX = []
    for i in range(dim):
        xi = Variable("x%d" % i)
        X.append(xi)
        dX.append(xi.dt)
    X = np.array(X)
    dX = np.array(dX)
    for i in range(Ntests):
        M = randomSymmetricMatrix(dim)
        V = np.random.rand(dim, dim)
        K = randomSymmetricMatrix(dim)
        A = np.random.rand(dim)
        B = np.random.rand(dim)
        C = np.random.rand()
        totalenergy = C
        totalenergy += np.dot(dX, np.dot(M, dX)) / 2
        totalenergy += np.dot(dX, np.dot(V, X))
        totalenergy += np.dot(X, np.dot(K, X)) / 2
        totalenergy += np.dot(A, dX)
        totalenergy += np.dot(B, X)
        E = Energy.frommatrix(X=X, M=M, V=V, K=K, A=A, B=B, C=C)
        assert E == totalenergy


@pytest.mark.timeout(20)
@pytest.mark.dependency(depends=["test_AllQuantity"])
def test_KineticEnergy_Standard():
    x = Variable("x")
    dx = x.dt
    mass = 1
    vector3D = (dx, 0, 0)
    E_good = mass*(dx**2)/2
    E_test = KineticEnergy(mass, vector3D)
    E_good = Energy(E_good)
    assert E_test == E_good

@pytest.mark.timeout(60)
@pytest.mark.dependency(depends=["test_KineticEnergy_Standard"])
def test_KineticEnergy_Random():
    Ntests = 1  # 10
    x = Variable("x")
    y = Variable("y")
    for i in range(Ntests):
        mass = np.random.rand()
        vector3D = np.zeros(3, dtype="object")
        vector3D += x.dt * np.random.rand(3)
        vector3D += y.dt * np.random.rand(3)
        E = KineticEnergy(mass, vector3D)

        assert E == mass * np.dot(vector3D, vector3D) / 2


@pytest.mark.dependency(depends=["test_SetMatrixDim5_Random", "test_KineticEnergy_Random"])
def test_allgood():
    pass

