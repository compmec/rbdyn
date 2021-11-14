import pytest
import numpy as np
from compmec.rbdyn.variable import Variable
from compmec.rbdyn.energy import KineticEnergy, Energy


def randomSymmetricMatrix(n):
    M = np.random.rand(n, n)
    return M + np.transpose(M)


@pytest.mark.dependency()
def test_Build():
    Energy()


@pytest.mark.dependency(depends=["test_Build"])
def test_Constant():
    Ntests = 10
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a)
        assert E == a


@pytest.mark.dependency(depends=["test_Constant"])
def test_LinearPosition():
    Ntests = 10
    x = Variable("x")
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x)
        assert E == a * x


@pytest.mark.dependency(depends=["test_Constant"])
def test_LinearSpeed():
    Ntests = 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * dx)
        assert E == a * dx


@pytest.mark.dependency(depends=["test_LinearPosition"])
def test_QuadPosition():
    Ntests = 10
    x = Variable("x")
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x**2)
        assert E == a * x**2


@pytest.mark.dependency(depends=["test_LinearPosition", "test_LinearSpeed"])
def test_QuadPositionSpeed():
    Ntests = 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * x * dx)
        assert E == a * x * dx


@pytest.mark.dependency(depends=["test_LinearSpeed"])
def test_QuadSpeed():
    Ntests = 10
    x = Variable("x")
    dx = x.dt
    for i in range(Ntests):
        a = np.random.rand()
        E = Energy(a * dx**2)
        assert E == a * dx**2


@pytest.mark.dependency(depends=["test_LinearPosition", "test_LinearSpeed"])
def test_Linear():
    Ntests = 10
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
def test_Quad():
    Ntests = 10
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
def test_AllQuantity():
    Ntests = 10
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
def test_CompareWithMatrixDim2():
    Ntests = 10
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


@pytest.mark.dependency(depends=["test_CompareWithMatrixDim2"])
def test_CompareWithMatrixDim5():
    Ntests = 2
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
        E = Energy(totalenergy)
        assert E == totalenergy


@pytest.mark.dependency(depends=["test_AllQuantity"])
def test_SetMatrixDim2():
    Ntests = 10
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
        E = Energy(X=X, M=M, V=V, K=K, A=A, B=B, C=C)
        assert E == totalenergy


@pytest.mark.dependency(depends=["test_SetMatrixDim2"])
def test_SetMatrixDim5():
    Ntests = 2
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
        E = Energy(X=X, M=M, V=V, K=K, A=A, B=B, C=C)
        assert E == totalenergy


@pytest.mark.dependency(depends=["test_AllQuantity"])
def test_KineticEnergy():
    Ntests = 100
    x = Variable("x")
    y = Variable("y")
    for i in range(Ntests):
        mass = np.random.rand()
        vector3D = np.zeros(3, dtype="object")
        vector3D += x.dt * np.random.rand(3)
        vector3D += y.dt * np.random.rand(3)
        E = KineticEnergy(mass, vector3D)
        assert E == mass * np.dot(vector3D, vector3D) / 2
