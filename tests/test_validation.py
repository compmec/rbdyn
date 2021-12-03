import pytest
import numpy as np
from compmec.rbdyn.__validation__ import Verify


def getRandomUnitVector3D():
    v = np.random.rand(3)
    norm_v = np.linalg.norm(v)
    return v / norm_v


@pytest.mark.dependency()
def test_VerifyVector3D():
    u = np.zeros(3)
    Verify.Vector3D(u)

    u = np.random.rand(3)
    Verify.Vector3D(u)


@pytest.mark.dependency()
def test_VerifyTensor3D():
    X = np.zeros((3, 3))
    Verify.Tensor3D(X)

    X = np.eye(3)
    Verify.Tensor3D(X)

    X = np.random.rand(3, 3)
    Verify.Tensor3D(X + X.T)


@pytest.mark.dependency(depends=["test_VerifyTensor3D"])
def test_VerifyAntiSymmetricMatrix():
    X = np.zeros((3, 3))
    Verify.AntiSymmetricMatrix(X)

    X = np.random.rand(3, 3)
    Xsym = (X + X.T) / 2
    Xant = X - Xsym
    Verify.AntiSymmetricMatrix(Xant)

    with pytest.raises(ValueError):
        Verify.AntiSymmetricMatrix(Xsym)


@pytest.mark.dependency(depends=["test_VerifyVector3D"])
def test_VerifyUnitVector3D():
    Verify.UnitVector3D((1, 0, 0))
    Verify.UnitVector3D((0, 1, 0))
    Verify.UnitVector3D((0, 0, 1))
    Verify.UnitVector3D((-1, 0, 0))
    Verify.UnitVector3D((0, -1, 0))
    Verify.UnitVector3D((0, 0, -1))
    Verify.UnitVector3D((0.6, 0.8, 0))
    Verify.UnitVector3D((0, 0.8, 0.6))

    Ntests = 100
    for i in range(Ntests):
        u = getRandomUnitVector3D()
        Verify.UnitVector3D(u)
