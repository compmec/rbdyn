import pytest
import numpy as np
import rbdyn.__validation__ as validation


def getRandomUnit3DVector():
    v = np.random.rand(3)
    norm_v = np.linalg.norm(v)
    return v / norm_v


@pytest.mark.dependency()
def test_Verify3DVector():
    u = np.zeros(3)
    validation.Verify3DVector(u)

    u = np.random.rand(3)
    validation.Verify3DVector(u)


@pytest.mark.dependency()
def test_Verify3DTensor():
    X = np.zeros((3, 3))
    validation.Verify3DTensor(X)

    X = np.random.rand(3, 3)
    validation.Verify3DTensor(X)


@pytest.mark.dependency(depends=["test_Verify3DTensor"])
def test_VerifyAntiSymmetric():
    X = np.zeros((3, 3))
    validation.VerifyAntiSymmetric(X)

    X = np.random.rand(3, 3)
    Xsym = (X + X.T) / 2
    Xant = X - Xsym
    validation.VerifyAntiSymmetric(Xant)

    with pytest.raises(ValueError):
        validation.VerifyAntiSymmetric(Xsym)


@pytest.mark.dependency(depends=["test_Verify3DVector"])
def test_VerifyUnit3DVector():
    validation.VerifyUnit3DVector((1, 0, 0))
    validation.VerifyUnit3DVector((0, 1, 0))
    validation.VerifyUnit3DVector((0, 0, 1))
    validation.VerifyUnit3DVector((-1, 0, 0))
    validation.VerifyUnit3DVector((0, -1, 0))
    validation.VerifyUnit3DVector((0, 0, -1))
    validation.VerifyUnit3DVector((0.6, 0.8, 0))
    validation.VerifyUnit3DVector((0, 0.8, 0.6))

    Ntests = 100
    for i in range(Ntests):
        u = getRandomUnit3DVector()
        validation.VerifyUnit3DVector(u)
