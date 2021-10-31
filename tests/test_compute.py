import pytest
import numpy as np
from rbdyn.composition import Compute


def getRandomUnit3DVector():
    v = np.random.rand(3)
    norm_v = np.linalg.norm(v)
    return v / norm_v


def randomNumberBetween(a, b):
    if a == b:
        return a
    if b < a:
        a, b = b, a
    diff = b - a
    return a + diff * np.random.rand()


def test_Ux2u():
    Ux = np.zeros((3, 3))
    ugood = np.zeros(3)
    utest = Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, 1, -1),
          (-1, 0, 1),
          (1, -1, 0))
    ugood = (-1, -1, -1)
    utest = Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, 1, 1),
          (-1, 0, 1),
          (-1, -1, 0))
    ugood = (-1, 1, -1)
    utest = Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, -5, 2),
          (5, 0, -7),
          (-2, 7, 0))
    ugood = (7, 2, 5)
    utest = Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)


def test_u2Ux():
    u = np.zeros(3)
    Uxgood = np.zeros((3, 3))
    Uxtest = Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (-1, -1, -1)
    Uxgood = ((0, 1, -1),
              (-1, 0, 1),
              (1, -1, 0))
    Uxtest = Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (-1, 1, -1)
    Uxgood = ((0, 1, 1),
              (-1, 0, 1),
              (-1, -1, 0))
    Uxtest = Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (7, 2, 5)
    Uxgood = ((0, -5, 2),
              (5, 0, -7),
              (-2, 7, 0))
    Uxtest = Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)


def test_u2Ux2u():
    Ntests = 1
    for i in range(Ntests):
        ugood = np.random.rand(3)
        Ux = Compute.u2Ux(ugood)
        utest = Compute.Ux2u(Ux)
        np.testing.assert_almost_equal(utest, ugood)


@pytest.mark.dependency()
def test_R_standard():
    angle = np.pi / 2
    u = (1, 0, 0)
    Rgood = ((1, 0, 0),
             (0, 0, -1),
             (0, 1, 0))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (1, 0, 0)
    Rgood = ((1, 0, 0),
             (0, -1, 0),
             (0, 0, -1))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi / 2
    u = (0, 1, 0)
    Rgood = ((0, 0, 1),
             (0, 1, 0),
             (-1, 0, 0))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (0, 1, 0)
    Rgood = ((-1, 0, 0),
             (0, 1, 0),
             (0, 0, -1))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi / 2
    u = (0, 0, 1)
    Rgood = ((0, -1, 0),
             (1, 0, 0),
             (0, 0, 1))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (0, 0, 1)
    Rgood = ((-1, 0, 0),
             (0, -1, 0),
             (0, 0, 1))
    Rtest = Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)


@pytest.mark.dependency(depends=["test_R_standard"])
def test_R_random():
    Ntests = 100
    angle = 0
    for i in range(Ntests):
        u = getRandomUnit3DVector()
        R = Compute.R(angle, u)
        np.testing.assert_almost_equal(R, np.eye(3))

    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        R = Compute.R(angle, u)
        RT = np.transpose(R)
        M = np.dot(R, RT)
        np.testing.assert_almost_equal(M, np.eye(3))


@pytest.mark.dependency(depends=["test_R_standard", "test_R_random"])
def test_r2R_standard():
    r = np.zeros(3)
    Rgood = np.eye(3)
    Rtest = Compute.r2R(r)
    np.testing.assert_almost_equal(Rtest, Rgood)


@pytest.mark.dependency(depends=["test_r2R_standard"])
def test_r2R_random():
    Ntests = 1000
    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        r = angle * u
        Rgood = Compute.R(angle, u)
        Rtest = Compute.r2R(r)
        np.testing.assert_almost_equal(Rtest, Rgood)


@pytest.mark.dependency(depends=["test_R_standard", "test_R_random"])
def test_R2r_standard():
    R = np.eye(3)
    rgood = np.zeros(3)
    rtest = Compute.R2r(R)
    np.testing.assert_almost_equal(rtest, rgood)


@pytest.mark.dependency(depends=["test_R2r_standard"])
def test_R2r_random():
    Ntests = 1000
    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        rgood = angle * u
        R = Compute.R(angle, u)
        rtest = Compute.R2r(R)
        np.testing.assert_almost_equal(rtest, rgood)
