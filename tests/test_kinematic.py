import pytest
import numpy as np
from dynamics import kinematic


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


def test_Verify3DVector():
    u = np.zeros(3)
    kinematic.Verify3DVector(u)

    u = np.random.rand(3)
    kinematic.Verify3DVector(u)


def test_Verify3DTensor():
    X = np.zeros((3, 3))
    kinematic.Verify3DTensor(X)

    X = np.random.rand(3, 3)
    kinematic.Verify3DTensor(X)


def test_VerifyAntiSymmetric():
    X = np.zeros((3, 3))
    kinematic.VerifyAntiSymmetric(X)

    X = np.random.rand(3, 3)
    Xsym = (X + X.T) / 2
    Xant = X - Xsym
    kinematic.VerifyAntiSymmetric(Xant)

    with pytest.raises(ValueError):
        kinematic.VerifyAntiSymmetric(Xsym)


def test_VerifyUnit3DVector():
    kinematic.VerifyUnit3DVector((1, 0, 0))
    kinematic.VerifyUnit3DVector((0, 1, 0))
    kinematic.VerifyUnit3DVector((0, 0, 1))
    kinematic.VerifyUnit3DVector((-1, 0, 0))
    kinematic.VerifyUnit3DVector((0, -1, 0))
    kinematic.VerifyUnit3DVector((0, 0, -1))
    kinematic.VerifyUnit3DVector((0.6, 0.8, 0))
    kinematic.VerifyUnit3DVector((0, 0.8, 0.6))

    Ntests = 100
    for i in range(Ntests):
        u = getRandomUnit3DVector()
        kinematic.VerifyUnit3DVector(u)


def test_Compute_Ux2u():
    Ux = np.zeros((3, 3))
    ugood = np.zeros(3)
    utest = kinematic.Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, 1, -1),
          (-1, 0, 1),
          (1, -1, 0))
    ugood = (-1, -1, -1)
    utest = kinematic.Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, 1, 1),
          (-1, 0, 1),
          (-1, -1, 0))
    ugood = (-1, 1, -1)
    utest = kinematic.Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)

    Ux = ((0, -5, 2),
          (5, 0, -7),
          (-2, 7, 0))
    ugood = (7, 2, 5)
    utest = kinematic.Compute.Ux2u(Ux)
    np.testing.assert_almost_equal(utest, ugood)


def test_Compute_u2Ux():
    u = np.zeros(3)
    Uxgood = np.zeros((3, 3))
    Uxtest = kinematic.Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (-1, -1, -1)
    Uxgood = ((0, 1, -1),
              (-1, 0, 1),
              (1, -1, 0))
    Uxtest = kinematic.Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (-1, 1, -1)
    Uxgood = ((0, 1, 1),
              (-1, 0, 1),
              (-1, -1, 0))
    Uxtest = kinematic.Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)

    u = (7, 2, 5)
    Uxgood = ((0, -5, 2),
              (5, 0, -7),
              (-2, 7, 0))
    Uxtest = kinematic.Compute.u2Ux(u)
    np.testing.assert_almost_equal(Uxtest, Uxgood)


def test_Compute_u2Ux2u():
    Ntests = 1
    for i in range(Ntests):
        ugood = np.random.rand(3)
        Ux = kinematic.Compute.u2Ux(ugood)
        utest = kinematic.Compute.Ux2u(Ux)
        np.testing.assert_almost_equal(utest, ugood)


def test_Compute_R_standard():
    angle = np.pi / 2
    u = (1, 0, 0)
    Rgood = ((1, 0, 0),
             (0, 0, -1),
             (0, 1, 0))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (1, 0, 0)
    Rgood = ((1, 0, 0),
             (0, -1, 0),
             (0, 0, -1))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi / 2
    u = (0, 1, 0)
    Rgood = ((0, 0, 1),
             (0, 1, 0),
             (-1, 0, 0))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (0, 1, 0)
    Rgood = ((-1, 0, 0),
             (0, 1, 0),
             (0, 0, -1))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi / 2
    u = (0, 0, 1)
    Rgood = ((0, -1, 0),
             (1, 0, 0),
             (0, 0, 1))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)

    angle = np.pi
    u = (0, 0, 1)
    Rgood = ((-1, 0, 0),
             (0, -1, 0),
             (0, 0, 1))
    Rtest = kinematic.Compute.R(angle, u)
    np.testing.assert_almost_equal(Rtest, Rgood)


def test_Compute_R_random():
    Ntests = 100
    angle = 0
    for i in range(Ntests):
        u = getRandomUnit3DVector()
        R = kinematic.Compute.R(angle, u)
        np.testing.assert_almost_equal(R, np.eye(3))

    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        R = kinematic.Compute.R(angle, u)
        RT = np.transpose(R)
        M = np.dot(R, RT)
        np.testing.assert_almost_equal(M, np.eye(3))


def test_Compute_r2R_standard():
    r = np.zeros(3)
    Rgood = np.eye(3)
    Rtest = kinematic.Compute.r2R(r)
    np.testing.assert_almost_equal(Rtest, Rgood)


def test_Compute_r2R_random():
    Ntests = 1000
    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        r = angle * u
        Rgood = kinematic.Compute.R(angle, u)
        Rtest = kinematic.Compute.r2R(r)
        np.testing.assert_almost_equal(Rtest, Rgood)


def test_Compute_R2r_standard():
    R = np.eye(3)
    rgood = np.zeros(3)
    rtest = kinematic.Compute.R2r(R)
    np.testing.assert_almost_equal(rtest, rgood)


def test_Compute_R2r_random():
    Ntests = 1000
    for i in range(Ntests):
        angle = randomNumberBetween(-np.pi, np.pi)
        u = getRandomUnit3DVector()
        rgood = angle * u
        R = kinematic.Compute.R(angle, u)
        rtest = kinematic.Compute.R2r(R)
        np.testing.assert_almost_equal(rtest, rgood)


def test_Kinematic_Build():
    kine = kinematic.Kinematic()


def test_Kinematic_InitialValues():
    kine = kinematic.Kinematic(initializezero=True)
    np.testing.assert_almost_equal(kine.p, np.zeros(3))
    np.testing.assert_almost_equal(kine.v, np.zeros(3))
    np.testing.assert_almost_equal(kine.a, np.zeros(3))
    np.testing.assert_almost_equal(kine.r, np.zeros(3))
    np.testing.assert_almost_equal(kine.w, np.zeros(3))
    np.testing.assert_almost_equal(kine.q, np.zeros(3))
    np.testing.assert_almost_equal(kine.R, np.eye(3))
    np.testing.assert_almost_equal(kine.W, np.zeros((3, 3)))
    np.testing.assert_almost_equal(kine.Q, np.zeros((3, 3)))


def test_Kinematic_NonInitialValues():
    kine = kinematic.Kinematic(initializezero=False)
    assert kine.p is None
    assert kine.v is None
    assert kine.a is None
    assert kine.r is None
    assert kine.w is None
    assert kine.q is None
    assert kine.R is None
    assert kine.W is None
    assert kine.Q is None


def test_Kinematic_SetGetLinearPosition_standard():
    kine = kinematic.Kinematic()
    kine.w = (0, 0, 5)


def test_Kinematic_SetGetLinearPosition_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.p = 1
    with pytest.raises(TypeError):
        kine.p = None
    with pytest.raises(ValueError):
        kine.p = (1, 0, 0, 0)


def test_Kinematic_SetGetLinearPosition_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_position = np.random.rand(3)
        kine.p = new_position
        np.testing.assert_almost_equal(new_position, kine.p)


def test_Kinematic_SetGetLinearSpeed_standard():
    kine = kinematic.Kinematic()
    kine.w = (0, 0, 5)


def test_Kinematic_SetGetLinearSpeed_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.v = 1
    with pytest.raises(TypeError):
        kine.v = None
    with pytest.raises(ValueError):
        kine.v = (1, 0, 0, 0)


def test_Kinematic_SetGetLinearSpeed_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_speed = np.random.rand(3)
        kine.v = new_speed
        np.testing.assert_almost_equal(new_speed, kine.v)


def test_Kinematic_SetGetLinearAcceleration_standard():
    kine = kinematic.Kinematic()
    kine.w = (0, 0, 5)


def test_Kinematic_SetGetLinearAcceleration_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.w = 1
    with pytest.raises(TypeError):
        kine.w = None
    with pytest.raises(ValueError):
        kine.w = (1, 0, 0, 0)


def test_Kinematic_SetGetLinearAcceleration_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_acceleration = np.random.rand(3)
        kine.a = new_acceleration
        np.testing.assert_almost_equal(new_acceleration, kine.a)


def test_Kinematic_SetGetAngularPositionVector_standard():
    kine = kinematic.Kinematic()
    kine.w = (0, 0, 5)


def test_Kinematic_SetGetAngularPositionVector_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.r = 1
    with pytest.raises(TypeError):
        kine.r = None
    with pytest.raises(ValueError):
        kine.r = (1, 0, 0, 0)


def test_Kinematic_SetGetAngularPositionVector_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_position = np.random.rand(3)
        kine.r = new_position
        np.testing.assert_almost_equal(new_position, kine.r)


def test_Kinematic_SetGetAngularSpeedVector_standard():
    kine = kinematic.Kinematic()
    kine.w = (0, 0, 3)


def test_Kinematic_SetGetAngularSpeedVector_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.w = 1
    with pytest.raises(TypeError):
        kine.w = None
    with pytest.raises(ValueError):
        kine.w = (1, 0, 0, 0)


def test_Kinematic_SetGetAngularSpeedVector_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_speed = np.random.rand(3)
        kine.w = new_speed
        np.testing.assert_almost_equal(new_speed, kine.w)


def test_Kinematic_SetGetAngularAccelerationVector_standard():
    kine = kinematic.Kinematic()
    kine.q = (1, 0, 0)


def test_Kinematic_SetGetAngularAccelerationVector_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.q = 1
    with pytest.raises(TypeError):
        kine.q = None
    with pytest.raises(ValueError):
        kine.q = (1, 0, 0, 0)


def test_Kinematic_SetGetAngularAccelerationVector_random():
    kine = kinematic.Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_acceleration = np.random.rand(3)
        kine.q = new_acceleration
        np.testing.assert_almost_equal(new_acceleration, kine.q)


def test_Kinematic_SetGetAngularPositionTensor_standard():
    kine = kinematic.Kinematic()
    kine.R = np.eye(3)
    np.testing.assert_almost_equal(kine.R, np.eye(3))


def test_Kinematic_SetGetAngularPositionTensor_fails():
    kine = kinematic.Kinematic(False)
    with pytest.raises(TypeError):
        kine.R = 1
    with pytest.raises(TypeError):
        kine.R = None
    with pytest.raises(ValueError):
        kine.R = np.zeros(3)
    with pytest.raises(ValueError):
        kine.R = np.zeros((3, 3))


def test_Kinematic_SetGetAngularSpeedTensor_standard():
    kine = kinematic.Kinematic()
    kine.W = np.zeros((3, 3))


def test_Kinematic_SetGetAngularSpeedTensor_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.W = 3
    with pytest.raises(ValueError):
        kine.W = np.eye(3)


def test_Kinematic_SetGetAngularAccelerationTensor_standard():
    kine = kinematic.Kinematic()
    kine.Q = np.zeros((3, 3))


def test_Kinematic_SetGetAngularAccelerationTensor_fails():
    kine = kinematic.Kinematic()
    with pytest.raises(TypeError):
        kine.Q = 3
    with pytest.raises(ValueError):
        kine.Q = np.eye(3)
