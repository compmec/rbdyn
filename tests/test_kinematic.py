import pytest
import numpy as np
from compmec.rbdyn.kinematic import Kinematic, ObjectKinematic


@pytest.mark.dependency()
@pytest.mark.timeout(2)
def test_BuildKinematic():
    kine = Kinematic()

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildKinematic"])
def test_InitialValues():
    kine = Kinematic(init=True)
    np.testing.assert_almost_equal(kine.p, np.zeros(3))
    np.testing.assert_almost_equal(kine.v, np.zeros(3))
    np.testing.assert_almost_equal(kine.a, np.zeros(3))
    np.testing.assert_almost_equal(kine.r, np.zeros(3))
    np.testing.assert_almost_equal(kine.w, np.zeros(3))
    np.testing.assert_almost_equal(kine.q, np.zeros(3))
    np.testing.assert_almost_equal(kine.R, np.eye(3))
    np.testing.assert_almost_equal(kine.W, np.zeros((3, 3)))
    np.testing.assert_almost_equal(kine.Q, np.zeros((3, 3)))

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildKinematic"])
def test_NonInitialValues():
    kine = Kinematic(init=False)
    assert kine.p is None
    assert kine.v is None
    assert kine.a is None
    assert kine.r is None
    assert kine.w is None
    assert kine.q is None
    assert kine.R is None
    assert kine.W is None
    assert kine.Q is None

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearPosition_standard():
    kine = Kinematic()
    kine.p = (0, 0, 5)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearPosition_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.p = 1
    with pytest.raises(TypeError):
        kine.p = None
    with pytest.raises(ValueError):
        kine.p = (1, 0, 0, 0)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_SetGetLinearPosition_standard"])
def test_SetGetLinearPosition_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_position = np.random.rand(3)
        kine.p = new_position
        np.testing.assert_almost_equal(new_position, kine.p)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearSpeed_standard():
    kine = Kinematic()
    kine.w = (0, 0, 5)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearSpeed_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.v = 1
    with pytest.raises(TypeError):
        kine.v = None
    with pytest.raises(ValueError):
        kine.v = (1, 0, 0, 0)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_SetGetLinearSpeed_standard"])
def test_SetGetLinearSpeed_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_speed = np.random.rand(3)
        kine.v = new_speed
        np.testing.assert_almost_equal(new_speed, kine.v)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearAcceleration_standard():
    kine = Kinematic()
    kine.w = (0, 0, 5)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetLinearAcceleration_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.w = 1
    with pytest.raises(TypeError):
        kine.w = None
    with pytest.raises(ValueError):
        kine.w = (1, 0, 0, 0)


@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_SetGetLinearAcceleration_standard"])
def test_SetGetLinearAcceleration_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_acceleration = np.random.rand(3)
        kine.a = new_acceleration
        np.testing.assert_almost_equal(new_acceleration, kine.a)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularPositionVector_standard():
    kine = Kinematic()
    kine.w = (0, 0, 5)


@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularPositionVector_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.r = 1
    with pytest.raises(TypeError):
        kine.r = None
    with pytest.raises(ValueError):
        kine.r = (1, 0, 0, 0)


@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_SetGetAngularPositionVector_standard"])
def test_SetGetAngularPositionVector_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_position = np.random.rand(3)
        kine.r = new_position
        np.testing.assert_almost_equal(new_position, kine.r)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularSpeedVector_standard():
    kine = Kinematic()
    kine.w = (0, 0, 3)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularSpeedVector_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.w = 1
    with pytest.raises(TypeError):
        kine.w = None
    with pytest.raises(ValueError):
        kine.w = (1, 0, 0, 0)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_SetGetAngularSpeedVector_standard"])
def test_SetGetAngularSpeedVector_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_speed = np.random.rand(3)
        kine.w = new_speed
        np.testing.assert_almost_equal(new_speed, kine.w)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularAccelerationVector_standard():
    kine = Kinematic()
    kine.q = (1, 0, 0)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularAccelerationVector_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.q = 1
    with pytest.raises(TypeError):
        kine.q = None
    with pytest.raises(ValueError):
        kine.q = (1, 0, 0, 0)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_SetGetAngularAccelerationVector_standard"])
def test_SetGetAngularAccelerationVector_random():
    kine = Kinematic()
    Ntests = 100
    for i in range(Ntests):
        new_acceleration = np.random.rand(3)
        kine.q = new_acceleration
        np.testing.assert_almost_equal(new_acceleration, kine.q)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularPositionTensor_standard():
    kine = Kinematic()
    kine.R = np.eye(3)
    np.testing.assert_almost_equal(kine.R, np.eye(3))

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularPositionTensor_fails():
    kine = Kinematic(False)
    with pytest.raises(TypeError):
        kine.R = 1
    with pytest.raises(TypeError):
        kine.R = None
    with pytest.raises(ValueError):
        kine.R = np.zeros(3)
    with pytest.raises(ValueError):
        kine.R = np.zeros((3, 3))

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularSpeedTensor_standard():
    kine = Kinematic()
    kine.W = np.zeros((3, 3))

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularSpeedTensor_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.W = 3
    with pytest.raises(ValueError):
        kine.W = np.eye(3)

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularAccelerationTensor_standard():
    kine = Kinematic()
    kine.Q = np.zeros((3, 3))

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InitialValues", "test_NonInitialValues"])
def test_SetGetAngularAccelerationTensor_fails():
    kine = Kinematic()
    with pytest.raises(TypeError):
        kine.Q = 3
    with pytest.raises(ValueError):
        kine.Q = np.eye(3)

@pytest.mark.timeout(1)
@pytest.mark.dependency()
def test_BuildObjectKinematic():
    kine = ObjectKinematic()

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_BuildObjectKinematic"])
def test_InheritanceObjectKinematic():
    kine = ObjectKinematic()
    assert isinstance(kine, Kinematic) is True

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InheritanceObjectKinematic"])
def test_ObjectKinematic_InitialValues():
    kine = ObjectKinematic(init=True)
    np.testing.assert_almost_equal(kine.p, np.zeros(3))
    np.testing.assert_almost_equal(kine.v, np.zeros(3))
    np.testing.assert_almost_equal(kine.a, np.zeros(3))
    np.testing.assert_almost_equal(kine.r, np.zeros(3))
    np.testing.assert_almost_equal(kine.w, np.zeros(3))
    np.testing.assert_almost_equal(kine.q, np.zeros(3))
    np.testing.assert_almost_equal(kine.R, np.eye(3))
    np.testing.assert_almost_equal(kine.W, np.zeros((3, 3)))
    np.testing.assert_almost_equal(kine.Q, np.zeros((3, 3)))
    np.testing.assert_almost_equal(kine.CM, np.zeros(3))
    np.testing.assert_almost_equal(kine.II, np.zeros((3, 3)))

@pytest.mark.timeout(1)
@pytest.mark.dependency(depends=["test_InheritanceObjectKinematic"])
def test_ObjectKinematic_NonInitialValues():
    kine = ObjectKinematic(init=False)
    assert kine.p is None
    assert kine.v is None
    assert kine.a is None
    assert kine.r is None
    assert kine.w is None
    assert kine.q is None
    assert kine.R is None
    assert kine.W is None
    assert kine.Q is None
    assert kine.CM is None
    assert kine.II is None
