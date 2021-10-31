import pytest
import numpy as np
from rbdyn.object import Object
from rbdyn.frames import FrameReference


@pytest.mark.dependency()
def test_Build():
    R0 = FrameReference()
    bar = Object(R0)


@pytest.mark.dependency(depends=["test_Build"])
def test_BuildWithName():
    R0 = FrameReference()
    bar = Object(R0, "bar")
    plate = Object(R0, name="plate")


@pytest.mark.dependency(depends=["test_Build"])
def test_SetMass():
    R0 = FrameReference()
    bar = Object(R0)
    bar.mass = 3


@pytest.mark.dependency(depends=["test_Build"])
def test_SetCenterMass():
    R0 = FrameReference()
    bar = Object(R0)
    bar.CM = (1, 2, 3)


@pytest.mark.dependency(depends=["test_Build"])
def test_SetInertia():
    R0 = FrameReference()
    bar = Object(R0)
    bar.II = ((1, 0, 0),
              (0, 1, 0),
              (0, 0, 1))


@pytest.mark.dependency(depends=["test_Build"])
def test_TranslatedPosition():
    R0 = FrameReference()
    R1 = FrameReference(R0, translation=(1, 0, 0))
    R2 = FrameReference(R0, translation=(0, 1, 0))
    R3 = FrameReference(R0, translation=(0, 0, 1))
    point1 = Object(R1)
    point2 = Object(R2)
    point3 = Object(R3)
    np.testing.assert_almost_equal(point1.get(R1, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point1.get(R0, "p"), (1, 0, 0))
    np.testing.assert_almost_equal(point2.get(R2, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point2.get(R0, "p"), (0, 1, 0))
    np.testing.assert_almost_equal(point3.get(R3, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point3.get(R0, "p"), (0, 0, 1))


@pytest.mark.dependency(depends=["test_Build"])
def test_RotationedPosition():
    R0 = FrameReference()
    R1 = FrameReference(R0, rotation=(np.pi / 2, (0, 0, 1)))
    R2 = FrameReference(R1, translation=(1, 0, 0))
    point = Object(R2)
    np.testing.assert_almost_equal(point.get(R2, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point.get(R1, "p"), (1, 0, 0))
    np.testing.assert_almost_equal(point.get(R0, "p"), (0, 1, 0))


@pytest.mark.dependency(depends=["test_TranslatedPosition"])
def test_CompositionTranslation():
    R0 = FrameReference()
    R1 = FrameReference(R0, translation=(1, 0, 0))
    R2 = FrameReference(R1, translation=(0, 1, 0))
    R3 = FrameReference(R2, translation=(0, 0, 1))
    point = Object(R3)
    np.testing.assert_almost_equal(point.get(R3, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point.get(R2, "p"), (0, 0, 1))
    np.testing.assert_almost_equal(point.get(R1, "p"), (0, 1, 1))
    np.testing.assert_almost_equal(point.get(R0, "p"), (1, 1, 1))


@pytest.mark.dependency(depends=["test_RotationedPosition"])
def test_CompositionRotationsZ():
    a, b, c = np.random.rand(3)
    R0 = FrameReference()
    R1 = FrameReference(R0, rotation=(np.pi / 2, "z"))
    R2 = FrameReference(R1, rotation=(np.pi / 2, "z"))
    R3 = FrameReference(R2, translation=(a, b, c))
    point = Object(R3)
    np.testing.assert_almost_equal(point.get(R3, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point.get(R2, "p"), (a, b, c))
    np.testing.assert_almost_equal(point.get(R1, "p"), (-b, a, c))
    np.testing.assert_almost_equal(point.get(R0, "p"), (-a, -b, c))


@pytest.mark.dependency(depends=["test_CompositionRotationsZ"])
def test_CompositionRotationsXYZ():
    a, b, c = np.random.rand(3)
    d, e, f = np.random.rand(3)
    R0 = FrameReference()
    R1 = FrameReference(R0, translation=(a, b, c))
    R2 = FrameReference(R1, rotation=(np.pi / 2, "x"))
    R3 = FrameReference(R2, rotation=(np.pi / 2, "y"))
    R4 = FrameReference(R3, rotation=(np.pi / 2, "z"))
    R5 = FrameReference(R4, translation=(d, e, f))
    point = Object(R5)
    np.testing.assert_almost_equal(point.get(R5, "p"), (0, 0, 0))
    np.testing.assert_almost_equal(point.get(R4, "p"), (d, e, f))
    np.testing.assert_almost_equal(point.get(R3, "p"), (-e, d, f))
    np.testing.assert_almost_equal(point.get(R2, "p"), (f, d, e))
    np.testing.assert_almost_equal(point.get(R1, "p"), (f, -e, d))
    np.testing.assert_almost_equal(point.get(R0, "p"), (a + f, b - e, c + d))
