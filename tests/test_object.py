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


# @pytest.mark.skip(reason="Not yet implemented fail")
# @pytest.mark.dependency(depends=["test_Build"])
# @pytest.mark.xfail(reason="Not yet implemented fail")
# def test_TranslatedPosition():
#     R0 = FrameReference()
#     R1 = FrameReference(R0, translation=(1, 0, 0))
#     point = Object(R1)
#     np.testing.assert_almost_equal(point.get(R1, "p"), (0, 0, 0))
#     np.testing.assert_almost_equal(point.get(R0, "p"), (1, 0, 0))
