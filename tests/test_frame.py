import pytest
import numpy as np
from compmec.rbdyn.frames import FrameReference


def getRandomUnit3DVector():
    v = np.random.rand(3)
    norm_v = np.linalg.norm(v)
    return v / norm_v


# @pytest.mark.dependency(depends=["tests/test_kinematic.py::test_allgood"], scope="session")
@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_BuildInertialFrame():
    R0 = FrameReference()  # Inertial Frame of Reference


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildInertialFrame"])
def test_BuildSecondFrame():
    R0 = FrameReference()
    R1 = FrameReference(R0)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildSecondFrame"])
def test_BuildTranslatedFrame_standard():
    R0 = FrameReference()
    R1 = FrameReference(R0, translation=(0, 0, 0))
    R2 = FrameReference(R0, translation=(1, 0, 0))
    R3 = FrameReference(R0, translation=(0, 1, 0))
    R4 = FrameReference(R0, translation=(0, 0, 1))
    R5 = FrameReference(R0, translation=(1, 0, 1))
    R6 = FrameReference(R0, translation=(2, 1, 2))
    R7 = FrameReference(R0, translation=(0, 1, 1))


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildTranslatedFrame_standard"])
def test_BuildTranslatedFrame_random():
    R0 = FrameReference()
    Ntests = 100
    for i in range(Ntests):
        translation = np.random.rand(3)
        Ri = FrameReference(R0, translation=translation)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_BuildSecondFrame"])
def test_BuildRotatedFrame_standard():
    R0 = FrameReference()
    R1 = FrameReference(R0, rotation=(0, "x"))
    R2 = FrameReference(R0, rotation=(0, "y"))
    R3 = FrameReference(R0, rotation=(0, "z"))
    R4 = FrameReference(R0, rotation=(np.pi / 2, "x"))
    R5 = FrameReference(R0, rotation=(np.pi / 2, "y"))
    R6 = FrameReference(R0, rotation=(np.pi / 2, "z"))
    R7 = FrameReference(R0, rotation=(2 * np.pi, "x"))
    R8 = FrameReference(R0, rotation=(2 * np.pi, "y"))
    R9 = FrameReference(R0, rotation=(2 * np.pi, "z"))

    R1 = FrameReference(R0, rotation=(0, (1, 0, 0)))
    R2 = FrameReference(R0, rotation=(0, (0, 1, 0)))
    R3 = FrameReference(R0, rotation=(0, (0, 0, 1)))
    R4 = FrameReference(R0, rotation=(np.pi / 2, (1, 0, 0)))
    R5 = FrameReference(R0, rotation=(np.pi / 2, (0, 1, 0)))
    R6 = FrameReference(R0, rotation=(np.pi / 2, (0, 0, 1)))
    R7 = FrameReference(R0, rotation=(2 * np.pi, (1, 0, 0)))
    R8 = FrameReference(R0, rotation=(2 * np.pi, (0, 1, 0)))
    R9 = FrameReference(R0, rotation=(2 * np.pi, (0, 0, 1)))

@pytest.mark.timeout(20)
@pytest.mark.dependency(depends=["test_BuildRotatedFrame_standard"])
def test_BuildRotatedFrame_random():
    R0 = FrameReference()
    Ntests = 100
    for i in range(Ntests):
        angle = np.random.rand(1)
        direction = getRandomUnit3DVector()
        rotation = (angle, direction)
        Ri = FrameReference(R0, rotation=rotation)

@pytest.mark.timeout(20)
@pytest.mark.dependency(depends=["test_BuildTranslatedFrame_random", "test_BuildRotatedFrame_random"])
def test_BuildTranslatedAndRotatedFrame_random():
    R0 = FrameReference()
    Ntests = 1000
    for i in range(Ntests):
        translation = np.random.rand(3)
        angle = np.random.rand(1)
        direction = getRandomUnit3DVector()
        rotation = (angle, direction)
        Ri = FrameReference(R0, translation=translation, rotation=rotation)


@pytest.mark.dependency(depends=["test_BuildTranslatedAndRotatedFrame_random"])
def test_allgood():
    pass