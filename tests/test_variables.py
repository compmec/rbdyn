import pytest
import numpy as np
from rbdyn.variable import Variable


@pytest.mark.dependency()
def test_Build():
    x = Variable("x")


@pytest.mark.dependency(depends=["test_Build"])
def test_BuildTwoVariables():
    x = Variable("x")
    y = Variable("y")


@pytest.mark.dependency(depends=["test_Build"])
def test_RebuildSameVariable():
    x = Variable("x")
    newx = Variable("x")
    assert x is newx


@pytest.mark.dependency(depends=["test_Build"])
def test_GetDerivative():
    x = Variable("x")
    dx = x.dt


@pytest.mark.dependency(depends=["test_Build"])
def test_Comparation():
    x = Variable("x")
    y = Variable("y")
    assert x == x
    assert x != y


@pytest.mark.dependency(depends=["test_BuildTwoVariables", "test_Comparation"])
def test_SortAllVariables():
    x = Variable("x")
    y = Variable("y")
    p = Variable("p")
    listgood = (x, y, p)

    listvars = [x, y, p]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    for i in range(3):
        assert type(listtest[i]) is type(listgood[i])
        assert listtest[i] == listgood[i]

    listvars = [x, p, y]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == listgood

    listvars = [y, x, p]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == listgood

    listvars = [y, p, x]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == listgood

    listvars = [p, x, y]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == listgood

    listvars = [p, y, x]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == listgood


@pytest.mark.dependency(depends=["test_SortAllVariables"])
def test_SortSomeVariables():
    x = Variable("x")
    y = Variable("y")
    p = Variable("p")

    listvars = [x, y]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (x, y)

    listvars = [x, p]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (x, p)

    listvars = [y, x]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (x, y)

    listvars = [p, x]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (x, p)

    listvars = [y, p]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (y, p)

    listvars = [p, y]
    listtest = Variable.sort(listvars)
    assert isinstance(listtest, tuple) is True
    assert listtest == (y, p)


@pytest.mark.dependency(depends=["test_BuildTwoVariables", "test_Comparation"])
def test_ErrorSortVariables():
    x = Variable("x")
    y = Variable("y")
    dx = x.dt

    with pytest.raises(TypeError):
        Variable.sort([1, y])

    with pytest.raises(ValueError):
        Variable.sort([dx, y])
