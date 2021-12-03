import pytest
import numpy as np
from compmec.rbdyn.variable import Variable, VariableList

@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.dependency(depends=["test_begin"])
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

    listgood = VariableList([x, y, p])

    assert VariableList([x, y, p]) == listgood
    assert VariableList([x, p, y]) == listgood
    assert VariableList([y, x, p]) == listgood
    assert VariableList([y, p, x]) == listgood
    assert VariableList([p, x, y]) == listgood
    assert VariableList([p, y, x]) == listgood



@pytest.mark.dependency(depends=["test_SortAllVariables"])
def test_SortSomeVariables():
    x = Variable("x")
    y = Variable("y")
    p = Variable("p")


    assert VariableList([x, y]) == VariableList([y, x])
    assert VariableList([x, p]) == VariableList([p, x])
    assert VariableList([y, p]) == VariableList([p, y])
    



@pytest.mark.dependency(depends=["test_SortSomeVariables", "test_SortSomeVariables"])
def test_allgood():
    pass