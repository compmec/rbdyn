import pytest 
import numpy as np 
from compmec.rbdyn.lagrangian import Lagrangian 
from compmec.rbdyn import Variable 
from compmec.rbdyn.energy import Energy 
from sympy import cos, sin 
 
 
 
# @pytest.mark.dependency(depends="tests/test_energy.py::test_AllQuantity") 
@pytest.mark.dependency() 
def test_begin(): 
	pass 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_begin"]) 
def test_Constant(): 
	E = Energy(1) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec is None 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_X(): 
	x = Variable("x") 
	E = Energy(x) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1,) 
	assert Lvec[0] == -1 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_dX(): 
	x = Variable("x") 
	dx = x.dt 
	E = Energy(dx) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1,) 
	assert Lvec[0] == 0 
 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_X2(): 
	x = Variable("x") 
	E = Energy(x**2) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1,) 
	assert Lvec[0] == -2*x 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_XdX(): 
	x = Variable("x") 
	dx = x.dt 
	E = Energy(x*dx) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1,) 
	assert Lvec[0] == 0 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_dX2(): 
	x = Variable("x") 
	dx = x.dt 
	ddx = x.ddt 
	E = Energy(dx**2) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1,) 
	assert Lvec[0] == 2*ddx 
 
 
# https://stackoverflow.com/questions/70126486/get-linear-and-quadratic-terms-from-sympy-expression 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_cosX(): 
	x = Variable("x") 
	f = cos(x) 
	df = -1*sin(x) 
 
	E = Energy(f) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1, ) 
	assert Lvec[0] == -1*df 
 
@pytest.mark.timeout(2) 
@pytest.mark.dependency(depends=["test_Constant"]) 
def test_sinX(): 
	x = Variable("x") 
	f = sin(x) 
	df = cos(x) 
 
	E = Energy(f) 
	L = Lagrangian(E) 
	Lvec = L.get() 
 
	assert Lvec.shape == (1, ) 
	assert Lvec[0] == -1*df 
 
 
 
 
@pytest.mark.dependency(depends=["test_dX2", "test_cosX", "test_sinX"]) 
def test_allgood(): 
    pass