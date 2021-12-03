import pytest
import numpy as np
import sympy as sp
from compmec.rbdyn import Variable
from compmec.rbdyn.lagrangian import Lagrangian
from compmec.rbdyn.energy import Energy
from compmec.rbdyn.solver import Euler


# @pytest.mark.dependency(
#     depends=["tests/test_frame.py::test_allgood",
#              "tests/test_object.py::test_allgood",
#              "tests/test_energy.py::test_allgood",
#              "tests/test_lagrangian.py::test_allgood"])
@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(60)
@pytest.mark.dependency(depends=["test_begin"])
def test_Spring():
    m = 1
    k = 1

    x = Variable("x")
    dx = x.dt
    X = [x]
    Ek = m*(dx**2)/2
    Eu = k*x**2/2
    E = Ek-Eu

    timesteps = np.linspace(0, 10, 100)
    IC = {x: (0.3, 0)}
    G = []
    F = [0]

    solver = Euler(E, IC, G, F, timesteps)
    result = solver.run(timeout=5)

@pytest.mark.timeout(60)
@pytest.mark.dependency(depends=["test_begin"])
def test_SimplePendulumAngular():
    m = 1
    g = 1
    L = 1

    theta = Variable("theta")
    dtheta = theta.dt
    X = [theta]
    Ek = m*(L**2) *(dtheta**2)/2
    Eu = -m*g*L*sp.cos(theta)
    E = Ek-Eu

    timesteps = np.linspace(0, 10, 100)
    IC = {theta: (0.9*np.pi, 0)}
    G = []
    F = [0]

    solver = Euler(E, IC, G, F, timesteps)
    result = solver.run(timeout=5)

@pytest.mark.timeout(60)
@pytest.mark.dependency(depends=["test_begin"])
def test_SimplePendulumCartesian():
    m = 1
    g = 1
    L = 1

    x = Variable("x")
    y = Variable("y")
    dx = x.dt
    dy = y.dt

    X = [x, y]

    Ek = m*(dx**2 + dy**2)/2
    Eu = m*g*y
    E = Ek - Eu

    timesteps = np.linspace(0, 10, 1000)
    x0 = L*np.sin(0.2)
    y0 = -L*np.cos(0.2)
    IC = {x: (x0, 0),
          y: (y0, 0)}
    G = [x**2 + y**2 - L**2]
    F = [0, 0]
    solver = Euler(E, IC, G, F, timesteps)
    results = solver.run(timeout=60)

@pytest.mark.timeout(60)
@pytest.mark.dependency(depends=["test_begin"])
def test_DoublePendulumCartesian():
    m1 = 1
    m2 = 1
    g = 1
    L1 = 1
    L2 = 1

    x1 = Variable("x1")
    y1 = Variable("y1")
    x2 = Variable("x2")
    y2 = Variable("y2")
    dx1 = x1.dt
    dy1 = y1.dt
    dx2 = x2.dt
    dy2 = y2.dt

    X = [x1, y1, x2, y2]

    Ek = m1*(dx1**2 + dy1**2)/2
    Ek += m2*(dx2**2 + dy2**2)/2
    Eu = m1*g*y1 + m2*g*y2
    E = Ek - Eu

    timesteps = np.linspace(0, 10, 1000)
    theta1 = 0.9*np.pi
    theta2 = 0.5*np.pi
    x10 = L1*np.sin(theta1)
    y10 = -L1*np.cos(theta1)
    x20 = L2*np.sin(theta2)
    y20 = -L2*np.cos(theta2)
    IC = {x1: (x10, 0),
          y1: (y10, 0),
          x2: (x20, 0),
          y2: (y20, 0)}
    G = [x1**2 + y1**2 - L1**2, 
         (x2)**2 + (y2)**2 - (L2)**2]
    F = [0, 0, 0, 0]
    solver = Euler(E, IC, G, F, timesteps)
    results = solver.run(timeout=60)

