# Rigid Body Dynamics


### Objective

Solve Rigid Body Dynamics' problems like:

* Get energy in algebraic form
* Given a initial condition, compute further moments and positions
* Get vibrational modules 

An example of with the initial condition, we can get a connecting rod of a motor:

<img src="https://raw.githubusercontent.com/carlos-adir/RigidBodyDynamics/master/docs/img/biela.gif" alt="biela" width="400"/>


### Introduction

Rigid body dynamics problems are problems that are described by the following requirements:

* There is a inertial frame of reference, absolute, that we call ```R0```, in the origin
* It's possible have some referential frames, which are relatively connected to the frame ```R0``` using ```translation``` and ```rotation``` vectors
* There are some bodies, with ```m```(mass), ```CM```(center of mass), ```II``` (inertia tensor) and the ```base``` frame that is attached.
* Initial configuration of the system

Once described the problem, the code finds the kinematics results, for example, energy.

### Exemple

The first example, we have a bar of mass ```m```, length ```l```, its center at the position ```(x, y, 0)``` and it's rotationed at an angle ```theta``` relative to the vector ```z```. So, we describe the problem as

```python
# Definition of the variables
t = sp.symbols("t", real=True)  # The time
m = sp.symbols("m", real=True, positive=True)  # The mass
l = sp.symbols("l", real=True, positive=True)  # The bar's length
x = sp.Function("x")(t)
y = sp.Function("y")(t)
theta = sp.Function("theta")(t)  # The rotation angle

# Definition of the frames of reference
R0 = FrameReference()  # Inertial frame of reference
R1 = FrameReference(base=R0, translation=(x, y, 0))
R2 = FrameReference(base=R1, rotation=(theta, "z"))

# Definition of the object
CM = (0, 0, 0)  # Center of mass
II = (m * l**2 / 12) * np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
bar = Objet(base=R2, name="bar")
bar.set_mass(m)  # Put the mass into the object
bar.set_CM(CM)  # Put the CM into the object
bar.set_II(II)  # Put the inertia tensor into the object

# Calculus of Kinetic Energy relative of the frame R0
E = bar.get_KineticEnergy(R0)
```

And it gives the result

```
[1] E = m * (0.0833333333333333 * l**2 * Derivative(theta(t), t)**2 + Derivative(x(t), t)**2 + Derivative(y(t), t)**2) / 2
```

### Coding and librarys

For implementation, we use Python with the following libraries:

* [Numpy][numpy_website]: Used for vector calculs
* [Sympy][sympy_website]: Used to calculate the algebric derivative

To use the codes, you just need these packages and the python installed. The easiest way to do it is using the [Anaconda][anaconda_website] that installs everything you need.


### Documentation

Until now, there is no documentation, but further it will be in [our wiki][github_wiki], with usage and examples.


### Authors

* Carlos Adir Ely Murussi Leite

[numpy_website]: https://numpy.org/doc/
[sympy_website]: https://www.sympy.org/en/index.html
[anaconda_website]: https://www.anaconda.com/
[github_wiki]: https://github.com/carlos-adir/RigidBodyDynamics/wiki