# Dynamics

## Introduction

Rigidy body dynamics problems are problems that are described by the following requirements:

* A inertial frame of reference, that que call ```R0```, in the origin
* Some referentials, which are relatively connected to the frame referential ```R0``` using ```translation``` and ```rotation``` vectors
* Body informations, like the base ```referential```, the ```mass```, position of ```Center of mass``` and the ```inertia``` of rotation

Once described the problem, the code finds the cinematics results, for exemple, energy.

## Exemple 1

The frist exemple, we have a bar of mass ```m```, length ```l```, its center at the position ```(x, y, 0)``` and it's rotationed at an angle ```theta``` relative to the vector ```z```. So, we describe the problem as

```python
# Definition of the variables
t = sp.symbols("t", real=True)  # The time
m = sp.symbols("m", real=True, positive=True)  # The mass
l = sp.symbols("l", real=True, positive=True)  # The length
x = sp.Function("x")(t)
y = sp.Function("y")(t)
theta = sp.Function("theta")(t)  # The rotation angle

# Definition of the frames of reference
R0 = FrameReference()  # Inertial frame of reference
R1 = FrameReference(base=R0, translation=(x, y, 0))
R2 = FrameReference(base=R1, rotation=(theta, "z"))
```

After that, we put the informations about the bar, like ```m```, ```CM``` and ```II```:

```python
# Definition of the object
CM = (0, 0, 0)  # Center of mass
II = (m * l**2 / 12) * np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
bar = Objet(base=R2, name="bar")
bar.set_mass(m)  # Put the mass into the object
bar.set_CM(CM)  # Put the CM into the object
bar.set_II(II)  # Put the inertia tensor into the object

# Calculing the energy relative of the frame R0
E = bar.get_energie_cinetique(R0)
```

And it gives the result

```
[1] E = m*(0.0833333333333333*l**2*Derivative(theta(t), t)**2 + Derivative(x(t), t)**2 + Derivative(y(t), t)**2)/2
```

## Coding and librarys

For implementation, we use Python with the following libraries:

* [Numpy](https://numpy.org/doc/): Used for vector calculs
* [Sympy](https://gmsh.info/): Used to calculate the algebric derivative

To use the codes, you just need these packages and the python installed. The easiest way to do it is using the [Anaconda](https://www.anaconda.com/) that installs everything you need.


### Documentation

Until now, there is no documentation, but further it will be in [our wiki](https://github.com/carlos-adir/Dynamics/wiki), with usage and examples.


### Authors

* Carlos Adir Ely Murussi Leite