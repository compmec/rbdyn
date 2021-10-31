# Rigid Body Dynamics

![Tests](https://github.com/compmec/rbdyn/actions/workflows/tests.yml/badge.svg)

## Rigid Body Dynamics problems

We can think a RBDyn problem as:

* We have some ```rigid objects``` in the space
* We know the relation between the objects
* We know the initial condition for each object.
* We want the values in further times:
    * Kinematic: ```position```, ```velocity``` and ```acceleration```
    * Reactions: ```Force``` and ```Moment```

An example is a connecting rod of a motor:

<img src="https://raw.githubusercontent.com/carlos-adir/RigidBodyDynamics/master/docs/img/biela.gif" alt="biela" width="400"/>

So, this library does the simulation and returns the requested values: position, forces and so on.

For more details, please see the page [RBDyn Problems][rbdynproblemlink] and our [Start Guide][startguidelink].


## Requirements and use

As requirements, we use Python with the following libraries:

* [numpy][numpy_website]: Used for vector calculs
* [sympy][sympy_website]: Used to calculate the algebric derivative

To use this library, you just need to install it using the command:

```
pip install rbdyn
```

Another way is clone the repository and installing it manually.

```
git clone https://github.com/compmec/rbdyn
cd rbdyn
pip install -e .
```

For the tests we use [pytest][pytestlink]. Type the command in the main folder

```
pytest
```

## Documentation

All the documentation is in [our wiki][github_wiki], with usage and examples.

## Contribuition

Please refer to the email ```carlos.adir.leite@gmail.com```.
I'm still new using GitHub tools.

[rbdynproblemlink]: https://github.com/compmec/rbdyn/wiki/RBDyn-problem
[wikipedialink]: https://en.wikipedia.org/w/index.php?title=Inertial_frame_of_reference&oldid=1050743548
[startguidelink]: https://github.com/compmec/rbdyn/wiki/Start-Guide
[numpy_website]: https://numpy.org/doc/
[sympy_website]: https://www.sympy.org/en/index.html
[pytestlink]: https://docs.pytest.org/
[anaconda_website]: https://www.anaconda.com/
[github_wiki]: https://github.com/carlos-adir/RigidBodyDynamics/wiki