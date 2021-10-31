import sympy as sp

time = sp.symbols("t")


def newVariable(name):
    if not isinstance(name, str):
        raise TypeError(
            "Name of the variable must be an string. type(name) = " + str(type(name)))
    if " " in name:
        raise ValueError("The name of the variable must not containt space")
    return sp.Function(name)(time)
