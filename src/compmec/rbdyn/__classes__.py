import sympy

timesymb = sympy.symbols("t", real=True, positive=True)
pi = sympy.pi

class VariableBaseClass(object):
	pass

class KinematicBaseClass(object):
    pass

class FrameReferenceBaseClass(object):
    pass

class ObjectBaseClass(object):
    pass

class EnergyBaseClass(object):
    pass

class ForceBaseClass(object):
	pass		