import sympy as sp
from compmec.rbdyn.__classes__ import VariableClass
from compmec.rbdyn import time


class Variable(VariableClass, sp.Function):

    def __new__(cls, name):
        if name in Variable.names:
            index = Variable.names.index(name)
            return Variable.instances[index]

        self = sp.Function(name)(time)
        self.time = time
        self.dt = sp.diff(self, self.time)

        Variable.names.append(name)
        Variable.instances.append(self)
        return self

    @staticmethod
    def sort(listvariables):
        indexs = []
        if isinstance(listvariables, set):
            listvariables = list(listvariables)
        for var in listvariables:
            if isinstance(var, Variable):
                pass
            elif isinstance(var, sp.Function):
                pass
            elif isinstance(var, sp.Derivative):
                pass
            else:
                raise TypeError("The received variable is '%s'" % type(var))

            if var not in Variable.instances:
                error = "The variable %s is not inside Variables instances"
                raise ValueError(error % var)

            newindex = Variable.instances.index(var)
            indexs.append(newindex)
        indexs.sort()
        for i, index in enumerate(indexs):
            listvariables[i] = Variable.instances[index]
        return tuple(listvariables)
