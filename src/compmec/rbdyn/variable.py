import sympy as sp
from compmec.rbdyn.__classes__ import timesymb, VariableBaseClass
from compmec.rbdyn.__validation__ import Validation_Variable, Validation_VariableList


class Variable(VariableBaseClass):

    X = []
    names = []
    validation = Validation_Variable

    def __new__(cls, name):
        cls.validation.new(cls, name)
        return cls.__new(name)

    @classmethod
    def __new(cls, name):
        if isinstance(name, int):
            return cls.X[name]
        if name not in cls.names:
            cls.__createnew(name)
        
        index = cls.index(name)
        return cls.X[index]


    @classmethod
    def index(cls, x):
        cls.validation.index(cls, x)
        return cls.__index(x)

    @classmethod
    def __index(cls, x):
        if isinstance(x, str):
            return cls.names.index(x)
        elif isinstance(type(x), sp.core.function.UndefinedFunction):
            return cls.X.index(x)
        else:
            raise TypeError("To get the index")

    @classmethod
    def __createnew(cls, name):
        x = sp.Function(name)(timesymb)
        dx = sp.diff(x, timesymb)
        ddx = sp.diff(dx, timesymb)
        x.dt = dx
        x.ddt = ddx
        cls.X.append(x)
        cls.names.append(name)
        
    @classmethod
    def __contains__(cls, var):
        return var in cls.X

    @classmethod
    def sort(cls, listvars):
        cls.validation.sort(cls, listvars)
        cls.__sort(cls, listvars)

    @classmethod
    def __sort(cls, listvars):
        return VariableList(listvars)



# https://stackoverflow.com/questions/24160831/how-to-create-an-list-of-a-specific-type-but-empty
class VariableList(list):
    validation = Validation_VariableList

    def __init__(self, iterable=None):
        """Override initializer which can accept iterable"""
        self.validation.init(self, iterable)
        super(VariableList, self).__init__()
        self.__insertnewitem(iterable)

    def __insertnewitem(self, item):

        try:
            for it in item:
                self.__insertnewitem(it)
            return
        except TypeError as e:
            pass

        if item not in Variable.X:
            return
            raise ValueError(f'Variable allowed only. Received {type(item)}')
        if item in self:
            return
        if not len(self):
            super(VariableList, self).append(item)
            return

        lastitem = self[len(self)-1]
        lastindex = Variable.index(lastitem)

        indexnewitem = Variable.index(item)
        if lastindex < indexnewitem:
            super(VariableList, self).append(item)
            return

        i = 0
        while True:
            index = Variable.index(self[i])
            if indexnewitem < index:
                break
            i += 1
        super(VariableList, self).insert(i, item)

    def append(self, item):
        self.__insertnewitem(item)

    def insert(self, index, item):
        raise ValueError("You cannot insert in a position. Please use append or +")

    def __add__(self, item):
        self.validation.add(self, item)
        self.__insertnewitem(item)
        return self

    def __iadd__(self, item):
        self.validation.iadd(self, item)
        self.__insertnewitem(item)
        return self


    @classmethod
    def fromexpression(cls, expression):
        cls.validation.fromexpression(cls, expression)
        return cls.__fromexpression(expression)

    @classmethod
    def __fromexpression(cls, expression):
        expression = sp.sympify(expression)
        listvars = list(expression.atoms(sp.Function))
        return cls(listvars)

