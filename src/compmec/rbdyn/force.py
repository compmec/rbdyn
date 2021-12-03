from compmec.rbdyn.__classes__ import timesymb, ForceBaseClass 
from compmec.rbdyn.__validation__ import Validation_Force 
import numpy as np 
import sympy as sp 
 
 
class Force(ForceBaseClass): 
    def __init__(self, F, X): 
        Validation_Force.init(self, F, X) 
        self.t = timesymb
        self.X = sp.Array(X) 
        self.dX = sp.Array([x.dt for x in self.X]) 
        self.F = sp.Array(F) 
        self.__getDependencies() 
 
    def __getDependencies(self): 
        self.isTDependent = not np.all(np.array(sp.diff(self.F, self.t)) == 0) 
        self.isXDependent = not np.all(np.array(sp.diff(self.F, self.X)) == 0) 
        self.isdXDependent = not np.all(np.array(sp.diff(self.F, self.dX)) == 0) 
 
    def __call__(self, t=None, X=None, dX=None): 
        Validation_Force.call(self, t, X, dX) 
        return self.__call(t, X, dX) 
 
    def __call(self, t, X, dX): 
        F0 = self.F 
        if self.isTDependent: 
            F0 = F0.subs(self.t, t) 
        if self.isXDependent: 
            F0 = F0.subs(self.X, X) 
        if self.isdXDependent: 
            F0 = F0.subs(self.dX, dX) 
        return F0 
         
 
 
 
