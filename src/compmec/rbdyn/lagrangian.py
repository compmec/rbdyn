import sympy as sp 
import numpy as np 
from compmec.rbdyn import time 
from compmec.rbdyn import energy 
from compmec.rbdyn.__validation__ import Validation_LagrangianMatrix, Validation_Lagrangian 
 
 
class LagrangianMatrix: 
 
	validation = Validation_LagrangianMatrix 
 
	@property 
	def X(self): 
		return self._X 
 
	@property 
	def dX(self): 
		if self.X is None: 
			return None 
		return sp.Array([x.dt for x in self.X]) 
	 
	@property 
	def ddX(self): 
		if self.X is None: 
			return None 
		return sp.Array([x.ddt for x in self.X]) 
 
	@property 
	def n(self): 
		if self.X is None: 
			return 0 
		return len(self.X) 
	 
 
	@property 
	def M(self): 
		if self._M is None: 
			return np.zeros([self.n, self.n], dtype="object") 
		return self._M 
 
	@property 
	def C(self): 
		if self._C is None: 
			return np.zeros([self.n, self.n], dtype="object") 
		return self._C 
 
	@property 
	def K(self): 
		if self._K is None: 
			return np.zeros([self.n, self.n], dtype="object") 
		return self._K 
 
	@property 
	def Z(self): 
		if self._M is None: 
			return np.zeros(self.n, dtype="object") 
		return self._Z 
 
	@property 
	def Mvv(self): 
		if self._Mvv is None: 
			return np.zeros([self.n, self.n, self.n], dtype="object")		 
		return self._Mvv 
 
	@property 
	def Mvp(self): 
		if self._Mvp is None: 
			return np.zeros([self.n, self.n, self.n], dtype="object")		 
		return self._Mvp 
 
	@property 
	def Mpp(self): 
		if self._Mpp is None: 
			return np.zeros([self.n, self.n, self.n], dtype="object")		 
		return self._Mpp 
 
	@X.setter 
	def X(self, value): 
		self.validation.Xsetter(self, value) 
		if value is None: 
			self._X = None 
		else: 
			self._X = sp.Array(value) 
 
	@M.setter 
	def M(self, value): 
		self.validation.Msetter(self, value) 
		if value is None: 
			self._M = None 
		else: 
			self._M = sp.Array(value) 
 
	@C.setter 
	def C(self, value): 
		self.validation.Csetter(self, value) 
		if value is None: 
			self._C = None 
		else: 
			self._C = sp.Array(value) 
 
	@K.setter 
	def K(self, value): 
		self.validation.Ksetter(self, value) 
		if value is None: 
			self._K = None 
		else: 
			self._K = sp.Array(value) 
 
	@Z.setter 
	def Z(self, value): 
		self.validation.Zsetter(self, value) 
		if value is None: 
			self._Z = None 
		else: 
			self._Z = sp.Array(value) 
 
	@Mvv.setter 
	def Mvv(self, value): 
		self.validation.Mvvsetter(self, value) 
		if value is None: 
			self._Mvv = None 
		else: 
			self._Mvv = sp.Array(value) 
 
	@Mvp.setter 
	def Mvp(self, value): 
		self.validation.Mvpsetter(self, value) 
		if value is None: 
			self._Mvp = None 
		else: 
			self._Mvp = sp.Array(value) 
		 
	@Mpp.setter 
	def Mpp(self, value): 
		self.validation.Mppsetter(self, value) 
		if value is None: 
			self._Mpp = None 
		else: 
			self._Mpp = sp.Array(value) 
 
		 
 
 
	 
 
class Lagrangian(LagrangianMatrix): 
 
	validation = Validation_Lagrangian 
 
	def __init__(self, energy): 
		self.validation.init(self, energy) 
		self.X = energy.X 
		self.initAllMatrix(energy) 
 
	@property 
	def n(self): 
		if self.X is None: 
			return 0 
		return len(self.X) 
	 
	def __initAllNone(self): 
		self.M = None 
		self.C = None 
		self.K = None 
		self.Z = None 
		self.Mvv = None 
		self.Mvp = None 
		self.Mpp = None 
 
	def initAllMatrix(self, energy): 
		if not self.n: 
			self.__initAllNone() 
		else: 
			self.initM(energy) 
			self.initC(energy) 
			self.initK(energy) 
			self.initZ(energy) 
			self.initMvv(energy) 
			self.initMvp(energy) 
			self.initMpp(energy) 
		 
	def initM(self, energy): 
		if energy.M is None: 
			self.M = None 
			return 
 
		M = sp.Matrix(energy.M) 
		self.M = (M + M.T)/2 
 
 
	def initC(self, energy): 
		C = np.zeros((self.n, self.n), dtype="object") 
 
 
		self.C = C 
 
 
	def initK(self, energy): 
		K = np.zeros((self.n, self.n), dtype="object") 
		if energy.K is not None: 
			Ktemp = sp.Matrix(energy.K) 
			K -= (Ktemp + Ktemp.T)/2 
 
		if energy.B is not None: 
			for i in range(self.n): 
				for j in range(self.n): 
					K[i, j] -= sp.diff(energy.B[j], self.X[i]) 
		for i in range(self.n): 
			for j in range(self.n): 
				K[i, j] = sp.simplify(K[i, j]) 
		self.K = K	 
 
	def initZ(self, energy): 
		Z = np.zeros((self.n), dtype="object") 
		if energy.B is not None: 
			Z = -np.copy(energy.B) 
		if energy.C is not None: 
			for i in range(self.n): 
				Z[i] -= sp.diff(energy.C, self.X[i]) 
		self.Z = Z 
 
	def initMvv(self, energy): 
		Mvv = np.zeros((self.n, self.n, self.n), dtype="object") 
		if energy.M is not None: 
			for i in range(self.n): 
				for j in range(self.n): 
					for k in range(self.n): 
						Mvv[i, j, k] += sp.diff(energy.M[i, j], self.X[k]) 
						Mvv[i, j, k] += sp.diff(energy.M[j, i], self.X[k]) 
						Mvv[i, j, k] -= sp.diff(energy.M[j, k], self.X[i]) 
		self.Mvv = (Mvv/2) 
 
			 
	def initMvp(self, energy): 
		Mvp = np.zeros((self.n, self.n, self.n), dtype="object") 
		if energy.V is not None: 
			for i in range(self.n): 
				for j in range(self.n): 
					for k in range(self.n): 
						Mvp[i, j, k] += sp.diff(energy.V[i, k], self.X[j]) 
						Mvp[i, j, k] -= sp.diff(energy.V[j, k], self.X[i]) 
					 
		self.Mvp = Mvp	 
			 
	def initMpp(self, energy): 
		Mpp = np.zeros((self.n, self.n, self.n), dtype="object") 
		if energy.K is None: 
			self.Mpp = Mpp 
			return 
 
		for i in range(self.n): 
			for j in range(self.n): 
				for k in range(self.n): 
					Mpp[i, j, k] -= sp.diff(energy.K[j, k], self.X[i]) 
		self.Mpp = Mpp 
 
	def __getfromM(self): 
		return np.dot(self.M, self.ddX) 
 
	def __getfromC(self): 
		return np.dot(self.C, self.dX) 
 
	def __getfromK(self): 
		return np.dot(self.K, self.X) 
 
	def __getfromZ(self): 
		return np.copy(self.Z) 
 
	def __getfromMvv(self): 
		return np.dot(np.dot(self.Mvv, self.dX),self.dX) 
 
	def __getfromMvp(self): 
		return np.dot(np.dot(self.Mvp, self.X),self.dX) 
 
	def __getfromMpp(self): 
		return np.dot(np.dot(self.Mpp, self.X),self.X) 
 
	def get(self): 
		Validation_Lagrangian.get(self) 
		return self.__get() 
 
	def __get(self): 
		if not self.n: 
			return None 
		lagrange = self.__getfromM() 
		lagrange += self.__getfromC() 
		lagrange += self.__getfromK() 
		lagrange += self.__getfromZ() 
		lagrange += self.__getfromMvv() 
		lagrange += self.__getfromMvp() 
		lagrange += self.__getfromMpp() 
		return lagrange 
 
 
			 
 
				 
 
 
