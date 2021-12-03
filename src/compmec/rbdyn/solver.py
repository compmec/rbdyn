from compmec.rbdyn.__validation__ import Validation_Solver, Validation_Euler 
from compmec.rbdyn.force import Force 
from compmec.rbdyn.energy import Energy 
from compmec.rbdyn.lagrangian import Lagrangian 
import sympy as sp 
import numpy as np 
from numpy import linalg as la 
import datetime 
 
def isMatrixConstant(M, X): 
	if X is None: 
		return True 
 
	try: 
		M = sp.Array(M) 
		X = sp.Array(X) 
		dMdX = sp.diff(M, X) 
		dMdX = np.array(dMdX) 
		return np.all(np.array(dMdX) == 0) 
	except Exception as e: 
		print("Error while verifing that M is constant:") 
		print("Type(M) = %s, type(X) = %s" % (str(type(M)), str(type(X)))) 
		print("Error = %s" % str(e)) 
		raise e 
 
 
class Solver: 
	def __init__(self, E, IC, G, F, timesteps): 
		Validation_Solver.init(self, E, IC, G, F, timesteps) 
		if not isinstance(E, Energy): 
			E = Energy(E) 
		lagrange = Lagrangian(E) 
		self.X = E.X 
 
		self.timesteps = timesteps 
 
		self.n = len(self.X) 
		self.m = len(G) 
		self.__computeG(G) 
		self.__computeDG() 
		self.__computeDDG() 
		self.__computeM(lagrange) 
		self.__computeC(lagrange) 
		self.__computeK(lagrange) 
		self.__computeZ(lagrange) 
		self.__computeMvv(lagrange) 
		self.__computeMvp(lagrange) 
		self.__computeMpp(lagrange) 
		self.__resetAllMatrix() 
		self.__computeF(F) 
		self.__computet0() 
		self.__computeX0(IC) 
		self.__computedX0(IC) 
 
 
	def __computet0(self): 
		self.t0 = self.timesteps[0] 
 
	def __computeX0(self, IC): 
		X0 = np.zeros(self.n) 
		for i, x in enumerate(self.X): 
			X0[i] = IC[x][0] 
		self.X0 = X0 
 
	def __computedX0(self, IC): 
		dX0 = np.zeros(self.n) 
		for i, x in enumerate(self.X): 
			dX0[i] = IC[x][1] 
		self.dX0 = dX0 
 
	def __computeF(self, F): 
		self.F = Force(F, self.X) 
 
	def __computeG(self, G): 
		if G == []: 
			self.G = None 
			self._G0 = np.array([]) 
			self.isGconst = True 
		else: 
			self.G = sp.Array(G) 
			self.isGconst = isMatrixConstant(self.G, self.X) 
			if not self.isGconst: 
				self._G0 = None 
			else: 
				self._G0 = np.array(self.G, dtype="float32") 
 
 
	def __computeDG(self): 
		""" 
		G is a list of lenght (m) 
		this function calculates the gradient of G 
		As we have (n) variables (X), the gradient of G is: 
		[DG]_{ij} = d(G_i)/d(X_j) 
		""" 
		if self.G is None: 
			self.DG = None 
			self._DG0 = np.zeros((0, self.n)) 
			self.isDGconst = True 
		else: 
			self.DG = np.zeros((self.m, self.n)) 
			self.DG = sp.MutableDenseNDimArray(self.DG) 
			for i, g in enumerate(self.G): 
				for j, x in enumerate(self.X): 
					self.DG[i, j] = sp.diff(g, x) 
			self.DG = sp.Array(self.DG) 
			self.isDGconst = isMatrixConstant(self.DG, self.X) 
			if not self.isDGconst: 
				self._DG0 = None 
			else: 
				self._DG0 = np.array(self.DG, dtype="float32") 
 
 
	def __computeDDG(self): 
		""" 
		G is a list of lenght (m) 
		this function calculates the double gradient of G 
		As we have (n) variables (X), the double gradient of G is: 
		[DDG]_{ijk} = d2(G_i)/d(X_j)d(X_k) 
		""" 
		if self.G is None: 
			self.DDG = None 
			self._DDG0 = np.zeros((0, self.n, self.n)) 
			self.isDDGconst = True 
		else: 
			self.DDG = np.zeros((self.m, self.n, self.n)) 
			self.DDG = sp.MutableDenseNDimArray(self.DDG) 
			for i, g in enumerate(self.G): 
				for j, xj in enumerate(self.X): 
					for k, xk in enumerate(self.X): 
						self.DDG[i, j, k] = sp.diff(g, xj, xk) 
			self.DDG = sp.Array(self.DDG) 
			 
			self.isDDGconst = isMatrixConstant(self.DDG, self.X) 
			if not self.isDDGconst: 
				self._DDG0 = None 
			else: 
				self._DDG0 = np.array(self.DDG, dtype="float32") 
 
	def __computeM(self, lagrange): 
		self.M = sp.Array(lagrange.M) 
		self.isMconst = isMatrixConstant(self.M, self.X) 
		if not self.isMconst: 
			self._M0 = None 
		else: 
			self._M0 = np.array(self.M, dtype="float32") 
		self._invM0 = None 
 
	def __computeC(self, lagrange): 
		self.C = sp.Array(lagrange.C) 
		self.isCconst = isMatrixConstant(self.C, self.X) 
		if not self.isCconst: 
			self._C0 = None 
		else: 
			self._C0 = np.array(self.C, dtype="float32") 
 
	def __computeK(self, lagrange): 
		self.K = sp.Array(lagrange.K) 
		self.isKconst = isMatrixConstant(self.K, self.X) 
		if not self.isKconst: 
			self._K0 = None 
		else: 
			self._K0 = np.array(self.K, dtype="float32") 
 
	def __computeZ(self, lagrange): 
		self.Z = sp.Array(lagrange.Z) 
		self.isZconst = isMatrixConstant(self.Z, self.X) 
		if not self.isZconst: 
			self._Z0 = None 
		else: 
			self._Z0 = np.array(self.Z, dtype="float32") 
 
 
	def __computeMvv(self, lagrange): 
		self.Mvv = sp.Array(lagrange.Mvv) 
		self.isMvvconst = isMatrixConstant(self.Mvv, self.X) 
		if not self.isMvvconst: 
			self._Mvv0 = None 
		else: 
			self._Mvv0 = np.array(self.Mvv, dtype="float32") 
 
	def __computeMvp(self, lagrange): 
		self.Mvp = sp.Array(lagrange.Mvp) 
		self.isMvpconst = isMatrixConstant(self.Mvp, self.X) 
		if not self.isMvpconst: 
			self._Mvp0 = None 
		else: 
			self._Mvp0 = np.array(self.Mvp, dtype="float32") 
 
	def __computeMpp(self, lagrange): 
		self.Mpp = sp.Array(lagrange.Mpp) 
		self.isMppconst = isMatrixConstant(self.Mpp, self.X) 
		if not self.isMvvconst: 
			self._Mpp0 = None 
		else: 
			self._Mpp0 = np.array(self.Mpp, dtype="float32") 
 
	def __subsF(self): 
		self._F0 = self.F(t=self.t0, X=self.X0, dX=self.dX0) 
		self._F0 = np.array(self._F0, dtype="float32") 
 
	def __subsG(self): 
		self._G0 = self.G.subs(zip(self.X, self.X0)) 
		self._G0 = np.array(self._G0, dtype="float32") 
 
	def __subsDG(self): 
		self._DG0 = self.DG.subs(zip(self.X, self.X0)) 
		self._DG0 = np.array(self._DG0, dtype="float32") 
 
	def __subsDDG(self): 
		self._DDG0 = self.DDG.subs(zip(self.X, self.X0))	 
		self._DDG0 = np.array(self._DDG0, dtype="float32") 
 
	def __subsM(self): 
		self._M0 = self.M.subs(zip(self.X, self.X0)) 
		self._M0 = np.array(self._M0, dtype="float32") 
		self._invM0 = None 
 
	def __subsC(self): 
		self._C0 = self.C.subs(zip(self.X, self.X0)) 
		self._C0 = np.array(self._C0, dtype="float32") 
 
	def __subsK(self): 
		self._K0 = self.K.subs(zip(self.X, self.X0)) 
		self._K0 = np.array(self._K0, dtype="float32") 
 
	def __subsZ(self): 
		self._Z0 = self.Z.subs(zip(self.X, self.X0)) 
		self._Z0 = np.array(self._Z0, dtype="float32") 
 
	def __subsMvv(self): 
		self._Mvv0 = self.Mvv.subs(zip(self.X, self.X0)) 
		self._Mvv0 = np.array(self._Mvv0, dtype="float32") 
 
	def __subsMvp(self): 
		self._Mvp0 = self.Mvp.subs(zip(self.X, self.X0)) 
		self._Mvp0 = np.array(self._Mvp0, dtype="float32") 
 
	def __subsMpp(self): 
		self._Mpp0 = self.Mpp.subs(zip(self.X, self.X0)) 
		self._Mpp0 = np.array(self._Mpp0, dtype="float32") 
 
	def __resetTDependentMatrix(self): 
		self._F0 = None 
		self._J0 = None 
 
	def __resetXDependentMatrix(self): 
		if not self.isGconst: 
			self._G0 = None 
		if not self.isDGconst: 
			self._DG0 = None 
		if not self.isDDGconst: 
			self._DDG0 = None 
		if not self.isMconst: 
			self._M0 = None 
			self._invM0 = None 
		if not self.isCconst: 
			self._C0 = None 
		if not self.isKconst: 
			self._K0 = None 
		if not self.isZconst: 
			self._Z0 = None 
		if not self.isMvvconst: 
			self._Mvv0 = None 
		if not self.isMvpconst: 
			self._Mvp0 = None 
		if not self.isMppconst: 
			self._Mpp0 = None 
		 
	 
 
	def __resetdXDependentMatrix(self): 
		self._F0 = None 
		self._H0 = None 
		self._J0 = None 
 
	def __resetAllMatrix(self): 
		self.__resetTDependentMatrix() 
		self.__resetXDependentMatrix() 
		self.__resetdXDependentMatrix() 
 
 
 
	@property 
	def t0(self): 
		return self._t0 
 
	@property 
	def X0(self): 
		return self._X0 
 
	@property 
	def dX0(self): 
		return self._dX0 
	 
	@t0.setter 
	def t0(self, value): 
		Validation_Solver.settert0(self, value) 
		self._t0 = value 
		self.__resetTDependentMatrix() 
 
	@X0.setter 
	def X0(self, value): 
		Validation_Solver.setterX0(self, value) 
		self._X0 = value 
		self.__resetXDependentMatrix() 
		# if np.any(np.abs(self.G0) > 1e-3): 
		# 	print("self.G0 = ") 
		# 	print(self.G0) 
		# 	raise ValueError("The constraint function got bigger than 1e-3") 
 
	@dX0.setter 
	def dX0(self, value): 
		Validation_Solver.setterdX0(self, value) 
		self._dX0 = value 
		self.__resetdXDependentMatrix() 
 
	@property 
	def G0(self): 
		if self._G0 is None: 
			self.__subsG() 
		return self._G0 
 
	@property 
	def DG0(self): 
		if self._DG0 is None: 
			self.__subsDG() 
		return self._DG0 
 
	@property 
	def DDG0(self): 
		if self._DDG0 is None: 
			self.__subsDDG() 
		return self._DDG0 
 
	@property 
	def M0(self): 
		if self._M0 is None: 
			self.__subsM() 
		return self._M0 
 
	@property 
	def C0(self): 
		if self._C0 is None: 
			self.__subsC() 
		return self._C0 
 
	@property 
	def K0(self): 
		if self._K0 is None: 
			self.__subsK() 
		return self._K0 
 
	@property 
	def Z0(self): 
		if self._Z0 is None: 
			self.__subsZ() 
		return self._Z0 
	 
 
	@property 
	def Mvv0(self): 
		return self._Mvv0 
 
	@property 
	def Mvp0(self): 
		if self._Mvp0 is None: 
			self.__subsMvp() 
		return self._Mvp0 
 
	@property 
	def Mpp0(self): 
		if self._Mpp0 is None: 
			self.__subsMpp() 
		return self._Mpp0 
 
	@property 
	def invM0(self): 
		if self._invM0 is None: 
			self._invM0 = np.linalg.inv(self.M0) 
		return self._invM0 
 
	@property 
	def F0(self): 
		if self._F0 is None: 
			self.__subsF() 
		return self._F0 
 
 
	@property 
	def H0(self): 
		alpha = 0.5 
		if self._H0 is None: 
			self._H0 = (-alpha**2) * self.G0 
			self._H0 -= 2*alpha * np.dot(self.DG0, self.dX0) 
			self._H0 -= np.dot(np.dot(self.DDG0, self.dX0), self.dX0) 
			self._H0 = np.array(self._H0, dtype="float32") 
		return self._H0 
	 
 
	@property 
	def J0(self): 
		if self._J0 is None: 
			self._J0 = self.F0 
			self._J0 -= self.Z0 
			self._J0 -= np.dot(self.K0, self.X0) 
			self._J0 -= np.dot(self.C0, self.dX0) 
			self._J0 -= np.dot(np.dot(self.Mvv0, self.dX0), self.dX0) 
			self._J0 -= np.dot(np.dot(self.Mvp0, self.X0), self.dX0) 
			self._J0 -= np.dot(np.dot(self.Mpp0, self.X0), self.X0) 
			self._J0 = np.array(self._J0, dtype="float32") 
		return self._J0 
	 
	 
	 
 
 
 
 
class Euler(Solver): 
	def __init__(self, energy, IC, G, F, timesteps): 
		Validation_Euler.init(self, energy, IC, G, F, timesteps) 
		super().__init__(energy, IC, G, F, timesteps) 
		 
 
 
 
 
	def run(self, timeout=None): 
		allX = [self.X0] 
		alldX = [self.dX0] 
 
		if timeout is not None:
			timeout = datetime.timedelta(seconds=timeout)


		finaltime = self.timesteps[-1] 
		initialtime = datetime.datetime.now() 
		i = 0 
		while True: 
			if self.G is None: 
				ddX0 = self.invM0 @ self.J0 
			else: 
				DG0T = np.transpose(self.DG0) 
				ML0 = np.dot(self.DG0, self.invM0) 
				ML0 = np.dot(ML0, DG0T) 
				Y0 = np.dot(np.dot(self.DG0, self.invM0) , self.J0) 
				invML0 = la.inv(ML0) 
				lamda = invML0 @ (Y0 - self.H0) 
				ddX0 = self.invM0 @ (self.J0 - DG0T @ lamda) 
 
			dt = self.timesteps[i+1]-self.timesteps[i] 
			X1 = self.X0 + dt * self.dX0 + ddX0 * (dt**2) /2 
			dX1 = self.dX0 + dt * ddX0 
			X1 = np.array(X1, dtype="float32") 
			dX1 = np.array(dX1, dtype="float32") 
			 
			allX.append(X1) 
			alldX.append(dX1) 
 
			self.t0 = self.t0 + dt 
			if self.t0 >= finaltime: 
				break 
			if timeout is not None: 
				if datetime.datetime.now() - initialtime > timeout: 
					break 
 
			self.X0 = X1 
			self.dX0 = dX1 
			i += 1 
 
		N = len(allX) 
		results = {} 
		for i, x in enumerate(self.X): 
			results[x] = np.zeros((2, N), dtype="float32") 
		for j, (X0, dX0) in enumerate(zip(allX, alldX)): 
			for i, x in enumerate(self.X): 
				results[x][0, j] = X0[i] 
				results[x][1, j] = dX0[i] 
 
		return results   
 
 
 
 
 
 
 
 
class RungeKutta2(Solver): 
	def __init__(self, energy): 
		pass 
 
 
 
 
 
 
 
class RungeKutta4(Solver): 
 
 
	def __init__(self, energy, IC, G, F, timesteps, adaptativetime = True): 
		""" 
		Energy is the Lagrange Energy: 
			E = K - U 
		IC is the initial conditions: 
			for each variable, we know the position and velocity 
		G is the constraint functions 
		F is the force function 
		""" 
		pass 
 
