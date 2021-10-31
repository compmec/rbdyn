import numpy as np
import numpy.linalg as la
import sympy as sp
from rbdyn.__classes__ import ObjectClass
from rbdyn.__validation__ import Validation_Object
from rbdyn.kinematic import ObjectKinematic
from rbdyn.frames import FrameReference, FrameComposition


class Object(ObjectClass):

    def __init__(self, baseframe, name=None):
        Validation_Object.init(baseframe, name)
        self.name = name
        self.baseframe = baseframe
        self._id = len(Object.instances)

        self.data = {}
        self.data[self.baseframe] = ObjectKinematic(init=True)

        self.mass = None

        Object.instances.append(self)

    @property
    def id(self):
        return self._id

    @property
    def mass(self):
        return self._mass

    @property
    def CM(self):
        return self.data[self.baseframe].CM

    @property
    def II(self):
        return self.data[self.baseframe].II

    @mass.setter
    def mass(self, value):
        Validation_Object.mass(value)
        self._mass = value

    @CM.setter
    def CM(self, value):
        Validation_Object.CM(value)
        self.data[self.baseframe].CM = value

    @II.setter
    def II(self, value):
        Validation_Object.II(value)
        CM = self.data[self.baseframe].CM
        # PAT is the Parallel Axis Theorem
        if CM is None:
            self.data[self.baseframe].CM = np.zeros(3)
            CM = np.zeros(3)
        PAT = FrameComposition.PAT(CM)
        self.data[self.baseframe].II = value - PAT

    def getCM(self, frame):
        Validation_Object.getCM(frame)
        if frame in self.data:
            CM = self.data[frame].CM
            if CM is not None:
                return CM
        return self.get(frame, "CM")

    def getII(self, frame):
        Validation_Object.getII(frame)
        if frame in self.data:
            II = self.data[frame].II
            if II is not None:
                return II
        return self.get(frame, "II")

    def get(self, frame, element):
        Validation_Object.get(frame, element)
        if frame not in self.data:
            self.__computeElement(frame, element)

        if element == "p":
            return self.data[frame].p
        elif element == "v":
            return self.data[frame].v
        elif element == "a":
            return self.data[frame].a
        elif element == "r":
            return self.data[frame].r
        elif element == "w":
            return self.data[frame].w
        elif element == "q":
            return self.data[frame].q
        elif element == "R":
            return self.data[frame].R
        elif element == "W":
            return self.data[frame].W
        elif element == "Q":
            return self.data[frame].Q
        elif element == "CM":
            return self.data[frame].CM
        elif element == "II":
            return self.data[frame].II

    def __computeElement(self, frame, element):
        listFoward = []
        listBackward = []
        f = self.baseframe
        while f is not None:
            listFoward.insert(0, f)
            f = f.base
        f = frame
        while f is not None:
            listBackward.insert(0, f)
            f = f.base
        while True:
            if len(listFoward) < 2:
                break
            elif len(listBackward) < 2:
                break
            if listFoward[1] != listBackward[1]:
                break
            listFoward.pop(0)
            listBackward.pop(0)

        self.__computeFoward(listFoward, element)
        self.__computeFoward(listBackward, element)

    def __computeFoward(self, listframes, element):
        """
        listframes = [rootframe, ..., branchframe]
        We suppose that branchframe is already know, that we know all the value
        """
        listframes = list(reversed(listframes))
        n = len(listframes)
        for i in range(n - 1):
            frame = listframes[i + 1]
            if frame not in self.data:
                self.data[frame] = ObjectKinematic(init=False)

            kine1 = self.data[listframes[i]]
            kine01 = listframes[i].kine
            kine0 = self.data[listframes[i + 1]]

            if kine0.p is None:
                kine0.p = FrameComposition.p0(kine01, kine1)
            if kine0.R is None:
                kine0.R = FrameComposition.R0(kine01, kine1)
            if element in ("v", "a"):
                if kine0.v is None:
                    kine0.v = FrameComposition.v0(kine01, kine1)
            if element == "a":
                if kine0.a is None:
                    kine0.a = FrameComposition.a0(kine01, kine1)
            if element in ("w", "q"):
                if kine0.w is None:
                    kine0.w = FrameComposition.w0(kine01, kine1)
            if element == "q":
                if kine0.q is None:
                    kine0.q = FrameComposition.q0(kine01, kine1)

    def __computeBackward(self, listframes, element):
        """
        listframes = [rootframe, ..., branchframe]
        We suppose that branchframe is already know, that we know all the value
        """
        n = len(listframes)
        print("n = ", n)
        for i in range(n - 1):
            print("   - loop i = %d" % i)
            frame = listframes[i + 1]
            if frame not in self.data:
                self.data[frame] = ObjectKinematic(init=False)

            kine0 = self.data[listframes[i]]
            kine01 = frame.kine
            kine1 = self.data[listframes[i + 1]]

            if kine1.p is None:
                kine1.p = FrameComposition.p1(kine01, kine0)
            if kine1.R is None:
                kine1.R = FrameComposition.R1(kine01, kine0)
            if element in ("v", "a"):
                if kine1.v is None:
                    kine1.v = FrameComposition.v1(kine01, kine0)
            if element == "a":
                if kine1.a is None:
                    kine1.a = FrameComposition.a1(kine01, kine0)
            if element in ("w", "q"):
                if kine1.w is None:
                    kine1.w = FrameComposition.w1(kine01, kine0)
            if element == "q":
                if kine1.q is None:
                    kine1.q = FrameComposition.q1(kine01, kine0)
