import numpy as np
import numpy.linalg as la
import sympy as sp
from frames import FrameReference


class FrameComposition:
    way = []

    @staticmethod
    def position(frame, p1):
        t_01 = frame.get_translation()
        R_01 = frame.get_rotation()
        R_p1 = R_01 @ p1
        p0 = t_01 + R_p1
        return p0

    @staticmethod
    def linear_speed(frame, p1, v1):
        v_01 = frame.get_linear_speed()
        R_01 = frame.get_rotation()
        W_01 = frame.get_rotation_speed()

        R_p1 = np.dot(R_01, p1)
        R_v1 = np.dot(R_01, v1)

        v0 = v_01 + np.dot(W_01, R_p1) + R_v1
        return v0

    @staticmethod
    def linear_acceleration(frame, p1, v1, a1):
        a_01 = frame.get_linear_speed()
        R_01 = frame.get_rotation()
        W_01 = frame.get_rotation_speed()
        Q_01 = frame.get_rotation_acceleration()

        R_p1 = R_01 @ p1
        R_v1 = R_01 @ v1
        R_a1 = R_01 @ a1
        W_p1 = W_01 @ R_p1
        W_v1 = W_01 @ R_v1
        Q_p1 = Q_01 @ R_p1
        WW_p1 = W_01 @ W_p1

        a0 = a_01 + Q_p1 + WW_p1 + 2 * W_v1 + R_a1
        return a0

    @staticmethod
    def rotation(frame, r1):
        """
        Not implemented yet
        """
        r0 = np.array([0, 0, 0])
        return r0

    @staticmethod
    def rotation_speed(frame, w1):
        # print("Calculing Rotation speed")
        w_01 = frame.get_vector_rotation_speed()
        R_01 = frame.get_rotation()

        R_w1 = R_01 @ w1

        w0 = w_01 + R_w1
        # print("w0 = " + str(w0))
        return w0

    @staticmethod
    def rotation_acceleration(frame, q1):
        q_01 = frame.get_vector_rotation_acceleration()
        R_01 = frame.get_rotation()

        R_q1 = R_01 @ q1

        q0 = q_01 + R_q1
        return q0

    @staticmethod
    def matrix_inertia(frame, II1):
        # Basically we have that
        # II0 = R_01 @ II1 @ R_01^T
        R_01 = frame.get_rotation()

        II0 = np.dot(II1, np.transpose(R_01))
        II0 = np.array(II0)
        II0 = II0.reshape((3, 3))
        for i in range(3):
            for j in range(3):
                II0[i, j] = sp.simplify(II0[i, j])
        II0 = np.dot(R_01, II0)
        II0 = np.array(II0)
        II0 = II0.reshape((3, 3))
        for i in range(3):
            for j in range(3):
                II0[i, j] = sp.simplify(II0[i, j])
        return II0

    @staticmethod
    def node_is_linked(frame, reference):
        # print("Function - reference = " + str(reference.get_id()))
        # print("  Searching frame = " + str(frame.get_id()))
        if frame == reference:
            return True
        for next_node in reference.next:
            result = FrameComposition.node_is_linked(frame, next_node)
            if result == True:
                return True
        return False

    @staticmethod
    def get_node_base(frame1, frame2, reference=None):
        if reference is None:
            reference = FrameReference.instances[0]
        linked_frame1 = FrameComposition.node_is_linked(frame1, reference)
        linked_frame2 = FrameComposition.node_is_linked(frame2, reference)
        if not (linked_frame1 and linked_frame2):
            node_base = None
        else:
            node_base = reference
            for next_node in reference.next:
                new_node_base = get_node_base(frame1, frame2, next_node)
                if new_node_base is not None:
                    node_base = new_node_base
        return node_base


class Objet:
    instances = []

    def __init__(self, base, name="None"):
        self.name = name
        self.set_base(base)
        self.id = len(Objet.instances)

        self.data = {}
        self.initializate()

        Objet.instances.append(self)

    def set_base(self, base):
        if base not in FrameReference.instances:
            raise Exception("The base is not known")
        self.base_id = base.get_id()
        self.base = base

    def set_mass(self, mass):
        self.mass = mass

    def set_CM(self, CM):
        self.data[self.base]["CM"] = CM

    def set_II(self, II):
        CM = self.__get_element(self.base, "CM")
        mass = self.mass
        PAT = np.eye(3) * np.dot(CM, CM) - np.tensordot(CM, CM, axes=0)
        # PAT is the Parallel Axis Theorem
        self.data[self.base]["II"] = II - mass * PAT

    def initializate(self):
        p = np.array([0, 0, 0])
        v = np.array([0, 0, 0])
        a = np.array([0, 0, 0])
        R = np.eye(3)
        # W = np.zeros((3, 3))
        w = np.array([0, 0, 0])
        # Q = np.zeros((3, 3))
        q = np.array([0, 0, 0])
        CM = np.array([0, 0, 0])
        II = np.zeros((3, 3))

        dicionario = {}
        dicionario["p"] = p
        dicionario["v"] = v
        dicionario["a"] = a
        dicionario["R"] = R
        dicionario["w"] = w
        dicionario["q"] = q
        dicionario["CM"] = CM
        dicionario["II"] = II
        self.data[self.base] = dicionario

    def __calc_element(self, reference, element):
        frame = self.base
        while frame.base is not None and frame != reference.base:
            # frame = frame.base
            if element == "p":
                p1 = self.__get_element(frame, "p")
                p0 = FrameComposition.position(frame, p1)
                value_element = p0
            elif element == "v":
                p1 = self.__get_element(frame, "p")
                v1 = self.__get_element(frame, "v")
                v0 = FrameComposition.linear_speed(frame, p1, v1)
                value_element = v0
            elif element == "a":
                p1 = self.__get_element(frame, "p")
                v1 = self.__get_element(frame, "v")
                a1 = self.__get_element(frame, "a")
                a0 = FrameComposition.linear_acceleration(frame, p1, v1, a1)
                value_element = a0
            elif element == "r":
                raise Exception("Not implemented")
            elif element == "w":
                w1 = self.__get_element(frame, "w")
                w0 = FrameComposition.rotation_speed(frame, w1)
                value_element = w0
            elif element == "q":
                q1 = self.__get_element(frame, "q")
                q0 = FrameComposition.rotation_acceleration(frame, q1)
                value_element = q0
            elif element == "CM":
                CM1 = self.__get_element(frame, "CM")
                CM0 = FrameComposition.position(frame, CM1)
                value_element = CM0
            elif element == "II":
                II1 = self.__get_element(frame, "II")
                II0 = FrameComposition.matrix_inertia(frame, II1)
                value_element = II0
            else:
                raise Exception("Ainda nao implementado")
            frame = frame.base
            if frame not in self.data:
                self.data[frame] = {}
            self.data[frame][element] = value_element
            # print("Writting value_element - frame = " +
            #       str(frame.get_id()) + "element = " + str(element))

    def calcule_elements_position(self, reference):
        self.__calc_element(reference, "p")
        # self.__calc_element(reference, "r")
        self.__calc_element(reference, "CM")
        self.__calc_element(reference, "II")

    def calcule_elements_speed(self, reference):
        self.__calc_element(reference, "v")
        self.__calc_element(reference, "w")

    def calcule_elements_acceleration(self, reference):
        self.__calc_element(reference, "a")
        self.__calc_element(reference, "q")

    def calcule_all_elements(self, reference):
        self.calcule_elements_position(reference)
        self.calcule_elements_speed(reference)
        self.calcule_elements_acceleration(reference)

    def __get_element(self, reference, element):
        if reference in self.data:
            if element in self.data[reference]:
                return self.data[reference][element]
        # print("Calculing with reference = " +
        #       str(reference) + " and element = " + str(element))
        self.__calc_element(reference, element)
        return self.data[reference][element]

    def print_cinematique_data(self, reference):
        if reference not in FrameReference.instances:
            raise Exception("Reference doesn't exist")
        # vetores = self.get_data(frame)
        # t, v, a, r, w, q, CM, I = vetores
        if reference not in self.data:
            self.data[reference] = {}
        print("Data of '" + str(self.name) +
              "' according the frame " + repr(reference))

        print("Translation: ")
        if "p" in self.data[reference]:
            p = self.data[reference]["p"]
            print("p = [" + str(p[0]) + "][" +
                  str(p[1]) + "][" + str(p[2]) + "]")
        if "v" in self.data[reference]:
            v = self.data[reference]["v"]
            print("v = [" + str(v[0]) + "][" +
                  str(v[1]) + "][" + str(v[2]) + "]")
        if "a" in self.data[reference]:
            a = self.data[reference]["a"]
            print("a = [" + str(a[0]) + "][" +
                  str(a[1]) + "][" + str(a[2]) + "]")
        print("Rotation: ")
        if "r" in self.data[reference]:
            r = self.data[reference]["r"]
            print("r = [" + str(r[0]) + "][" +
                  str(r[1]) + "][" + str(r[2]) + "]")
        if "w" in self.data[reference]:
            w = self.data[reference]["w"]
            print("w = [" + str(w[0]) + "][" +
                  str(w[1]) + "][" + str(w[2]) + "]")
        if "q" in self.data[reference]:
            q = self.data[reference]["q"]
            print("q = [" + str(q[0]) + "][" +
                  str(q[1]) + "][" + str(q[2]) + "]")
        print("Center of Mass: ")
        if "CM" in self.data[reference]:
            CM = self.data[reference]["CM"]
            print("CM = [" + str(CM[0]) + "][" +
                  str(CM[1]) + "][" + str(CM[2]) + "]")
        print("Inertia: ")
        if "II" in self.data[reference]:
            II = self.data[reference]["II"]
            print("     [" + str(II[0, 0]) + "][" +
                  str(II[0, 1]) + "][" + str(II[0, 2]) + "]")
            print("II = [" + str(II[1, 0]) + "][" +
                  str(II[1, 1]) + "][" + str(II[1, 2]) + "]")
            print("     [" + str(II[2, 0]) + "][" +
                  str(II[2, 1]) + "][" + str(II[2, 2]) + "]")

    def get_energie_cinetique(self, frame):
        if not frame in FrameReference.instances:
            raise Exception("FrameReference doesn't exist")
        v = self.__get_element(frame, "v")
        w = self.__get_element(frame, "w")
        II = self.__get_element(frame, "II")
        E_c_translation = sp.Rational(1, 2) * self.mass * np.dot(v, v)
        E_c_rotation = sp.Rational(1, 2) * np.dot(w, np.dot(II, w))
        E_c_total = E_c_translation + E_c_rotation
        E_c_total = sp.expand(E_c_total)
        E_c_total = sp.simplify(E_c_total)
        return E_c_total

    def print_all_data(self):
        print("############")
        for key in self.data:
            self.print_cinematique_data(key)
        print("############")

    def print_indices_known(self):
        for key in self.data:
            lista = []
            for var in self.data[key]:
                lista.append(var)
            print({key: lista})
