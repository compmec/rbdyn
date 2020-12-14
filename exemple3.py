
import numpy as np
import sympy as sp
from frames import FrameReference
from objet import Objet

sp.init_printing(use_unicode=True)
t = sp.symbols("t", real=True)

if __name__ == "__main__":

    ###########################################################################
    #                       Definition of the variables                       #
    ###########################################################################
    l, a, g, m, MB, k = sp.symbols("l a g m M_{B} k")

    b = sp.Function("b")(t)
    theta1 = sp.Function("t1")(t)
    theta2 = sp.Function("t2")(t)
    theta3 = sp.Function("t3")(t)

    ###########################################################################
    #                   Definition of the Reference Frames                    #
    ###########################################################################
    R0 = FrameReference()
    R1 = FrameReference(R0, translation=(b, 0, 0))

    R2 = FrameReference(R1, translation=(-l, 0, 0),
                        rotation=(theta1 - sp.pi / 2, (0, 0, 1)))
    R3 = FrameReference(R2, translation=(a, 0, 0))

    R4 = FrameReference(R1, rotation=(theta2 - sp.pi / 2, (0, 0, 1)))
    R5 = FrameReference(R4, translation=(a, 0, 0))

    R6 = FrameReference(R1, translation=(l, 0, 0),
                        rotation=(theta3 - sp.pi / 2, (0, 0, 1)))
    R7 = FrameReference(R6, translation=(a, 0, 0))

    Corpo_MB = Objet(R1, "Corpo MB")
    Corpo_MB.set_mass(MB)
    Corpo_P1 = Objet(R3, "Corpo P1")
    Corpo_P1.set_mass(m)
    Corpo_P2 = Objet(R5, "Corpo P2")
    Corpo_P2.set_mass(m)
    Corpo_P3 = Objet(R7, "Corpo P3")
    Corpo_P3.set_mass(m)

    # Corpo_MB.calcule_elements_speed(R0)
    # Corpo_P1.calcule_elements_speed(R0)
    # Corpo_P2.calcule_elements_speed(R0)
    # Corpo_P3.calcule_elements_speed(R0)

    # Corpo_P1.print_cinematique_data(R0)

    Energ = []
    E = Corpo_MB.get_energie_cinetique(R0)
    Energ.append(E)
    print("Gotten Energie1")
    E = Corpo_P1.get_energie_cinetique(R0)
    Energ.append(E)
    print("Gotten Energie2")
    E = Corpo_P2.get_energie_cinetique(R0)
    Energ.append(E)
    print("Gotten Energie3")
    E = Corpo_P3.get_energie_cinetique(R0)
    Energ.append(E)
    print("Gotten Energie4")
    # Corpo_MB.print_all_data()
    # Corpo_P1.print_all_data()
    # Corpo_P2.print_all_data()
    # Corpo_P3.print_all_data()

    Etotal = 0
    for i, E in enumerate(Energ):
        Etotal += E

    Etotal = sp.expand(Etotal)
    Etotal = sp.simplify(Etotal)
    Etotal = sp.factor(Etotal)

    print("Energie totale: ")
    print(Etotal)
