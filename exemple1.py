
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
    l = sp.symbols("l", real=True, positive=True)
    x = sp.Function("x")(t)
    y = sp.Function("y")(t)
    q = sp.Function("q")(t)
    u = sp.Function("u")(t)
    beta = sp.Function("beta")(t)
    dbeta = sp.diff(beta, t)

    ###########################################################################
    #                   Definition of the Reference Frames                    #
    ###########################################################################
    R0 = FrameReference()  # Reference absolut
    R1 = FrameReference(base=R0, translation=(x, y, 0))
    R2 = FrameReference(base=R1, translation=(l / 2, 0, 0))
    R3 = FrameReference(base=R2, rotation=(beta, (0, 0, 1)))

    print(R0)
    print(R1)
    print(R2)
    print(R3)

    ###########################################################################
    #                            Bar calculations                             #
    ###########################################################################
    if 1:
        mass_bar = sp.symbols("m2")
        # CM_bar = (0, 0, 0)
        II_bar = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        II_bar = II_bar * sp.Rational(1, 12) * m_bar * l**2

        bar = Objet(base=R3, name="bar")
        bar.set_mass(mass_bar)
        # bar.set_CM(CM_bar)
        bar.set_II(II_bar)

        E = bar.get_energie_cinetique(R0)
        if 1:  # Print informations
            print("########################")
            print("#          Bar         #")
            print("########################")
            print("E = " + str(E))
            bar.print_cinematique_data(R1)
            bar.print_cinematique_data(R0)
