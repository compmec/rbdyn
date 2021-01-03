
import numpy as np
import sympy as sp
import sys
sys.path.append("../src/")
from frames import FrameReference
from objet import Objet

sp.init_printing(use_unicode=True)
t = sp.symbols("t", real=True)

if __name__ == "__main__":

    ###########################################################################
    #                       Definition of the variables                       #
    ###########################################################################
    M, L, a, m = sp.symbols("M L a m", real=True, positive=True)
    rho = sp.Function("rho")(t)
    phi = sp.Function("phi")(t)
    theta = sp.Function("theta")(t)

    ###########################################################################
    #                   Definition of the Reference Frames                    #
    ###########################################################################
    R0 = FrameReference()
    R1 = FrameReference(R0, rotation=(phi, (0, 0, 1)))
    R2 = FrameReference(R1, translation=(rho, 0, 0),
                        rotation=(theta, (1, 0, 0)))
    print(R0)
    print(R1)
    print(R2)

    ###########################################################################
    #                            Bar calculations                             #
    ###########################################################################
    if 1:
        mass_bar = M
        CM_bar = sp.Rational(1, 2) * np.array([L, 0, 0])
        II_bar = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        II_bar = II_bar * sp.Rational(1, 3) * M * L**2

        bar = Objet(base=R1, name="bar")
        bar.set_mass(mass_bar)
        bar.set_CM(CM_bar)
        bar.set_II(II_bar)
        if 1:  # Print informations
            print("########################")
            print("#          Bar         #")
            print("########################")
            # E = bar.get_energie_cinetique(R0)
            # print("E = " + str(E))
            bar.print_cinematique_data(R1)
            bar.print_cinematique_data(R0)

    ###########################################################################
    #                           Plate calculations                            #
    ###########################################################################
    if 1:

        mass_plate = m
        CM_plate = (0, a, 0)
        II_plate = np.array([[1, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        II_plate = II_plate * sp.Rational(1, 3) * m * a**2

        plate = Objet(base=R2, name="plate")
        plate.set_mass(mass_plate)
        plate.set_CM(CM_plate)
        plate.set_II(II_plate)

        # E = plate.get_energie_cinetique(R0)
        if 1:  # Print informations
            print("########################")
            print("#        Plaque        #")
            print("########################")
            # print("E = " + str(E))
            plate.print_cinematique_data(R2)
            plate.print_cinematique_data(R1)
            plate.print_cinematique_data(R0)
