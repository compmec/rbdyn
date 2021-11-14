__import__("pkg_resources").declare_namespace(__name__)
import sympy as sp


time = sp.symbols("t", real=True, positive=True)
