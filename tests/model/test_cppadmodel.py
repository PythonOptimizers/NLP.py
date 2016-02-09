# Tests relative to algorithmic differentiation with CppAD.
import sys
try:
    from nlp.model.cppadmodel import CppADModel
except ImportError as exc:
    print "Failed to import: ", exc, "  No tests run!"
    sys.exit(0)

from helper import *
from numpy.testing import *


class CppADRosenbrock(CppADModel):
    "The standard Rosenbrock function."

    def obj(self, x, **kwargs):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class CppADHS7(CppADModel):
    "Problem #7 in the Hock and Schittkowski collection."

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return np.array([(1 + x[0]**2)**2 + x[1]**2])


class Test_CppADRosenbrock(TestCase, Rosenbrock):    # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        self.model = CppADRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))


class Test_CppADHS7(TestCase, Hs7):    # Test def'd in Hs7

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        self.model = CppADHS7(n=2, m=1, name='HS7',
                              x0=2*np.ones(2), pi0=np.ones(1),
                              Lcon=np.array([4.]), Ucon=np.array([4.]))
