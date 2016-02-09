# Tests relative to algorithmic differentiation with ADOL-C.
import sys
try:
    from nlp.model.adolcmodel import BaseAdolcModel, SparseAdolcModel
    from nlp.model.adolcmodel import PySparseAdolcModel, SciPyAdolcModel
except ImportError as exc:
    print "Failed to import: ", exc, "  No tests run!"
    sys.exit(0)

from numpy.testing import *
from helper import *
import numpy as np


class AdolcRosenbrock(BaseAdolcModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class AdolcHS7(BaseAdolcModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return np.array([(1 + x[0]**2)**2 + x[1]**2])


class SparseRosenbrock(SparseAdolcModel, AdolcRosenbrock):
    pass


class PySparseRosenbrock(PySparseAdolcModel, AdolcRosenbrock):
    pass


class SciPyRosenbrock(SciPyAdolcModel, AdolcRosenbrock):
    pass


class Test_AdolcRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        self.model = AdolcRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))


class Test_AdolcHS7(TestCase, Hs7):  # Test def'd in Hs7

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        self.model = AdolcHS7(n=2, m=1, name='HS7',
                              x0=2*np.ones(2), pi0=np.ones(1),
                              Lcon=np.array([4.]), Ucon=np.array([4.]))
