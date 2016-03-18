"""Tests relative to algorithmic differentiation with CppAD."""

from helper import *
import numpy as np

if not module_missing('pycppad'):
    from nlp.model.cppadmodel import CppADModel

    class CppADRosenbrock(CppADModel):
        """The standard Rosenbrock function."""

        def obj(self, x, **kwargs):
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    class CppADHS7(CppADModel):
        """Problem #7 in the Hock and Schittkowski collection."""

        def obj(self, x, **kwargs):
            return np.log(1 + x[0]**2) - x[1]

        def cons(self, x, **kwargs):
            return np.array([(1 + x[0]**2)**2 + x[1]**2])

    class Test_CppADRosenbrock(TestCase, Rosenbrock):
        # Test def'd in Rosenbrock

        def get_derivatives(self, model):
            return get_derivatives_plain(model)

        def setUp(self):
            n = 5
            self.model = CppADRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))

    class Test_CppADHS7(TestCase, Hs7):  # Test def'd in Hs7
        def get_derivatives(self, model):
            return get_derivatives_plain(model)

        def setUp(self):
            n = 2
            m = 1
            self.model = CppADHS7(n,
                                  m=m,
                                  name='HS7',
                                  x0=2 * np.ones(n),
                                  pi0=np.ones(m),
                                  Lcon=4 * np.ones(m),
                                  Ucon=4 * np.ones(m))
