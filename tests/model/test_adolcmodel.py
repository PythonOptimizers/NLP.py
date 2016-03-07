# Tests relative to algorithmic differentiation with ADOL-C.
import sys
from unittest import TestCase
from numpy.testing import dec
from .helper import *
import numpy as np


if not module_missing('adolc'):
    from nlp.model.adolcmodel import BaseAdolcModel, SparseAdolcModel

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

    if not module_missing('pysparse'):
        from nlp.model.adolcmodel import PySparseAdolcModel
        class PySparseRosenbrock(PySparseAdolcModel, AdolcRosenbrock):
            pass

    if not module_missing('scipy'):
        from nlp.model.adolcmodel import SciPyAdolcModel
        class SciPyRosenbrock(SciPyAdolcModel, AdolcRosenbrock):
            pass


class Test_AdolcRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    @dec.skipif(module_missing('adolc'), "Test skipped because ADOL-C is not available.")
    def setUp(self):
        n = 5
        self.model = AdolcRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


class Test_AdolcHS7(TestCase, Hs7):  # Test def'd in Hs7

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    @dec.skipif(module_missing('adolc'), "Test skipped because ADOL-C is not available.")
    def setUp(self):
        n = 2
        m = 1
        self.model = AdolcHS7(2, m=m, name='HS7',
                              x0=2 * np.ones(n), pi0=np.ones(m),
                              Lcon=4 * np.ones(m), Ucon=4 * np.ones(m))

if __name__ == '__main__':

    import unittest
    unittest.main()
