# Tests relative to algorithmic differentiation with ADOL-C.

from unittest import TestCase
from helper import *
import numpy as np

if not module_missing('adolc'):
    from nlp.model.adolcmodel import BaseAdolcModel, SparseAdolcModel

    class AdolcRosenbrock(BaseAdolcModel):
        """The standard Rosenbrock function."""

        def obj(self, x, **kwargs):
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    class AdolcHs7(BaseAdolcModel):
        """Problem #7 in the Hock and Schittkowski collection."""

        def obj(self, x, **kwargs):
            return np.log(1 + x[0]**2) - x[1]

        def cons(self, x, **kwargs):
            return np.array([(1 + x[0]**2)**2 + x[1]**2])

    class Test_AdolcRosenbrock(TestCase, Rosenbrock):
        # Test def'd in Rosenbrock

        def get_derivatives(self, model):
            return get_derivatives_plain(model)

        def setUp(self):
            n = 5
            self.model = AdolcRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))

    class SparseRosenbrock(SparseAdolcModel, AdolcRosenbrock):
        pass

    class Test_SparseAdolcRosenbrock(TestCase, Rosenbrock):

        def get_derivatives(self, model):
            return get_derivatives_coord(model)

        def setUp(self):
            n = 5
            self.model = SparseRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))

    class SparseHs7(SparseAdolcModel, AdolcHs7):
        pass

    class Test_AdolcHs7(TestCase, Hs7):  # Test def'd in Hs7

        def get_derivatives(self, model):
            return get_derivatives_plain(model)

        def setUp(self):
            n = 2
            m = 1
            self.model = AdolcHs7(2,
                                  m=m,
                                  name='HS7',
                                  x0=2 * np.ones(n),
                                  pi0=np.ones(m),
                                  Lcon=4 * np.ones(m),
                                  Ucon=4 * np.ones(m))

    class TestSparseAdolcHs7(TestCase, Hs7):

        def get_derivatives(self, model):
            return get_derivatives_coord(model)

        def setUp(self):
            n = 2
            m = 1
            self.model = SparseHs7(2,
                                   m=m,
                                   name='HS7',
                                   x0=2 * np.ones(n),
                                   pi0=np.ones(m),
                                   Lcon=4 * np.ones(m),
                                   Ucon=4 * np.ones(m))

    if not module_missing('pysparse'):
        from nlp.model.adolcmodel import PySparseAdolcModel

        class PySparseRosenbrock(PySparseAdolcModel, AdolcRosenbrock):
            pass

        class Test_PySparseAdolcRosenbrock(TestCase, Rosenbrock):

            def get_derivatives(self, model):
                return get_derivatives_llmat(model)

            def setUp(self):
                n = 5
                self.model = PySparseRosenbrock(n,
                                                name='Rosenbrock',
                                                x0=-np.ones(n))

        class PySparseHs7(PySparseAdolcModel, AdolcHs7):
            pass

        class Test_PySparseAdolcHs7(TestCase, Hs7):

            def get_derivatives(self, model):
                return get_derivatives_llmat(model)

            def setUp(self):
                n = 2
                m = 1
                self.model = PySparseHs7(2,
                                         m=m,
                                         name='HS7',
                                         x0=2 * np.ones(n),
                                         pi0=np.ones(m),
                                         Lcon=4 * np.ones(m),
                                         Ucon=4 * np.ones(m))

    if not module_missing('scipy'):
        from nlp.model.adolcmodel import SciPyAdolcModel

        class SciPyRosenbrock(SciPyAdolcModel, AdolcRosenbrock):
            pass

        class Test_SciPyAdolcRosenbrock(TestCase, Rosenbrock):

            def get_derivatives(self, model):
                return get_derivatives_scipy(model)

            def setUp(self):
                n = 5
                self.model = SciPyRosenbrock(n,
                                             name='Rosenbrock',
                                             x0=-np.ones(n))

        class SciPyHs7(SciPyAdolcModel, AdolcHs7):
            pass

        class Test_SciPyAdolcHs7(TestCase, Hs7):

            def get_derivatives(self, model):
                return get_derivatives_scipy(model)

            def setUp(self):
                n = 2
                m = 1
                self.model = SciPyHs7(2,
                                      m=m,
                                      name='HS7',
                                      x0=2 * np.ones(n),
                                      pi0=np.ones(m),
                                      Lcon=4 * np.ones(m),
                                      Ucon=4 * np.ones(m))
