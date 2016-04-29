"""Tests relative to algorithmic differentiation with ADOL-C."""

from unittest import TestCase
from helper import *
import numpy as np
import pytest


class Test_AdolcRosenbrock(TestCase, Rosenbrock):
    # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("adolc")
        n = 5
        self.model = AdolcRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


class Test_SparseAdolcRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("adolc")
        n = 5
        self.model = SparseRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


class Test_AdolcHs7(TestCase, Hs7):  # Test def'd in Hs7

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("adolc")
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
        pytest.importorskip("adolc")
        n = 2
        m = 1
        self.model = SparseHs7(2,
                               m=m,
                               name='HS7',
                               x0=2 * np.ones(n),
                               pi0=np.ones(m),
                               Lcon=4 * np.ones(m),
                               Ucon=4 * np.ones(m))


class Test_PySparseAdolcRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("adolc")
        pytest.importorskip("pysparse")
        n = 5
        self.model = PySparseRosenbrock(n,
                                        name='Rosenbrock',
                                        x0=-np.ones(n))


class Test_PySparseAdolcHs7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("adolc")
        pytest.importorskip("pysparse")
        n = 2
        m = 1
        self.model = PySparseHs7(2,
                                 m=m,
                                 name='HS7',
                                 x0=2 * np.ones(n),
                                 pi0=np.ones(m),
                                 Lcon=4 * np.ones(m),
                                 Ucon=4 * np.ones(m))


class Test_SciPyAdolcRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("adolc")
        pytest.importorskip("scipy")
        n = 5
        self.model = SciPyRosenbrock(n,
                                     name='Rosenbrock',
                                     x0=-np.ones(n))


class Test_SciPyAdolcHs7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("adolc")
        pytest.importorskip("scipy")
        n = 2
        m = 1
        self.model = SciPyHs7(2,
                              m=m,
                              name='HS7',
                              x0=2 * np.ones(n),
                              pi0=np.ones(m),
                              Lcon=4 * np.ones(m),
                              Ucon=4 * np.ones(m))
