"""Tests relative to algorithmic differentiation with CppAD."""

from helper import *
import numpy as np
from unittest import TestCase
import pytest


class Test_CppADRosenbrock(TestCase, Rosenbrock):
    # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("pycppad")
        n = 5
        self.model = CppADRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


class Test_CppADHS7(TestCase, Hs7):  # Test def'd in Hs7
    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("pycppad")
        n = 2
        m = 1
        self.model = CppADHS7(n,
                              m=m,
                              name='HS7',
                              x0=2 * np.ones(n),
                              pi0=np.ones(m),
                              Lcon=4 * np.ones(m),
                              Ucon=4 * np.ones(m))
