"""Tests relative to AlgoPy."""

from unittest import TestCase
from helper import *
import numpy as np
import pytest


class Test_AlgopyRosenbrock(TestCase, Rosenbrock):
    # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("algopy")
        n = 5
        self.model = AlgopyRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


class Test_AlgopyHS7(TestCase, Hs7):  # Test def'd in Hs7
    def get_expected(self):
        hs7_data = Hs7Data()
        # AlgoPy doesn't support cons_pos()
        hs7_data.expected_c = np.array([25.])
        return hs7_data

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        pytest.importorskip("algopy")
        n = 2
        m = 1
        self.model = AlgopyHS7(n,
                               m=m,
                               name='HS7',
                               x0=2 * np.ones(n),
                               pi0=np.ones(m),
                               Lcon=np.zeros(m),
                               Ucon=np.zeros(m))
