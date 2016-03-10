"""Tests relative to AlgoPy."""

from unittest import TestCase
from helper import *
import numpy as np
import pytest

algopy_missing = False

try:
    import algopy
    from nlp.model.algopymodel import AlgopyModel
except:
    algopy_missing = True


algo = pytest.mark.skipif(algopy_missing, reason="requires ALGOPY")


@algo
class AlgopyRosenbrock(AlgopyModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        return algopy.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


@algo
class AlgopyHS7(AlgopyModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def obj(self, x, **kwargs):
        return algopy.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        c = algopy.zeros(1, dtype=x)

        # AlgoPy doesn't support cons_pos()
        c[0] = (1 + x[0]**2)**2 + x[1]**2 - 4
        return c


@algo
class Test_AlgopyRosenbrock(TestCase, Rosenbrock):
    # Test def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        n = 5
        self.model = AlgopyRosenbrock(n, name='Rosenbrock', x0=-np.ones(n))


@algo
class Test_AlgopyHS7(TestCase, Hs7):  # Test def'd in Hs7
    def get_expected(self):
        hs7_data = Hs7Data()
        # AlgoPy doesn't support cons_pos()
        hs7_data.expected_c = np.array([25.])
        return hs7_data

    def get_derivatives(self, model):
        return get_derivatives_plain(model)

    def setUp(self):
        n = 2
        m = 1
        self.model = AlgopyHS7(n,
                               m=m,
                               name='HS7',
                               x0=2 * np.ones(n),
                               pi0=np.ones(m),
                               Lcon=np.zeros(m),
                               Ucon=np.zeros(m))
