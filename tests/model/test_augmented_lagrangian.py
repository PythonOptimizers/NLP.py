"""Tests relative to :class:`AugmentedLagrangian`."""

import os
from unittest import TestCase
from nlp.model.nlpmodel import NLPModel
from python_models import Rosenbrock
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.tools.dercheck import DerivativeChecker
from nlp.tools.logs import config_logger
import logging
import numpy as np
import pytest

this_path = os.path.dirname(os.path.realpath(__file__))


class SimpleQP(NLPModel):

    def __init__(self, **kwargs):
        super(SimpleQP, self).__init__(
            n=2, m=1, Lcon=np.zeros(1), Ucon=np.zeros(1), **kwargs)

    def obj(self, x):
        return .5 * (x[0]**2 + x[1]**2)

    def grad(self, x):
        n = self.nvar
        g = np.empty(n)
        g[0] = x[0]
        g[1] = x[1]
        return g

    def cons(self, x):
        c = np.empty(self.m)
        c[0] = x[1]**3 + 1
        return c

    def jac(self, x):
        J = np.empty([1, self.nvar])
        J[0, 0] = 0
        J[0, 1] = 3 * x[1]**2
        return J

    def jprod(self, x, v):
        return np.dot(self.jac(x), v)

    def jtprod(self, x, v):
        return np.dot(self.jac(x).T, v)

    def hess(self, x, z=None, *args, **kwargs):

        if z is None:
            z = np.zeros(self.m)
        H = np.empty([self.n, self.n])
        H[0, 0] = 1
        H[0, 1] = H[1, 0] = 0
        H[1, 1] = 1 - 6 * z * x[1]
        return H

    def hprod(self, x, z, v):
        hv = np.empty(self.n)
        hv[0] = v[0]
        hv[1] = (1 - 6 * z * x[1]) * v[1]
        return hv


@pytest.fixture(params=[2, 5, 10])
def rosenbrock(request):
    return AugmentedLagrangian(Rosenbrock(request.param))


def test_rosenbrock(rosenbrock):
    model = rosenbrock.model
    x = np.zeros(model.nvar)
    assert (rosenbrock.nvar == model.nvar)
    assert (rosenbrock.ncon == 0)
    assert np.allclose(rosenbrock.obj(x), model.obj(x))

    assert np.allclose(rosenbrock.grad(x),
                       model.grad(x))

    assert np.allclose(rosenbrock.hess(x, 0).to_array(),
                       model.hess(x, 0))


class test_augmented_lagrangian_simpleqp(TestCase):

    def setUp(self):
        self.model = AugmentedLagrangian(SimpleQP())
        self.model.pi = 3
        self.x = np.array([1, 2], dtype=np.float)

    def test_init(self):
        assert(self.model.nvar == 2)
        assert(self.model.ncon == 0)
        assert(self.model.pi == 3)

    def test_obj(self):
        assert(self.model.obj(self.x) == 380.5)

    def test_grad(self):
        assert np.allclose(self.model.grad(self.x), np.array([1, 1046]))

    def test_hess(self):
        assert np.allclose(self.model.hess(self.x).to_array(),
                           np.array([[1., 0], [0, 2485]]))

    def test_derivatives(self):
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(self.model, self.x)
        dcheck.check(chess=False)
        assert (len(dcheck.grad_errs) == 0)
        assert (len(dcheck.hess_errs) == 0)
