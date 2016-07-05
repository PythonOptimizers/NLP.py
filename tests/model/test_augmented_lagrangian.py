"""Tests relative to :class:`AugmentedLagrangian`."""

import os
from unittest import TestCase
from helper import *
from python_models import Rosenbrock, SimpleQP
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.tools.dercheck import DerivativeChecker
from nlp.tools.logs import config_logger
import logging
import numpy as np
import pytest

this_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(params=[2, 5, 10])
def rosenbrock(request):
    return AugmentedLagrangian(Rosenbrock(request.param), prox=1.0)


def test_rosenbrock(rosenbrock):
    model = rosenbrock.model
    x = np.zeros(model.nvar)
    assert (rosenbrock.nvar == model.nvar)
    assert (rosenbrock.ncon == 0)
    assert np.allclose(rosenbrock.obj(x),
                       model.obj(x) + 0.5 * np.linalg.norm(x)**2)

    assert np.allclose(rosenbrock.grad(x),
                       model.grad(x) + x)

    assert np.allclose(rosenbrock.hess(x, 0).to_array(),
                       model.hop(x, 0).to_array() + np.eye(model.nvar))


class test_augmented_lagrangian_simpleqp(TestCase):

    def setUp(self):
        self.model = AugmentedLagrangian(SimpleQP(), prox=1.0)
        self.model.pi = 3
        self.x = np.array([1, 2], dtype=np.float)

    def test_init(self):
        assert(self.model.nvar == 2)
        assert(self.model.ncon == 0)
        assert(self.model.pi == 3)

    def test_obj(self):
        assert self.model.obj(self.x) == 380.5 + 0.5 * np.linalg.norm(self.x)**2

    def test_grad(self):
        assert np.allclose(self.model.grad(self.x),
                           np.array([1, 1046]) + self.x)

    def test_hess(self):
        assert np.allclose(self.model.hess(self.x).to_array(),
                           np.array([[1.,    0],
                                     [0., 2485]]) + np.eye(2))

    def test_derivatives(self):
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(self.model, self.x)
        dcheck.check(chess=False)
        assert (len(dcheck.grad_errs) == 0)
        assert (len(dcheck.hess_errs) == 0)


class AugmentedLagrangianHS7(object):

    def setUp(self):
        raise NotImplementedError("Please subclass.")

    def test_init(self):
        assert self.model.nvar == 2
        assert self.model.ncon == 0
        assert self.model.pi == 3
        assert self.model.penalty == 10
        assert self.model.prox == 1

    def test_obj(self):
        assert np.allclose(self.model.obj(self.x),
                           3049.61 + 0.5 * np.linalg.norm(self.x)**2)

    def test_grad(self):
        assert np.allclose(self.model.grad(self.x),
                           np.array([9880.8, 987]) + self.x)

    def test_hop(self):
        assert np.allclose(self.model.hop(self.x).to_array(),
                           np.array([[28843.8, 1600],
                                     [1600,    654]]) + np.eye(2))

    def test_derivatives(self):
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(self.model, self.x)
        dcheck.check(chess=False)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0


class TestAugmentedLagrangianPySparseAmplModelHS7(AugmentedLagrangianHS7, TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AugmentedLagrangian(PySparseAmplModel(model), prox=1.0)
        self.model.pi = 3
        self.x = np.array([2, 2], dtype=np.float)


class TestAugmentedLagrangianCySparseAmplModelHS7(AugmentedLagrangianHS7, TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AugmentedLagrangian(CySparseAmplModel(model), prox=1.0)
        self.model.pi = 3
        self.x = np.array([2, 2], dtype=np.float)


class TestAugmentedLagrangianScipyAmplModelHS7(AugmentedLagrangianHS7,
                                               TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AugmentedLagrangian(SciPyAmplModel(model),
                                         prox=1.0)
        self.model.pi = 3
        self.x = np.array([2, 2], dtype=np.float)


class AugmentedLagrangianHS10(object):

    def setUp(self):
        raise NotImplementedError("Please subclass.")

    def test_init(self):
        assert self.model.nvar == 3
        assert self.model.ncon == 0
        assert self.model.pi == 3
        assert self.model.penalty == 10
        assert self.model.prox == 1
        assert self.model.Lvar[2] == -1

    def test_setters(self):
        self.model.penalty = 2
        assert self.model.penalty == 2
        self.model.penalty = -2
        assert self.model.penalty == 0
        self.model.penalty = 10  # reset

        self.model.prox = 2
        assert self.model.prox == 2
        self.model.prox = -2
        assert self.model.prox == 0
        self.model.penalty = 1  # reset

    def test_obj(self):
        assert np.allclose(self.model.obj(self.x),
                           432. + 0.5 * np.linalg.norm(self.x)**2)

    def test_grad(self):
        assert np.allclose(self.model.grad(self.x),
                           np.array([745., -1, 93]) + self.x)

    def test_dual_feasibility(self):
        print self.model.model.grad(self.x)

        print self.model.model.jop(self.x).to_array()
        assert np.allclose(self.model.dual_feasibility(self.x),
                           np.array([25., -1., 3.]))

    def test_hop(self):
        assert np.allclose(self.model.hop(self.x).to_array(),
                           np.array([[1198, -186, 80],
                                     [-186,  186,  0],
                                     [80,    0, 10]]) + np.eye(3))

    def test_derivatives(self):
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(self.model, self.x)
        dcheck.check(chess=False)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0


class TestAugmentedLagrangianPySparseAmplModelHS10(AugmentedLagrangianHS10,
                                                   TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = AugmentedLagrangian(PySparseAmplModel(model), prox=1.0)
        self.model.pi = np.array([3.])
        self.x = np.array([2, 2, 1], dtype=np.float)


class TestAugmentedLagrangianCySparseAmplModelHS10(AugmentedLagrangianHS10, TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = AugmentedLagrangian(CySparseAmplModel(model), prox=1.0)
        self.model.pi = np.array([3.])
        self.x = np.array([2, 2, 1], dtype=np.float)


class TestAugmentedLagrangianScipyAmplModelHS10(AugmentedLagrangianHS10,
                                                TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = AugmentedLagrangian(SciPyAmplModel(model), prox=1.0)
        self.model.pi = np.array([3.])
        self.x = np.array([2, 2, 1], dtype=np.float)
