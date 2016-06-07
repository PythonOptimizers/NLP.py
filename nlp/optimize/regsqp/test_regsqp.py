"""Test damped LBFGS linear operators."""

from __future__ import division
from unittest import TestCase
import numpy as np
import pytest
import os
import pysparse.sparse.pysparseMatrix as ps
try:
    from nlp.model.pysparsemodel import PySparseAmplModel
except ImportError:
    pass
from nlp.tools.dercheck import DerivativeChecker

from new_regsqp import RegSQPSolver

this_path = os.path.dirname(os.path.realpath(__file__))


class TestAugmentedLagrangianMeritFunction(TestCase):
    """"""

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")

    def test_derivatives(self):
        model_name = os.path.join(this_path, "..", "..", "..",
                                  "tests", "model", 'hs007.nl')
        model = PySparseAmplModel(model_name)
        dcheck = DerivativeChecker(model, model.x0, tol=1e-4)
        dcheck.check()
        dcheck.check(cheap_check=True, hess=True)
        assert (len(dcheck.grad_errs) == 0)
        assert (len(dcheck.hess_errs) == 0)
        assert (len(dcheck.cheap_grad_errs) == 0)


class TestRegSQP(TestCase):
    """Test RegSQP solver."""

    def setUp(self):
        """Initialize."""
        pytest.importorskip("nlp.model.amplmodel")
        self.n = 10
        self.npairs = 5
        model = PySparseAmplModel('hs007.nl')
        self.solver = RegSQPSolver(model)

    def test_solve_linear_system(self):
        """Check that H = B = I initially."""

        delta = 1
        K = ps.PysparseMatrix(nrow=3, ncol=3, symmetric=True)
        K[0, 0] = K[2, 0] = 1
        K[2, 2] = -1
        self.solver.K = K
        np.testing.assert_raises(
            ValueError, self.solver.solve_linear_system, np.array([4., 5., 3.]), 0)

        self.solver.solve_linear_system(np.array([4., 5., 3.]), 0,
                                        np.array([1., 0.]))
        print self.solver.LBL.inertia
        print self.solver.K
        K = ps.PysparseMatrix(nrow=3, ncol=3, symmetric=True)
        K[0, 0] = K[2, 0] = 1
        self.solver.K = K
        status, short_status, finished, nb_bump, dx, dy = self.solver.solve_linear_system(
            np.array([4. + self.solver.rho_min, 2 * self.solver.rho_min, 1.]), delta, np.array([1., 0.]))
        assert self.solver.LBL.isFullRank == True
        assert self.solver.K[1, 1] == self.solver.rho_min
        assert finished == False
        assert np.allclose(self.solver.LBL.x, np.array([1., 2., 3.]))

        K = ps.PysparseMatrix(nrow=3, ncol=3, symmetric=True)
        K[0, 0] = K[1, 1] = 1
        self.solver.K = K
        status, short_status, finished, nb_bump, dx, dy = self.solver.solve_linear_system(
            np.array([4. + self.solver.rho_min, 2 * self.solver.rho_min, 1.]), delta, np.array([1., 1.]))
        assert self.solver.LBL.isFullRank == False
        assert short_status == "degn"
        assert finished == True

    def test_backtracking_linesearch(self):
        x = self.solver.model.x0
        y = np.ones(self.solver.model.m)
        delta = 1
        grad_phi = self.solver.merit_function.grad(x, y, 0, delta, x)

        np.testing.assert_raises(ValueError, self.solver.backtracking_linesearch,
                                 x, y, grad_phi, delta)
