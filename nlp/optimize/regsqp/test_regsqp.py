"""Test damped LBFGS linear operators."""

from __future__ import division
import unittest
import numpy as np
from numpy.testing import *

from new_regsqp import RegSQPSolver
import pysparse.sparse.pysparseMatrix as ps
from nlp.model.pysparsemodel import PySparseAmplModel


class TestRegSQP(unittest.TestCase):
    """Test RegSQP solver."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        model = PySparseAmplModel('hs007.nl')
        self.solver = RegSQPSolver(model)

    def test_solve_linear_system(self):
        """Check that H = B = I initially."""

        delta = 1
        K = ps.PysparseMatrix(nrow=3, ncol=3, symmetric=True)
        K[0, 0] = K[1, 1] = K[2, 0] = K[2, 1] = 1
        self.solver.K = K
        self.solver.solve_linear_system(np.array([4., 5., 3.]), 0)
        assert np.allclose(self.solver.LBL.x, np.array([1., 2., 3.]))

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
        grad_phi = self.solver.dphi(x, y, 0, delta, x)

        assert_raises(ValueError, self.solver.backtracking_linesearch,
                      x, y, grad_phi, delta)
