"""Test damped LBFGS linear operators."""

from __future__ import division
import unittest
import numpy as np
from damped_lbfgs import DampedLBFGSOperator, DampedInverseLBFGSOperator
from pykrylov.tools import check_symmetric, check_positive_definite


class TestDampedLBFGSOperator(unittest.TestCase):
    """Test the damped LBFGS linear operators."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        self.B = DampedLBFGSOperator(self.n, self.npairs)
        self.H = DampedInverseLBFGSOperator(self.n, self.npairs)

    def test_init(self):
        """Check that H = B = I initially."""
        assert self.B.insert == 0
        assert self.H.insert == 0
        assert np.allclose(self.B.full(), np.eye(self.n))
        assert np.allclose(self.H.full(), np.eye(self.n))

    def test_structure(self):
        """Test that B and H are spd and inverses of each other."""
        # Insert a few {s,y} pairs.
        for _ in range(self.npairs + 2):
            s = np.random.random(self.n)
            y = np.random.random(self.n)
            self.B.store(s, y)
            self.H.store(s, y)

        assert self.B.insert == 2
        assert self.H.insert == 2

        assert check_symmetric(self.B)
        assert check_symmetric(self.H)
        assert check_positive_definite(self.B)
        assert check_positive_definite(self.H)

        C = self.B * self.H
        assert np.allclose(C.full(), np.eye(self.n))
